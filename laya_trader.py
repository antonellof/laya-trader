"""laya-trader live: Laya MLX reads market signals for crypto and S&P 500 stocks, and each
market's strategy decides LONG / SHORT / CLOSE / HOLD every interval.
Paper positions only: no API keys, no orders.

Open http://127.0.0.1:8765 for the live dashboard (backtests at /backtest).
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import tomllib
import webbrowser
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from backtest import memory_of, paper_of, prompt_of, setup_text
from core import (
    Account,
    Cache,
    ask_laya,
    compute_signals,
    describe,
    explain,
    funding_times_between,
    load_agent,
    memory_text,
)
from markets import INTERVAL_MS, load_markets

HERE = Path(__file__).resolve().parent
DASHBOARD = HERE / "dashboard.html"


def keep(entry, row, t_ms):
    """Keep a decision in the on-screen log only if it's a trade or something changed
    (what Laya read, or the reason). Unchanged HOLDs would otherwise push trades out
    within minutes; every round is still in the saved log file."""
    rows, new = entry["decisions"], decision_row(t_ms, row)
    if rows and new[3] == "HOLD" and rows[-1][5] == new[5] and rows[-1][4] == new[4]:
        return False
    rows.append(new)
    return True


def decision_row(t_ms, row):
    """[time, price, P(bullish), action, reason, what Laya read, equity]"""
    return [
        t_ms,
        row["signals"]["price"],
        row["p_bullish"],
        row["action"],
        row["reason"],
        row["state"],
        row["equity"],
    ]


class Live:
    """Everything the dashboard shows, guarded by one lock."""

    def __init__(self, config, markets, keep=900, keep_decisions=2000):
        self.lock = threading.Lock()
        self.keep_decisions = keep_decisions
        ids = [f"{m.kind}:{s}" for m in markets for s in m.symbols]
        # Full context per decision, fetched on demand when a log row is clicked.
        self.details = {i: OrderedDict() for i in ids}
        self.data = {
            "mode": "live",
            "title": "Live paper trading",
            "capital": config["paper"]["capital_usdt"],
            "interval_seconds": config["interval_seconds"],
            "prompt": dict(zip(("question", "format"), prompt_of(config))),
            "markets": {
                m.kind: {
                    "memory_trades": memory_of(config, m),
                    "label": m.label,
                    "rules": m.cfg["strategy"],
                    "fee_pct": m.cfg["fee_pct"],
                    "interval": m.interval,
                    "setup": setup_text(m.cfg["strategy"], m.interval),
                    "open": m.is_open(),
                }
                for m in markets
            },
            "assets": {
                f"{m.kind}:{s}": {
                    "market": m.kind,
                    "symbol": s,
                    "points": deque(maxlen=keep),
                    "trades": deque(maxlen=200),
                    "decisions": deque(maxlen=keep_decisions),
                    "last": None,
                }
                for m in markets
                for s in m.symbols
            },
        }

    def preload(self, path):
        """Show earlier decisions from the saved log, so a restart keeps the history."""
        if not path or not path.exists():
            return 0
        count = 0
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                asset = row.get("asset") or f"crypto:{row['coin']}"
                t_ms = int(datetime.fromisoformat(row["at"]).timestamp() * 1000)
                entry = self.data["assets"][asset]
            except (ValueError, KeyError):
                continue
            if keep(entry, row, t_ms):
                self.remember(asset, t_ms, row)
            count += 1
        return count

    def remember(self, asset, t_ms, row):
        details = self.details[asset]
        details[t_ms] = row
        while len(details) > self.keep_decisions:
            details.popitem(last=False)

    def detail(self, asset, t_ms):
        with self.lock:
            return self.details.get(asset, {}).get(t_ms)

    def json(self):
        with self.lock:
            assets = {
                a: {
                    **v,
                    "points": list(v["points"]),
                    "trades": list(v["trades"]),
                    "decisions": list(v["decisions"]),
                }
                for a, v in self.data["assets"].items()
            }
            return json.dumps({**self.data, "assets": assets}).encode()


class BacktestRunner:
    """Runs backtest.py as a separate process: it loads its own model copy, so the live
    loop's model is never shared between threads."""

    def __init__(self, config_path):
        self.config_path = config_path.resolve()
        self.report = HERE / "backtest.html"
        self.lock = threading.Lock()
        self.process, self.days, self.started, self.output = None, None, None, deque(maxlen=20)
        self.exit_code = None

    def start(self, days, markets=None):
        with self.lock:
            if self.process and self.process.poll() is None:
                return False
            self.days, self.started, self.exit_code = days, time.time(), None
            self.output.clear()
            command = [
                sys.executable,
                str(HERE / "backtest.py"),
                "--days",
                str(days),
                "--no-open",
                "--config",
                str(self.config_path),
                "--out",
                str(self.report),
            ]
            if markets:
                command += ["--markets", markets]
            self.process = subprocess.Popen(
                command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            threading.Thread(target=self._read, args=(self.process,), daemon=True).start()
            return True

    def _read(self, process):
        for line in process.stdout:
            if line.strip():
                self.output.append(line.rstrip())
        self.exit_code = process.wait()

    def status(self):
        running = bool(self.process and self.process.poll() is None)
        return {
            "running": running,
            "days": self.days,
            "started": self.started,
            "exit_code": None if running else self.exit_code,
            "output": list(self.output),
            "report_time": self.report.stat().st_mtime if self.report.exists() else None,
        }


def serve(live, port, log_path=None, runner=None):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, kind, code=200, extra=()):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            for name, value in extra:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path, query = urlsplit(self.path).path, parse_qs(urlsplit(self.path).query)
            if path == "/state.json":
                self.reply(live.json(), "application/json")
            elif path == "/decisions.jsonl" and log_path and log_path.exists():
                disposition = f'attachment; filename="{log_path.name}"'
                self.reply(
                    log_path.read_bytes(),
                    "application/x-ndjson",
                    extra=[("Content-Disposition", disposition)],
                )
            elif path in ("/", "/index.html"):
                self.reply(DASHBOARD.read_bytes(), "text/html; charset=utf-8")
            elif path in ("/backtest", "/backtest/"):
                page = runner.report if runner and runner.report.exists() else DASHBOARD
                self.reply(page.read_bytes(), "text/html; charset=utf-8")
            elif path == "/decision":
                try:
                    row = live.detail(query["asset"][0], int(query["t"][0]))
                except (KeyError, ValueError):
                    row = None
                if row is None:
                    self.send_error(404, "decision not kept")
                else:
                    self.reply(json.dumps(row).encode(), "application/json")
            elif path == "/backtest/status" and runner:
                self.reply(json.dumps(runner.status()).encode(), "application/json")
            else:
                self.send_error(404)

        def do_POST(self):
            query = parse_qs(urlsplit(self.path).query)
            if urlsplit(self.path).path != "/backtest/run" or not runner:
                self.send_error(404)
                return
            try:
                days = min(max(float(query.get("days", ["7"])[0]), 0.05), 55)
            except ValueError:
                self.send_error(400, "days must be a number")
                return
            markets = query.get("markets", [None])[0]
            started = runner.start(days, markets)
            self.reply(
                json.dumps({"started": started}).encode(),
                "application/json",
                202 if started else 409,
            )

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class Laya:
    """Laya with a small cache: an unchanged state (common between stock polls) reuses the
    last answer instead of running the model again."""

    def __init__(self, agent, question, size=4096):
        self.agent, self.question, self.cache, self.size = agent, question, OrderedDict(), size

    def __call__(self, state):
        if state in self.cache:
            self.cache.move_to_end(state)
            return self.cache[state], True
        answer = ask_laya(self.agent, state, self.question)
        self.cache[state] = answer
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return answer, False


def round_trip(laya, fmt, markets, config, cache, accounts, pool, live, log):
    stamp = datetime.now(timezone.utc)
    now = stamp.timestamp()
    t_ms = int(now * 1000)
    previous_ms = live.data.get("last_ms") or t_ms
    context = config.get("context", {})

    started = time.perf_counter()
    jobs = []
    for market in markets:
        extras = market.extras(cache, context)
        for symbol in market.symbols:
            jobs.append((market, symbol, pool.submit(market.snapshot, symbol, cache, extras)))
    snapshots = []
    for market, symbol, job in jobs:
        try:
            klines, ctx = job.result()
            if len(klines) > 30:
                snapshots.append((market, symbol, compute_signals(klines[-100:], ctx)))
        except Exception as error:  # one asset failing must not stop the others
            print(f"{market.kind}:{symbol} skipped: {error}", file=sys.stderr)
    fetch_ms = (time.perf_counter() - started) * 1000

    started, calls, lines = time.perf_counter(), 0, []
    for market, symbol, s in snapshots:
        asset = f"{market.kind}:{symbol}"
        account = accounts[asset]
        state = describe(s, market.noun, fmt)
        memory = memory_of(config, market)
        if memory:  # the asset's recent trades and their outcome, in words
            state = f"{state} {memory_text(account, memory, s['price'], now)}"
        answer, reused = laya(state)
        calls += not reused
        p = answer["p"]
        if market.cfg["strategy"].get("signal") == "votes":  # rules instead of Laya
            votes = sum(s["votes"])
            p = 1.0 if votes >= 2 else 0.0 if votes <= -2 else 0.5
        if s.get("funding") is not None:
            for _ in funding_times_between(previous_ms, t_ms):
                account.pay_funding(s["funding"], s["price"])
        before = account.position
        if market.is_open(now):
            action, reason, fill = account.decide(p, s, now)
        else:
            action, reason, fill = "HOLD", "market closed", s["price"]
        trade = account.apply(action, reason, fill, s, now, p)
        equity = account.equity(s["price"])
        held = account.position
        status = "flat"
        if held:
            move = held["side"] * (s["price"] / held["entry"] - 1) * 100
            status = (
                f"{'long' if held['side'] > 0 else 'short'} {held['leverage']:.1f}x {move:+.2f}%"
            )
        lines.append(
            f"{asset:<13} {s['price']:>11,.4f}  P {p:.2f} -> {action:<5} ({reason:<14}) {status:<20} "
            f"equity {equity:>9.2f} ({(equity / account.start - 1) * 100:+.2f}%)"
        )
        strategy = market.cfg["strategy"]
        row = {
            "at": stamp.isoformat(),
            "asset": asset,
            "market": market.kind,
            "coin": symbol,
            "signals": s,
            "state": state,
            "p_bullish": p,
            "action": action,
            "reason": reason,
            "trade": trade,
            "laya": {k: v for k, v in answer.items() if k != "p"},
            "strategy": explain(p, strategy, before, action, reason),
            "position_before": before,
            "position": held,
            "equity": equity,
        }
        with live.lock:
            entry = live.data["assets"][asset]
            entry["points"].append([t_ms, s["price"], p, equity])
            if action != "HOLD":
                entry["trades"].append([t_ms, action, fill, reason])
            entry["last"] = {
                "state": state,
                "p": p,
                "action": action,
                "reason": reason,
                "position": held,
                "signals": s,
            }
            if keep(entry, row, t_ms):
                live.remember(asset, t_ms, row)
        if log:
            log.write(json.dumps(row) + "\n")
    infer_ms = (time.perf_counter() - started) * 1000
    if log:
        log.flush()  # saved as it happens, not only on exit
    with live.lock:
        live.data["timing"] = {"fetch_ms": fetch_ms, "laya_ms": infer_ms, "laya_calls": calls}
        live.data["last_ms"] = t_ms
        for market in markets:
            live.data["markets"][market.kind]["open"] = market.is_open(now)
    trades = sum(len(a.trades) for a in accounts.values())
    print(
        f"{stamp:%H:%M:%S}  fetch {fetch_ms:.0f} ms  laya {infer_ms:.0f} ms ({calls} calls)  "
        f"trades {trades}\n" + "\n".join(lines) + "\n",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--rounds", type=int, help="Stop after this many rounds")
    parser.add_argument(
        "--log",
        type=Path,
        default=HERE / "logs" / f"decisions-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl",
        help="Append every decision as JSON lines (default: logs/decisions-<UTC date>.jsonl)",
    )
    parser.add_argument("--no-log", action="store_true", help="Do not save decisions")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    markets = load_markets(config)
    question, fmt = prompt_of(config)

    laya = Laya(load_agent(config["model"]), question)
    cache, live = Cache(), Live(config, markets)
    accounts = {
        f"{m.kind}:{s}": Account(
            m.cfg["strategy"], paper_of(config, m), INTERVAL_MS[m.interval] / 1000
        )
        for m in markets
        for s in m.symbols
    }
    log = None
    if not args.no_log:
        loaded = live.preload(args.log)
        if loaded:
            print(f"Loaded {loaded} earlier decisions from {args.log}", file=sys.stderr)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log = args.log.open("a")
        print(f"Saving decisions to {args.log}", file=sys.stderr)
    if not args.no_dashboard:
        serve(live, args.port, None if args.no_log else args.log, BacktestRunner(args.config))
        url = f"http://127.0.0.1:{args.port}"
        print(f"Dashboard: {url}  ·  backtest: {url}/backtest", file=sys.stderr)
        if not args.no_open:
            webbrowser.open(url)
    print("Paper trading only: no orders are sent anywhere. Ctrl-C to stop.\n", file=sys.stderr)
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            while args.rounds is None or done < args.rounds:
                tick = time.monotonic()
                try:
                    round_trip(laya, fmt, markets, config, cache, accounts, pool, live, log)
                except Exception as error:  # A network hiccup skips one round, not the run.
                    print(f"round skipped: {error}", file=sys.stderr)
                done += 1
                time.sleep(max(0.0, config["interval_seconds"] - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if log:
            log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
