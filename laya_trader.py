"""laya-trader live: Laya MLX reads market signals and a rule layer decides
LONG / SHORT / CLOSE / HOLD every interval. Paper positions only: no API keys, no orders.

Open http://127.0.0.1:8765 for the live dashboard.
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

from core import (
    Account,
    Cache,
    ask_laya,
    compute_signals,
    describe,
    explain,
    funding_times_between,
    get_json,
    load_agent,
)

HERE = Path(__file__).resolve().parent
DASHBOARD = HERE / "dashboard.html"


def fetch(coin, config, cache):
    spot, futures = config["binance"]["spot"], config["binance"]["futures"]
    ttl = config["refresh"]["context_seconds"]
    symbol = f"{coin}USDT"
    klines = get_json(
        f"{spot}/api/v3/klines?symbol={symbol}&interval={config['kline_interval']}&limit=100"
    )
    closes_4h = cache.get(
        f"{coin}:4h",
        ttl,
        lambda: [
            float(k[4])
            for k in get_json(f"{spot}/api/v3/klines?symbol={symbol}&interval=4h&limit=60")
        ],
        default=[],
    )
    funding = cache.get(
        f"{coin}:funding",
        ttl,
        lambda: float(
            get_json(f"{futures}/fapi/v1/premiumIndex?symbol={symbol}")["lastFundingRate"]
        ),
    )
    long_short = cache.get(
        f"{coin}:ls",
        ttl,
        lambda: float(
            get_json(
                f"{futures}/futures/data/globalLongShortAccountRatio"
                f"?symbol={symbol}&period=5m&limit=1"
            )[0]["longShortRatio"]
        ),
    )
    return compute_signals(klines, closes_4h, funding, long_short)


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

    def __init__(self, config, keep=900, keep_decisions=2000):
        self.lock = threading.Lock()
        self.keep_decisions = keep_decisions
        # Full context per decision, fetched on demand when a log row is clicked.
        self.details = {c: OrderedDict() for c in config["symbols"]}
        self.data = {
            "mode": "live",
            "title": "Laya trader · live paper trading",
            "rules": config["strategy"],
            "capital": config["paper"]["capital_usdt"],
            "interval_seconds": config["interval_seconds"],
            "coins": {
                c: {
                    "points": deque(maxlen=keep),
                    "trades": deque(maxlen=200),
                    "decisions": deque(maxlen=keep_decisions),
                    "last": None,
                }
                for c in config["symbols"]
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
                t_ms = int(datetime.fromisoformat(row["at"]).timestamp() * 1000)
                entry = self.data["coins"][row["coin"]]
            except (ValueError, KeyError):
                continue
            entry["decisions"].append(decision_row(t_ms, row))
            self.remember(row["coin"], t_ms, row)
            count += 1
        return count

    def remember(self, coin, t_ms, row):
        details = self.details[coin]
        details[t_ms] = row
        while len(details) > self.keep_decisions:
            details.popitem(last=False)

    def detail(self, coin, t_ms):
        with self.lock:
            return self.details.get(coin, {}).get(t_ms)

    def json(self):
        with self.lock:
            coins = {
                c: {
                    **v,
                    "points": list(v["points"]),
                    "trades": list(v["trades"]),
                    "decisions": list(v["decisions"]),
                }
                for c, v in self.data["coins"].items()
            }
            return json.dumps({**self.data, "coins": coins}).encode()


class BacktestRunner:
    """Runs backtest.py as a separate process: it loads its own model copy, so the live
    loop's model is never shared between threads."""

    def __init__(self, config_path):
        self.config_path = config_path.resolve()
        self.report = HERE / "backtest.html"
        self.lock = threading.Lock()
        self.process, self.days, self.started, self.output = None, None, None, deque(maxlen=20)
        self.exit_code = None

    def start(self, days):
        with self.lock:
            if self.process and self.process.poll() is None:
                return False
            self.days, self.started, self.exit_code = days, time.time(), None
            self.output.clear()
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(HERE / "backtest.py"),
                    "--days",
                    str(days),
                    "--no-open",
                    "--config",
                    str(self.config_path),
                    "--out",
                    str(self.report),
                ],
                cwd=HERE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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
            path = urlsplit(self.path).path
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
                query = parse_qs(urlsplit(self.path).query)
                try:
                    row = live.detail(query["coin"][0], int(query["t"][0]))
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
                days = min(max(float(query.get("days", ["1"])[0]), 0.05), 30)
            except ValueError:
                self.send_error(400, "days must be a number")
                return
            started = runner.start(days)
            self.reply(
                json.dumps({"started": started}).encode(),
                "application/json",
                code=202 if started else 409,
            )

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def round_trip(agent, config, cache, accounts, pool, live, log):
    coins = config["symbols"]
    fear_greed = cache.get(
        "fng",
        config["refresh"]["fear_greed_seconds"],
        lambda: int(get_json("https://api.alternative.me/fng/?limit=1")["data"][0]["value"]),
    )
    started = time.perf_counter()
    data = dict(zip(coins, pool.map(lambda c: fetch(c, config, cache), coins)))
    fetch_ms = (time.perf_counter() - started) * 1000

    stamp = datetime.now(timezone.utc)
    now = stamp.timestamp()
    t_ms = int(now * 1000)
    previous_ms = live.data.get("last_ms") or t_ms
    started = time.perf_counter()
    lines = []
    for coin, s in data.items():
        state = describe(s, fear_greed)
        laya = ask_laya(agent, state)
        p = laya["p"]
        account = accounts[coin]
        for _ in funding_times_between(previous_ms, t_ms):
            account.pay_funding(s["funding"], s["price"])
        before = account.position
        action, reason, fill = account.decide(p, s, now)
        trade = account.apply(action, reason, fill, s, now)
        equity = account.equity(s["price"])
        change = (equity / account.start - 1) * 100
        held = account.position
        status = "flat"
        if held:
            side = "long" if held["side"] > 0 else "short"
            move = held["side"] * (s["price"] / held["entry"] - 1) * 100
            status = f"{side} {held['leverage']:.1f}x {move:+.2f}%"
        lines.append(
            f"{coin:<5} {s['price']:>12,.4f}  rsi {s['rsi14']:5.1f}  votes {sum(s['votes']):+d}  "
            f"P(bullish) {p:.2f} -> {action:<5} ({reason:<14}) {status:<20} "
            f"equity {equity:>9.2f} ({change:+.2f}%)"
        )
        with live.lock:
            entry = live.data["coins"][coin]
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
                "fear_greed": fear_greed,
            }
        row = {
            "at": stamp.isoformat(),
            "coin": coin,
            "signals": s,
            "fear_greed": fear_greed,
            "state": state,
            "p_bullish": p,
            "action": action,
            "reason": reason,
            "trade": trade,
            "laya": {k: v for k, v in laya.items() if k != "p"},
            "strategy": explain(p, config["strategy"], before, action, reason),
            "position_before": before,
            "position": held,
            "equity": equity,
        }
        with live.lock:
            live.data["coins"][coin]["decisions"].append(decision_row(t_ms, row))
            live.remember(coin, t_ms, row)
        if log:
            log.write(json.dumps(row) + "\n")
    infer_ms = (time.perf_counter() - started) * 1000
    if log:
        log.flush()  # saved as it happens, not only on exit
    with live.lock:
        live.data["timing"] = {"fetch_ms": fetch_ms, "laya_ms": infer_ms, "fear_greed": fear_greed}
        live.data["last_ms"] = t_ms
    print(
        f"{stamp:%H:%M:%S}  fear&greed {fear_greed}  fetch {fetch_ms:.0f} ms  "
        f"laya {infer_ms:.0f} ms  trades {sum(len(a.trades) for a in accounts.values())}\n"
        + "\n".join(lines)
        + "\n",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--rounds", type=int, help="Stop after this many rounds")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(f"logs/decisions-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"),
        help="Append every decision as JSON lines (default: logs/decisions-<UTC date>.jsonl)",
    )
    parser.add_argument("--no-log", action="store_true", help="Do not save decisions")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    config["symbols"] = [c.upper() for c in config["symbols"]]

    agent = load_agent(config["model"])
    cache, live = Cache(), Live(config)
    accounts = {c: Account(config["strategy"], config["paper"]) for c in config["symbols"]}
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
        with ThreadPoolExecutor(max_workers=max(4, len(config["symbols"]))) as pool:
            while args.rounds is None or done < args.rounds:
                tick = time.monotonic()
                try:
                    round_trip(agent, config, cache, accounts, pool, live, log)
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
