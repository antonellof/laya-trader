"""Backtest laya-trader on historical Binance candles and write an HTML report.

Replays closed candles one by one with the same signals, Laya question, strategy and
paper account as the live loop, and compares against two baselines on the same data:
  rules only  - the four trend votes instead of Laya (sum >= +2 / <= -2), same strategy
  buy & hold  - buy at the first candle, hold to the end
Fear & Greed (daily) and funding (8-hourly) use their historical values. The long/short
ratio has no long history on Binance and is left out, so the backtest sees slightly less
than the live loop. Stops and take profits trigger on the candle's high/low and fill at
their level (a gap past the level is not modelled).
"""

import argparse
import json
import sys
import time
import tomllib
import webbrowser
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

from core import (
    Account,
    ask_laya,
    compute_signals,
    describe,
    explain,
    funding_times_between,
    get_json,
    load_agent,
)

DASHBOARD = Path(__file__).with_name("dashboard.html")
PLACEHOLDER = "/*__DATA__*/null"
INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
WARMUP = 100  # candles of history each decision sees, as in the live loop


def klines(spot, symbol, interval, start_ms, end_ms):
    rows = []
    while start_ms < end_ms:
        batch = get_json(
            f"{spot}/api/v3/klines?symbol={symbol}&interval={interval}"
            f"&startTime={start_ms}&endTime={end_ms}&limit=1000",
            timeout=10,
        )
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
    return rows


def funding_history(futures, symbol, start_ms, end_ms):
    rows, cursor = [], start_ms - 86_400_000
    try:
        while cursor < end_ms:
            batch = get_json(
                f"{futures}/fapi/v1/fundingRate?symbol={symbol}"
                f"&startTime={cursor}&endTime={end_ms}&limit=1000",
                timeout=10,
            )
            if not batch:
                break
            rows.extend((r["fundingTime"], float(r["fundingRate"])) for r in batch)
            cursor = batch[-1]["fundingTime"] + 1
    except Exception:
        pass
    return rows


def fear_greed_history(days):
    try:
        rows = get_json(f"https://api.alternative.me/fng/?limit={days + 2}", timeout=10)["data"]
        return sorted((int(r["timestamp"]) * 1000, int(r["value"])) for r in rows)
    except Exception:
        return []


def value_at(series, t):
    """Last known value at time t from [(time, value)], or None."""
    i = bisect_right(series, (t, float("inf")))
    return series[i - 1][1] if i else None


def prepare(config, coin, interval, start_ms, end_ms, agent, memo, quiet=False):
    """Signals and Laya's P for every closed candle. Expensive part, done once."""
    spot, futures = config["binance"]["spot"], config["binance"]["futures"]
    step = INTERVAL_MS[interval]
    symbol = f"{coin}USDT"
    if not quiet:
        days = (end_ms - start_ms) / 86_400_000
        print(f"{coin}: downloading {days:g} day(s) of {interval} candles...", file=sys.stderr)
    fast = klines(spot, symbol, interval, start_ms - WARMUP * step, end_ms)
    slow = klines(spot, symbol, "4h", start_ms - 60 * 14_400_000, end_ms)
    funding = funding_history(futures, symbol, start_ms, end_ms)
    sentiment = fear_greed_history(int((end_ms - start_ms) / 86_400_000) + 1)
    steps, j, calls = [], 0, 0
    for i in range(WARMUP, len(fast)):
        close_ms = fast[i][6] + 1
        while j < len(slow) and slow[j][6] < close_ms:
            j += 1
        closes_4h = [float(k[4]) for k in slow[max(0, j - 60) : j]]
        s = compute_signals(fast[i - WARMUP + 1 : i + 1], closes_4h, value_at(funding, close_ms))
        state = describe(s, value_at(sentiment, close_ms))
        if state not in memo:
            memo[state] = ask_laya(agent, state)
            calls += 1
        votes = sum(s["votes"])
        steps.append(
            {
                "t": close_ms,
                "s": s,
                "state": state,
                "p": memo[state]["p"],
                "laya": memo[state],
                "rule": 1.0 if votes >= 2 else 0.0 if votes <= -2 else 0.5,
            }
        )
        if not quiet and (i - WARMUP) % 1000 == 0:
            print(f"  {coin}: {i - WARMUP}/{len(fast) - WARMUP} candles", file=sys.stderr)
    if not quiet:
        print(f"  {coin}: {len(steps)} candles, {calls} new Laya calls", file=sys.stderr)
    return {"steps": steps, "funding": funding}


def simulate(prepared, strategy, paper, score_key="p", record=False):
    """Run one strategy over prepared candles. Cheap: no network, no model."""
    account = Account(strategy, paper)
    funding, equities, decisions = prepared["funding"], [], []
    previous_t = None
    for step in prepared["steps"]:
        s, now = step["s"], step["t"] / 1000
        if previous_t is not None:
            for t in funding_times_between(previous_t, step["t"]):
                account.pay_funding(value_at(funding, t), s["price"])
        previous_t = step["t"]
        before = account.position
        action, reason, fill = account.decide(step[score_key], s, now, s["high"], s["low"])
        trade = account.apply(action, reason, fill, s, now)
        equity = account.equity(s["price"])
        equities.append(equity)
        if record and trade:
            detail = {
                "signals": s,
                "laya": {k: v for k, v in step["laya"].items() if k != "p"},
                "strategy": explain(step[score_key], strategy, before, action, reason),
                "position_before": before,
                "trade": trade,
                "equity": equity,
                "fill": "candle high/low"
                if reason in ("stop loss", "take profit")
                else "candle close",
            }
            decisions.append(
                [
                    step["t"],
                    fill,
                    round(step[score_key], 4),
                    action,
                    reason,
                    step["state"],
                    equity,
                    detail,
                ]
            )
    return account, equities, decisions


def buy_and_hold(prepared, paper):
    steps, fee = prepared["steps"], paper["fee_pct"] / 100
    first = steps[0]["s"]["price"]
    return [paper["capital_usdt"] * (1 - fee) * step["s"]["price"] / first for step in steps]


def max_drawdown(equities):
    peak, worst = equities[0], 0.0
    for value in equities:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return worst * 100


def stats(account, equities, start):
    closes = [t for t in account.trades if t["action"] == "CLOSE"] if account else []
    return {
        "return_pct": (equities[-1] / start - 1) * 100,
        "max_drawdown_pct": max_drawdown(equities),
        "trades": len(account.trades) if account else 1,
        "win_rate": sum(t["pnl"] > 0 for t in closes) / len(closes) if closes else None,
        "funding": account.funding_paid if account else 0.0,
    }


def report(prepared, config):
    paper, strategy = config["paper"], config["strategy"]
    laya, laya_eq, decisions = simulate(prepared, strategy, paper, "p", record=True)
    rules, rules_eq, _ = simulate(prepared, strategy, paper, "rule")
    hold_eq = buy_and_hold(prepared, paper)
    steps = prepared["steps"]
    points = [
        [st["t"], st["s"]["price"], round(st["p"], 4), round(a, 4), round(b, 4), round(c, 4)]
        for st, a, b, c in zip(steps, laya_eq, rules_eq, hold_eq)
    ]

    def timed(account):
        return [
            [round(t["at"] * 1000), t["action"], t["price"], t["reason"]] for t in account.trades
        ]

    start = paper["capital_usdt"]
    return {
        "points": points,
        "trades": timed(laya),
        "trades_rules": timed(rules),
        "decisions": decisions,
        "summary": {
            "Laya": stats(laya, laya_eq, start),
            "Rules only": stats(rules, rules_eq, start),
            "Buy & hold": stats(None, hold_eq, start),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--days", type=float, default=1.0)
    parser.add_argument("--end", help="UTC end date, YYYY-MM-DD (default: now)")
    parser.add_argument("--out", type=Path, default=Path("backtest.html"))
    parser.add_argument("--json", type=Path, help="Also write the raw results here")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    coins = [c.upper() for c in config["symbols"]]
    interval = config["kline_interval"]
    if interval not in INTERVAL_MS:
        parser.error(f"kline_interval must be one of {', '.join(INTERVAL_MS)} for backtests")
    end = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc)
    )
    end_ms = int(end.timestamp() * 1000)
    start_ms = int(end_ms - args.days * 86_400_000)

    agent, memo = load_agent(config["model"]), {}
    started = time.perf_counter()
    results = {
        coin: report(prepare(config, coin, interval, start_ms, end_ms, agent, memo), config)
        for coin in coins
    }
    st = config["strategy"]
    setup = (
        f"{st['market']}, {st['direction']}, max {st['max_leverage'] if st['market'] == 'futures' else 1}x, "
        f"risk {st['risk_pct']}% per trade"
    )
    data = {
        "mode": "backtest",
        "days": args.days,
        "title": f"Laya trader · backtest · {args.days:g} day(s) of {interval} candles to "
        f"{end:%Y-%m-%d %H:%M} UTC · {setup}",
        "rules": st,
        "capital": config["paper"]["capital_usdt"],
        "fee_pct": config["paper"]["fee_pct"],
        "coins": results,
    }
    page = DASHBOARD.read_text().replace(
        PLACEHOLDER, json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    )
    args.out.write_text(page)
    if args.json:
        args.json.write_text(json.dumps(data))

    print(f"\n{setup} · {time.perf_counter() - started:.0f} s")
    print(f"{'coin':<6}{'strategy':<12}{'return':>9}{'max DD':>9}{'trades':>8}{'win rate':>10}")
    for coin, result in results.items():
        for name, row in result["summary"].items():
            win = f"{row['win_rate'] * 100:.0f}%" if row["win_rate"] is not None else "-"
            print(
                f"{coin:<6}{name:<12}{row['return_pct']:>+8.2f}%{row['max_drawdown_pct']:>+8.2f}%"
                f"{row['trades']:>8}{win:>10}"
            )
    print(f"\nReport: {args.out.resolve()}")
    if not args.no_open:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
