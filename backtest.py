"""Backtest laya-trader on historical candles and write an HTML report.

Replays closed candles one by one with the same signals, Laya question, strategy and
paper account as the live loop, for every market in config.toml, and compares against:
  rules only  - the four trend votes in place of Laya (sum >= +2 / <= -2), same strategy
  buy & hold  - buy at the first candle, hold to the end
Every candle sees only data that existed when it closed. Crypto futures statistics
(open interest, long/short, taker ratio) exist for the last 30 days only; stocks have
15-minute history for about 60 days. Whale alerts and headlines are never included.
Stops and take profits trigger on the candle's high/low and fill at their level.
"""

import argparse
import json
import sys
import time
import tomllib
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from core import (
    QUESTION,
    Account,
    ask_laya,
    compute_signals,
    describe,
    explain,
    funding_times_between,
    load_agent,
)
from markets import INTERVAL_MS, load_markets, value_at

HERE = Path(__file__).resolve().parent
DASHBOARD = HERE / "dashboard.html"
PLACEHOLDER = "/*__DATA__*/null"
WARMUP = 100  # candles of history each decision sees, as in the live loop


def prompt_of(config):
    prompt = config.get("prompt", {})
    return prompt.get("question", QUESTION), prompt.get("format", "good_bad")


def prepare(market, symbol, interval, start_ms, end_ms, agent, memo, prompt, quiet=False):
    """Signals and Laya's P for every closed candle in [start, end]. The expensive part,
    done once; identical states reuse Laya's earlier answer."""
    question, fmt = prompt
    if not quiet:
        days = (end_ms - start_ms) / 86_400_000
        print(f"{symbol}: downloading {days:g} day(s) of {interval} candles...", file=sys.stderr)
    history = market.history(symbol, interval, start_ms, end_ms, WARMUP)
    fast = history["fast"]
    steps, calls = [], 0
    for i in range(WARMUP, len(fast)):
        close_ms = fast[i][6] + 1
        if close_ms < start_ms or close_ms > end_ms:
            continue
        s = compute_signals(fast[i - WARMUP + 1 : i + 1], history["ctx_at"](close_ms))
        state = describe(s, market.noun, fmt)
        key = (state, question)
        if key not in memo:
            memo[key] = ask_laya(agent, state, question)
            calls += 1
        votes = sum(s["votes"])
        steps.append(
            {
                "t": close_ms,
                "s": s,
                "state": state,
                "p": memo[key]["p"],
                "laya": memo[key],
                "rule": 1.0 if votes >= 2 else 0.0 if votes <= -2 else 0.5,
            }
        )
    if market.kind == "stocks":  # mark the last candle of each session
        days = [datetime.fromtimestamp(st["t"] / 1000, timezone.utc).date() for st in steps]
        for step, day, next_day in zip(steps, days, days[1:] + [None]):
            step["s"]["closes_soon"] = next_day is not None and next_day != day
    if not quiet:
        print(f"  {symbol}: {len(steps)} candles, {calls} new Laya calls", file=sys.stderr)
    return {"steps": steps, "funding": history["funding"]}


def simulate(prepared, strategy, paper, score_key="p", record=False, candle_seconds=900):
    """Run one strategy over prepared candles. Cheap: no network, no model."""
    account = Account(strategy, paper, candle_seconds)
    funding, equities, decisions = prepared["funding"], [], []
    previous_t = None
    for step in prepared["steps"]:
        s, now = step["s"], step["t"] / 1000
        if previous_t is not None and funding:
            for t in funding_times_between(previous_t, step["t"]):
                account.pay_funding(value_at(funding, t), s["price"])
        previous_t = step["t"]
        before = account.position
        action, reason, fill = account.decide(step[score_key], s, now, s["high"], s["low"])
        trade = account.apply(action, reason, fill, s, now)
        equity = account.equity(s["price"])
        equities.append(equity)
        account.exposure_steps = getattr(account, "exposure_steps", 0) + bool(account.position)
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


def paper_of(config, market):
    return {"capital_usdt": config["paper"]["capital_usdt"], "fee_pct": market.cfg["fee_pct"]}


def report(prepared, strategy, paper, candle_seconds):
    laya, laya_eq, decisions = simulate(prepared, strategy, paper, "p", True, candle_seconds)
    rules, rules_eq, _ = simulate(prepared, strategy, paper, "rule", False, candle_seconds)
    hold_eq = buy_and_hold(prepared, paper)
    points = [
        [st["t"], st["s"]["price"], round(st["p"], 4), round(a, 4), round(b, 4), round(c, 4)]
        for st, a, b, c in zip(prepared["steps"], laya_eq, rules_eq, hold_eq)
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


def setup_text(strategy, interval):
    lev = strategy["max_leverage"] if strategy["market"] == "futures" else 1
    extra = []
    if strategy.get("trend_filter", "none") != "none":
        extra.append("only with the higher-timeframe trend")
    if strategy.get("flat_at_close"):
        extra.append("flat at the close")
    if strategy.get("max_hold_candles"):
        extra.append(f"exit after {strategy['max_hold_candles']} candles")
    return ", ".join(
        [f"{interval} candles", strategy["market"], strategy["direction"], f"max {lev:g}x", *extra]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--end", help="UTC end date, YYYY-MM-DD (default: now)")
    parser.add_argument("--markets", help="Comma-separated: crypto,stocks (default: all)")
    parser.add_argument("--out", type=Path, default=HERE / "backtest.html")
    parser.add_argument("--json", type=Path, help="Also write the raw results here")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    markets = load_markets(config)
    if args.markets:
        wanted = {m.strip() for m in args.markets.split(",")}
        markets = [m for m in markets if m.kind in wanted]
    end = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc)
    )
    end_ms = int(end.timestamp() * 1000)
    start_ms = int(end_ms - args.days * 86_400_000)

    agent, memo, prompt = load_agent(config["model"]), {}, prompt_of(config)
    started = time.perf_counter()
    assets, market_info = {}, {}
    for market in markets:
        if market.interval not in INTERVAL_MS:
            parser.error(f"{market.kind}.kline_interval must be one of {', '.join(INTERVAL_MS)}")
        strategy, paper = market.cfg["strategy"], paper_of(config, market)
        market_info[market.kind] = {
            "label": market.label,
            "rules": strategy,
            "fee_pct": paper["fee_pct"],
            "interval": market.interval,
            "setup": setup_text(strategy, market.interval),
        }
        for symbol in market.symbols:
            try:
                prepared = prepare(
                    market, symbol, market.interval, start_ms, end_ms, agent, memo, prompt
                )
            except Exception as error:
                print(f"  {symbol}: skipped ({error})", file=sys.stderr)
                continue
            if not prepared["steps"]:
                print(f"  {symbol}: no candles in the window", file=sys.stderr)
                continue
            assets[f"{market.kind}:{symbol}"] = {
                "market": market.kind,
                "symbol": symbol,
                **report(prepared, strategy, paper, INTERVAL_MS[market.interval] / 1000),
            }

    data = {
        "mode": "backtest",
        "days": args.days,
        "title": f"Backtest · {args.days:g} day(s) to {end:%Y-%m-%d %H:%M} UTC",
        "capital": config["paper"]["capital_usdt"],
        "prompt": {"question": prompt[0], "format": prompt[1]},
        "markets": market_info,
        "assets": assets,
    }
    page = DASHBOARD.read_text().replace(
        PLACEHOLDER, json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    )
    args.out.write_text(page)
    if args.json:
        args.json.write_text(json.dumps(data))

    print(f"\n{time.perf_counter() - started:.0f} s")
    print(f"{'asset':<14}{'strategy':<12}{'return':>9}{'max DD':>9}{'trades':>8}{'win rate':>10}")
    for asset_id, result in assets.items():
        for name, row in result["summary"].items():
            win = f"{row['win_rate'] * 100:.0f}%" if row["win_rate"] is not None else "-"
            print(
                f"{asset_id:<14}{name:<12}{row['return_pct']:>+8.2f}%"
                f"{row['max_drawdown_pct']:>+8.2f}%{row['trades']:>8}{win:>10}"
            )
    print(f"\nReport: {args.out.resolve()}")
    if not args.no_open:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
