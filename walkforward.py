"""Walk-forward search: pick a strategy on older data, then test it on newer data it never saw.

Per market, signals and Laya's P are computed once over the whole window. Every strategy
in the grid is simulated on the first part (train); the best few by average return across
assets are then run on the last part (test), next to buy & hold. A strategy that only
looks good on the train part was fitted to noise.

The grid encodes what the backtests taught so far: fees dominate (so fewer, stronger
trades: stricter thresholds, longer cooldowns), tight stops hurt dip-buying (so wider,
time-based or no stops, exit when the signal flips), and markets drift (so an optional
filter that only trades with the higher-timeframe trend).
"""

import argparse
import itertools
import statistics
import sys
import time
import tomllib
from pathlib import Path

from backtest import buy_and_hold, max_drawdown, paper_of, prepare, prompt_of, simulate
from core import load_agent
from markets import INTERVAL_MS, load_markets

HERE = Path(__file__).resolve().parent
GRID = {
    "score": ["p", "rule"],  # Laya's P, or the four trend votes alone
    "direction": ["trend", "reversion"],
    "thresholds": [(0.55, 0.30), (0.65, 0.20), (0.75, 0.15)],
    "trend_filter": ["none", "htf"],
    # (stop ATR, take-profit ATR, max hold candles); 0 = off. The last exits on flip only.
    "exits": [(1.5, 3.0, 0), (3.0, 0, 16), (0, 0, 16), (0, 0, 0)],
    "cooldown_candles": [1, 4, 16],
}
SETUPS = {"crypto": [("spot", 1.0), ("futures", 1.0)], "stocks": [("spot", 1.0)]}
DEFAULT_DAYS = {"crypto": (20, 10), "stocks": (37, 18)}  # train, test


def split(prepared, cut_ms):
    def part(keep):
        return {
            "steps": [s for s in prepared["steps"] if keep(s["t"])],
            "funding": prepared["funding"],
        }

    return part(lambda t: t <= cut_ms), part(lambda t: t > cut_ms)


def evaluate(parts, strategy, paper, score, candle_seconds):
    rows = []
    for prepared in parts.values():
        if not prepared["steps"]:
            continue
        account, equities, _ = simulate(prepared, strategy, paper, score, False, candle_seconds)
        rows.append(
            (
                (equities[-1] / paper["capital_usdt"] - 1) * 100,
                max_drawdown(equities),
                len(account.trades),
            )
        )
    return {
        "return": statistics.mean(r[0] for r in rows),
        "worst": min(r[0] for r in rows),
        "drawdown": min(r[1] for r in rows),
        "trades": sum(r[2] for r in rows),
    }


def label(choice):
    (above, below), (stop, take, hold) = choice["thresholds"], choice["exits"]
    market, lev = choice["setup"]
    who = "Laya" if choice["score"] == "p" else "votes"
    exits = f"sl{stop:g}/tp{take:g}/h{hold}"
    filt = "htf" if choice["trend_filter"] == "htf" else "any"
    return (
        f"{who:<5} {choice['direction']:<9} {above:.2f}/{below:.2f} {filt:<3} {exits:<14} "
        f"cd{choice['cooldown_candles']:<2} {market}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--markets", help="Comma-separated: crypto,stocks (default: all)")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    markets = load_markets(config)
    if args.markets:
        markets = [m for m in markets if m.kind in {x.strip() for x in args.markets.split(",")}]
    agent, memo, prompt = load_agent(config["model"]), {}, prompt_of(config)
    end_ms = int(time.time() * 1000)

    for market in markets:
        train_days, test_days = DEFAULT_DAYS[market.kind]
        cut_ms = int(end_ms - test_days * 86_400_000)
        start_ms = int(cut_ms - train_days * 86_400_000)
        paper, base = paper_of(config, market), market.cfg["strategy"]
        candle_seconds = INTERVAL_MS[market.interval] / 1000
        print(f"Preparing {market.label} ({market.interval})...", file=sys.stderr)
        train, test = {}, {}
        for symbol in market.symbols:
            try:
                prepared = prepare(
                    market,
                    symbol,
                    market.interval,
                    start_ms,
                    end_ms,
                    agent,
                    memo,
                    prompt,
                    quiet=True,
                )
            except Exception as error:
                print(f"  {symbol}: skipped ({error})", file=sys.stderr)
                continue
            train[symbol], test[symbol] = split(prepared, cut_ms)

        results = []
        grid = {**GRID, "setup": SETUPS[market.kind]}
        for values in itertools.product(*grid.values()):
            choice = dict(zip(grid, values))
            (above, below), (stop, take, hold) = choice["thresholds"], choice["exits"]
            market_kind, leverage = choice["setup"]
            strategy = {
                **base,
                "direction": choice["direction"],
                "enter_above": above,
                "enter_below": below,
                "trend_filter": choice["trend_filter"],
                "stop_loss_atr": stop,
                "take_profit_atr": take,
                "max_hold_candles": hold,
                "market": market_kind,
                "max_leverage": leverage,
                "cooldown_seconds": int(choice["cooldown_candles"] * candle_seconds),
            }
            results.append(
                (
                    choice,
                    strategy,
                    evaluate(train, strategy, paper, choice["score"], candle_seconds),
                )
            )
        results.sort(key=lambda r: r[2]["return"], reverse=True)

        hold = [
            (buy_and_hold(p, paper)[-1] / paper["capital_usdt"] - 1) * 100
            for p in test.values()
            if p["steps"]
        ]
        hold_train = [
            (buy_and_hold(p, paper)[-1] / paper["capital_usdt"] - 1) * 100
            for p in train.values()
            if p["steps"]
        ]
        print(
            f"\n{market.label}: {len(results)} strategies on {len(train)} assets. Train {train_days} days, "
            f"test the next {test_days} days (never seen during selection). Prompt: {prompt[0]!r} ({prompt[1]})\n"
        )
        print(f"{'strategy':<58}{'train':>9}{'test':>9}{'worst':>9}{'DD':>9}{'trades':>8}")
        shown = results[: args.top]
        best_laya = next((r for r in results if r[0]["score"] == "p"), None)
        if best_laya and best_laya not in shown:
            shown = [*shown, best_laya]
        for choice, strategy, tr in shown:
            te = evaluate(test, strategy, paper, choice["score"], candle_seconds)
            print(
                f"{label(choice):<58}{tr['return']:>+8.2f}%{te['return']:>+8.2f}%"
                f"{te['worst']:>+8.2f}%{te['drawdown']:>+8.2f}%{te['trades']:>8}"
            )
        print(
            f"{'buy & hold':<58}{statistics.mean(hold_train):>+8.2f}%{statistics.mean(hold):>+8.2f}%"
            f"{min(hold):>+8.2f}%"
        )
        profitable = sum(r[2]["return"] > 0 for r in results)
        print(f"\n{profitable} of {len(results)} strategies made money on the train window.")
        if best_laya:
            print(f"Best Laya strategy on the train window, as [{market.kind}.strategy] values:")
            for key in (
                "market",
                "direction",
                "enter_above",
                "enter_below",
                "trend_filter",
                "stop_loss_atr",
                "take_profit_atr",
                "max_hold_candles",
                "max_leverage",
                "cooldown_seconds",
            ):
                value = best_laya[1][key]
                print(f'  {key} = "{value}"' if isinstance(value, str) else f"  {key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
