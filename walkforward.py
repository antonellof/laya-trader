"""Walk-forward search: pick a strategy on older data, then test it on newer data it never saw.

For each candle size, signals and Laya's P are computed once over train + test days.
Every strategy in the grid is simulated on the train window; the best few by average
return across coins are then run on the test window, next to buy & hold. A strategy
that only looks good on the train window was fitted to noise.
"""

import argparse
import itertools
import statistics
import sys
import time
import tomllib
from pathlib import Path

from backtest import buy_and_hold, max_drawdown, prepare, simulate
from core import load_agent

GRID = {
    "kline_interval": ["1m", "5m", "15m"],
    "score": ["p", "rule"],  # Laya's P, or the four trend votes alone
    "direction": ["trend", "reversion"],
    "thresholds": [(0.55, 0.30), (0.65, 0.20), (0.50, 0.35)],
    "setup": [("spot", 1.0), ("futures", 1.0), ("futures", 3.0)],
    "cooldown_candles": [1, 5],
}
CANDLE_SECONDS = {"1m": 60, "5m": 300, "15m": 900}


def split(prepared, cut_ms):
    def part(keep):
        return {
            "steps": [s for s in prepared["steps"] if keep(s["t"])],
            "funding": prepared["funding"],
        }

    return part(lambda t: t <= cut_ms), part(lambda t: t > cut_ms)


def evaluate(parts, strategy, paper, score):
    rows = []
    for prepared in parts.values():
        account, equities, _ = simulate(prepared, strategy, paper, score)
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
        "per_coin": [r[0] for r in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--train-days", type=float, default=14)
    parser.add_argument("--test-days", type=float, default=7)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    coins = [c.upper() for c in config["symbols"]]
    paper, base = config["paper"], config["strategy"]

    end_ms = int(time.time() * 1000)
    cut_ms = int(end_ms - args.test_days * 86_400_000)
    start_ms = int(cut_ms - args.train_days * 86_400_000)
    agent, memo = load_agent(config["model"]), {}

    data = {}
    for interval in GRID["kline_interval"]:
        print(f"Preparing {interval} candles...", file=sys.stderr)
        data[interval] = {"train": {}, "test": {}}
        for coin in coins:
            prepared = prepare(config, coin, interval, start_ms, end_ms, agent, memo, quiet=True)
            data[interval]["train"][coin], data[interval]["test"][coin] = split(prepared, cut_ms)

    results = []
    keys = [k for k in GRID if k != "kline_interval"]
    for interval in GRID["kline_interval"]:
        for values in itertools.product(*(GRID[k] for k in keys)):
            choice = dict(zip(keys, values))
            (above, below), (market, leverage) = choice["thresholds"], choice["setup"]
            strategy = {
                **base,
                "direction": choice["direction"],
                "enter_above": above,
                "enter_below": below,
                "market": market,
                "max_leverage": leverage,
                "cooldown_seconds": choice["cooldown_candles"] * CANDLE_SECONDS[interval],
            }
            train = evaluate(data[interval]["train"], strategy, paper, choice["score"])
            results.append((interval, choice, strategy, train))
    results.sort(key=lambda r: r[3]["return"], reverse=True)

    def label(interval, choice):
        (above, below), (market, leverage) = choice["thresholds"], choice["setup"]
        who = "Laya" if choice["score"] == "p" else "votes"
        return (
            f"{interval:>3} {who:<5} {choice['direction']:<9} {above:.2f}/{below:.2f} "
            f"{market:<7} {leverage:.0f}x cd{choice['cooldown_candles']}"
        )

    print(
        f"\n{len(results)} strategies. Train {args.train_days:g} days, "
        f"test the next {args.test_days:g} days (never seen during selection).\n"
    )
    print(
        f"{'strategy':<48}{'train':>9}{'test':>9}{'test worst coin':>17}{'test DD':>9}{'trades':>8}"
    )
    for interval, choice, strategy, train in results[: args.top]:
        test = evaluate(data[interval]["test"], strategy, paper, choice["score"])
        print(
            f"{label(interval, choice):<48}{train['return']:>+8.2f}%{test['return']:>+8.2f}%"
            f"{test['worst']:>+16.2f}%{test['drawdown']:>+8.2f}%{test['trades']:>8}"
        )
    for interval in GRID["kline_interval"][:1]:
        hold = []
        for prepared in data[interval]["test"].values():
            hold.append((buy_and_hold(prepared, paper)[-1] / paper["capital_usdt"] - 1) * 100)
        print(
            f"{'buy & hold (test window)':<48}{'':>9}{statistics.mean(hold):>+8.2f}%{min(hold):>+16.2f}%"
        )

    best_laya = next(r for r in results if r[1]["score"] == "p")
    interval, choice, strategy, train = best_laya
    print("\nBest Laya strategy on the train window, as config.toml values:")
    print(f'kline_interval = "{interval}"\n[strategy]')
    for key in (
        "market",
        "direction",
        "enter_above",
        "enter_below",
        "max_leverage",
        "cooldown_seconds",
    ):
        value = strategy[key]
        print(f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
