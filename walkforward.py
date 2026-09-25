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

from backtest import asker, buy_and_hold, max_drawdown, paper_of, prepare, prompt_of, simulate
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


def evaluate(parts, strategy, paper, score, candle_seconds, ask=None, memory=0):
    rows = []
    for prepared in parts.values():
        if not prepared["steps"]:
            continue
        account, equities, _ = simulate(
            prepared, strategy, paper, score, False, candle_seconds, ask, memory
        )
        rows.append(
            (
                (equities[-1] / paper["capital_usdt"] - 1) * 100,
                max_drawdown(equities),
                len(account.trades),
                getattr(account, "exposure_steps", 0) / len(equities),
                min(
                    (t["pnl"] / paper["capital_usdt"] * 100 for t in account.trades if "pnl" in t),
                    default=0.0,
                ),
            )
        )
    return {
        "return": statistics.mean(r[0] for r in rows),
        "worst": min(r[0] for r in rows),
        "drawdown": min(r[1] for r in rows),
        "trades": sum(r[2] for r in rows),
        "exposure": statistics.mean(r[3] for r in rows),
        "worst_trade": min((r[4] for r in rows), default=0.0),
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


# A few predefined variants of the default, compared on every test month without any
# search (so they can't be fitted to the test months).
FIXED_VARIANTS = {
    "default (config)": {},
    "hold ~1 week, ignore flips": {"exit_on_flip": False, "max_hold_candles": 35},
    "hold ~2 weeks, ignore flips": {"exit_on_flip": False, "max_hold_candles": 70},
    "default + daily trend filter": {"trend_filter": "htf"},
    "hold ~2 weeks + daily trend filter": {
        "exit_on_flip": False,
        "max_hold_candles": 70,
        "trend_filter": "htf",
    },
    "hold ~1 week + 3 ATR stop": {
        "exit_on_flip": False,
        "max_hold_candles": 35,
        "stop_loss_atr": 3.0,
    },
}

# Protection variants on top of the market's default strategy. Sizing is the same in all
# of them (sizing_atr), so any difference comes from the exits and entry filters alone.
RISK_VARIANTS = {
    "current (no stop)": {},
    "stop 5% from entry": {"stop_loss_pct": 5},
    "stop 8% from entry (crash only)": {"stop_loss_pct": 8},
    "stop 4 ATR": {"stop_loss_atr": 4},
    "trailing stop 3 ATR": {"trailing_stop_atr": 3},
    "trailing stop 6 ATR": {"trailing_stop_atr": 6},
    "breakeven after 2 ATR + stop 8%": {"breakeven_after_atr": 2, "stop_loss_pct": 8},
    "1-day cooldown after a loss": {"loss_cooldown_seconds": 86_400},
    "no entries when ATR3/ATR14 > 2": {"max_entry_atr_ratio": 2},
    "pause 3 days after a 5% drawdown": {"pause_drawdown_pct": 5, "pause_seconds": 259_200},
    "stop 8% + trailing 6 ATR + no volatile entries": {
        "stop_loss_pct": 8,
        "trailing_stop_atr": 6,
        "max_entry_atr_ratio": 2,
    },
}

# Laya reads the asset's last N closed trades and their outcome ("_memory" is not a
# strategy key; it sets how many trades go into the text).
MEMORY_VARIANTS = {
    "no memory": {"_memory": 0},
    "last 3 trades": {"_memory": 3},
    "last 10 trades": {"_memory": 10},
}

ROLLING_GRID = {
    "score": ["p", "rule"],
    "direction": ["trend", "reversion"],
    "thresholds": [(0.55, 0.30), (0.65, 0.20), (0.75, 0.15)],
    "trend_filter": ["none", "htf"],
    # (stop ATR, take ATR, max hold candles, flat at the close, exit when the signal flips)
    "exits": [
        (0, 0, 0, False, True),
        (0, 0, 4, False, True),
        (0, 0, 0, True, True),
        (3.0, 0, 0, False, True),
        (0, 0, 35, False, False),  # hold about a week (1h candles), ignore flips
        (0, 0, 70, False, False),  # about two weeks
        (3.0, 0, 70, False, False),
    ],
    "cooldown_candles": [1, 4],
}


def grid_strategies(grid, base, candle_seconds, market_kind):
    for values in itertools.product(*grid.values()):
        choice = dict(zip(grid, values))
        (above, below), exits = choice["thresholds"], choice["exits"]
        stop, take, hold = exits[:3]
        flat = exits[3] if len(exits) > 3 else False
        flip = exits[4] if len(exits) > 4 else True
        market, leverage = choice.get("setup", ("spot", 1.0))
        yield (
            choice,
            {
                **base,
                "direction": choice["direction"],
                "enter_above": above,
                "enter_below": below,
                "trend_filter": choice["trend_filter"],
                "stop_loss_atr": stop,
                "take_profit_atr": take,
                "max_hold_candles": hold,
                "flat_at_close": flat,
                "exit_on_flip": flip,
                "market": market,
                "max_leverage": leverage,
                "cooldown_seconds": int(choice["cooldown_candles"] * candle_seconds),
            },
        )


def describe_choice(choice):
    (above, below), exits = choice["thresholds"], choice["exits"]
    stop, _, hold = exits[:3]
    flat = len(exits) > 3 and exits[3]
    flip = exits[4] if len(exits) > 4 else True
    parts = [
        text
        for ok, text in (
            (flat, "close at session end"),
            (hold, f"hold max {hold} candles"),
            (stop, f"stop {stop:g} ATR"),
            (flip, "exit on flip"),
        )
        if ok
    ]
    exit_text = ", ".join(parts)
    who = "Laya" if choice["score"] == "p" else "votes"
    filt = ", with daily trend" if choice["trend_filter"] == "htf" else ""
    return f"{who} {choice['direction']} {above:.2f}/{below:.2f}{filt}, {exit_text}, cd{choice['cooldown_candles']}"


def window(prepared, start_ms, end_ms):
    return {
        "steps": [s for s in prepared["steps"] if start_ms <= s["t"] < end_ms],
        "funding": prepared["funding"],
    }


def rolling(args, config, market, agent, memo, prompt):
    """Rolling walk-forward: choose on train_days, test on the next test_days, slide by
    test_days, repeat. Every test month is data the chosen strategy never saw."""
    if args.symbols:
        market.symbols = [x.strip().upper() for x in args.symbols.split(",")]
    market.interval = args.interval
    candle_seconds = INTERVAL_MS[market.interval] / 1000
    paper, base = paper_of(config, market), market.cfg["strategy"]
    day = 86_400_000
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(args.days * day)
    print(
        f"Preparing {len(market.symbols)} {market.label.lower()} on {market.interval} candles, "
        f"{args.days:g} days...",
        file=sys.stderr,
    )
    prepared = {}
    for symbol in market.symbols:
        started = time.perf_counter()
        try:
            prepared[symbol] = prepare(
                market, symbol, market.interval, start_ms, end_ms, agent, memo, prompt, quiet=True
            )
        except Exception as error:
            print(f"  {symbol}: skipped ({error})", file=sys.stderr)
            continue
        print(
            f"  {symbol}: {len(prepared[symbol]['steps'])} candles, "
            f"{time.perf_counter() - started:.0f} s",
            file=sys.stderr,
        )
    strategies = (
        []
        if args.no_search
        else list(grid_strategies(ROLLING_GRID, base, candle_seconds, market.kind))
    )
    default = {**base, "cooldown_seconds": base["cooldown_seconds"]}

    ask = asker(agent, memo, prompt[0])

    def mean_return(start, end, strategy, score, memory=0):
        parts = {k: window(v, start, end) for k, v in prepared.items()}
        return evaluate(parts, strategy, paper, score, candle_seconds, ask, memory)

    def hold_return(start, end):
        values = []
        for v in prepared.values():
            part = window(v, start, end)
            if part["steps"]:
                values.append((buy_and_hold(part, paper)[-1] / paper["capital_usdt"] - 1) * 100)
        return statistics.mean(values)

    train_ms, test_ms = int(args.train_days * day), int(args.test_days * day)
    folds, cursor = [], start_ms + train_ms
    while cursor + test_ms <= end_ms + day:
        folds.append((cursor - train_ms, cursor, min(cursor + test_ms, end_ms)))
        cursor += test_ms

    def compound(values):
        total = 1.0
        for v in values:
            total *= 1 + v / 100
        return (total - 1) * 100

    def hold_drawdown(start, end):
        values = []
        for v in prepared.values():
            part = window(v, start, end)
            if part["steps"]:
                values.append(max_drawdown(buy_and_hold(part, paper)))
        return min(values)

    print(
        f"\n{market.label}: {len(prepared)} assets, {len(folds)} test months "
        f"(each after {args.train_days:g} days of history)."
    )
    totals = {"chosen": [], "laya": [], "default": [], "hold": []}
    chosen_log = []
    if strategies:
        print(
            f"Monthly re-choice among {len(strategies)} strategies:\n\n"
            f"{'test month':<12}{'chosen on the train window':<58}{'test':>8}{'best Laya':>11}"
            f"{'default':>9}{'hold':>8}"
        )
    for train_start, test_start, test_end in folds:
        totals["hold"].append(hold_return(test_start, test_end))
        if not strategies:
            continue
        scored = [
            (mean_return(train_start, test_start, st, ch["score"])["return"], ch, st)
            for ch, st in strategies
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        _, best_choice, best = scored[0]
        _, laya_choice, laya = next(x for x in scored if x[1]["score"] == "p")
        results = {
            "chosen": mean_return(test_start, test_end, best, best_choice["score"])["return"],
            "laya": mean_return(test_start, test_end, laya, "p")["return"],
            "default": mean_return(test_start, test_end, default, "p")["return"],
        }
        for k, v in results.items():
            totals[k].append(v)
        chosen_log.append(laya_choice)
        month = time.strftime("%Y-%m-%d", time.gmtime(test_start / 1000))
        print(
            f"{month:<12}{describe_choice(best_choice):<58}{results['chosen']:>+7.2f}%"
            f"{results['laya']:>+10.2f}%{results['default']:>+8.2f}%{totals['hold'][-1]:>+7.2f}%"
        )
    if strategies:
        print(f"\n{'':<70}{'chosen':>8}{'best Laya':>11}{'default':>9}{'hold':>8}")
        print(
            f"{'compounded':<70}{compound(totals['chosen']):>+7.2f}%"
            f"{compound(totals['laya']):>+10.2f}%{compound(totals['default']):>+8.2f}%"
            f"{compound(totals['hold']):>+7.2f}%"
        )

    variants = {"strategy": FIXED_VARIANTS, "risk": RISK_VARIANTS, "memory": MEMORY_VARIANTS}[
        args.variants
    ]
    print(f"\nPredefined {args.variants} variants on every test month (no search involved):")
    print(
        f"{'variant':<48}{'compounded':>11}{'months +':>10}{'beat hold':>10}{'worst month':>13}"
        f"{'worst DD':>10}{'worst trade':>13}{'in market':>11}"
    )
    for name, change in variants.items():
        change = dict(change)
        memory = change.pop("_memory", 0)
        strategy = {**base, **change}
        returns, exposure, drawdowns, worst_trades = [], [], [], []
        for _, test_start, test_end in folds:
            result = mean_return(test_start, test_end, strategy, "p", memory)
            returns.append(result["return"])
            exposure.append(result["exposure"])
            drawdowns.append(result["drawdown"])
            worst_trades.append(result["worst_trade"])
        beat = sum(x > y for x, y in zip(returns, totals["hold"]))
        print(
            f"{name:<48}{compound(returns):>+10.2f}%{sum(v > 0 for v in returns):>7}/{len(returns)}"
            f"{beat:>7}/{len(returns)}{min(returns):>+12.2f}%{min(drawdowns):>+9.2f}%"
            f"{min(worst_trades):>+12.2f}%{statistics.mean(exposure) * 100:>10.0f}%"
        )
    hold_dd = min(hold_drawdown(ts, te) for _, ts, te in folds)
    print(
        f"{'buy & hold':<48}{compound(totals['hold']):>+10.2f}%"
        f"{sum(v > 0 for v in totals['hold']):>7}/{len(folds)}{'':>10}"
        f"{min(totals['hold']):>+12.2f}%{hold_dd:>+9.2f}%{'':>13}{100:>10}%"
    )
    print(
        "\nworst DD: the deepest drawdown of any single asset within a test month. "
        "worst trade: the biggest loss on one trade, as % of that asset's capital."
    )

    if chosen_log:
        counts = {}
        for choice in chosen_log:
            counts[describe_choice(choice)] = counts.get(describe_choice(choice), 0) + 1
        print("\nBest Laya strategy per train window (how stable the choice is):")
        for text, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>2} x {text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--markets", help="Comma-separated: crypto,stocks (default: all)")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument(
        "--rolling", action="store_true", help="Rolling walk-forward over many months"
    )
    parser.add_argument("--interval", default="1h", help="Candle size for --rolling (default 1h)")
    parser.add_argument("--days", type=float, default=365, help="Total history for --rolling")
    parser.add_argument("--train-days", type=float, default=90)
    parser.add_argument("--test-days", type=float, default=30)
    parser.add_argument("--symbols", help="Comma-separated symbols for --rolling (default: config)")
    parser.add_argument(
        "--variants",
        choices=("strategy", "risk", "memory"),
        default="strategy",
        help="Predefined variants to compare in --rolling",
    )
    parser.add_argument(
        "--no-search", action="store_true", help="--rolling: skip the grid, only compare variants"
    )
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    markets = load_markets(config)
    if args.markets:
        markets = [m for m in markets if m.kind in {x.strip() for x in args.markets.split(",")}]
    agent, memo, prompt = load_agent(config["model"]), {}, prompt_of(config)
    end_ms = int(time.time() * 1000)
    if args.rolling:
        for market in markets:
            rolling(args, config, market, agent, memo, prompt)
        return 0

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
