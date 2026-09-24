"""Prompt lab: which question and wording make Laya's P most predictive of the next move?

For every closed candle in the window, each candidate prompt gets Laya's P. Then, per
market and horizon, it measures:
  IC      rank correlation between P and the forward return (+ trend-like, - reversal-like)
  spread  average forward return of the top 20% of P minus the bottom 20%, in basis points
on the first two thirds of the window (used to choose) and on the last third (the check).
A prompt worth using keeps the same sign, and most of its size, on the unseen third.
"""

import argparse
import json
import statistics
import sys
import time
import tomllib
from pathlib import Path

from core import PROMPT_FORMATS, compute_signals, describe, load_agent
from markets import INTERVAL_MS, load_markets

HERE = Path(__file__).resolve().parent
WARMUP = 100
QUESTIONS = {
    "outlook": "Is the short-term outlook for this asset bullish?",
    "higher": "Will the price most likely be higher in a few hours?",
    "stretched": "Is the price stretched and likely to pull back soon?",
    "buy_now": "Is this a good moment to buy?",
    "exhausted": "Is the selling pressure exhausted, so that a bounce is likely?",
}
HORIZONS = (4, 16)  # candles ahead: 1 hour and 4 hours on 15-minute candles


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2
        i = j + 1
    return result


def spearman(x, y):
    if len(x) < 30 or len(set(x)) < 3:
        return None
    return statistics.correlation(ranks(x), ranks(y))


def spread_bp(p, fwd):
    pairs = sorted(zip(p, fwd))
    n = max(1, len(pairs) // 5)
    return (
        statistics.mean(f for _, f in pairs[-n:]) - statistics.mean(f for _, f in pairs[:n])
    ) * 1e4


def score_candles(agent, market, symbol, start_ms, end_ms, memo):
    """P for every (question, format) on every candle, plus forward returns."""
    history = market.history(symbol, market.interval, start_ms, end_ms, WARMUP)
    fast = history["fast"]
    rows = []
    questions = {k: {"type": "noul", "instructions": q} for k, q in QUESTIONS.items()}
    for i in range(WARMUP, len(fast) - max(HORIZONS)):
        close_ms = fast[i][6] + 1
        if close_ms < start_ms:
            continue
        s = compute_signals(fast[i - WARMUP + 1 : i + 1], history["ctx_at"](close_ms))
        price = s["price"]
        row = {"t": close_ms, "fwd": {h: float(fast[i + h][4]) / price - 1 for h in HORIZONS}}
        for fmt in PROMPT_FORMATS:
            state = describe(s, market.noun, fmt)
            if state not in memo:
                answers = agent.predict(state, questions)["answers"]
                memo[state] = {k: answers[k]["noul"] for k in QUESTIONS}
            for k, p in memo[state].items():
                row[f"{k}|{fmt}"] = p
        rows.append(row)
    return rows


def evaluate(rows_by_asset, key, horizon, part):
    ics, spreads = [], []
    for rows in rows_by_asset.values():
        cut = int(len(rows) * 2 / 3)
        chosen = rows[:cut] if part == "train" else rows[cut:]
        p = [r[key] for r in chosen]
        f = [r["fwd"][horizon] for r in chosen]
        ic = spearman(p, f)
        if ic is not None:
            ics.append(ic)
            spreads.append(spread_bp(p, f))
    if not ics:
        return None, None
    return statistics.mean(ics), statistics.mean(spreads)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--crypto-days", type=float, default=30)
    parser.add_argument("--stocks-days", type=float, default=55)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    agent = load_agent(config["model"])
    agent.batch_size = len(QUESTIONS)
    end_ms = int(time.time() * 1000)
    results = {}
    for market in load_markets(config):
        days = args.crypto_days if market.kind == "crypto" else args.stocks_days
        start_ms = int(end_ms - days * 86_400_000)
        memo, rows_by_asset = {}, {}
        for symbol in market.symbols:
            started = time.perf_counter()
            try:
                rows_by_asset[symbol] = score_candles(agent, market, symbol, start_ms, end_ms, memo)
            except Exception as error:
                print(f"{symbol}: skipped ({error})", file=sys.stderr)
                continue
            print(
                f"{market.kind}:{symbol}: {len(rows_by_asset[symbol])} candles, "
                f"{len(memo)} distinct states so far, {time.perf_counter() - started:.0f} s",
                file=sys.stderr,
            )
        table = []
        for fmt in PROMPT_FORMATS:
            for q in QUESTIONS:
                key = f"{q}|{fmt}"
                entry = {"question": q, "format": fmt}
                for h in HORIZONS:
                    for part in ("train", "test"):
                        entry[f"ic_{h}_{part}"], entry[f"spread_{h}_{part}"] = evaluate(
                            rows_by_asset, key, h, part
                        )
                table.append(entry)
        results[market.kind] = table

        step_minutes = INTERVAL_MS[market.interval] // 60_000
        print(
            f"\n{market.label}: {len(rows_by_asset)} assets, {days:g} days of {market.interval} candles"
        )
        header = "".join(
            f"{f'IC {h * step_minutes}m train':>16}{'test':>8}{'spread bp tr/te':>18}"
            for h in HORIZONS
        )
        print(f"{'question':<11}{'format':<10}{header}")
        for e in sorted(table, key=lambda e: -abs(e[f"ic_{HORIZONS[-1]}_train"] or 0)):
            cells = ""
            for h in HORIZONS:
                tr, te = e[f"ic_{h}_train"], e[f"ic_{h}_test"]
                sp_tr, sp_te = e[f"spread_{h}_train"], e[f"spread_{h}_test"]
                cells += f"{tr:>+16.3f}{te:>+8.3f}{sp_tr:>+10.1f}/{sp_te:<+7.1f}"
            print(f"{e['question']:<11}{e['format']:<10}{cells}")
    out = HERE / "logs" / "promptlab.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
