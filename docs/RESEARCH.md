# Research log

Every experiment behind the defaults in `config.toml`, in the order they were run, with the numbers as measured. Later results sometimes overturn earlier ones; those places are marked. For how the system works, see [HOW-IT-WORKS.md](HOW-IT-WORKS.md).

**Where things stand** (tests on 9–20 unseen months, 1h candles):

- **Stocks:** buy Laya-oversold dips, exit when the signal flips, full-size positions, no stop, 1h cooldown. Roughly +10–12% over 9 months against +12–18% for buy & hold, with much smaller losses in down months. Laya beats the same strategy run on plain indicator rules.
- **Crypto:** futures 1×, trade with the 4h trend, exit within ~3 days, 4 ATR stop, 2h cooldown, detailed wording with indicator values and memory. Clearly positive in tests where buy & hold lost, but single numbers move a lot between runs.
- **What failed:** 1-minute and 15-minute trading (fees), leverage above 1×, tight stops (and any stop on stocks), shorts in rising markets, re-picking the strategy every month, and detailed wording or memory on stocks.

## What the backtests taught

**Your 30-day run** (24 Aug – 24 Sept 2026, crypto, 15m, spot, reversion, stop 1.5 ATR, target 3 ATR, 15-minute cooldown) lost 26.7% (BTC), 17.7% (ETH) and 11.7% (SOL), while buy & hold made +7% to +19%. Taking the trades apart:

| Per coin | BTC | ETH | SOL |
|---|---|---|---|
| Round trips | 148 | 122 | 119 |
| Fees paid, of 1,000 | ~255 | ~223 | ~221 |
| Stop-loss exits, total P&L | 64, −254 | 51, −267 | 47, −314 |
| "Signal flipped" exits, total P&L | 69, **+43** | 59, **+120** | 59, **+205** |
| Take-profit exits, total P&L | 15, +71 | 12, +81 | 13, +103 |

1. **Fees were most of the loss:** about a quarter of the capital every month. Before fees the result was roughly flat.
2. **Tight stops kill dip-buying.** A stop 1.5 ATR under a dip gets hit at the low. The stops lost 250–310 per coin; exiting when the signal flipped made money.
3. **Markets drift.** Trading against the move without a filter fights that drift.

These lessons went into the strategy options (no stop, time exits, trend filter) and into the search grid below.

## Prompt lab

`promptlab.py` measures how well each candidate prompt's P predicts the next move. It uses the rank correlation (IC) with the forward return 1 h and 4 h ahead, plus the return spread between the top and bottom 20% of P. Each is measured on the first two-thirds of the window (used to choose) and on the last third (the check). There are 5 questions (*outlook*, *higher in a few hours*, *stretched*, *good moment to buy*, *selling exhausted*) × 2 wordings (Good/Bad vs neutral lists).

Results on 24 Sept 2026 (crypto 30 days, stocks 55 days, 15m candles):

| Market | Best prompt | IC 4h, choose / check | Top−bottom spread 4h |
|---|---|---|---|
| Crypto | *outlook*, Good/Bad | −0.14 / **+0.09** (sign flips) | −34 bp / +19 bp |
| Stocks | *outlook*, Good/Bad | −0.05 / **−0.12** (same sign) | −23 bp / −36 bp |

- **None of the 9 alternatives beat the current prompt.** The neutral wording mostly weakened the signal. So the prompt stays, now with evidence.
- **Stocks:** a low P (Laya reads bearish) is followed by a bounce over the next 4 hours, consistently. The spread is much bigger than the ~4 bp stock fees. **That's the case for reversion.**
- **Crypto:** the relation reversed between the two parts: reversal first, then trend. A crypto strategy built on P rests on a signal whose direction isn't stable.

## Walk-forward

`walkforward.py` searches, per market, over:

| Setting | Options |
|---|---|
| Score | Laya's P or the trend votes |
| Direction | trend or reversion |
| Thresholds | 0.55/0.30, 0.65/0.20, 0.75/0.15 |
| Trend filter | off or higher timeframe |
| Exits | 1.5 ATR stop + 3 ATR target; 3 ATR stop + 16-candle limit; 16-candle limit only; exit on signal flip only |
| Cooldown | 1, 4 or 16 candles |
| Setup | spot, futures 1× (crypto only) |

That's 576 strategies for crypto and 288 for stocks. It picks by average return on the older part (**crypto 20 days, stocks 37 days**), then runs the top picks on the newer part (**10 and 18 days**) that played no part in the choice.

Result on 24 Sept 2026:

| | Crypto | Stocks |
|---|---|---|
| Strategies profitable on the training part | **36 of 576** (the previous grid: 0 of 216) | 147 of 288 |
| Best Laya strategy | reversion, 0.75/0.15, no stop or target, exit on flip, 16-candle cooldown, spot | reversion, 0.55/0.30, same exits and cooldown, spot |
| … training part | +5.6% | +2.6% |
| … **unseen part** | **+1.2%** (worst coin −4.1%) | **+0.5%** (worst −3.8%) |
| Buy & hold, unseen part | +8.6% | +3.0% |
| Same Laya strategy with shorts (futures) | −6.3% on the unseen part | – |

> **Update:** the year-long tests below overturned the crypto part of this. Over 9 unseen months this crypto strategy lost 27–31%, so crypto now uses a different setup (see [Stop losses](#stop-losses-and-protection-two-years-of-evidence)). The 10-day +1.2% was luck.

These two were the defaults at that point. On the same 30-day crypto window as the run above they give −4.0% (BTC), +17.5% (ETH), +7.6% (SOL), versus −26.7%, −17.7% and −11.7% before. That window overlaps the one used to choose them, though, so the unseen-part numbers are the honest ones.

**The honest conclusion:**
- The changes turned heavy losses into small out-of-sample gains, with smaller drawdowns than holding. In rising markets, they still trail buy & hold.
- 10–18 unseen days is a short test, and the crypto signal changed direction within a month.
- **No stop loss** means an open position is exposed to a crash until the signal flips. The tested alternative with stops did worse, but a crash was not in the sample.
- Rerun `promptlab.py` and `walkforward.py` on fresh data before trusting any setting.

## Stocks: a year of out-of-sample tests

Fifteen-minute stock history only goes back about 60 days, too short to trust. Yahoo keeps two years of hourly candles, so `walkforward.py --rolling` tests on **1h candles over a year**, using **20 large S&P 500 names**:

- AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM, BRK-B, V
- UNH, JNJ, PG, HD, MA, COST, LLY, AVGO, WMT, KO

Each fold picks the best of 336 strategies on 90 days and tests it on the next 30, then slides a month forward: **9 unseen test months** in total. The grid now also includes closing before the session ends, holding for one or two weeks while ignoring signal flips, and stop + time exits.

Results (Dec 2025 – Sep 2026, average across the 20 stocks):

| Variant | Compounded over 9 months | Months positive | Worst month | Time in market |
|---|---|---|---|---|
| **Default: Laya reversion 0.55/0.30, exit on flip, 4h cooldown** | **+5.0%** | 7/9 | **−0.7%** | 68% |
| Same, re-picked every month from the 336 | +2.1% | 6/9 | – | – |
| Same strategy with trend votes instead of Laya | +3.6% | 6/9 | −0.3% | 47% |
| Hold ~1 week + 3 ATR stop | +4.2% | 5/9 | – | 57% |
| Hold ~2 weeks, ignore flips | +3.1% | 5/9 | – | 68% |
| Default + daily trend filter | +1.4% | 5/9 | – | 36% |
| Stay long, step aside when P ≥ 0.85 | +4.7% | 5/9 | −1.6% | 90% |
| Buy & hold | +12.2% | 7/9 | −4.3% | 100% |

- **The default is the best active variant.** None of the ideas to hold longer, filter by trend or stay invested improved on it. Those ideas came from reasoning about stocks and from the earlier crypto lessons, and all were tested on the same unseen months. The "stay long" one was proposed after seeing the first results, so it's exploratory.
- **Re-picking every month is worse than a fixed rule.** Choosing the month's best strategy chases noise.
- **Laya adds value over plain rules:** +5.0% vs +3.6% on the same strategy.
- **With 1% risk it doesn't come close to buy & hold.** It trades about 60% of the rally's return for losing much less in down months. In February, buy & hold lost 4.3% and the default 0.7%; in May, −1.9% vs 0.0%.

**Position size was the real bottleneck.** With no stop, a trade is sized as if its stop were 3 ATR away. On hourly stock candles that gave positions of only about 0.5× equity, so even good calls earned half. Same 9 test months, only the risk per trade changed:

| Variant | Compounded | Months beating buy & hold | Worst month |
|---|---|---|---|
| Default, risk 1% (positions ~0.5×) | +5.0% | 2/9 | −0.7% |
| Default, risk 2% | +9.4% | 3/9 | −1.4% |
| **Default, risk 3% (full size, capped at 1×)** | **+10.3%** | **5/9** | **−1.6%** |
| Rules only, risk 3% | +8.2% | 5/9 | −0.7% |
| Buy & hold | +12.2% | – | −4.3% |

At full size, the Laya strategy gets close to buy & hold's return, beats it in 5 of 9 months, and its worst month is about a third as bad, while in the market 68% of the time. This changes only how much to buy, not when, so it's hard to overfit. **`stocks.strategy.risk_pct` is now 3.0.**

Because this is what was validated, live stocks now use **1h candles** (they were on 15m). A 6-month backtest of the 8 default stocks on 1h candles:

| | Laya, risk 1% | **Laya, risk 3% (default)** | Rules only, risk 3% | Buy & hold |
|---|---|---|---|---|
| Average return | +5.8% | **+13.1%** | +5.5% | +21.4% |
| Average max drawdown | – | **−12.9%** | −10.7% | −18.2% |
| Worst max drawdown | −5.4% | −16.1% | −15.6% | −24.2% |

At full size, Laya beats rules only on 6 of 8 stocks and buy & hold on 1 of 8 (XOM). It earns about 60% of buy & hold's return with about 70% of its drawdown. Larger positions mean larger swings: the worst drawdown went from −5.4% to −16.1%.

## Stop losses and protection: two years of evidence

`walkforward.py --rolling --no-search --variants risk` runs the market's default strategy on every test month, once with each protection option. Sizing is identical in all variants, so any difference comes from the exits alone. The test windows include real crashes: the April 2025 tariff selloff (stocks) and the October 2025 liquidation cascade (crypto).

**Stocks:** 20 names, 1h candles, 20 test months (Dec 2024 – Sep 2026).

| Variant | Compounded | Worst month | Worst drawdown (one stock) | Worst trade |
|---|---|---|---|---|
| **No stop (default)** | **+19.6%** | −3.8% | −28.9% | −20.3% |
| Stop 8% from entry | +15.3% | −6.5% | −27.2% | −18.2% |
| Stop 5% from entry | +13.7% | −5.8% | −27.0% | −18.0% |
| Stop 4 ATR | +13.5% | −4.9% | −27.7% | −18.0% |
| Trailing stop 6 ATR | +14.5% | −4.4% | −26.6% | −18.2% |
| Breakeven after 2 ATR + stop 8% | +16.6% | −5.9% | −27.2% | −18.2% |
| No entries when volatility spikes | +19.4% | −3.7% | −28.9% | −20.3% |
| Pause 3 days after a 5% drawdown | +17.5% | −4.9% | −28.9% | −20.3% |
| Buy & hold | +25.7% | −5.5% | −41.5% | – |

**Every stop made stocks worse.** Returns fell 3–9 points, and even the worst month got deeper, because stops sold dips at the low and missed the rebound the strategy buys them for. Stops barely changed the worst trade (−20% → −18%): the big single-stock losses are **overnight gaps** after news or earnings, where the price opens far below any stop. **Stocks keep no stop.** The protection that actually limits single-stock gap risk is holding several stocks with smaller positions each, not a price stop.

**Crypto:** BTC/ETH/SOL, 9 test months (Dec 2025 – Sep 2026).

The earlier crypto default (15m, reversion, flip exit) lost **−30.8%** over the year, against −9.2% for buy & hold. On 15-minute candles, every stop made it worse (−33% to −56%): many small stop-outs on top of about 7% a month in fees. The loss cooldown and the drawdown pause softened the worst month (from −17% to −12%) but not the year.

On **1h candles**, the monthly re-chosen strategies made **+8.5%** while buy & hold lost 8.7%. They picked "hold at most ~3 days" 9 of 9 times and "only with the 4h trend" 8 of 9 times. Tested as **fixed** strategies on the same months (mild hindsight, since the pattern came from those picks):

| Variant (1h, 9 months) | Compounded | Worst month | Worst drawdown |
|---|---|---|---|
| Old crypto default | −27.4% | −18.6% | −27.5% |
| Reversion 0.55/0.30, with 4h trend, hold ≤ 3 days | −6.7% | −4.4% | −8.2% |
| … + stop 4 ATR | −7.8% | −2.5% | −5.8% |
| Reversion 0.65/0.20, with 4h trend, hold ≤ 3 days | −2.3% | −3.7% | −8.5% |
| Trend 0.65/0.20, with 4h trend, hold ≤ 3 days | +0.4% | −3.6% | −8.9% |
| **… + stop 4 ATR (new default)** | **+0.5%** | **−3.3%** | **−8.3%** |
| … same, trend votes instead of Laya (`signal = "votes"`) | **+8.6%** | −2.3% | −7.4% |
| Buy & hold | −8.7% | −30.3% | −41.5% |

- **What fixed crypto was structure, not the stop.** Hourly candles, trading only with the 4h trend, and a 3-day time limit took a −27% year to about flat. The worst month went from −18.6% to −3.3%, while buy & hold's worst was −30%.
- **On crypto a 4 ATR stop helps a little.** Without overnight gaps, stops fill near their level: similar return, a slightly better worst month. **Crypto now uses it.**
- **On crypto, the plain trend votes did better than Laya** with the same structure (+8.6% vs +0.5%). It's one comparison on the same months, so it's not proof, but it matches the prompt lab: Laya's crypto signal isn't stable. Set `signal = "votes"` under `[crypto.strategy]` to trade on the votes; Laya's reading still shows on the dashboard.

## A faster trader: cooldowns, futures and leverage

`walkforward.py --rolling --no-search --variants futures` (crypto) and `--variants speed` (stocks). Same 9 unseen months, 1h candles, without memory:

| Crypto variant | Compounded | Worst month | Worst drawdown |
|---|---|---|---|
| Spot, 4h cooldown (previous default) | −0.3% | −3.8% | −8.4% |
| Spot, 1h cooldown | −0.8% | −3.9% | −9.3% |
| Futures 1×, 4h cooldown, fee 0.05%, risk 3% | +0.5% | −10.5% | −23.2% |
| Futures 1×, 2h cooldown, fee 0.05%, risk 3% | −3.1% | −9.5% | −23.7% |
| Futures 1×, 1h cooldown, fee 0.05%, risk 3% | −4.1% | −9.6% | −22.2% |
| **Futures 1×, 2h cooldown, fee 0.05%, risk 1.5% (new default)** | **+6.4%** | −6.2% | −17.1% |
| Futures 1×, 2h cooldown, maker fee 0.02% | +7.4% | −8.5% | −23.2% |
| Futures 2×, 1h cooldown | −17.5% | −18.9% | −39.6% |
| Futures 3×, 1h cooldown | −35.3% | −27.7% | −53.1% |
| Buy & hold | −12.0% | −29.8% | −41.5% |

| Stocks cooldown | Compounded | Worst month |
|---|---|---|
| 4h (previous default) | +11.5% | −3.6% |
| **1h (new default)** | **+11.8%** | **−3.3%** |
| none | +11.8% | −3.3% |
| Buy & hold | +18.2% | −4.9% |

- **Leverage above 1× destroyed crypto results.** 2× and 3× lost 17–35%, with drawdowns of 40–53%. Futures stay at 1×; their benefits are the lower fee (0.05% taker vs 0.1% spot) and being able to short.
- **Faster works if positions are smaller.** At 1.5% risk (about half of equity per position), a 2h cooldown made +6.4% with a smaller worst drawdown than full size.
- **Maker fees** (limit orders, 0.02%) would add about a point, but market orders pay taker fees, so the paper account assumes 0.05%.
- **Stocks can trade every hour** at no cost. Their fees are tiny, and the result barely changes with the cooldown.
- The crypto default was picked after seeing this table and was tested without memory, so treat +6.4% as optimistic.

