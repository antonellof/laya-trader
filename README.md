# laya-trader

An experiment: every second, [Laya MLX](https://github.com/mizorewww/laya-mlx) reads live market signals for **crypto** (Binance) and **S&P 500 stocks** (Yahoo Finance) and says how bullish they look. Each market's strategy turns that into **LONG**, **SHORT**, **CLOSE** or **HOLD** on a paper account, sizes the position from risk and caps leverage. A live dashboard shows every decision and its full context, a backtest replays history with the same logic, and two research tools pick the prompt and the strategy from evidence.

**Paper trading only.** No API keys, no orders, nothing is sent to an exchange or broker. Not financial advice.

![Live dashboard: crypto and stocks together, with P(bullish), action and paper equity per asset](docs/screenshots/live-overview.png)

```bash
./run.sh                               # live loop + dashboard at http://127.0.0.1:8765
uv run python backtest.py --days 7     # historical replay of every market, writes backtest.html
uv run python promptlab.py             # which prompt makes Laya's P most predictive
uv run python walkforward.py           # which strategy holds up on data it never saw
uv run python walkforward.py --rolling --markets stocks   # a year of monthly out-of-sample tests
uv run python walkforward.py --rolling --no-search --variants risk --days 700 --markets stocks   # stop-loss comparison
```

Needs an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/). The Laya checkpoint (~650 MB) downloads on first run.

## How it works

Each round, for every asset in `config.toml`:

```
public market data ──► signals ──► one sentence ──► Laya: P(bullish) ──► strategy ──► LONG / SHORT / CLOSE / HOLD
 Binance / Yahoo        indicators,  "Good: …          one forward pass    direction,     size from risk,
                        trend votes   Bad: …"           on the Mac GPU      exits, filter  paper account, log, dashboard
```

### 1. Markets and data

Everything comes from public endpoints, with no account needed.

| | Crypto (`[crypto]`) | Stocks (`[stocks]`) |
|---|---|---|
| Default assets | BTC, ETH, SOL | AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM |
| Candles | Binance spot 1h, every round | Yahoo Finance chart API 1h, every `poll_seconds` (30 s) |
| Trading hours | 24/7 | regular US session only; outside it: HOLD "market closed" |
| Higher timeframe | 4h candles | daily candles |
| Futures data | funding, open interest, long/short ratio, taker buy/sell ratio | – |
| Sentiment | Fear & Greed index | VIX |
| Benchmark | – | SPY, for relative strength |
| Fees (paper) | 0.1% per side | 0.02% per side (approximates the spread) |
| Optional, live only | Whale Alert transfers, CoinJournal headlines | Yahoo Finance headlines per symbol |

Yahoo's chart API is public but unofficial, so stocks are polled every 30 seconds rather than every second. Requests reuse open HTTPS connections. A round for all 11 assets takes about 0.3 s (median) to fetch. Laya's answers are cached by sentence, so an unchanged market doesn't run the model again. If one asset fails, only that asset is skipped.

### 2. Signals

- **Fast candles** (`kline_interval`, default 1h):
  - EMA20
  - MACD, and whether it's rising or falling
  - RSI14 and RSI7 (Wilder)
  - ATR3/ATR14 (volatility)
  - volume vs its 20-candle average
  - buyer share of volume (crypto only; Yahoo has no taker data)
- **Higher timeframe** (4h crypto, daily stocks): EMA20 vs EMA50, RSI14, MACD, ATR3/ATR14.
- **1h:** the last hour's volume vs the 23 before it.
- **Day:**
  - crypto: 24h change and position in the 24h range
  - stocks: change since yesterday's close and position in today's range
- **Daily pivots** from yesterday: PP, S1, S2, R1, R2.
- **Crypto futures:** funding, 1h open-interest change, long/short account ratio, taker buy/sell ratio.
- **Sentiment and relative strength:** Fear & Greed (crypto), VIX (stocks), stock change minus SPY's.

Four of them become **trend votes** of +1, 0 or −1:

| Vote | +1 when | −1 when | 0 (neutral band) |
|---|---|---|---|
| Price vs EMA20 | more than 0.12% above | more than 0.12% below | within ±0.12% |
| Higher-timeframe EMA20 vs EMA50 | EMA20 above | EMA20 below | gap under 0.05% |
| MACD | positive | negative | smaller than 2% of ATR14 |
| RSI14 | above 58 | below 42 | 42–58 |

### 3. What Laya reads

The signals become one sentence, with bullish facts under *Good* and bearish facts under *Bad*. A signal a market doesn't have simply never appears.

```
Stock market signals. Good: price at daily support. Bad: trend votes bearish (-2 of 4),
price below EMA20, MACD negative, MACD falling, daily momentum bearish.
```

| Bullish (Good) when | Bearish (Bad) when |
|---|---|
| trend votes sum ≥ +2 | trend votes sum ≤ −2 |
| price > 0.12% above EMA20 | price > 0.12% below EMA20 |
| MACD positive / rising beyond its noise band | MACD negative / falling beyond its noise band |
| buyers > 55% of recent volume | buyers < 45% of recent volume |
| volume > 1.5× average, mostly buying | volume > 1.5× average, mostly selling |
| RSI14 < 30, RSI7 < 20 | RSI14 > 70, RSI7 > 80 |
| higher-timeframe RSI > 55 and MACD > 0 | higher-timeframe RSI < 45 and MACD < 0 |
| last hour's volume > 2× average while rising | the same while falling |
| price in the top 10% of the day's range | price in the bottom 10% |
| price above R1 / within 0.3% of S1 or S2 | price below S1 / within 0.3% of R1 or R2 |
| open interest up > 1% in an hour, price up | open interest up > 1%, price down |
| futures taker buy/sell > 1.2, long/short < 0.8 | taker buy/sell < 0.8, funding > 0.05% |
| Fear & Greed ≤ 25, VIX ≥ 30 | Fear & Greed ≥ 75, VIX ≤ 13 |
| stock beating the S&P 500 by > 1% today | stock lagging it by > 1% |
| whales moved coins off exchanges (24h) | whales moved coins onto exchanges (24h) |
| | ATR3 > 2 × ATR14, fast or higher timeframe |

With `news = true`, up to three recent headlines are appended as plain text. Whale Alert and the news feeds have no history, so backtests never include them. With them on, the live loop sees more than any backtest did; turn them on to explore, not to trust.

### 4. Laya's answer

Laya is a typed-decision model: it doesn't generate text. Given a situation and a yes/no question, it returns a probability in one forward pass, in about 15 ms on an M2 Pro, fully local. The question (`[prompt]`) is *"Is the short-term outlook for this asset bullish?"*, and the answer is **P(bullish)**. Why this question is explained under [Prompt lab](#prompt-lab).

### 4b. Memory: Laya reads its own recent trades

Laya doesn't learn between calls. To let it see what its earlier readings led to, the state text can end with the asset's last N **closed trades and their outcome**, most recent first, plus the open position. It's the last N *trades*, not decisions: the last N decisions are mostly identical HOLDs and carry no information.

```
… Your recent trades on this asset: bought on a bullish reading (P 0.69), lost 3.4% (stop loss);
bought on a bullish reading (P 0.68), made 8.6% (time exit); … 4 of the last 5 lost money.
Now: long for 8 hours, down 3.2%.
```

Set per market with `memory_trades` (defaults: crypto 10, stocks 3; 0 = off).

`memory_details = true` adds the whole book as well: trade counts by side, winners and losers, realized net, and the open position's quantity, size, leverage, entry and stop. **It made things worse.** On 8 stocks over the same 9 unseen months, the default strategy went from +11.5% (no memory) to +1.1% with the detailed memory, while time in the market fell from 66% to 33%. Laya became far too cautious. It's off by default.

**Detailed state wording** (`format = "detailed"`: the Good/Bad summary plus every indicator value) was tested the same way. **It depends on the market**, so wording and memory detail are set per market:

| 9 unseen months, previous defaults | Current wording, no memory | Current wording + detailed memory | Detailed wording, no memory | Detailed wording + detailed memory |
|---|---|---|---|---|
| Stocks (8) | **+11.5%** | +1.1% | +1.9% | +0.3% |
| Crypto (3) | −0.3% | −5.0% | +4.9% | **+6.3%** |

On stocks, Laya reads the summary well, and the numbers and the detailed memory only make it hesitate. On crypto, the values add about 5 points, and with them the detailed memory helps too. Defaults: **crypto `format = "detailed"`, `memory_details = true`; stocks `good_bad` with the short memory**. The crypto numbers were measured on the earlier spot/4h setup; a confirmation on the futures setup is in `logs/`. Because Laya's answer now depends on the account's own history, backtests ask Laya during the simulation instead of precomputing it.

Tested on 9 unseen months (a year of 1h candles; `walkforward.py --rolling --no-search --variants memory`):

| | Compounded | Worst month | Worst drawdown | Worst single trade | Time in market |
|---|---|---|---|---|---|
| Crypto, no memory | +0.95% | −3.2% | −8.4% | −1.4% | 28% |
| Crypto, last 3 trades | −0.97% | −2.9% | −7.8% | −1.5% | 25% |
| **Crypto, last 10 trades** | **+1.62%** | **−2.2%** | **−6.0%** | −1.4% | 23% |
| Stocks, no memory | +10.0% | −1.8% | −22.1% | −19.3% | 68% |
| **Stocks, last 3 trades** | **+10.1%** | **−1.2%** | −24.0% | **−5.0%** | 38% |
| Stocks, last 10 trades | +8.4% | −1.6% | −24.0% | −5.0% | 40% |

Memory mostly makes Laya **more cautious after losses**. Stocks spend 38% of the time in the market instead of 68%, and the worst single trade shrinks from −19% to −5%, for the same return. Crypto gains a little and loses less in its worst month. The effects are modest and come from one test. Choosing 10 for crypto and 3 for stocks after seeing it is mild hindsight.

### 5. Strategy, position size and leverage

Each market has its own `[<market>.strategy]`.

**Direction.**

| `direction` | Bullish signal | Bearish signal |
|---|---|---|
| `trend` | P ≥ `enter_above` | P ≤ `enter_below` |
| `reversion` | P ≤ `enter_below` (oversold) | P ≥ `enter_above` (overbought) |

**Actions**, in order:

| Situation | Action |
|---|---|
| open position, losses would eat the margin (futures) | CLOSE (liquidated) |
| stop-loss or take-profit level touched (if set) | CLOSE |
| open longer than `max_hold_candles` (if set) | CLOSE (time exit) |
| stocks with `flat_at_close`: last candle before the session ends | CLOSE (session close) |
| last trade on this asset less than `cooldown_seconds` ago | HOLD (cooldown) |
| open position and the signal points the other way (unless `exit_on_flip = false`) | CLOSE (signal flipped) |
| flat and bullish (and with the higher-timeframe trend, if `trend_filter = "htf"`) | LONG |
| flat and bearish, `market = "futures"` (same filter) | SHORT |
| anything else | HOLD |

**Position size comes from risk, not from the model:** size = equity × `risk_pct` ÷ (`sizing_atr` × ATR14). The size is then capped at `max_leverage` × equity on futures and 1× on spot. Stops don't change the size, so turning a stop on or off only changes the exits. Futures also pay or receive funding at 00:00, 08:00 and 16:00 UTC and can be liquidated.

**Protection options** (per market, all off when 0):

| Option | What it does |
|---|---|
| `stop_loss_atr`, `stop_loss_pct` | fixed stop from the entry, in ATR or percent; if both are set, the nearer one wins |
| `trailing_stop_atr` | a stop that follows the best price since entry, and only ever tightens |
| `breakeven_after_atr` | once the trade is this far in profit, the stop moves up to the entry |
| `loss_cooldown_seconds` | no new entry for this long after a losing trade |
| `max_entry_atr_ratio` | no new entry while short-term volatility (ATR3/ATR14) is above this |
| `pause_drawdown_pct`, `pause_seconds` | circuit breaker: after equity falls this far below its peak, no new entries for a while |

In backtests, stops trigger on the candle's high/low. A candle that *opens* beyond the stop (an overnight gap, for example) fills at its open, not at the stop, because that's what a real stop order would get. Which options help is in [Stop losses](#stop-losses-and-protection-two-years-of-evidence).

**Could Laya pick the size or the leverage?** Only if a bigger P meant a bigger move, and it doesn't: P's rank correlation with the next move is ±0.1 at best, and its sign changes between periods (see below). Size stays a risk rule; leverage stays a cap you choose.

### 6. Logging

Every decision is saved as it happens to `logs/decisions-<UTC date>.jsonl`. Each line has:
- the asset and every signal value
- the full Laya exchange: request, encoder input, raw answer
- how the strategy decided
- the trade and position
- the equity

On restart, the dashboard reloads that day's history.

## Dashboard

`./run.sh` opens **http://127.0.0.1:8765**. The page updates every second. The header stays at the top while you scroll, with the Live / Backtest switch always in the same place and the connection status on the right.

- **Summary bar:** paper equity, session P&L, open positions, trades, and how many markets are open, for whatever the filter shows.
- **Filter, right above the asset list:** **All** (the default: crypto and stocks together), **Crypto** or **Stocks**. Each option shows a count and a live dot when its market is open.
- **Asset tiles:** price and change, a sparkline, a P(bullish) gauge, the current action or position, and equity. Under All, each tile is tagged *crypto* or *stock*. Click a tile to open it. The URL (`#all/stocks:AAPL`) keeps the filter and the selection.
- **Strategy line** under the tiles: each shown market's strategy in plain words.
- **Detail panel:**
  - price chart with ▲ long, ▼ short and ● close markers, and hover to inspect any point
  - P(bullish) against the thresholds
  - stats, and the sentence Laya read with the raw values
- **Decision log**, filtered as Trades or All changes. On screen it keeps every trade and every round where something changed; unchanged repeats are only in the saved file, so trades never scroll away. **Click any row for its full context:**
  - how the strategy chose the action
  - every signal
  - the exact request sent to Laya
  - the token sequence its encoder read
  - its raw answer
  - the trade and position

  **copy JSON** copies it all.

![Detail panel for SOL: price, Laya's P(bullish) over time against the thresholds, position, what Laya read, and the decision log](docs/screenshots/live-detail.png)

Clicking a decision opens its full context: how the strategy chose the action, the trade and resulting position, and every signal value…

![An expanded LONG decision: the rule that fired, the trade, the position after it, and all signals](docs/screenshots/decision-context.png)

…and exactly what Laya was asked, the token sequence its encoder read, and its raw answer:

![The request sent to Laya, the decoded encoder input and Laya's answer for the same decision](docs/screenshots/decision-prompt.png)

**/backtest** shows the latest backtest report in the same layout, plus a summary table and the return curves of Laya vs rules-only vs buy & hold. Pick the days and markets and press **Run**. The backtest runs as a separate process, so live trading is unaffected, and the page reloads when it's done.

![Backtest report: averages for Laya, rules only and buy & hold, and each asset's result against holding it](docs/screenshots/backtest-overview.png)

![Backtest detail for META: trades on the price chart, P(bullish) over time, return curves of Laya vs rules only vs buy & hold, and the trade log](docs/screenshots/backtest-detail.png)

## Backtest

`backtest.py --days N [--markets crypto,stocks]` replays closed candles one at a time with exactly the live logic, so nothing from the future leaks in:

- **Slower data:** each candle only sees higher-timeframe candles, statistics and sentiment that existed when it closed.
- **Crypto futures statistics:** they exist for 30 days. Stock 15-minute candles for about 60 days.
- **Stops and targets:** they trigger on the candle's high/low and fill at their level.
- **Speed:** identical sentences reuse Laya's answer, so a week of all 11 assets takes a minute or two.

It compares Laya with **rules only** (the same strategy driven by the trend votes instead of Laya) and **buy & hold**.

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

## Configuration (`config.toml`)

| Key | Default | Meaning |
|---|---|---|
| `interval_seconds` | 1.0 | time between rounds |
| `model` | aac6fef/laya-multilingual-mlx | Laya checkpoint |
| `paper.capital_usdt` | 1000 | paper balance per asset |
| `prompt.question` / `format` | *outlook* / good_bad | what Laya is asked, and how the sentence is worded |
| `<market>.format` | detailed crypto, good_bad stocks | per-market wording: `good_bad`, `lists`, `detailed` (with values), `values` |
| `<market>.memory_details` | true crypto, false stocks | add the open position's size/entry/stop and trade counts to the memory |
| `<market>.memory_trades` | 10 crypto, 3 stocks | how many recent closed trades Laya reads (0 = none) |
| `<market>.symbols` | see above | assets (crypto quoted in USDT) |
| `<market>.kline_interval` | 1h | candle size: 1m, 5m, 15m, 1h |
| `<market>.fee_pct` | 0.05 crypto (futures taker), 0.02 stocks | fee per side, in percent |
| `stocks.benchmark` / `poll_seconds` | SPY / 30 | relative-strength benchmark, Yahoo polling rate |
| `<market>.strategy.market` | futures crypto, spot stocks | `spot` (long only) or `futures` (long + short, leverage, funding) |
| `<market>.strategy.signal` | laya | `laya`, or `votes` to trade on the four trend votes |
| `<market>.strategy.direction` | trend crypto, reversion stocks | `trend` or `reversion` |
| `<market>.strategy.enter_above` / `enter_below` | 0.65/0.20 crypto, 0.55/0.30 stocks | P(bullish) thresholds |
| `<market>.strategy.trend_filter` | htf crypto, none stocks | `htf`: only trade with the higher-timeframe trend |
| `<market>.strategy.risk_pct` | 1.5 crypto, 3.0 stocks | % of equity at risk per trade; sets the size (capped by leverage) |
| `<market>.strategy.max_leverage` | 1 | size cap (futures only); 2× and 3× lost heavily in tests |
| `<market>.strategy.sizing_atr` | 3.0 | size = equity × risk_pct ÷ (sizing_atr × ATR14) |
| `<market>.strategy.stop_loss_atr` / `take_profit_atr` | 4 / 0 crypto, 0 / 0 stocks | exits in ATR14 from the entry; 0 = off |
| `<market>.strategy.stop_loss_pct` | 0 | fixed stop in percent from the entry |
| `<market>.strategy.trailing_stop_atr` / `breakeven_after_atr` | 0 / 0 | trailing stop, move stop to entry after this profit |
| `<market>.strategy.loss_cooldown_seconds` | 0 | no new entry for this long after a loss |
| `<market>.strategy.max_entry_atr_ratio` | 0 | no new entry while ATR3/ATR14 is above this |
| `<market>.strategy.pause_drawdown_pct` / `pause_seconds` | 0 / 259200 | circuit breaker on equity drawdown |
| `<market>.strategy.max_hold_candles` | 70 crypto, 0 stocks | time exit (70 × 1h ≈ 3 days); 0 = off |
| `<market>.strategy.exit_on_flip` | true | close when the signal turns the other way |
| `<market>.strategy.flat_at_close` | false | stocks: close before the session ends, no overnight positions |
| `<market>.strategy.cooldown_seconds` | 7200 crypto, 3600 stocks | minimum time between trades on one asset |
| `context.whale_alerts` / `news` | false / false | live-only sources |
| `refresh.*` | 60 / 600 / 300 s | refresh rates of the slow sources |

## Files

```
config.toml     markets, assets, prompt, strategies, paper account
markets.py      data providers: Crypto (Binance) and Stocks (Yahoo Finance), live and history
core.py         indicators, trend votes, the sentence, Laya's question, strategy, paper account
laya_trader.py  live loop, dashboard server, backtest runner
backtest.py     historical replay and report (prepare once, simulate any strategy)
promptlab.py    which prompt makes P most predictive, checked on unseen data
walkforward.py  strategy search, chosen on older data and tested on newer data
dashboard.html  one page for live (polls the server) and backtest (data embedded)
run.sh          uv run python laya_trader.py "$@"
```
