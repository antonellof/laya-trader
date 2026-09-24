# laya-trader

An experiment: every second, [Laya MLX](https://github.com/mizorewww/laya-mlx) reads live market signals for **crypto** (Binance) and **S&P 500 stocks** (Yahoo Finance) and says how bullish they look. Each market's strategy turns that into **LONG**, **SHORT**, **CLOSE** or **HOLD** on a paper account, sizes the position from risk and caps leverage. A live dashboard shows every decision and its full context, a backtest replays history with the same logic, and two research tools pick the prompt and the strategy from evidence.

**Paper trading only.** No API keys, no orders, nothing is sent to an exchange or broker. Not financial advice.

```bash
./run.sh                               # live loop + dashboard at http://127.0.0.1:8765
uv run python backtest.py --days 7     # historical replay of every market, writes backtest.html
uv run python promptlab.py             # which prompt makes Laya's P most predictive
uv run python walkforward.py           # which strategy holds up on data it never saw
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
| Candles | Binance spot, every round | Yahoo Finance chart API, every `poll_seconds` (15 s) |
| Trading hours | 24/7 | regular US session only; outside it: HOLD "market closed" |
| Higher timeframe | 4h candles | daily candles |
| Futures data | funding, open interest, long/short ratio, taker buy/sell ratio | – |
| Sentiment | Fear & Greed index | VIX |
| Benchmark | – | SPY, for relative strength |
| Fees (paper) | 0.1% per side | 0.02% per side (approximates the spread) |
| Optional, live only | Whale Alert transfers, CoinJournal headlines | Yahoo Finance headlines per symbol |

Yahoo's chart API is public but unofficial, so stocks are polled every 15 seconds rather than every second. Requests reuse open HTTPS connections. A round for all 11 assets takes about 0.3 s (median) to fetch. Laya's answers are cached by sentence, so an unchanged market doesn't run the model again. If one asset fails, only that asset is skipped.

### 2. Signals

- **Fast candles** (`kline_interval`, default 15m):
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
| last trade on this asset less than `cooldown_seconds` ago | HOLD (cooldown) |
| open position and the signal points the other way | CLOSE (signal flipped) |
| flat and bullish (and with the higher-timeframe trend, if `trend_filter = "htf"`) | LONG |
| flat and bearish, `market = "futures"` (same filter) | SHORT |
| anything else | HOLD |

**Position size comes from risk, not from the model.** A trade is sized so that hitting the stop loses `risk_pct` of equity; with no stop, it's sized as if the stop were 3 ATR away. The size is then capped at `max_leverage` × equity on futures and 1× on spot. Futures also pay or receive funding at 00:00, 08:00 and 16:00 UTC and can be liquidated.

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

`./run.sh` opens **http://127.0.0.1:8765**. The page updates every second.

- **Market tabs** (Crypto / Stocks): each has a live dot when its market is open.
- **Market bar:** paper equity, session P&L, open positions, trades, and the strategy in plain words.
- **Asset tiles:** price and change, a sparkline, a P(bullish) gauge, the current action or position, and equity. Click one to open it. The URL (`#stocks/AAPL`) keeps the selection.
- **Detail panel:**
  - price chart with ▲ long, ▼ short and ● close markers, and hover to inspect any point
  - P(bullish) against the thresholds
  - stats, and the sentence Laya read with the raw values
- **Decision log**, filtered as Trades / Changes / All. **Click any row for its full context:**
  - how the strategy chose the action
  - every signal
  - the exact request sent to Laya
  - the token sequence its encoder read
  - its raw answer
  - the trade and position

  **copy JSON** copies it all.

**/backtest** shows the latest backtest report in the same layout, plus a summary table and the return curves of Laya vs rules-only vs buy & hold. Pick the days and markets and press **Run**. The backtest runs as a separate process, so live trading is unaffected, and the page reloads when it's done.

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

These two are the defaults in `config.toml`. On the same 30-day crypto window as the run above they give −4.0% (BTC), +17.5% (ETH), +7.6% (SOL), versus −26.7%, −17.7% and −11.7% before. That window overlaps the one used to choose them, though, so the unseen-part numbers are the honest ones.

**The honest conclusion:**
- The changes turned heavy losses into small out-of-sample gains, with smaller drawdowns than holding. In rising markets, they still trail buy & hold.
- 10–18 unseen days is a short test, and the crypto signal changed direction within a month.
- **No stop loss** means an open position is exposed to a crash until the signal flips. The tested alternative with stops did worse, but a crash was not in the sample.
- Rerun `promptlab.py` and `walkforward.py` on fresh data before trusting any setting.

## Configuration (`config.toml`)

| Key | Default | Meaning |
|---|---|---|
| `interval_seconds` | 1.0 | time between rounds |
| `model` | aac6fef/laya-multilingual-mlx | Laya checkpoint |
| `paper.capital_usdt` | 1000 | paper balance per asset |
| `prompt.question` / `format` | *outlook* / good_bad | what Laya is asked, and how the sentence is worded |
| `<market>.symbols` | see above | assets (crypto quoted in USDT) |
| `<market>.kline_interval` | 15m | candle size: 1m, 5m, 15m, 1h |
| `<market>.fee_pct` | 0.1 crypto, 0.02 stocks | fee per side, in percent |
| `stocks.benchmark` / `poll_seconds` | SPY / 15 | relative-strength benchmark, Yahoo polling rate |
| `<market>.strategy.market` | spot | `spot` (long only) or `futures` (long + short, leverage, funding) |
| `<market>.strategy.direction` | reversion | `trend` or `reversion` |
| `<market>.strategy.enter_above` / `enter_below` | 0.75/0.15 crypto, 0.55/0.30 stocks | P(bullish) thresholds |
| `<market>.strategy.trend_filter` | none | `htf`: only trade with the higher-timeframe trend |
| `<market>.strategy.risk_pct` | 1.0 | % of equity at risk per trade; sets the size |
| `<market>.strategy.max_leverage` | 3 crypto, 1 stocks | size cap (futures only) |
| `<market>.strategy.stop_loss_atr` / `take_profit_atr` | 0 / 0 | exits in ATR14 from the entry; 0 = off |
| `<market>.strategy.max_hold_candles` | 0 | time exit; 0 = off |
| `<market>.strategy.cooldown_seconds` | 14400 | minimum time between trades on one asset |
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
