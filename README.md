# laya-trader

A small experiment: every second, [Laya MLX](https://github.com/mizorewww/laya-mlx) reads live crypto market signals and says how bullish they look. A strategy layer turns that into **LONG**, **SHORT**, **CLOSE** or **HOLD** on a paper account, sizes each position from risk and caps leverage. A live dashboard shows each decision and its reason, and a backtest replays history with the same logic.

**Paper trading only.** No API keys, no orders, nothing is sent to an exchange. Not financial advice.

```bash
./run.sh                             # live loop + dashboard at http://127.0.0.1:8765
                                     # backtest dashboard at http://127.0.0.1:8765/backtest
uv run python backtest.py --days 3   # backtest from the terminal, writes backtest.html
```

Needs an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/). The Laya checkpoint (~650 MB) downloads on first run.

## How it works

Each round, for every coin in `config.toml`:

```
Binance public data ──► signals ──► one sentence ──► Laya: P(bullish) ──► strategy ──► LONG / SHORT / CLOSE / HOLD
   candles, funding,     indicators,   "Good: …           one forward pass    direction,      size from risk, leverage cap,
   long/short, F&G       trend votes    Bad: …"            on the Mac GPU      stops, cooldown  paper account + dashboard + log
```

### 1. Market data

Everything comes from public endpoints, no account needed.

| Data | Source | Refreshed |
|---|---|---|
| Last 100 candles of `kline_interval` (default 15m) | Binance spot `/api/v3/klines` | every round |
| Last 60 four-hour candles | Binance spot | every 60 s |
| Funding rate | Binance futures `/fapi/v1/premiumIndex` | every 60 s |
| Long/short account ratio | Binance futures | every 60 s |
| Fear & Greed index | alternative.me | every 10 min |

Requests reuse open HTTPS connections. Opening a new one costs 0.3–0.9 s, which alone would break a one-second loop. With reuse, a round for three coins takes about 0.3 s. If a request fails, that round is skipped and the next one tries again.

### 2. Signals

From the candles:

- **EMA20** (exponential moving average)
- **MACD** (EMA12 − EMA26)
- **RSI14** (Wilder smoothing, as exchange charts use)
- **ATR3 and ATR14** (average true range, i.e. volatility)
- **Volume ratio**: the last closed candle against the 20 before it
- **Buyer share of volume** over the last five closed candles, from Binance's taker-buy volume

Four of them become **trend votes** of +1, 0 or −1:

| Vote | +1 when | −1 when | 0 (neutral band) |
|---|---|---|---|
| Price vs EMA20 | more than 0.12% above | more than 0.12% below | within ±0.12% |
| 4h EMA20 vs EMA50 | EMA20 above | EMA20 below | gap under 0.05% |
| MACD | positive | negative | smaller than 2% of ATR14 |
| RSI14 | above 58 | below 42 | 42–58 |

The neutral bands keep tiny wiggles from counting as trend. Most of the time the sum is between −1 and +1, which reads as "no clear trend".

### 3. What Laya reads

The signals are turned into one sentence. Bullish facts go under *Good*, bearish facts under *Bad*:

```
Crypto market signals. Good: MACD positive, buyers dominate recent order flow.
Bad: RSI overbought, funding high, longs crowded.
```

| Bullish (Good) when | Bearish (Bad) when |
|---|---|
| trend votes sum ≥ +2 | trend votes sum ≤ −2 |
| price > 0.12% above EMA20 | price > 0.12% below EMA20 |
| MACD positive | MACD negative |
| buyers > 55% of recent volume | buyers < 45% of recent volume |
| volume > 1.5× average, mostly buying | volume > 1.5× average, mostly selling |
| RSI below 30 (oversold) | RSI above 70 (overbought) |
| Fear & Greed ≤ 25 (extreme fear) | Fear & Greed ≥ 75 (extreme greed) |
| long/short ratio < 0.8 (crowd is short) | funding > 0.05% (longs crowded) |
| | ATR3 > 2 × ATR14 (extreme volatility) |

### 4. Laya's answer

Laya is a typed-decision model. It doesn't generate text. Given a situation and a yes/no question, it returns a probability in a single forward pass: about 15 ms on an M2 Pro, fully local.

Every round, each coin gets the same question:

> *Is the short-term outlook for this coin bullish?*

The answer is **P(bullish)**, between 0 and 1.

Why this question: asking Laya directly "BUY or HOLD?" was mostly noise. On 36 hand-labelled scenarios, the best action wording picked the right action 25 times, and leaned heavily toward HOLD. Other wordings did no better than a coin flip. Asking it to *judge the signals* worked much better. It scored bull scenarios above bear scenarios in 94% of pairs, with average P of 0.57 for bull, 0.47 for mixed and 0.19 for bear. So Laya judges and fixed rules act.

### 5. Strategy, position size and leverage

**Direction.** P(bullish) becomes a signal:

| `direction` | Bullish signal | Bearish signal |
|---|---|---|
| `trend` | P ≥ `enter_above` | P ≤ `enter_below` |
| `reversion` | P ≤ `enter_below` (oversold, expect a bounce) | P ≥ `enter_above` (overbought, expect a pullback) |

**Actions**, checked in this order every round:

| Situation | Action |
|---|---|
| open position, losses would eat the margin | CLOSE (liquidated) |
| open position, stop-loss level touched | CLOSE (stop loss) |
| open position, take-profit level touched | CLOSE (take profit) |
| a trade on this coin less than `cooldown_seconds` ago | HOLD (cooldown) |
| open position and the signal points the other way | CLOSE (signal flipped) |
| flat and bullish | LONG |
| flat and bearish, `market = "futures"` | SHORT |
| anything else | HOLD |

Stops sit `stop_loss_atr` × ATR14 from the entry, targets `take_profit_atr` × ATR14. In a backtest they trigger on the candle's high/low and fill at their level. Live, they're checked against the price every round.

**Position size comes from risk, not from the model.** Each trade is sized so that hitting the stop loses `risk_pct` of equity:

```
quantity = equity × risk_pct / stop distance
```

The size is then capped: `max_leverage` × equity on futures, 1× equity on spot. Quiet markets have a small ATR and tight stops, so the risk formula asks for large positions and the cap usually decides. Volatile markets get smaller positions automatically.

**Futures** add shorts, leverage, funding and liquidation:
- Funding is paid or received at 00:00, 08:00 and 16:00 UTC at the current rate: longs pay when it's positive, shorts receive.
- A position is liquidated when equity falls below the 0.5% maintenance margin.

**Paper account.** One per coin (`capital_usdt`, default 1,000). `fee_pct` is charged on every entry and exit. The accounts reset when the program restarts.

**Could Laya pick the size or the leverage?** Only if a higher P meant a bigger next move. I measured that on 7–45 days of history: P against the return over the next 15 minutes to 4 hours, for BTC, ETH and SOL. On 1-minute and 5-minute candles the correlation is slightly *negative*, between −0.01 and −0.08: when Laya reads the market as bearish, price tends to bounce. On 15-minute candles it's near zero, at best +0.07. Sizing by P would scale positions with noise, so size stays a risk rule and leverage stays a cap you choose.

### 6. Logging

Every decision is saved to `logs/decisions-<UTC date>.jsonl` as it happens. Each line holds the time, coin, all signal values, the full Laya exchange (request, encoder input, raw answer), how the strategy decided, the action and reason, the trade and position, and the equity. On restart, the dashboard reloads that day's history.

## Live dashboard

`./run.sh` starts the loop and opens **http://127.0.0.1:8765**. The page updates every second. For each coin:

- price chart with ▲ longs, ▼ shorts and ● closes
- P(bullish) chart with the buy and sell thresholds
- paper equity, position (side, leverage, move since entry) and trade count
- the sentence Laya read in the latest round, plus the raw values (RSI, votes, buyer share, funding, long/short, Fear & Greed)
- a **decision log**, filtered three ways:
  - **Trades**: only LONG, SHORT and CLOSE
  - **Changes**: trades, plus every round where Laya's input changed
  - **All**: every round
- **click any row** for its full context:
  - how the strategy turned P into the action: zone, signal, position before, rule that fired
  - every signal value
  - the exact request sent to Laya
  - the token sequence its encoder actually read, decoded back to text
  - Laya's raw answer
  - the trade and resulting position, if there was one

  A **copy JSON** button copies all of it. Live, the last 2,000 decisions per coin keep this detail and are fetched on click; backtest reports embed it for every trade.
- a link to download the saved log

Options: `--rounds N`, `--log path.jsonl`, `--no-log`, `--no-open`, `--port`, `--config`.

## Backtest

**http://127.0.0.1:8765/backtest** shows the latest report. Pick the number of days and press **Run**. The backtest runs as a separate process with its own copy of the model, so the live loop is unaffected. The page shows progress, then reloads with the new report. From the terminal, `uv run python backtest.py --days N` writes `backtest.html`, which also works opened as a file.

How the replay works:

- It downloads historical candles from Binance and replays them one closed candle at a time.
- Each step sees only the 100 candles before it and the four-hour candles already closed. Nothing from the future leaks in.
- Signals, sentence, question, strategy and paper account are exactly the same as live, including funding on futures.
- Fear & Greed (daily) and funding (every 8 hours) use their historical values.
- The long/short ratio has no long history on Binance, so backtests run without it.
- When two candles produce the same sentence, Laya's earlier answer is reused. A day of 1-minute candles takes a few seconds.

Every run compares three strategies on the same candles, fees and stops:

| Strategy | What it does |
|---|---|
| **Laya** | the full loop above |
| **Rules only** | the same strategy with the trend votes in place of Laya (sum ≥ +2 bullish, ≤ −2 bearish). Shows what Laya adds on top of plain indicators. |
| **Buy & hold** | buy at the first candle, sell at the last |

The report has a summary table (return, maximum drawdown, trades, win rate of closed trades), per-coin charts, the three return curves, and a log of every trade with what Laya read at that moment.

## Walk-forward: picking a strategy honestly

`uv run python walkforward.py` tries 216 strategies:

- candles: 1m, 5m, 15m
- score: Laya's P, or the trend votes alone
- direction: trend or reversion
- three pairs of thresholds
- setup: spot, futures 1×, futures 3×
- cooldown: 1 or 5 candles

It picks the best by average return on the older **14 days**, then runs the top picks on the most recent **7 days**, which played no part in the choice. A strategy that shines only on the first window was fitted to noise. The run takes about 90 seconds; signals and Laya's answers are computed once per candle size.

**Result (run on 24 Sept 2026, BTC/ETH/SOL):**

- **No strategy made money, even on the 14 days it was chosen on.** The best lost 1.5%.
- **On the unseen 7 days**, the top picks lost 2–14%, while buy & hold made +11.6% in a rising week.
- **What lost least:** 15-minute candles, reversion, spot, few trades.
- **What lost most:** 1-minute candles and leverage. Fees are paid on the full leveraged size, so 3× leverage triples them. The old default (1m, trend, futures 3×) lost 31–40% in a single day.
- **Best Laya strategy:** −5.5% on the training window, −3.5% on the test window. Trend votes alone did slightly better on both.

The defaults in `config.toml` are that best Laya strategy: 15m candles, reversion, spot, thresholds 0.65 / 0.20, one trade per 15 minutes. On the last 7 days it returned −4.9% (BTC), −6.3% (ETH) and −7.1% (SOL), against +9% to +16% for buy & hold.

**The honest conclusion:** these public signals, read by Laya or by plain rules, don't carry an edge large enough to beat trading fees. Leverage and faster trading make that worse, not better. Improving results needs better information, not more tuning:
- data the market hasn't priced in yet
- cheaper execution (maker fees, fewer trades)
- longer holding periods

Rerun `walkforward.py` on new data before trusting any setting.

## Configuration (`config.toml`)

| Key | Default | Meaning |
|---|---|---|
| `symbols` | BTC, ETH, SOL | coins, quoted in USDT |
| `interval_seconds` | 1.0 | time between rounds |
| `kline_interval` | 15m | candle size for the indicators |
| `model` | aac6fef/laya-multilingual-mlx | Laya checkpoint |
| `paper.capital_usdt` / `fee_pct` | 1000 / 0.1 | paper balance per coin, fee per side in percent |
| `strategy.market` | spot | `spot` (long only, 1×) or `futures` (long + short, leverage, funding) |
| `strategy.direction` | reversion | `trend` or `reversion` |
| `strategy.enter_above` / `enter_below` | 0.65 / 0.20 | P(bullish) thresholds |
| `strategy.risk_pct` | 1.0 | % of equity lost if the stop is hit; sets the size |
| `strategy.max_leverage` | 3.0 | size cap as a multiple of equity (futures only) |
| `strategy.stop_loss_atr` / `take_profit_atr` | 1.5 / 3.0 | exits, in ATR14 multiples from the entry |
| `strategy.cooldown_seconds` | 900 | minimum time between trades on one coin |
| `refresh.context_seconds` / `fear_greed_seconds` | 60 / 600 | refresh rate of the slow signals |
| `binance.spot` / `futures` | public Binance hosts | data endpoints |

## Files

```
config.toml     coins, timing, thresholds, paper account
core.py         Binance data, indicators, trend votes, the sentence, Laya's question, strategy, paper account
laya_trader.py  live loop, dashboard server, backtest runner
backtest.py     historical replay and report (prepare once, simulate any strategy)
walkforward.py  216-strategy search, chosen on older data and tested on newer data
dashboard.html  one page for live (polls the server) and backtest (data embedded)
run.sh          uv run python laya_trader.py "$@"
```
