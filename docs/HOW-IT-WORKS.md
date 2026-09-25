# How laya-trader works

Everything behind the short overview in the [README](../README.md): where the data comes from, every signal, the sentence Laya reads, its memory, the strategy rules, protection options, logging, how backtests avoid seeing the future, and every setting. For the experiments behind the defaults, see [RESEARCH.md](RESEARCH.md).


Each round, for every asset in `config.toml`:

```
public market data ──► signals ──► one sentence ──► Laya: P(bullish) ──► strategy ──► LONG / SHORT / CLOSE / HOLD
 Binance / Yahoo        indicators,  "Good: …          one forward pass    direction,     size from risk,
                        trend votes   Bad: …"           on the Mac GPU      exits, filter  paper account, log, dashboard
```

## 1. Markets and data

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

## 2. Signals

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

## 3. What Laya reads

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

## 4. Laya's answer

Laya is a typed-decision model: it doesn't generate text. Given a situation and a yes/no question, it returns a probability in one forward pass, in about 15 ms on an M2 Pro, fully local. The question (`[prompt]`) is *"Is the short-term outlook for this asset bullish?"*, and the answer is **P(bullish)**. Why this question is explained under [Prompt lab](RESEARCH.md#prompt-lab).

## 5. Memory: Laya reads its own recent trades

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

On stocks, Laya reads the summary well, and the numbers and the detailed memory only make it hesitate. On crypto, the values add about 5 points, and with them the detailed memory helps too. Defaults: **crypto `format = "detailed"`, `memory_details = true`; stocks `good_bad` with the short memory**. Confirmed on the new crypto futures setup (1×, 2h cooldown), same 9 months:

| Crypto, futures setup | Compounded | Worst month | Worst drawdown |
|---|---|---|---|
| Current wording, no memory | +12.1% | −4.8% | −17.1% |
| Current wording + memory | −11.9% | −7.7% | −13.2% |
| Detailed wording, no memory | +28.4% | −5.0% | −14.8% |
| **Detailed wording + detailed memory (default)** | **+26.1%** | **−4.4%** | **−10.2%** |
| Buy & hold | −8.1% | −29.8% | – |

**How much to trust single numbers:** the same "current wording, no memory" futures setup made +6.4% in a run a few hours earlier and +12.1% here. The only difference was that the test months started a few hours later. Single results are fragile. What held in every run is the *ordering*: on crypto the values help, and memory without them hurts. Because Laya's answer now depends on the account's own history, backtests ask Laya during the simulation instead of precomputing it.

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

## 6. Strategy, position size and leverage

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

In backtests, stops trigger on the candle's high/low. A candle that *opens* beyond the stop (an overnight gap, for example) fills at its open, not at the stop, because that's what a real stop order would get. Which options help is in [Stop losses](RESEARCH.md#stop-losses-and-protection-two-years-of-evidence).

**Could Laya pick the size or the leverage?** Only if a bigger P meant a bigger move, and it doesn't: P's rank correlation with the next move is ±0.1 at best, and its sign changes between periods (see below). Size stays a risk rule; leverage stays a cap you choose.

## 7. Logging

Every decision is saved as it happens to `logs/decisions-<UTC date>.jsonl`. Each line has:
- the asset and every signal value
- the full Laya exchange: request, encoder input, raw answer
- how the strategy decided
- the trade and position
- the equity

On restart, the dashboard reloads that day's history.

## 8. Backtests

`backtest.py --days N [--markets crypto,stocks]` replays closed candles one at a time with exactly the live logic, so nothing from the future leaks in:

- **Slower data:** each candle only sees higher-timeframe candles, statistics and sentiment that existed when it closed.
- **Crypto futures statistics:** they exist for 30 days. Stock 15-minute candles for about 60 days.
- **Stops and targets:** they trigger on the candle's high/low and fill at their level.
- **Speed:** identical sentences reuse Laya's answer, so a week of all 11 assets takes a minute or two.

It compares Laya with **rules only** (the same strategy driven by the trend votes instead of Laya) and **buy & hold**.

## 9. Configuration (`config.toml`)

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

