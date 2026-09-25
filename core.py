"""Shared pieces: Binance data, indicators, Laya's question, trading rules, paper book."""

import http.client
import json
import os
import sys
import threading
import time
from urllib.parse import urlsplit

QUESTION = "Is the short-term outlook for this asset bullish?"


# --- Data ---------------------------------------------------------------------------------

_local = threading.local()


def get_json(url, timeout=3):
    """GET over a kept-alive HTTPS connection per host and thread (a new TLS handshake
    per request costs 0.3-0.9 s, too slow for one decision per second)."""
    parts = urlsplit(url)
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    connections = getattr(_local, "connections", None)
    if connections is None:
        connections = _local.connections = {}
    for attempt in (0, 1):
        connection = connections.get(parts.netloc)
        if connection is None:
            connection = http.client.HTTPSConnection(parts.netloc, timeout=timeout)
            connections[parts.netloc] = connection
        try:
            connection.request("GET", path, headers={"User-Agent": "laya-trader"})
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} from {parts.netloc}: {body[:120]!r}")
            return json.loads(body)
        except (http.client.HTTPException, OSError):
            connection.close()
            connections.pop(parts.netloc, None)
            if attempt:
                raise
    raise AssertionError("unreachable")


class Cache:
    """Slow signals (4h stack, funding, sentiment) refresh on their own schedule."""

    def __init__(self):
        self.values = {}

    def get(self, key, ttl, fetch, default=None):
        value, stamp = self.values.get(key, (default, float("-inf")))
        if time.monotonic() - stamp >= ttl:
            try:
                value = fetch()
            except Exception:  # Keep the last good value; the fast loop must not stop.
                pass
            self.values[key] = (value, time.monotonic())
        return value


# --- Indicators ---------------------------------------------------------------------------


def ema(prices, period):
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2 / (period + 1)
    value = sum(prices[:period]) / period
    for price in prices[period:]:
        value = price * k + value * (1 - k)
    return value


def macd(prices):
    return ema(prices, 12) - ema(prices, 26) if len(prices) >= 26 else 0.0


def rsi(prices, period=14):
    """Wilder's smoothing, matching exchange charts."""
    if len(prices) < period + 1:
        return 50.0
    deltas = [b - a for a, b in zip(prices, prices[1:])]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_gain, avg_loss = sum(gains[:period]) / period, sum(losses[:period]) / period
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (gain + (period - 1) * avg_gain) / period
        avg_loss = (loss + (period - 1) * avg_loss) / period
    return 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)


def atr(klines, period):
    if len(klines) < period + 1:
        return 0.0
    ranges = [
        max(
            float(k[2]) - float(k[3]),
            abs(float(k[2]) - float(p[4])),
            abs(float(k[3]) - float(p[4])),
        )
        for p, k in zip(klines, klines[1:])
    ]
    return sum(ranges[-period:]) / period


def trend_votes(price, ema20, ema20_4h, ema50_4h, macd_value, rsi14, atr14):
    """Four +1/0/-1 votes. The neutral bands keep noise from counting as trend."""
    votes = []
    pct = (price - ema20) / ema20 * 100 if ema20 else 0
    votes.append(0 if abs(pct) < 0.12 else 1 if pct > 0 else -1)
    gap = abs(ema20_4h - ema50_4h) / ema50_4h * 100 if ema50_4h else 0
    votes.append(0 if gap < 0.05 else 1 if ema20_4h > ema50_4h else -1)
    noise = max(atr14 * 0.02, price * 1e-5, 1e-12)
    votes.append(0 if abs(macd_value) < noise else 1 if macd_value > 0 else -1)
    votes.append(1 if rsi14 > 58 else -1 if rsi14 < 42 else 0)
    return votes


def pivots(daily):
    """Classic floor pivots from the last completed day: PP, S1, S2, R1, R2."""
    if not daily:
        return None
    high, low, close = float(daily[-1][2]), float(daily[-1][3]), float(daily[-1][4])
    pp = (high + low + close) / 3
    return {
        "pp": pp,
        "r1": 2 * pp - low,
        "s1": 2 * pp - high,
        "r2": pp + high - low,
        "s2": pp - high + low,
    }


def compute_signals(klines, ctx):
    """klines: the latest ~100 candles of the fast timeframe, oldest first.

    ctx holds the slower context; every item is optional (None when a market lacks it):
      k_htf, htf_label   higher-timeframe candles (crypto: 4h, stocks: daily)
      k1h                completed 1-hour candles (25)
      k1d                the last completed daily candle, for pivots
      day_ref, day_label reference close and today's range (stocks); else the last 24 hours
      funding, long_short, taker_ratio, oi   futures statistics (crypto only)
      sentiment          {"kind": "fear_greed" | "vix", "value": ...}
      relative_strength  today's change minus the benchmark's (stocks)
      whales, headlines  live-only extras
    """
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    taker = [k[9] for k in klines]
    price = closes[-1]
    ema20, macd_value, rsi14 = ema(closes, 20), macd(closes), rsi(closes, 14)
    atr3, atr14 = atr(klines, 3), atr(klines, 14)
    k_htf = ctx.get("k_htf") or []
    closes_htf = [float(k[4]) for k in k_htf]
    ema20_htf, ema50_htf = (ema(closes_htf, 20), ema(closes_htf, 50)) if closes_htf else (0.0, 0.0)
    recent = slice(-6, -1)  # last five closed candles
    base_volume = sum(volumes[-22:-2]) / 20
    buy_ratio = None
    if all(t is not None for t in taker[recent]) and sum(volumes[recent]):
        buy_ratio = sum(float(t) for t in taker[recent]) / sum(volumes[recent])

    k1h = ctx.get("k1h") or []
    volumes_1h = [float(k[5]) for k in k1h]
    day_ref = ctx.get("day_ref")
    if day_ref:
        change = (price / day_ref["reference"] - 1) * 100
        high, low = max(day_ref["high"], price), min(day_ref["low"], price)
    else:
        day = k1h[-24:]
        change = (price / float(day[0][1]) - 1) * 100 if len(day) >= 24 else None
        high = max((float(k[2]) for k in day), default=None)
        low = min((float(k[3]) for k in day), default=None)
    oi = ctx.get("oi") or []
    return {
        "price": price,
        "open": float(klines[-1][1]),
        "high": float(klines[-1][2]),
        "low": float(klines[-1][3]),
        "ema20": ema20,
        "dist_ema20_pct": (price - ema20) / ema20 * 100 if ema20 else 0.0,
        "macd": macd_value,
        "macd_slope": macd_value - macd(closes[:-3]),  # change over the last 3 candles
        "macd_noise": max(atr14 * 0.02, price * 1e-5),
        "rsi7": rsi(closes, 7),
        "rsi14": rsi14,
        "rsi14_slope": rsi14 - rsi(closes[:-3], 14),
        "atr14": atr14,
        "atr_ratio": atr3 / atr14 if atr14 else 1.0,
        "volume_ratio": volumes[-2] / base_volume if base_volume else 1.0,
        "buy_ratio": buy_ratio,
        "htf_label": ctx.get("htf_label", "4h"),
        "rsi14_htf": rsi(closes_htf, 14) if closes_htf else None,
        "macd_htf": macd(closes_htf) if closes_htf else None,
        "atr_ratio_htf": atr(k_htf, 3) / atr(k_htf, 14)
        if len(k_htf) > 15 and atr(k_htf, 14)
        else None,
        "volume_ratio_1h": (
            volumes_1h[-1] / (sum(volumes_1h[-24:-1]) / 23)
            if len(volumes_1h) >= 24 and sum(volumes_1h[-24:-1])
            else None
        ),
        "day_label": ctx.get("day_label", "24h"),
        "change_day_pct": change,
        "range_day_pos": (price - low) / (high - low) if high is not None and high > low else None,
        "pivots": pivots(ctx.get("k1d")),
        "oi_change_1h_pct": (oi[-1] / oi[0] - 1) * 100 if len(oi) >= 13 and oi[0] else None,
        "taker_ratio": ctx.get("taker_ratio"),
        "funding": ctx.get("funding"),
        "long_short": ctx.get("long_short"),
        "sentiment": ctx.get("sentiment"),
        "relative_strength": ctx.get("relative_strength"),
        "whales": ctx.get("whales"),
        "headlines": ctx.get("headlines"),
        "closes_soon": bool(ctx.get("closes_soon")),  # last candle before the session ends
        "votes": trend_votes(price, ema20, ema20_htf, ema50_htf, macd_value, rsi14, atr14),
    }


# --- Signals to Laya ------------------------------------------------------------------------


def facts(s):
    """(bullish facts, bearish facts) in plain words. A missing signal never counts."""
    trend = sum(s["votes"])
    price, pv, htf, day = s["price"], s.get("pivots"), s["htf_label"], s["day_label"]
    oi, change, flow = s.get("oi_change_1h_pct"), s.get("change_day_pct"), s.get("buy_ratio")
    sentiment, whales, rs = (
        s.get("sentiment") or {},
        s.get("whales") or {},
        s.get("relative_strength"),
    )
    fear_greed = sentiment.get("value") if sentiment.get("kind") == "fear_greed" else None
    vix = sentiment.get("value") if sentiment.get("kind") == "vix" else None

    def near(level):
        return abs(price / level - 1) < 0.003  # within 0.3%

    def known(*values):
        return all(v is not None for v in values)

    bullish = [
        (trend >= 2, f"trend votes bullish ({trend:+d} of 4)"),
        (s["dist_ema20_pct"] > 0.12, "price above EMA20"),
        (s["macd"] > 0, "MACD positive"),
        (s["macd_slope"] > s["macd_noise"], "MACD rising"),
        (known(flow) and flow > 0.55, "buyers dominate recent order flow"),
        (known(flow) and s["volume_ratio"] > 1.5 and flow > 0.5, "rising volume on buying"),
        (s["rsi14"] < 30, "RSI oversold"),
        (s["rsi7"] < 20, "short-term RSI deeply oversold"),
        (
            known(s["rsi14_htf"], s["macd_htf"]) and s["rsi14_htf"] > 55 and s["macd_htf"] > 0,
            f"{htf} momentum bullish",
        ),
        (
            known(s["volume_ratio_1h"]) and s["volume_ratio_1h"] > 2 and (change or 0) > 0,
            "hourly volume surge while rising",
        ),
        (known(s["range_day_pos"]) and s["range_day_pos"] > 0.9, f"price near its {day} high"),
        (known(pv) and price > pv["r1"], "price broke above daily resistance R1"),
        (known(pv) and (near(pv["s1"]) or near(pv["s2"])), "price at daily support"),
        (
            known(oi, change) and oi > 1 and change > 0,
            "open interest rising with price (new longs)",
        ),
        (known(s["taker_ratio"]) and s["taker_ratio"] > 1.2, "futures buyers aggressive"),
        (known(s["long_short"]) and s["long_short"] < 0.8, "crowd is short"),
        (known(fear_greed) and fear_greed <= 25, "extreme fear in the market"),
        (known(vix) and vix >= 30, f"market fear high (VIX {vix:.0f})" if vix else ""),
        (known(rs) and rs > 1, f"outperforming the S&P 500 {day}"),
        (
            whales.get("outflow", 0) > whales.get("inflow", 0),
            "whales moved coins off exchanges in the last day",
        ),
    ]
    bearish = [
        (trend <= -2, f"trend votes bearish ({trend:+d} of 4)"),
        (s["dist_ema20_pct"] < -0.12, "price below EMA20"),
        (s["macd"] < 0, "MACD negative"),
        (s["macd_slope"] < -s["macd_noise"], "MACD falling"),
        (known(flow) and flow < 0.45, "sellers dominate recent order flow"),
        (known(flow) and s["volume_ratio"] > 1.5 and flow < 0.5, "rising volume on selling"),
        (s["rsi14"] > 70, "RSI overbought"),
        (s["rsi7"] > 80, "short-term RSI extremely overbought"),
        (
            known(s["rsi14_htf"], s["macd_htf"]) and s["rsi14_htf"] < 45 and s["macd_htf"] < 0,
            f"{htf} momentum bearish",
        ),
        (
            known(s["volume_ratio_1h"]) and s["volume_ratio_1h"] > 2 and (change or 0) < 0,
            "hourly volume surge while falling",
        ),
        (known(s["range_day_pos"]) and s["range_day_pos"] < 0.1, f"price near its {day} low"),
        (known(pv) and price < pv["s1"], "price broke below daily support S1"),
        (known(pv) and (near(pv["r1"]) or near(pv["r2"])), "price at daily resistance"),
        (
            known(oi, change) and oi > 1 and change < 0,
            "open interest rising as price falls (new shorts)",
        ),
        (known(s["taker_ratio"]) and s["taker_ratio"] < 0.8, "futures sellers aggressive"),
        (known(s["funding"]) and s["funding"] > 0.0005, "funding high, longs crowded"),
        (known(fear_greed) and fear_greed >= 75, "extreme greed in the market"),
        (known(vix) and vix <= 13, f"market complacent (VIX {vix:.0f})" if vix else ""),
        (known(rs) and rs < -1, f"underperforming the S&P 500 {day}"),
        (s["atr_ratio"] > 2.0, "volatility extreme"),
        (known(s["atr_ratio_htf"]) and s["atr_ratio_htf"] > 2.0, f"{htf} volatility extreme"),
        (
            whales.get("inflow", 0) > whales.get("outflow", 0),
            "whales moved coins onto exchanges in the last day",
        ),
    ]
    return [t for ok, t in bullish if ok], [t for ok, t in bearish if ok]


PROMPT_FORMATS = ("good_bad", "lists")


def describe(s, market="Crypto", fmt="good_bad"):
    """The signals as Laya's state text.

    good_bad: "Good: <bullish facts>. Bad: <bearish facts>."
    lists:    "Bullish signals: ... Bearish signals: ..." (neutral wording)
    """
    good, bad = facts(s)
    if fmt == "lists":
        text = f"Bullish signals: {', '.join(good) or 'none'}. Bearish signals: {', '.join(bad) or 'none'}."
    else:
        text = f"Good: {', '.join(good)}." if good else ""
        if bad:
            text += f" Bad: {', '.join(bad)}."
        text = text.strip() or "No clear signals."
    state = f"{market} market signals. {text}"
    if s.get("headlines"):
        state += " Recent headlines: " + " | ".join(s["headlines"][:3]) + "."
    return state


# --- Memory: the asset's own recent trades, in words ------------------------------------------


def memory_text(account, n, price, now):
    """The last n closed trades on this asset with their outcome, most recent first, plus
    the open position. Laya doesn't learn between calls; this is how it sees what its
    earlier readings led to."""
    trips, entry = [], None
    for trade in account.trades:
        if trade["action"] in ("LONG", "SHORT"):
            entry = trade
        elif trade["action"] == "CLOSE" and entry:
            result = trade["pnl"] / entry["notional"] * 100 if entry["notional"] else 0.0
            trips.append((entry, result, trade["reason"]))
            entry = None
    parts = []
    for entry, result, reason in reversed(trips[-n:]):
        verb = "bought" if entry["action"] == "LONG" else "sold short"
        reading = f" (P {entry['score']:.2f})" if entry.get("score") is not None else ""
        outcome = f"made {result:.1f}%" if result >= 0 else f"lost {-result:.1f}%"
        parts.append(f"{verb} on a {entry['reason']} reading{reading}, {outcome} ({reason})")
    if parts:
        losses = sum(r < 0 for _, r, _ in trips[-n:])
        text = f"Your recent trades on this asset: {'; '.join(parts)}. {losses} of the last {len(parts)} lost money."
    else:
        text = "No earlier trades on this asset."
    held = account.position
    if held:
        hours = (now - held["opened"]) / 3600
        move = held["side"] * (price / held["entry"] - 1) * 100
        side = "long" if held["side"] > 0 else "short"
        text += (
            f" Now: {side} for {hours:.0f} hours, {'up' if move >= 0 else 'down'} {abs(move):.1f}%."
        )
    return text


# --- Live-only context: whale transfers and news ----------------------------------------------

EXCHANGES = (
    "binance",
    "coinbase",
    "kraken",
    "okx",
    "okex",
    "bybit",
    "bitfinex",
    "bitstamp",
    "kucoin",
    "gemini",
    "htx",
    "huobi",
    "gate",
    "bitget",
    "upbit",
    "crypto.com",
)
NAMES = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "ether", "eth"),
    "SOL": ("solana", "sol"),
    "XRP": ("xrp", "ripple"),
    "BNB": ("bnb",),
    "DOGE": ("dogecoin", "doge"),
}


def fetch_text(url, timeout=6):
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "laya-trader"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def whale_flows(coins, window_seconds=86400):
    """Large transfers in the last 24 hours from Whale Alert's public feed, per coin:
    USD moved onto exchanges (inflow, often before selling) and off them (outflow)."""
    import csv
    import io

    alerts = json.loads(fetch_text("https://whale-alert.io/data.json?alerts=50"))["alerts"]
    flows = {c: {"inflow": 0.0, "outflow": 0.0} for c in coins}
    cutoff = time.time() - window_seconds
    for line in alerts:
        try:
            stamp, _, amount, usd, text = next(csv.reader(io.StringIO(line)))[:5]
            if int(stamp) < cutoff:
                continue
            value = float(usd.replace(" USD", "").replace(",", ""))
        except (ValueError, StopIteration):
            continue
        coin = next((c for c in coins if f"#{c}" in amount), None)
        if not coin:
            continue
        lowered = text.lower()
        source, _, target = lowered.partition(" to ")
        from_exchange = any(e in source for e in EXCHANGES)
        to_exchange = any(e in target for e in EXCHANGES)
        if to_exchange and not from_exchange:
            flows[coin]["inflow"] += value
        elif from_exchange and not to_exchange:
            flows[coin]["outflow"] += value
    return flows


def headlines(coins, url="https://coinjournal.net/news/feed/", per_coin=3):
    """Latest RSS titles that mention each coin."""
    import re
    from html import unescape

    titles = [
        unescape(t)
        for t in re.findall(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", fetch_text(url))[1:]
    ]
    found = {}
    for coin in coins:
        words = NAMES.get(coin, (coin.lower(),))
        found[coin] = [
            t for t in titles if any(re.search(rf"\b{re.escape(w)}\b", t.lower()) for w in words)
        ][:per_coin]
    return found


def load_agent(model):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from laya_mlx import Agent

    print(f"Loading {model} (FP16, MLX)...", file=sys.stderr)
    agent = Agent(model, dtype="float16", device="gpu", batch_size=1)
    agent.predict("warmup", {"q": {"type": "noul", "instructions": QUESTION}})
    return agent


def ask_laya(agent, state, question=QUESTION):
    """Ask the question and keep the whole exchange: the request as sent, the token
    sequence the encoder actually reads (decoded back to text), and the raw answer."""
    questions = {"q": {"type": "noul", "instructions": question}}
    items, _ = agent.prepare(state, questions)
    ids = items[0]["ids"]
    output = agent.predict(state, questions)
    answer = output["answers"]["q"]
    return {
        "p": answer["noul"],
        "request": {"state": state, "questions": questions},
        "encoder_input": agent.tok.backend.decode(ids, skip_special_tokens=False),
        "tokens": len(ids),
        "answer": answer,
    }


# --- Strategy and paper account ------------------------------------------------------------


def signal(score, strategy):
    """+1 bullish, -1 bearish, 0 neutral from a 0-1 score (Laya's P, or a rule score).

    "trend" follows the score; "reversion" fades it (short-term moves often snap back)."""
    high = score >= strategy["enter_above"]
    low = score <= strategy["enter_below"]
    if strategy["direction"] == "reversion":
        high, low = low, high
    return 1 if high else -1 if low else 0


def explain(score, strategy, held, action, reason):
    """How the strategy turned the score into this action, in words."""
    direction = signal(score, strategy)
    above, below = strategy["enter_above"], strategy["enter_below"]
    zone = (
        f"P {score:.3f} ≥ {above}"
        if score >= above
        else f"P {score:.3f} ≤ {below}"
        if score <= below
        else f"{below} < P {score:.3f} < {above}"
    )
    word = {1: "bullish", -1: "bearish", 0: "neutral"}[direction]
    return {
        "direction_mode": strategy["direction"],
        "zone": zone,
        "signal": word,
        "position_before": (
            "flat"
            if not held
            else f"{'long' if held['side'] > 0 else 'short'} {held['qty']:.6g} @ {held['entry']:.6g}"
        ),
        "rule": f"{action} ({reason})",
        "market": strategy["market"],
    }


class Account:
    """Paper account for one coin.

    spot:    long only, no leverage.
    futures: long and short, leverage up to max_leverage, funding every 8 hours,
             liquidation if losses eat the margin.
    Size comes from risk, not from the model: size = equity x risk_pct / (sizing_atr x
    ATR14), capped by the leverage limit. Stops only protect; they don't change the size.

    Protection, each optional (0 / off by default):
      stop_loss_atr, stop_loss_pct   fixed stop from the entry (the nearer one wins)
      trailing_stop_atr              stop follows the best price since entry
      breakeven_after_atr            once this far in profit, the stop moves to the entry
      loss_cooldown_seconds          no new entry for this long after a losing trade
      max_entry_atr_ratio            no new entry while ATR3 / ATR14 is above this
      pause_drawdown_pct, pause_seconds   circuit breaker: after equity falls this far
                                     below its peak, no new entries for pause_seconds
    """

    MAINTENANCE = 0.005  # maintenance margin, share of notional

    def __init__(self, strategy, paper, candle_seconds=900):
        self.strategy = strategy
        self.candle_seconds = candle_seconds
        self.fee = paper["fee_pct"] / 100
        self.start = self.balance = paper["capital_usdt"]
        self.position = None
        self.last_trade = float("-inf")
        self.trades = []
        self.funding_paid = 0.0
        self.peak = self.start
        self.blocked_until, self.blocked_reason = float("-inf"), ""

    @property
    def futures(self):
        return self.strategy["market"] == "futures"

    def equity(self, price):
        held = self.position
        return self.balance + (held["side"] * held["qty"] * (price - held["entry"]) if held else 0)

    def decide(self, score, s, now, high=None, low=None):
        """Returns (action, reason, fill price). Actions: LONG, SHORT, CLOSE, HOLD.

        high/low: the candle's range, so a backtest sees stops touched inside the candle.
        The live loop checks the current price every round instead."""
        st, held, price = self.strategy, self.position, s["price"]
        in_candle = high is not None  # backtest: the whole candle is known
        high = price if high is None else high
        low = price if low is None else low
        opened_at = s.get("open", price) if in_candle else price
        equity = self.equity(price)
        self.peak = max(self.peak, equity)
        pause = st.get("pause_drawdown_pct", 0)
        if pause and not held and equity < self.peak * (1 - pause / 100):
            # Circuit breaker: stop opening trades for a while, then start from here.
            self.blocked_until = now + st.get("pause_seconds", 86_400)
            self.blocked_reason, self.peak = "drawdown pause", equity
        if held:
            worst = low if held["side"] > 0 else high
            if self.equity(worst) <= self.MAINTENANCE * held["qty"] * worst:
                return "CLOSE", "liquidated", worst
            stop, reason = self._stop(held)
            take = held["take"]
            # A candle that opens beyond the stop (a gap) fills at its open, not the stop.
            if stop is not None and held["side"] > 0 and low <= stop:
                return "CLOSE", reason, min(stop, opened_at)
            if stop is not None and held["side"] < 0 and high >= stop:
                return "CLOSE", reason, max(stop, opened_at)
            # The best price since entry is updated after the stop check, so a candle
            # never raises its own trailing stop.
            held["best"] = max(held["best"], high) if held["side"] > 0 else min(held["best"], low)
            if take is not None and held["side"] > 0 and high >= take:
                return "CLOSE", "take profit", take
            if take is not None and held["side"] < 0 and low <= take:
                return "CLOSE", "take profit", take
            if st.get("flat_at_close") and s.get("closes_soon"):
                return "CLOSE", "session close", price
            max_hold = st.get("max_hold_candles", 0)
            if max_hold and now - held["opened"] >= max_hold * self.candle_seconds:
                return "CLOSE", "time exit", price
        direction = signal(score, st)
        if now - self.last_trade < st["cooldown_seconds"]:
            return "HOLD", "cooldown", price
        if not held and now < self.blocked_until:
            return "HOLD", self.blocked_reason, price
        max_ratio = st.get("max_entry_atr_ratio", 0)
        if not held and direction and max_ratio and s["atr_ratio"] > max_ratio:
            return "HOLD", "too volatile", price
        if held and direction == -held["side"] and st.get("exit_on_flip", True):
            return "CLOSE", "signal flipped", price
        if st.get("flat_at_close") and s.get("closes_soon"):
            return "HOLD", "session closing", price
        # Higher-timeframe trend filter: only buy dips in an uptrend, only short in a downtrend.
        htf_trend = s["votes"][1]  # +1 EMA20 above EMA50 on the higher timeframe, -1 below
        with_trend = st.get("trend_filter", "none") == "none"
        if not held and direction > 0:
            if not (with_trend or htf_trend > 0):
                return "HOLD", "against the trend", price
            return "LONG", "bullish" if st["direction"] == "trend" else "oversold", price
        if not held and direction < 0 and self.futures:
            if not (with_trend or htf_trend < 0):
                return "HOLD", "against the trend", price
            return "SHORT", "bearish" if st["direction"] == "trend" else "overbought", price
        return "HOLD", "no signal", price

    def _stop(self, held):
        """The active stop and its name: fixed, trailing or breakeven, whichever is
        closest to the price (a stop only ever tightens)."""
        side, atr_entry = held["side"], held["atr"]
        candidates = []
        if held["stop"] is not None:
            candidates.append((held["stop"], "stop loss"))
        trail = self.strategy.get("trailing_stop_atr", 0)
        if trail:
            candidates.append((held["best"] - side * trail * atr_entry, "trailing stop"))
        breakeven = self.strategy.get("breakeven_after_atr", 0)
        if breakeven and side * (held["best"] - held["entry"]) >= breakeven * atr_entry:
            candidates.append((held["entry"], "breakeven stop"))
        if not candidates:
            return None, None
        pick = max if side > 0 else min
        return pick(candidates, key=lambda c: c[0])

    def apply(self, action, reason, price, s, now, score=None):
        if action == "HOLD":
            return None
        st = self.strategy
        if action in ("LONG", "SHORT"):
            equity = self.balance
            sizing = max(st.get("sizing_atr", 3.0) * s["atr14"], price * 0.001)
            qty = equity * st["risk_pct"] / 100 / sizing
            distances = []  # fixed stops: ATR-based and/or percentage, the nearer one wins
            if st["stop_loss_atr"]:
                distances.append(st["stop_loss_atr"] * s["atr14"])
            if st.get("stop_loss_pct"):
                distances.append(price * st["stop_loss_pct"] / 100)
            has_stop = bool(distances)
            stop_distance = min(distances) if distances else None
            max_leverage = st["max_leverage"] if self.futures else 1.0
            qty = min(qty, equity * max_leverage / price * (1 - self.fee))
            side = 1 if action == "LONG" else -1
            self.balance -= qty * price * self.fee
            self.position = {
                "side": side,
                "qty": qty,
                "entry": price,
                "stop": price - side * stop_distance if has_stop else None,
                "take": price + side * st["take_profit_atr"] * s["atr14"]
                if st["take_profit_atr"]
                else None,
                "leverage": qty * price / equity,
                "opened": now,
                "atr": s["atr14"],
                "best": price,
            }
            trade = {"qty": qty, "notional": qty * price, "leverage": qty * price / equity}
        else:  # CLOSE
            held = self.position
            pnl = held["side"] * held["qty"] * (price - held["entry"])
            fee = held["qty"] * price * self.fee
            self.balance = max(self.balance + pnl - fee, 0.0)
            self.position = None
            trade = {"qty": held["qty"], "notional": held["qty"] * price, "pnl": pnl - fee}
            loss_cooldown = st.get("loss_cooldown_seconds", 0)
            if loss_cooldown and pnl - fee < 0:
                self.blocked_until = max(self.blocked_until, now + loss_cooldown)
                self.blocked_reason = "cooling off after a loss"
        self.last_trade = now
        trade.update(action=action, reason=reason, price=price, at=now, score=score)
        self.trades.append(trade)
        return trade

    def pay_funding(self, rate, price):
        """Futures funding: longs pay shorts when the rate is positive."""
        if self.position and self.futures and rate:
            amount = self.position["side"] * self.position["qty"] * price * rate
            self.balance -= amount
            self.funding_paid += amount


def funding_times_between(t0_ms, t1_ms):
    """Binance funding timestamps (00:00, 08:00, 16:00 UTC) in (t0, t1]."""
    period = 8 * 3_600_000
    first = (t0_ms // period + 1) * period
    return range(int(first), int(t1_ms) + 1, period)
