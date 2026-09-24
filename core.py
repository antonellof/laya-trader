"""Shared pieces: Binance data, indicators, Laya's question, trading rules, paper book."""

import http.client
import json
import os
import sys
import threading
import time
from urllib.parse import urlsplit

QUESTION = "Is the short-term outlook for this coin bullish?"


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


def compute_signals(klines, closes_4h, funding=None, long_short=None):
    """klines: the latest ~100 candles of the fast timeframe, oldest first."""
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    taker_buy = [float(k[9]) for k in klines]
    price = closes[-1]
    ema20, macd_value, rsi14 = ema(closes, 20), macd(closes), rsi(closes, 14)
    atr3, atr14 = atr(klines, 3), atr(klines, 14)
    ema20_4h, ema50_4h = (ema(closes_4h, 20), ema(closes_4h, 50)) if closes_4h else (0.0, 0.0)
    recent = slice(-6, -1)  # last five closed candles
    base_volume = sum(volumes[-22:-2]) / 20
    return {
        "price": price,
        "high": float(klines[-1][2]),
        "low": float(klines[-1][3]),
        "ema20": ema20,
        "dist_ema20_pct": (price - ema20) / ema20 * 100 if ema20 else 0.0,
        "macd": macd_value,
        "rsi14": rsi14,
        "atr14": atr14,
        "atr_ratio": atr3 / atr14 if atr14 else 1.0,
        "volume_ratio": volumes[-2] / base_volume if base_volume else 1.0,
        "buy_ratio": sum(taker_buy[recent]) / max(sum(volumes[recent]), 1e-12),
        "funding": funding,
        "long_short": long_short,
        "votes": trend_votes(price, ema20, ema20_4h, ema50_4h, macd_value, rsi14, atr14),
    }


# --- Signals to Laya ------------------------------------------------------------------------


def describe(s, fear_greed=None):
    """The signals as one "Good: ... Bad: ..." sentence (Good = bullish)."""
    trend = sum(s["votes"])
    bullish = [
        (trend >= 2, f"trend votes bullish ({trend:+d} of 4)"),
        (s["dist_ema20_pct"] > 0.12, "price above EMA20"),
        (s["macd"] > 0, "MACD positive"),
        (s["buy_ratio"] > 0.55, "buyers dominate recent order flow"),
        (s["volume_ratio"] > 1.5 and s["buy_ratio"] > 0.5, "rising volume on buying"),
        (s["rsi14"] < 30, "RSI oversold"),
        (fear_greed is not None and fear_greed <= 25, "extreme fear in the market"),
        (s["long_short"] is not None and s["long_short"] < 0.8, "crowd is short"),
    ]
    bearish = [
        (trend <= -2, f"trend votes bearish ({trend:+d} of 4)"),
        (s["dist_ema20_pct"] < -0.12, "price below EMA20"),
        (s["macd"] < 0, "MACD negative"),
        (s["buy_ratio"] < 0.45, "sellers dominate recent order flow"),
        (s["volume_ratio"] > 1.5 and s["buy_ratio"] < 0.5, "rising volume on selling"),
        (s["rsi14"] > 70, "RSI overbought"),
        (s["funding"] is not None and s["funding"] > 0.0005, "funding high, longs crowded"),
        (fear_greed is not None and fear_greed >= 75, "extreme greed in the market"),
        (s["atr_ratio"] > 2.0, "volatility extreme"),
    ]
    good = [text for ok, text in bullish if ok]
    bad = [text for ok, text in bearish if ok]
    text = f"Good: {', '.join(good)}." if good else ""
    if bad:
        text += f" Bad: {', '.join(bad)}."
    return "Crypto market signals. " + (text.strip() or "No clear signals.")


def load_agent(model):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from laya_mlx import Agent

    print(f"Loading {model} (FP16, MLX)...", file=sys.stderr)
    agent = Agent(model, dtype="float16", device="gpu", batch_size=1)
    agent.predict("warmup", {"q": {"type": "noul", "instructions": QUESTION}})
    return agent


def ask_laya(agent, state):
    """Ask the question and keep the whole exchange: the request as sent, the token
    sequence the encoder actually reads (decoded back to text), and the raw answer."""
    questions = {"q": {"type": "noul", "instructions": QUESTION}}
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
    Size comes from risk, not from the model: a stop-loss hit costs risk_pct of equity,
    so size = equity x risk_pct / stop distance, capped by the leverage limit.
    """

    MAINTENANCE = 0.005  # maintenance margin, share of notional

    def __init__(self, strategy, paper):
        self.strategy = strategy
        self.fee = paper["fee_pct"] / 100
        self.start = self.balance = paper["capital_usdt"]
        self.position = None
        self.last_trade = float("-inf")
        self.trades = []
        self.funding_paid = 0.0

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
        high = price if high is None else high
        low = price if low is None else low
        if held:
            worst = low if held["side"] > 0 else high
            if self.equity(worst) <= self.MAINTENANCE * held["qty"] * worst:
                return "CLOSE", "liquidated", worst
            if held["side"] > 0 and low <= held["stop"]:
                return "CLOSE", "stop loss", held["stop"]
            if held["side"] < 0 and high >= held["stop"]:
                return "CLOSE", "stop loss", held["stop"]
            if held["side"] > 0 and high >= held["take"]:
                return "CLOSE", "take profit", held["take"]
            if held["side"] < 0 and low <= held["take"]:
                return "CLOSE", "take profit", held["take"]
        direction = signal(score, st)
        if now - self.last_trade < st["cooldown_seconds"]:
            return "HOLD", "cooldown", price
        if held and direction == -held["side"]:
            return "CLOSE", "signal flipped", price
        if not held and direction > 0:
            return "LONG", "bullish" if st["direction"] == "trend" else "oversold", price
        if not held and direction < 0 and self.futures:
            return "SHORT", "bearish" if st["direction"] == "trend" else "overbought", price
        return "HOLD", "no signal", price

    def apply(self, action, reason, price, s, now):
        if action == "HOLD":
            return None
        st = self.strategy
        if action in ("LONG", "SHORT"):
            equity = self.balance
            stop_distance = max(st["stop_loss_atr"] * s["atr14"], price * 0.001)
            qty = equity * st["risk_pct"] / 100 / stop_distance
            max_leverage = st["max_leverage"] if self.futures else 1.0
            qty = min(qty, equity * max_leverage / price * (1 - self.fee))
            side = 1 if action == "LONG" else -1
            self.balance -= qty * price * self.fee
            self.position = {
                "side": side,
                "qty": qty,
                "entry": price,
                "stop": price - side * stop_distance,
                "take": price + side * st["take_profit_atr"] * s["atr14"],
                "leverage": qty * price / equity,
                "opened": now,
            }
            trade = {"qty": qty, "notional": qty * price, "leverage": qty * price / equity}
        else:  # CLOSE
            held = self.position
            pnl = held["side"] * held["qty"] * (price - held["entry"])
            fee = held["qty"] * price * self.fee
            self.balance = max(self.balance + pnl - fee, 0.0)
            self.position = None
            trade = {"qty": held["qty"], "notional": held["qty"] * price, "pnl": pnl - fee}
        self.last_trade = now
        trade.update(action=action, reason=reason, price=price, at=now)
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
