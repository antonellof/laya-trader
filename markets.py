"""Market data providers. Each one gives the same shapes to the rest of the program:

  candles  Binance-style rows: [open ms, open, high, low, close, volume, close ms,
           quote volume, trades, taker buy volume or None]
  ctx      the slower context compute_signals() reads (see core.compute_signals)

Crypto: Binance public API (spot candles, futures statistics), Fear & Greed.
Stocks: Yahoo Finance's public chart API (unofficial, no key; polled politely), VIX as
        the sentiment gauge, SPY as the benchmark. Regular session only.
"""

import re
import sys
import time
from bisect import bisect_right
from datetime import datetime, timezone
from html import unescape

from core import fetch_text, get_json, headlines, whale_flows

DAY_MS, HOUR_MS, FIVE_MINUTES = 86_400_000, 3_600_000, 300_000
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def value_at(series, t):
    """Last known value at time t from sorted [(time, value)], or None."""
    i = bisect_right(series, (t, float("inf")))
    return series[i - 1][1] if i else None


class Candles:
    """Candles with their close times indexed, for fast 'closed before t' lookups."""

    def __init__(self, rows):
        self.rows, self.closes = rows, [r[6] for r in rows]

    def before(self, t, n):
        """The last n candles that closed before time t."""
        i = bisect_right(self.closes, t)
        return self.rows[max(0, i - n) : i]


# --- Crypto: Binance ------------------------------------------------------------------------


class Crypto:
    kind, label, noun = "crypto", "Crypto", "Crypto"
    SPOT, FUTURES = "https://api.binance.com", "https://fapi.binance.com"
    # Binance blocks some regions (HTTP 451, e.g. US servers). Its public market-data
    # mirror serves the same spot candles; futures statistics have no mirror, so there
    # they're simply missing and the sentence leaves them out.
    SPOT_MIRROR = "https://data-api.binance.vision"
    FUTURES_HISTORY_MS = 29 * DAY_MS  # Binance keeps futures/data statistics for 30 days

    def __init__(self, cfg, refresh):
        self.cfg, self.refresh = cfg, refresh
        self.symbols = [s.upper() for s in cfg["symbols"]]
        self.interval = cfg["kline_interval"]
        self.spot = self.SPOT

    def _spot(self, path, timeout=3):
        base = self.spot  # assets are fetched in parallel: another thread may switch it
        try:
            return get_json(f"{base}{path}", timeout=timeout)
        except RuntimeError as error:
            if base == self.SPOT and ("HTTP 451" in str(error) or "HTTP 403" in str(error)):
                if self.spot == self.SPOT:
                    print("Binance API blocked here; using the market-data mirror", file=sys.stderr)
                    self.spot = self.SPOT_MIRROR
                return get_json(f"{self.SPOT_MIRROR}{path}", timeout=timeout)
            raise

    def is_open(self, now=None):
        return True

    # Live ---------------------------------------------------------------------------------

    def _candles(self, symbol, interval, limit):
        return self._spot(f"/api/v3/klines?symbol={symbol}USDT&interval={interval}&limit={limit}")

    def _stats(self, symbol, path, field):
        rows = get_json(
            f"{self.FUTURES}/futures/data/{path}?symbol={symbol}USDT&period=5m&limit=13"
        )
        return [float(r[field]) for r in rows]

    def extras(self, cache, context):
        """Live-only sources shared by all coins: whale flows and headlines."""
        ttl = self.refresh["context_seconds"]
        return {
            "whales": cache.get("whales", ttl, lambda: whale_flows(self.symbols))
            if context.get("whale_alerts")
            else None,
            "headlines": cache.get(
                "news:crypto", self.refresh["news_seconds"], lambda: headlines(self.symbols)
            )
            if context.get("news")
            else None,
            "sentiment": cache.get(
                "fng",
                self.refresh["fear_greed_seconds"],
                lambda: {
                    "kind": "fear_greed",
                    "value": int(
                        get_json("https://api.alternative.me/fng/?limit=1")["data"][0]["value"]
                    ),
                },
            ),
        }

    def snapshot(self, symbol, cache, extras):
        ttl = self.refresh["context_seconds"]
        klines = self._candles(symbol, self.interval, 100)
        ctx = {
            "htf_label": "4h",
            "k_htf": cache.get(f"{symbol}:4h", ttl, lambda: self._candles(symbol, "4h", 60), []),
            # 1h and 1d: completed candles only (the last one returned is still forming)
            "k1h": cache.get(f"{symbol}:1h", ttl, lambda: self._candles(symbol, "1h", 26)[:-1], []),
            "k1d": cache.get(f"{symbol}:1d", ttl, lambda: self._candles(symbol, "1d", 2)[:-1], []),
            "funding": cache.get(
                f"{symbol}:funding",
                ttl,
                lambda: float(
                    get_json(f"{self.FUTURES}/fapi/v1/premiumIndex?symbol={symbol}USDT")[
                        "lastFundingRate"
                    ]
                ),
            ),
            "long_short": cache.get(
                f"{symbol}:ls",
                ttl,
                lambda: self._stats(symbol, "globalLongShortAccountRatio", "longShortRatio")[-1],
            ),
            "oi": cache.get(
                f"{symbol}:oi",
                ttl,
                lambda: self._stats(symbol, "openInterestHist", "sumOpenInterest"),
            ),
            "taker_ratio": cache.get(
                f"{symbol}:taker",
                ttl,
                lambda: self._stats(symbol, "takerlongshortRatio", "buySellRatio")[-1],
            ),
            "sentiment": extras.get("sentiment"),
            "whales": (extras.get("whales") or {}).get(symbol),
            "headlines": (extras.get("headlines") or {}).get(symbol),
        }
        return klines, ctx

    # History ------------------------------------------------------------------------------

    def _history(self, symbol, interval, start_ms, end_ms):
        rows = []
        while start_ms < end_ms:
            batch = self._spot(
                f"/api/v3/klines?symbol={symbol}USDT&interval={interval}"
                f"&startTime={start_ms}&endTime={end_ms}&limit=1000",
                timeout=10,
            )
            if not batch:
                break
            rows.extend(batch)
            start_ms = batch[-1][0] + 1
        return rows

    def _stats_history(self, symbol, path, field, start_ms, end_ms):
        """5-minute futures statistics as [(time, value)]. The endpoint returns the latest
        records in a range, so walk backwards; nothing exists before the 30-day limit."""
        start_ms = max(start_ms, int(time.time() * 1000) - self.FUTURES_HISTORY_MS)
        rows, cursor = [], end_ms
        try:
            while cursor > start_ms:
                batch = get_json(
                    f"{self.FUTURES}/futures/data/{path}?symbol={symbol}USDT&period=5m"
                    f"&startTime={start_ms}&endTime={cursor}&limit=500",
                    timeout=10,
                )
                if not batch:
                    break
                rows.extend((int(r["timestamp"]), float(r[field])) for r in batch)
                first = min(int(r["timestamp"]) for r in batch)
                if first >= cursor:
                    break
                cursor = first - 1
        except Exception:
            pass
        return sorted(set(rows))

    def _funding_history(self, symbol, start_ms, end_ms):
        rows, cursor = [], start_ms - DAY_MS
        try:
            while cursor < end_ms:
                batch = get_json(
                    f"{self.FUTURES}/fapi/v1/fundingRate?symbol={symbol}USDT"
                    f"&startTime={cursor}&endTime={end_ms}&limit=1000",
                    timeout=10,
                )
                if not batch:
                    break
                rows.extend((r["fundingTime"], float(r["fundingRate"])) for r in batch)
                cursor = batch[-1]["fundingTime"] + 1
        except Exception:
            pass
        return rows

    def history(self, symbol, interval, start_ms, end_ms, warmup):
        step = INTERVAL_MS[interval]
        fast = self._history(symbol, interval, start_ms - warmup * step, end_ms)
        k4h = self._history(symbol, "4h", start_ms - 60 * 4 * HOUR_MS, end_ms)
        k1h = self._history(symbol, "1h", start_ms - 26 * HOUR_MS, end_ms)
        k1d = self._history(symbol, "1d", start_ms - 3 * DAY_MS, end_ms)
        funding = self._funding_history(symbol, start_ms, end_ms)
        stats = {
            key: self._stats_history(symbol, path, field, start_ms - HOUR_MS, end_ms)
            for key, path, field in (
                ("long_short", "globalLongShortAccountRatio", "longShortRatio"),
                ("oi", "openInterestHist", "sumOpenInterest"),
                ("taker", "takerlongshortRatio", "buySellRatio"),
            )
        }
        try:
            days = int((end_ms - start_ms) / DAY_MS) + 2
            rows = get_json(f"https://api.alternative.me/fng/?limit={days}", timeout=10)["data"]
            fng = sorted((int(r["timestamp"]) * 1000, int(r["value"])) for r in rows)
        except Exception:
            fng = []
        oi_times = [t for t, _ in stats["oi"]]
        k4h, k1h, k1d = Candles(k4h), Candles(k1h), Candles(k1d)

        def ctx_at(close_ms):
            known = close_ms - FIVE_MINUTES  # 5-minute statistics: only periods already over
            upto = bisect_right(oi_times, known)
            sentiment = value_at(fng, close_ms)
            return {
                "htf_label": "4h",
                "k_htf": k4h.before(close_ms, 60),
                "k1h": k1h.before(close_ms, 25),
                "k1d": k1d.before(close_ms, 1),
                "funding": value_at(funding, close_ms),
                "long_short": value_at(stats["long_short"], known),
                "oi": [v for _, v in stats["oi"][max(0, upto - 13) : upto]],
                "taker_ratio": value_at(stats["taker"], known),
                "sentiment": {"kind": "fear_greed", "value": sentiment} if sentiment else None,
            }

        return {"fast": fast, "ctx_at": ctx_at, "funding": funding}


# --- Stocks: Yahoo Finance --------------------------------------------------------------------


class Stocks:
    kind, label, noun = "stocks", "Stocks", "Stock"
    CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"
    RANGE_LIMIT_DAYS = {"1m": 7, "5m": 59, "15m": 59, "1h": 729}

    def __init__(self, cfg, refresh):
        self.cfg, self.refresh = cfg, refresh
        self.symbols = [s.upper() for s in cfg["symbols"]]
        self.interval = cfg["kline_interval"]
        self.benchmark = cfg.get("benchmark", "SPY")
        self.session = None  # (start, end) of today's regular session, epoch seconds

    def _chart(self, symbol, interval, session=False, **params):
        """session=True only for the traded stocks' own candles: indexes such as ^VIX report
        much longer hours, and taking the session from them made the market look open at
        3 am New York time."""
        query = "&".join(f"{k}={v}" for k, v in {"interval": interval, **params}.items())
        data = get_json(f"{self.CHART}{symbol.replace('^', '%5E')}?{query}", timeout=10)
        result = data["chart"]["result"][0]
        meta = result.get("meta", {})
        regular = meta.get("currentTradingPeriod", {}).get("regular")
        if session and regular:
            self.session = (regular["start"], regular["end"])
        step = INTERVAL_MS.get(interval, DAY_MS)
        quote = result["indicators"]["quote"][0]
        rows = []
        for i, t in enumerate(result.get("timestamp") or []):
            o, h, low, c, v = (quote[k][i] for k in ("open", "high", "low", "close", "volume"))
            if None in (o, h, low, c):
                continue
            rows.append([t * 1000, o, h, low, c, v or 0, t * 1000 + step - 1, 0, 0, None])
        return rows

    def is_open(self, now=None):
        now = now or time.time()
        return bool(self.session) and self.session[0] <= now < self.session[1]

    # Live ---------------------------------------------------------------------------------

    def extras(self, cache, context):
        ttl = self.refresh["context_seconds"]

        def vix():
            rows = self._chart("^VIX", "1d", range="5d")
            return {"kind": "vix", "value": rows[-1][4]} if rows else None

        def benchmark_change():
            rows = self._chart(self.benchmark, "1d", range="5d")
            return (rows[-1][4] / rows[-2][4] - 1) * 100 if len(rows) >= 2 else None

        return {
            "sentiment": cache.get("vix", ttl, vix),
            "benchmark_change": cache.get("benchmark", ttl, benchmark_change),
            "headlines": cache.get("news:stocks", self.refresh["news_seconds"], self._headlines)
            if context.get("news")
            else None,
        }

    def _headlines(self, per_symbol=3):
        """Latest titles from Yahoo Finance's per-symbol RSS feed."""
        found = {}
        for symbol in self.symbols:
            try:
                feed = fetch_text(
                    f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
                )
                titles = [unescape(t) for t in re.findall(r"<title>(.*?)</title>", feed)[1:]]
                found[symbol] = titles[:per_symbol]
            except Exception:
                found[symbol] = []
        return found

    def snapshot(self, symbol, cache, extras):
        ttl = self.refresh["context_seconds"]
        poll = self.cfg.get("poll_seconds", 15)
        klines = cache.get(
            f"{symbol}:fast",
            poll,
            lambda: self._chart(symbol, self.interval, session=True, range="5d"),
            [],
        )
        daily = cache.get(f"{symbol}:1d", ttl, lambda: self._chart(symbol, "1d", range="6mo"), [])
        hourly = cache.get(f"{symbol}:1h", ttl, lambda: self._chart(symbol, "1h", range="1mo"), [])
        now_ms = time.time() * 1000
        return klines, self._ctx(klines, daily, hourly, now_ms, extras, symbol)

    def _ctx(self, fast, daily, hourly, now_ms, extras, symbol=None):
        """Completed daily candles (today's is still forming during the session)."""
        today = datetime.fromtimestamp(now_ms / 1000, timezone.utc).date()
        done_days = [
            d for d in daily if datetime.fromtimestamp(d[0] / 1000, timezone.utc).date() < today
        ]
        session = [
            k for k in fast if datetime.fromtimestamp(k[0] / 1000, timezone.utc).date() == today
        ]
        price = fast[-1][4] if fast else None
        day_ref = None
        if done_days and session and price:
            day_ref = {
                "reference": done_days[-1][4],
                "high": max(k[2] for k in session),
                "low": min(k[3] for k in session),
            }
        change = (price / done_days[-1][4] - 1) * 100 if done_days and price else None
        benchmark = extras.get("benchmark_change")
        return {
            "htf_label": "daily",
            "k_htf": done_days[-60:],
            "k1h": [h for h in hourly if h[6] < now_ms][-25:],
            "k1d": done_days[-1:],
            "day_ref": day_ref,
            "day_label": "today",
            "relative_strength": change - benchmark
            if change is not None and benchmark is not None
            else None,
            "sentiment": extras.get("sentiment"),
            "headlines": (extras.get("headlines") or {}).get(symbol),
            # Live: the session ends within one candle. Backtests mark each day's last
            # candle instead (see backtest.prepare).
            "closes_soon": bool(self.session)
            and 0 <= self.session[1] - now_ms / 1000 < INTERVAL_MS[self.interval] / 1000,
        }

    # History ------------------------------------------------------------------------------

    def history(self, symbol, interval, start_ms, end_ms, warmup):
        limit = self.RANGE_LIMIT_DAYS[interval] * DAY_MS
        first = max(start_ms - warmup * INTERVAL_MS[interval] * 4, int(time.time() * 1000) - limit)
        fast = self._chart(symbol, interval, period1=first // 1000, period2=end_ms // 1000)
        daily = self._chart(
            symbol, "1d", period1=(start_ms - 120 * DAY_MS) // 1000, period2=end_ms // 1000
        )
        hourly = self._chart(
            symbol, "1h", period1=(start_ms - 10 * DAY_MS) // 1000, period2=end_ms // 1000
        )
        vix_rows = self._chart(
            "^VIX", "1d", period1=(start_ms - 5 * DAY_MS) // 1000, period2=end_ms // 1000
        )
        vix = [(r[6], r[4]) for r in vix_rows]
        bench = self._chart(self.benchmark, interval, period1=first // 1000, period2=end_ms // 1000)
        bench_daily = self._chart(
            self.benchmark, "1d", period1=(start_ms - 10 * DAY_MS) // 1000, period2=end_ms // 1000
        )
        bench_close = [(r[6], r[4]) for r in bench]

        fast_c, bench_daily_c, hourly_c = Candles(fast), Candles(bench_daily), Candles(hourly)
        day_opens = [d[0] for d in daily]

        def ctx_at(close_ms):
            upto = fast_c.before(close_ms, 100)
            prev_days = bench_daily_c.before(close_ms - 12 * HOUR_MS, 1)
            b_now = value_at(bench_close, close_ms)
            b_change = (
                (b_now / prev_days[-1][4] - 1) * 100 if b_now is not None and prev_days else None
            )
            ctx = self._ctx(
                upto or fast[:1],
                daily[: bisect_right(day_opens, close_ms - 1)][-70:],
                hourly_c.before(close_ms, 30),
                close_ms,
                {"benchmark_change": b_change},
            )
            level = value_at(vix, close_ms - DAY_MS)  # yesterday's close: known at the time
            ctx["sentiment"] = {"kind": "vix", "value": level} if level else None
            return ctx

        return {"fast": fast, "ctx_at": ctx_at, "funding": []}


MARKETS = {"crypto": Crypto, "stocks": Stocks}


def load_markets(config):
    return [MARKETS[kind](config[kind], config["refresh"]) for kind in MARKETS if kind in config]
