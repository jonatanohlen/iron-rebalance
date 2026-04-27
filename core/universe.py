"""
Universe management.

UniverseFilter downloads OHLCV for all candidate tickers then drops:
  1. Tickers missing from yfinance (exchange not supported or delisted)
  2. Tickers with > max_gap_pct missing trading days
  3. Tickers whose 20-day ADV falls below min_adv_sek

Returns clean prices and log-returns DataFrames ready for downstream agents.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_FX_FALLBACK_USDSEK = 10.5   # only used if live rate is unavailable


def get_usd_sek_rate() -> float:
    try:
        hist = yf.Ticker("USDSEK=X").history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.warning("FX fetch failed: %s — using fallback %.2f", exc, _FX_FALLBACK_USDSEK)
    return _FX_FALLBACK_USDSEK


class UniverseFilter:
    def __init__(
        self,
        se_tickers: list[str],
        us_tickers: list[str],
        min_adv_sek: float = 10_000_000.0,
        max_gap_pct: float = 0.05,
        lookback_days: int = 273,
    ):
        self.se_tickers = se_tickers
        self.us_tickers = us_tickers
        self.min_adv_sek = min_adv_sek
        self.max_gap_pct = max_gap_pct
        self.lookback_days = lookback_days

    def fetch_and_filter(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          prices     — adjusted close (rows=dates, cols=tickers)
          log_returns — daily log-returns (rows=dates, cols=tickers)
        """
        all_tickers = self.se_tickers + self.us_tickers
        end = datetime.today()
        # Download extra buffer so momentum calculations have full 252 days
        start = end - timedelta(days=int(self.lookback_days * 1.5))

        logger.info(
            "Downloading %d tickers (%s → %s)",
            len(all_tickers), start.date(), end.date(),
        )

        raw = yf.download(
            all_tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if raw.empty:
            raise RuntimeError("yfinance returned empty dataset — check network/API.")

        # Normalise to flat Close / Volume DataFrames
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
            volumes = raw["Volume"]
        else:
            # Single ticker (shouldn't happen in practice)
            closes = raw[["Close"]].rename(columns={"Close": all_tickers[0]})
            volumes = raw[["Volume"]].rename(columns={"Volume": all_tickers[0]})

        usd_sek = get_usd_sek_rate()
        passed: list[str] = []

        for ticker in all_tickers:
            if ticker not in closes.columns:
                logger.info("Drop %s: not returned by yfinance", ticker)
                continue

            close = closes[ticker].dropna()
            vol = volumes[ticker].dropna()

            if close.empty:
                logger.info("Drop %s: empty price series", ticker)
                continue

            # Data gap check — compare actual trading days to expected
            gap = max(0, self.lookback_days - len(close)) / self.lookback_days
            if gap > self.max_gap_pct:
                logger.info("Drop %s: data gap %.1f%% > %.1f%%", ticker, gap * 100, self.max_gap_pct * 100)
                continue

            # Liquidity check — 20-day ADV in SEK
            adv_20 = vol.tail(20).mean()
            price_local = close.tail(20).mean()
            is_se = ticker.endswith(".ST")
            adv_sek = adv_20 * price_local if is_se else adv_20 * price_local * usd_sek

            if adv_sek < self.min_adv_sek:
                logger.info(
                    "Drop %s: ADV %.0f SEK < %.0f SEK minimum",
                    ticker, adv_sek, self.min_adv_sek,
                )
                continue

            passed.append(ticker)

        logger.info("Universe: %d / %d tickers passed filters", len(passed), len(all_tickers))

        if not passed:
            raise RuntimeError("All tickers dropped by universe filters.")

        prices = closes[passed].dropna(how="all")
        log_returns = np.log(prices / prices.shift(1)).dropna(how="all")

        return prices, log_returns
