"""
Data_Architect Agent

1. Delegates universe download + liquidity/quality filtering to UniverseFilter
2. Computes technical indicators on the clean price series:
     RSI-14, MA-50, MA-200, rolling annualised vol (21d / 63d)
3. Returns a DataBundle consumed by QuantResearcher

No look-ahead bias: all indicators use only data up to row t to produce
a value at t. The only usage in the backtest context is via .shift(1)
before merging with the alpha factor frame.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from core.universe import UniverseFilter

logger = logging.getLogger(__name__)


# ── Technical indicator helpers ───────────────────────────────────────────────

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_ma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=window).mean()


def compute_rolling_vol(log_returns: pd.Series, window: int = 21) -> pd.Series:
    return log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)


# ── DataBundle ────────────────────────────────────────────────────────────────

class DataBundle:
    """Immutable snapshot passed between pipeline stages."""

    def __init__(
        self,
        prices: pd.DataFrame,
        log_returns: pd.DataFrame,
        indicators: dict[str, pd.DataFrame],
        fetch_date: str,
    ):
        self.prices = prices
        self.log_returns = log_returns
        self.indicators = indicators
        self.fetch_date = fetch_date
        self.tickers: list[str] = list(prices.columns)

    def latest_prices(self) -> pd.Series:
        return self.prices.iloc[-1]


# ── Agent class ───────────────────────────────────────────────────────────────

class DataArchitect:
    def __init__(
        self,
        se_tickers: list[str],
        us_tickers: list[str],
        min_adv_sek: float = 10_000_000.0,
        max_gap_pct: float = 0.05,
        lookback_days: int = 273,
    ):
        self._filter = UniverseFilter(
            se_tickers=se_tickers,
            us_tickers=us_tickers,
            min_adv_sek=min_adv_sek,
            max_gap_pct=max_gap_pct,
            lookback_days=lookback_days,
        )

    def run(self) -> DataBundle:
        prices, log_returns = self._filter.fetch_and_filter()

        indicators: dict[str, pd.DataFrame] = {}
        for ticker in prices.columns:
            close = prices[ticker].dropna()
            lr = log_returns[ticker].dropna()
            indicators[ticker] = pd.DataFrame(
                {
                    "close": close,
                    "rsi_14": compute_rsi(close, 14),
                    "ma_50": compute_ma(close, 50),
                    "ma_200": compute_ma(close, 200),
                    "vol_21d": compute_rolling_vol(lr, 21),
                    "vol_63d": compute_rolling_vol(lr, 63),
                }
            )

        logger.info(
            "DataArchitect complete: %d tickers, %d trading days (%s)",
            len(prices.columns),
            len(prices),
            datetime.today().strftime("%Y-%m-%d"),
        )

        return DataBundle(
            prices=prices,
            log_returns=log_returns,
            indicators=indicators,
            fetch_date=datetime.today().strftime("%Y-%m-%d"),
        )
