"""
Alpha factor calculators.

All methods are point-in-time: they use only data available at index[0]
(most-recent fiscal year) and index[1] (prior fiscal year) to prevent
look-ahead bias in any backtest that calls them historically.

PiotroskiFScore  — 9-signal binary score, requires ≥ 7 to pass
ROICCalculator   — 3-year average ROIC must exceed 15%
MomentumCalculator — 12M return minus 1M return (skip-1 momentum)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _safe_get(df: pd.DataFrame, *keys: str, col: int = 0) -> float | None:
    """Case-insensitive partial-match row lookup from a yfinance financial table."""
    for k in keys:
        matches = [idx for idx in df.index if k.lower() in str(idx).lower()]
        if matches:
            try:
                val = df.loc[matches[0]].iloc[col]
                return float(val) if pd.notna(val) else None
            except (IndexError, TypeError):
                continue
    return None


class PiotroskiFScore:
    """
    Signals (each = 1 point):
      F1  ROA > 0
      F2  Operating cash flow > 0
      F3  ΔROA > 0
      F4  Cash earnings > accrual earnings  (CF quality)
      F5  Leverage decreasing
      F6  Current ratio improving
      F7  No share dilution
      F8  Gross margin improving
      F9  Asset turnover improving
    """

    def score(self, ticker: str) -> int | None:
        try:
            t = yf.Ticker(ticker)
            fin = t.financials      # income statement; col 0 = most recent
            bal = t.balance_sheet
            cf = t.cashflow

            if any(df is None or df.empty for df in [fin, bal, cf]):
                return None
            if fin.shape[1] < 2 or bal.shape[1] < 2:
                return None

            def g(df, *keys, col=0):
                return _safe_get(df, *keys, col=col)

            # Fetch raw items
            assets_0 = g(bal, "Total Assets", col=0)
            assets_1 = g(bal, "Total Assets", col=1)
            ni_0 = g(fin, "Net Income", col=0)
            ni_1 = g(fin, "Net Income", col=1)
            ocf_0 = g(cf, "Operating Cash Flow", "Total Cash From Operating", col=0)
            debt_0 = g(bal, "Long Term Debt", col=0) or 0.0
            debt_1 = g(bal, "Long Term Debt", col=1) or 0.0
            curr_a_0 = g(bal, "Current Assets", col=0)
            curr_l_0 = g(bal, "Current Liabilities", col=0)
            curr_a_1 = g(bal, "Current Assets", col=1)
            curr_l_1 = g(bal, "Current Liabilities", col=1)
            shares_0 = g(bal, "Share Issued", "Common Stock Shares Outstanding", col=0)
            shares_1 = g(bal, "Share Issued", "Common Stock Shares Outstanding", col=1)
            gp_0 = g(fin, "Gross Profit", col=0)
            gp_1 = g(fin, "Gross Profit", col=1)
            rev_0 = g(fin, "Total Revenue", "Revenue", col=0)
            rev_1 = g(fin, "Total Revenue", "Revenue", col=1)

            s = 0

            # F1: ROA > 0
            if assets_0 and ni_0:
                s += int(ni_0 / assets_0 > 0)

            # F2: Operating CF > 0
            if ocf_0:
                s += int(ocf_0 > 0)

            # F3: ΔROA > 0
            if all(v is not None for v in [ni_0, ni_1, assets_0, assets_1]):
                roa_0 = ni_0 / assets_0
                roa_1 = ni_1 / (assets_1 or 1)
                s += int(roa_0 > roa_1)

            # F4: CF quality (OCF > Net Income, asset-scaled)
            if ocf_0 and ni_0 and assets_0:
                s += int((ocf_0 - ni_0) / assets_0 > 0)

            # F5: Leverage falling
            if assets_0 and assets_1:
                s += int(debt_0 / assets_0 < debt_1 / (assets_1 or 1))

            # F6: Current ratio improving
            if all(v is not None for v in [curr_a_0, curr_l_0, curr_a_1, curr_l_1]):
                cr_0 = curr_a_0 / (curr_l_0 or 1)
                cr_1 = curr_a_1 / (curr_l_1 or 1)
                s += int(cr_0 > cr_1)

            # F7: No dilution
            if shares_0 and shares_1:
                s += int(shares_0 <= shares_1)

            # F8: Gross margin improving
            if all(v is not None for v in [gp_0, gp_1, rev_0, rev_1]) and rev_0 and rev_1:
                s += int(gp_0 / rev_0 > gp_1 / rev_1)

            # F9: Asset turnover improving
            if all(v is not None for v in [rev_0, rev_1, assets_0, assets_1]) and assets_0 and assets_1:
                s += int(rev_0 / assets_0 > rev_1 / assets_1)

            return s

        except Exception as exc:
            logger.debug("F-Score failed for %s: %s", ticker, exc)
            return None


class ROICCalculator:
    """
    ROIC = NOPAT / Invested Capital
    NOPAT  = Operating Income × (1 − effective_tax_rate)
    IC     = Total Equity + Total Debt − Cash
    Returns the 3-year arithmetic mean; None if < 1 year of data.
    """

    _TAX_RATE = 0.22  # conservative approximation

    def compute(self, ticker: str) -> float | None:
        try:
            t = yf.Ticker(ticker)
            fin = t.financials
            bal = t.balance_sheet

            if any(df is None or df.empty for df in [fin, bal]):
                return None

            years = min(3, fin.shape[1], bal.shape[1])
            roics: list[float] = []

            for yr in range(years):
                ebit = _safe_get(fin, "EBIT", "Operating Income", col=yr)
                if ebit is None:
                    continue
                nopat = ebit * (1 - self._TAX_RATE)

                equity = _safe_get(bal, "Stockholders Equity", "Total Equity", col=yr) or 0.0
                ltd = _safe_get(bal, "Long Term Debt", col=yr) or 0.0
                std = _safe_get(bal, "Short Term Debt", "Current Portion Of Long Term Debt", col=yr) or 0.0
                cash = _safe_get(bal, "Cash And Cash Equivalents", col=yr) or 0.0

                ic = equity + ltd + std - cash
                if ic > 0:
                    roics.append(nopat / ic)

            return float(np.mean(roics)) if roics else None

        except Exception as exc:
            logger.debug("ROIC failed for %s: %s", ticker, exc)
            return None


class MomentumCalculator:
    """
    Skip-1-month momentum: 12M total return minus 1M total return.
    Avoids the short-term reversal effect documented in the literature.
    Requires at least 252 trading days of price history.
    """

    def compute(self, prices: pd.Series) -> float | None:
        prices = prices.dropna()
        if len(prices) < 252:
            return None
        try:
            ret_12m = prices.iloc[-1] / prices.iloc[-252] - 1
            ret_1m = prices.iloc[-1] / prices.iloc[-21] - 1
            return float(ret_12m - ret_1m)
        except (IndexError, ZeroDivisionError):
            return None
