"""
Alpha factor calculators.

All methods are point-in-time: they use only data available at index[0]
(most-recent fiscal year) and index[1] (prior fiscal year) to prevent
look-ahead bias in any backtest that calls them historically.

PiotroskiFScore  — 9-signal binary score, requires >= 7 to pass
ROICCalculator   — 3-year average ROIC must exceed 15%
MomentumCalculator — 12M return minus 1M return (skip-1 momentum)

Feature D — Robustness improvements:
  - DataCoverageReport tracks which tickers fail and why
  - F-Score records how many of the 9 signals had usable data (coverage %)
  - Tickers with < MIN_F_SCORE_COVERAGE usable signals return None
  - All failures logged at WARNING for .ST tickers (common yfinance gap),
    DEBUG for US tickers (rarer)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

MIN_F_SCORE_SIGNALS = 5   # need at least 5/9 signals with data to produce a score


@dataclass
class DataCoverageReport:
    """Accumulated across all tickers in a QuantResearcher run."""
    total: int = 0
    passed_f_score: int = 0
    passed_roic: int = 0
    passed_momentum: int = 0
    failed_no_fundamentals: list[str] = field(default_factory=list)
    failed_f_score_filter: list[str] = field(default_factory=list)
    failed_f_score_coverage: list[str] = field(default_factory=list)
    failed_roic_filter: list[str] = field(default_factory=list)
    failed_roic_coverage: list[str] = field(default_factory=list)
    failed_momentum: list[str] = field(default_factory=list)

    def log_summary(self) -> None:
        logger.info(
            "Factor coverage: %d tickers | F-Score pass=%d | ROIC pass=%d | Momentum pass=%d",
            self.total, self.passed_f_score, self.passed_roic, self.passed_momentum,
        )
        if self.failed_no_fundamentals:
            logger.warning("No fundamentals (yfinance gap): %s", self.failed_no_fundamentals)
        if self.failed_f_score_coverage:
            logger.warning("F-Score: insufficient signal coverage (<5/9): %s", self.failed_f_score_coverage)
        if self.failed_f_score_filter:
            logger.info("F-Score: below threshold: %s", self.failed_f_score_filter)
        if self.failed_roic_coverage:
            logger.warning("ROIC: insufficient data: %s", self.failed_roic_coverage)

    def to_markdown(self) -> str:
        lines = [
            "## Factor Data Coverage",
            f"| Stage | Count |",
            f"|-------|-------|",
            f"| Universe | {self.total} |",
            f"| F-Score >= threshold | {self.passed_f_score} |",
            f"| ROIC >= threshold | {self.passed_roic} |",
            f"| Momentum available | {self.passed_momentum} |",
            f"| No fundamentals (data gap) | {len(self.failed_no_fundamentals)} |",
            f"| F-Score coverage < {MIN_F_SCORE_SIGNALS}/9 | {len(self.failed_f_score_coverage)} |",
            f"| F-Score below threshold | {len(self.failed_f_score_filter)} |",
            f"| ROIC data gap | {len(self.failed_roic_coverage)} |",
            f"| ROIC below threshold | {len(self.failed_roic_filter)} |",
        ]
        if self.failed_no_fundamentals:
            lines.append(f"\n**No fundamentals:** {', '.join(self.failed_no_fundamentals)}")
        return "\n".join(lines)


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
      F3  delta-ROA > 0
      F4  Cash earnings > accrual earnings  (CF quality)
      F5  Leverage decreasing
      F6  Current ratio improving
      F7  No share dilution
      F8  Gross margin improving
      F9  Asset turnover improving

    Returns (score, signals_with_data) or (None, 0) on data failure.
    Tickers with < MIN_F_SCORE_SIGNALS usable signals are treated as None.
    """

    def score(self, ticker: str) -> tuple[int | None, int]:
        """Returns (f_score, n_signals_with_data). Returns (None, 0) on failure."""
        is_se = ticker.endswith(".ST")
        log = logger.warning if is_se else logger.debug

        try:
            t = yf.Ticker(ticker)
            fin = t.financials
            bal = t.balance_sheet
            cf  = t.cashflow

            if any(df is None or df.empty for df in [fin, bal, cf]):
                log("F-Score: no fundamentals returned for %s", ticker)
                return None, 0
            if fin.shape[1] < 2 or bal.shape[1] < 2:
                log("F-Score: < 2 years of data for %s", ticker)
                return None, 0

            def g(df, *keys, col=0):
                return _safe_get(df, *keys, col=col)

            assets_0 = g(bal, "Total Assets", col=0)
            assets_1 = g(bal, "Total Assets", col=1)
            ni_0     = g(fin, "Net Income", col=0)
            ni_1     = g(fin, "Net Income", col=1)
            ocf_0    = g(cf,  "Operating Cash Flow", "Total Cash From Operating", col=0)
            debt_0   = g(bal, "Long Term Debt", col=0) or 0.0
            debt_1   = g(bal, "Long Term Debt", col=1) or 0.0
            curr_a_0 = g(bal, "Current Assets", col=0)
            curr_l_0 = g(bal, "Current Liabilities", col=0)
            curr_a_1 = g(bal, "Current Assets", col=1)
            curr_l_1 = g(bal, "Current Liabilities", col=1)
            shares_0 = g(bal, "Share Issued", "Common Stock Shares Outstanding", col=0)
            shares_1 = g(bal, "Share Issued", "Common Stock Shares Outstanding", col=1)
            gp_0     = g(fin, "Gross Profit", col=0)
            gp_1     = g(fin, "Gross Profit", col=1)
            rev_0    = g(fin, "Total Revenue", "Revenue", col=0)
            rev_1    = g(fin, "Total Revenue", "Revenue", col=1)

            s = 0
            n = 0   # signals where we had enough data to evaluate

            # F1: ROA > 0
            if assets_0 and ni_0:
                s += int(ni_0 / assets_0 > 0); n += 1

            # F2: Operating CF > 0
            if ocf_0 is not None:
                s += int(ocf_0 > 0); n += 1

            # F3: delta-ROA > 0
            if all(v is not None for v in [ni_0, ni_1, assets_0, assets_1]):
                s += int(ni_0 / assets_0 > ni_1 / (assets_1 or 1)); n += 1

            # F4: CF quality
            if ocf_0 is not None and ni_0 is not None and assets_0:
                s += int((ocf_0 - ni_0) / assets_0 > 0); n += 1

            # F5: Leverage falling
            if assets_0 and assets_1:
                s += int(debt_0 / assets_0 < debt_1 / (assets_1 or 1)); n += 1

            # F6: Current ratio improving
            if all(v is not None for v in [curr_a_0, curr_l_0, curr_a_1, curr_l_1]):
                s += int(curr_a_0 / (curr_l_0 or 1) > curr_a_1 / (curr_l_1 or 1)); n += 1

            # F7: No dilution
            if shares_0 and shares_1:
                s += int(shares_0 <= shares_1); n += 1

            # F8: Gross margin improving
            if all(v is not None for v in [gp_0, gp_1, rev_0, rev_1]) and rev_0 and rev_1:
                s += int(gp_0 / rev_0 > gp_1 / rev_1); n += 1

            # F9: Asset turnover improving
            if all(v is not None for v in [rev_0, rev_1, assets_0, assets_1]) and assets_0 and assets_1:
                s += int(rev_0 / assets_0 > rev_1 / assets_1); n += 1

            if n < MIN_F_SCORE_SIGNALS:
                log("F-Score: only %d/9 signals had data for %s — skipping", n, ticker)
                return None, n

            return s, n

        except Exception as exc:
            logger.debug("F-Score failed for %s: %s", ticker, exc)
            return None, 0


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
