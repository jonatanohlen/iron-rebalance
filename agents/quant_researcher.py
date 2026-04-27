"""
Quant_Researcher Agent

Hard filters (both must pass):
  - Piotroski F-Score >= 7  AND >= MIN_F_SCORE_SIGNALS signals with data
  - 3-year average ROIC > 15%

Ranking signal (Z-score combination, equal-weight):
  z_composite = mean(z_f_score, z_roic, z_momentum)

Output: top-N DataFrame  (index=ticker)
  columns: f_score, roic, momentum, z_f_score, z_roic, z_momentum, z_score, sector

Feature D: emits a DataCoverageReport after each run so you can see
exactly which tickers failed and at which stage.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from agents.data_architect import DataBundle
from core.factors import DataCoverageReport, MomentumCalculator, PiotroskiFScore, ROICCalculator

logger = logging.getLogger(__name__)


def _zscore(series: pd.Series) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    return (series - mu) / (sigma if sigma > 0 else 1.0)


class QuantResearcher:
    def __init__(
        self,
        f_score_min: int = 7,
        roic_min: float = 0.15,
        top_n: int = 30,
        workers: int = 8,
    ):
        self.f_score_min = f_score_min
        self.roic_min = roic_min
        self.top_n = top_n
        self.workers = workers
        self._f_scorer = PiotroskiFScore()
        self._roic_calc = ROICCalculator()
        self._momentum_calc = MomentumCalculator()

    def run(
        self,
        bundle: DataBundle,
        sector_map: dict[str, str],
        coverage: DataCoverageReport | None = None,
    ) -> pd.DataFrame:
        """
        Returns a ranked DataFrame; empty if no candidates survive the hard filters.
        `coverage` is updated in-place if provided.
        """
        cov = coverage or DataCoverageReport()
        cov.total = len(bundle.tickers)
        raw: dict[str, dict] = {}

        def _evaluate(ticker: str) -> tuple[str, dict | None, str]:
            # F-Score — now returns (score, n_signals)
            f_score, n_signals = self._f_scorer.score(ticker)

            if f_score is None:
                reason = "no_fundamentals" if n_signals == 0 else "f_coverage"
                return ticker, None, reason

            if f_score < self.f_score_min:
                return ticker, None, "f_filter"

            roic = self._roic_calc.compute(ticker)
            if roic is None:
                return ticker, None, "roic_coverage"
            if roic < self.roic_min:
                return ticker, None, "roic_filter"

            mom = self._momentum_calc.compute(bundle.prices[ticker].dropna())
            if mom is None:
                return ticker, None, "momentum"

            return ticker, {
                "f_score": float(f_score),
                "f_signals": n_signals,
                "roic": float(roic),
                "momentum": float(mom),
                "sector": sector_map.get(ticker, "Unknown"),
            }, "ok"

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_evaluate, t): t for t in bundle.tickers}
            for fut in as_completed(futures):
                ticker, data, reason = fut.result()
                if data:
                    raw[ticker] = data
                    cov.passed_f_score += 1
                    cov.passed_roic += 1
                    cov.passed_momentum += 1
                else:
                    if reason == "no_fundamentals":
                        cov.failed_no_fundamentals.append(ticker)
                    elif reason == "f_coverage":
                        cov.failed_f_score_coverage.append(ticker)
                    elif reason == "f_filter":
                        cov.failed_f_score_filter.append(ticker)
                    elif reason == "roic_coverage":
                        cov.passed_f_score += 1
                        cov.failed_roic_coverage.append(ticker)
                    elif reason == "roic_filter":
                        cov.passed_f_score += 1
                        cov.failed_roic_filter.append(ticker)
                    elif reason == "momentum":
                        cov.passed_f_score += 1
                        cov.passed_roic += 1
                        cov.failed_momentum.append(ticker)

        cov.log_summary()

        if not raw:
            logger.warning("QuantResearcher: no tickers survived hard alpha filters.")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(raw, orient="index")

        for col in ("f_score", "roic", "momentum"):
            df[f"z_{col}"] = _zscore(df[col])

        df["z_score"] = df[["z_f_score", "z_roic", "z_momentum"]].mean(axis=1)
        df = df.sort_values("z_score", ascending=False).head(self.top_n)

        logger.info(
            "QuantResearcher: %d candidates | z_score [%.2f, %.2f]",
            len(df), df["z_score"].min(), df["z_score"].max(),
        )
        return df
