"""
Quant_Researcher Agent

Hard filters (both must pass):
  • Piotroski F-Score ≥ 7
  • 3-year average ROIC > 15%

Ranking signal (Z-score combination, equal-weight):
  z_composite = mean(z_f_score, z_roic, z_momentum)

Output: top-N DataFrame  (index=ticker)
  columns: f_score, roic, momentum, z_f_score, z_roic, z_momentum, z_score, sector

Fundamental data fetched via yfinance in parallel threads.
Tickers that fail to return data are silently dropped (logged at DEBUG).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from agents.data_architect import DataBundle
from core.factors import MomentumCalculator, PiotroskiFScore, ROICCalculator

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

    def run(self, bundle: DataBundle, sector_map: dict[str, str]) -> pd.DataFrame:
        """
        Returns a ranked DataFrame; empty if no candidates survive the hard filters.
        """
        raw: dict[str, dict] = {}

        def _evaluate(ticker: str) -> tuple[str, dict | None]:
            f = self._f_scorer.score(ticker)
            if f is None or f < self.f_score_min:
                logger.debug("%s: F-Score=%s — dropped", ticker, f)
                return ticker, None

            roic = self._roic_calc.compute(ticker)
            if roic is None or roic < self.roic_min:
                logger.debug("%s: ROIC=%s — dropped", ticker, roic)
                return ticker, None

            mom = self._momentum_calc.compute(bundle.prices[ticker].dropna())
            if mom is None:
                logger.debug("%s: insufficient price history for momentum", ticker)
                return ticker, None

            return ticker, {
                "f_score": float(f),
                "roic": float(roic),
                "momentum": float(mom),
                "sector": sector_map.get(ticker, "Unknown"),
            }

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_evaluate, t): t for t in bundle.tickers}
            for fut in as_completed(futures):
                ticker, data = fut.result()
                if data:
                    raw[ticker] = data

        if not raw:
            logger.warning("QuantResearcher: no tickers survived hard alpha filters.")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(raw, orient="index")

        # Z-score individual factors then average
        for col in ("f_score", "roic", "momentum"):
            df[f"z_{col}"] = _zscore(df[col])

        df["z_score"] = df[["z_f_score", "z_roic", "z_momentum"]].mean(axis=1)
        df = df.sort_values("z_score", ascending=False).head(self.top_n)

        logger.info(
            "QuantResearcher: %d candidates  |  z_score range [%.2f, %.2f]",
            len(df), df["z_score"].min(), df["z_score"].max(),
        )
        return df
