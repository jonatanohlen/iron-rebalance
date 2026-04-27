"""
Risk_Supervisor Agent
═════════════════════
Pure-Python, zero LLM opinion.

Pipeline (in order):
  1. Trailing-stop enforcement  — forced SELL for positions that have
                                  breached peak × (1 − stop_pct)
  2. Correlation de-duplication — greedy removal of lower-Z-score twin
                                  when pair corr > max_correlation
  3. Baseline weight computation — Inverse-Vol (default) or Kelly
  4. CVXPY constraint enforcement
       • sum(w) ≤ 1
       • w_i ≤ max_single_weight  (8%)
       • per-sector sum ≤ sector_cap (25%)
       • portfolio_vol² ≤ vol_target²
  5. Volatility scaling          — rescale to hit vol_target exactly;
                                  never applies leverage (scalar ≤ 1)
  6. Sector exposure audit       — emitted in WeightResult for downstream

WeightResult separates tickers removed by correlation from those
removed by trailing stops so Signal_Commander can tag them correctly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.risk import (
    ConstraintOptimizer,
    CorrelationFilter,
    InverseVolWeighter,
    KellyWeighter,
    RiskConfig,
    TrailingStopMonitor,
    VolatilityScaler,
    WeightResult,
    compute_portfolio_metrics,
)
from core.portfolio import PortfolioState

logger = logging.getLogger(__name__)


@dataclass
class RiskInput:
    candidates: pd.DataFrame       # index=ticker; columns: z_score, sector, …
    returns: pd.DataFrame           # daily log-returns; columns=tickers
    portfolio: Optional[PortfolioState] = None
    sector_map: dict[str, str] = None


class RiskSupervisor:
    """
    Instantiate once per run; call `.run(RiskInput)` to get a `WeightResult`.
    `use_kelly=True` switches the baseline sizer from Inverse-Vol to Kelly.
    """

    def __init__(self, config: RiskConfig, use_kelly: bool = False):
        self.config = config
        self._corr_filter = CorrelationFilter(
            threshold=config.max_correlation,
            lookback=config.lookback_days,
        )
        self._sizer = (
            KellyWeighter(
                max_fraction=config.kelly_max_fraction,
                lookback=config.lookback_days,
            )
            if use_kelly
            else InverseVolWeighter(lookback=63)
        )
        self._vol_scaler = VolatilityScaler()
        self._optimizer = ConstraintOptimizer()
        self._stop_monitor = TrailingStopMonitor(stop_pct=config.trailing_stop_pct)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, inp: RiskInput) -> WeightResult:
        sector_map: dict[str, str] = inp.sector_map or {}
        candidates = inp.candidates.copy()
        returns = inp.returns.copy()

        # ── 1. Trailing-stop enforcement ─────────────────────────────────────
        stop_tickers = self._enforce_trailing_stops(candidates, inp.portfolio)

        # Remove stopped tickers from candidate set before any further work
        candidates = candidates.drop(
            index=[t for t in stop_tickers if t in candidates.index],
            errors="ignore",
        )

        # ── 2. Correlation de-duplication ────────────────────────────────────
        kept, corr_dropped = self._corr_filter.filter(candidates, returns)
        if not kept:
            raise RuntimeError("Correlation filter removed all candidates.")

        logger.info(
            "Correlation filter: %d kept | %d dropped: %s",
            len(kept), len(corr_dropped), corr_dropped,
        )
        candidates = candidates.loc[kept]
        returns_filtered = returns[[t for t in kept if t in returns.columns]]

        # ── 3. Baseline weights ───────────────────────────────────────────────
        raw_weights = self._sizer.compute(returns_filtered)
        raw_weights = raw_weights.reindex(candidates.index).fillna(0.0)
        raw_weights /= raw_weights.sum()

        # ── 4. CVXPY constraint enforcement ──────────────────────────────────
        optimized_weights = self._run_optimizer(
            raw_weights, returns_filtered, sector_map
        )

        # ── 5. Volatility scaling ─────────────────────────────────────────────
        scaled_weights = self._vol_scaler.scale(
            optimized_weights,
            returns_filtered,
            vol_target=self.config.vol_target,
        )
        # Drop dust weights (< 0.1 bp)
        scaled_weights = scaled_weights[scaled_weights > 1e-4]

        # ── 6. Sector exposures ───────────────────────────────────────────────
        sector_exposures: dict[str, float] = {}
        for ticker, w in scaled_weights.items():
            sec = sector_map.get(ticker, "Unknown")
            sector_exposures[sec] = sector_exposures.get(sec, 0.0) + float(w)

        metrics = compute_portfolio_metrics(
            scaled_weights, returns_filtered[scaled_weights.index]
        )
        self._log_summary(scaled_weights, sector_exposures, metrics)

        return WeightResult(
            weights=scaled_weights,
            portfolio_vol=metrics["annual_vol"],
            sector_exposures=sector_exposures,
            dropped_by_correlation=corr_dropped,
            trailing_stop_triggered=stop_tickers,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _enforce_trailing_stops(
        self,
        candidates: pd.DataFrame,
        portfolio: Optional[PortfolioState],
    ) -> list[str]:
        if portfolio is None:
            return []

        check_list = [
            {
                "ticker": h.ticker,
                "peak_price_sek": h.peak_price_sek,
                "current_price_sek": h.current_price_sek,
            }
            for h in portfolio.holdings
            if h.ticker in candidates.index
        ]
        stopped = self._stop_monitor.check(check_list)
        if stopped:
            logger.warning("Trailing stops triggered — forced SELL: %s", stopped)
        return stopped

    def _run_optimizer(
        self,
        initial_weights: pd.Series,
        returns: pd.DataFrame,
        sector_map: dict[str, str],
    ) -> pd.Series:
        try:
            return self._optimizer.optimize(
                initial_weights=initial_weights,
                returns=returns,
                sector_map=sector_map,
                max_weight=self.config.max_weight,
                sector_cap=self.config.sector_cap,
                vol_target=self.config.vol_target,
            )
        except RuntimeError as exc:
            # Graceful degradation: clip-and-renormalize rather than crash
            logger.error("CVXPY failed (%s) — using clipped inv-vol weights", exc)
            capped = initial_weights.clip(upper=self.config.max_weight)
            return capped / capped.sum()

    def _log_summary(
        self,
        weights: pd.Series,
        sector_exposures: dict[str, float],
        metrics: dict,
    ) -> None:
        top5 = weights.sort_values(ascending=False).head(5)
        logger.info(
            "RiskSupervisor result | positions=%d | vol=%.1f%% | sharpe=%.2f\n"
            "  Top-5 weights: %s\n"
            "  Sector exposure: %s",
            len(weights),
            metrics["annual_vol"] * 100,
            metrics["sharpe_ratio"],
            {t: f"{w:.2%}" for t, w in top5.items()},
            {s: f"{e:.1%}" for s, e in sorted(sector_exposures.items(), key=lambda x: -x[1])},
        )
