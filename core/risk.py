"""
Core risk management primitives.

Components:
  CorrelationFilter      — greedy de-duplication of correlated pairs
  InverseVolWeighter     — 1/σ position weights
  KellyWeighter          — fractional Kelly, capped at max_fraction
  VolatilityScaler       — scale weights to hit a portfolio vol target
  ConstraintOptimizer    — CVXPY: enforce per-asset and sector caps
  TrailingStopMonitor    — flag holdings that have breached their peak
  compute_portfolio_metrics — summary dict for downstream reporting

No LLM logic lives here. All decisions are closed-form or convex optimisation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    max_weight: float = 0.08
    sector_cap: float = 0.25
    vol_target: float = 0.15
    max_correlation: float = 0.70
    kelly_max_fraction: float = 0.25
    trailing_stop_pct: float = 0.15
    lookback_days: int = 252
    min_adv_sek: float = 10_000_000.0
    max_data_gap_pct: float = 0.05
    top_n_candidates: int = 30


@dataclass
class WeightResult:
    weights: pd.Series                      # final target weights (sums ≤ 1)
    portfolio_vol: float                    # annualised realised vol
    sector_exposures: dict[str, float]      # {sector: weight_sum}
    dropped_by_correlation: list[str]       # tickers removed by corr filter
    trailing_stop_triggered: list[str]      # tickers removed by stop rule
    constraint_violations: list[str] = field(default_factory=list)


# ── Correlation filter ─────────────────────────────────────────────────────────

class CorrelationFilter:
    """
    Greedy pair-wise de-correlator.
    Iterates until no remaining pair exceeds `threshold`.
    In each violating pair, the ticker with the lower composite Z-score is dropped.
    """

    def __init__(self, threshold: float = 0.70, lookback: int = 252):
        self.threshold = threshold
        self.lookback = lookback

    def filter(
        self,
        candidates: pd.DataFrame,   # index=ticker, must contain 'z_score' column
        returns: pd.DataFrame,       # daily log-returns, columns=tickers
    ) -> tuple[list[str], list[str]]:
        """Returns (kept_tickers, dropped_tickers)."""
        available = [t for t in candidates.index if t in returns.columns]
        if len(available) < 2:
            return available, []

        corr = returns[available].tail(self.lookback).corr()
        kept = available.copy()
        dropped: list[str] = []

        changed = True
        while changed:
            changed = False
            for i, t1 in enumerate(kept):
                for t2 in kept[i + 1:]:
                    if corr.loc[t1, t2] > self.threshold:
                        z1 = candidates.loc[t1, "z_score"]
                        z2 = candidates.loc[t2, "z_score"]
                        loser = t2 if z1 >= z2 else t1
                        winner = t1 if loser == t2 else t2
                        logger.info(
                            "Corr filter: dropping %s (z=%.2f) — corr %.2f with %s (z=%.2f)",
                            loser, candidates.loc[loser, "z_score"],
                            corr.loc[t1, t2],
                            winner, candidates.loc[winner, "z_score"],
                        )
                        kept.remove(loser)
                        dropped.append(loser)
                        changed = True
                        break
                if changed:
                    break

        return kept, dropped


# ── Position sizers ──────────────────────────────────────────────────────────

class InverseVolWeighter:
    """
    Weight proportional to 1 / realised_volatility.
    Lower-volatility assets receive larger allocations.
    """

    def __init__(self, lookback: int = 63):
        self.lookback = lookback

    def compute(self, returns: pd.DataFrame) -> pd.Series:
        vols = returns.tail(self.lookback).std() * np.sqrt(TRADING_DAYS)
        vols = vols.replace(0, np.nan).dropna()
        inv_vol = 1.0 / vols
        return inv_vol / inv_vol.sum()


class KellyWeighter:
    """
    Fractional Kelly: f_i = μ_i / σ_i²  clipped to [0, max_fraction].
    Normalised to sum to 1.
    """

    def __init__(self, max_fraction: float = 0.25, lookback: int = 252):
        self.max_fraction = max_fraction
        self.lookback = lookback

    def compute(self, returns: pd.DataFrame) -> pd.Series:
        r = returns.tail(self.lookback)
        mu = r.mean() * TRADING_DAYS
        var = r.var() * TRADING_DAYS
        kelly = (mu / var.replace(0, np.nan)).clip(lower=0.0, upper=self.max_fraction)
        kelly = kelly.fillna(0.0)
        total = kelly.sum()
        if total <= 0:
            n = len(returns.columns)
            return pd.Series(1.0 / n, index=returns.columns)
        return kelly / total


# ── Vol scaler ───────────────────────────────────────────────────────────────

class VolatilityScaler:
    """
    Scales weights so realised portfolio vol equals vol_target.
    Scalar is capped at 1.0 — never applies leverage.
    """

    def scale(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        vol_target: float = 0.15,
    ) -> pd.Series:
        aligned = weights.reindex(returns.columns).fillna(0.0)
        cov = returns.tail(TRADING_DAYS).cov() * TRADING_DAYS
        port_var = float(aligned @ cov @ aligned)
        port_vol = np.sqrt(max(port_var, 1e-12))
        scalar = min(vol_target / port_vol, 1.0)
        logger.info(
            "VolScaler: port_vol=%.2f%% → scalar=%.4f (target=%.2f%%)",
            port_vol * 100, scalar, vol_target * 100,
        )
        return weights * scalar


# ── Constraint optimiser ─────────────────────────────────────────────────────

class ConstraintOptimizer:
    """
    CVXPY minimises deviation from the baseline (inv-vol / Kelly) weights
    subject to:
      - sum(w) ≤ 1          (may hold cash)
      - w_i ≤ max_weight     (single-asset cap)
      - w_i ≥ 0              (long-only)
      - sector sum ≤ sector_cap
      - portfolio variance ≤ vol_target²
    """

    def optimize(
        self,
        initial_weights: pd.Series,
        returns: pd.DataFrame,
        sector_map: dict[str, str],
        max_weight: float = 0.08,
        sector_cap: float = 0.25,
        vol_target: float = 0.15,
    ) -> pd.Series:
        tickers = initial_weights.index.tolist()
        n = len(tickers)

        ret_slice = returns[tickers].tail(TRADING_DAYS)
        cov = ret_slice.cov().values * TRADING_DAYS

        # Guarantee strict positive-definiteness before Cholesky
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(cov_pd)   # cov_pd = L @ L.T

        w = cp.Variable(n, nonneg=True)
        objective = cp.Minimize(cp.sum_squares(w - initial_weights.values))

        # cp.norm(L.T @ w, 2) ≤ vol_target  ≡  sqrt(w'Σw) ≤ vol_target
        # Expressed as a second-order cone constraint — well-behaved in all solvers.
        constraints: list = [
            cp.sum(w) <= 1.0,
            w <= max_weight,
            cp.norm(L.T @ w, 2) <= vol_target,
        ]

        # Per-sector caps
        sectors: dict[str, list[int]] = {}
        for idx, ticker in enumerate(tickers):
            sec = sector_map.get(ticker, "Unknown")
            sectors.setdefault(sec, []).append(idx)

        for sec, indices in sectors.items():
            constraints.append(cp.sum([w[i] for i in indices]) <= sector_cap)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except cp.SolverError as exc:
            logger.warning("CLARABEL failed (%s) — falling back to SCS", exc)
            prob.solve(solver=cp.SCS, verbose=False)

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Portfolio optimisation infeasible: status={prob.status}")

        # Clip numerical dust; do NOT renormalise — that would violate max_weight.
        raw = np.maximum(w.value, 0.0)
        return pd.Series(raw, index=tickers)


# ── Trailing stop monitor ────────────────────────────────────────────────────

class TrailingStopMonitor:
    """
    Compares each holding's current price against its recorded peak.
    Emits a SELL signal if drawdown from peak ≥ stop_pct.
    """

    def __init__(self, stop_pct: float = 0.15):
        self.stop_pct = stop_pct

    def check(
        self,
        holdings: list[dict],   # [{ticker, peak_price_sek, current_price_sek}, ...]
    ) -> list[str]:
        triggered: list[str] = []
        for h in holdings:
            ticker = h.get("ticker", "")
            peak = h.get("peak_price_sek", 0.0)
            current = h.get("current_price_sek", 0.0)
            if peak > 0 and current > 0:
                drawdown = (peak - current) / peak
                if drawdown >= self.stop_pct:
                    logger.warning(
                        "Trailing stop triggered: %s  drawdown=%.1f%%  peak=%.2f  now=%.2f",
                        ticker, drawdown * 100, peak, current,
                    )
                    triggered.append(ticker)
        return triggered


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_portfolio_metrics(
    weights: pd.Series,
    returns: pd.DataFrame,
) -> dict:
    aligned = weights.reindex(returns.columns).fillna(0.0)
    cov = returns.tail(TRADING_DAYS).cov() * TRADING_DAYS
    port_var = float(aligned @ cov @ aligned)
    port_vol = float(np.sqrt(max(port_var, 1e-12)))
    port_ret = float((returns.tail(TRADING_DAYS).mean() * TRADING_DAYS * aligned).sum())
    sharpe = port_ret / port_vol if port_vol > 0 else 0.0
    return {
        "annual_vol": round(port_vol, 4),
        "annual_return_est": round(port_ret, 4),
        "sharpe_ratio": round(sharpe, 3),
    }
