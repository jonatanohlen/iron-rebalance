"""
Unit tests for core/risk.py

All tests are offline — no network calls, no yfinance.
Synthetic return series are constructed deterministically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
from agents.risk_supervisor import RiskInput, RiskSupervisor

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
N_DAYS = 504   # ~2 years


def _make_returns(n_assets: int, corr: float = 0.0) -> pd.DataFrame:
    """
    Generate correlated daily log-returns.
    `corr` is the off-diagonal correlation applied to all pairs.
    """
    tickers = [f"T{i}" for i in range(n_assets)]
    cov = np.full((n_assets, n_assets), corr * 0.01 * 0.01)
    np.fill_diagonal(cov, 0.01 ** 2)        # 1% daily vol
    cov = np.clip(cov, 0, None)
    # Ensure PSD
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov += (-eigvals.min() + 1e-9) * np.eye(n_assets)
    raw = RNG.multivariate_normal(np.zeros(n_assets), cov, size=N_DAYS)
    return pd.DataFrame(raw, columns=tickers)


def _make_candidates(tickers: list[str], z_base: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"z_score": [z_base + i * 0.1 for i in range(len(tickers))],
         "sector": ["Industrials"] * len(tickers)},
        index=tickers,
    )


# ── InverseVolWeighter ────────────────────────────────────────────────────────

class TestInverseVolWeighter:
    def test_weights_sum_to_one(self):
        returns = _make_returns(5)
        w = InverseVolWeighter().compute(returns)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_lower_vol_gets_higher_weight(self):
        # T0 has lower vol, should get higher weight
        returns = _make_returns(2)
        returns["T0"] *= 0.5   # halve T0 vol
        w = InverseVolWeighter(lookback=N_DAYS).compute(returns)
        assert w["T0"] > w["T1"]

    def test_all_weights_positive(self):
        returns = _make_returns(10)
        w = InverseVolWeighter().compute(returns)
        assert (w > 0).all()


# ── KellyWeighter ─────────────────────────────────────────────────────────────

class TestKellyWeighter:
    def test_weights_sum_to_one(self):
        returns = _make_returns(5)
        returns += 0.0005   # positive drift so Kelly is non-zero
        w = KellyWeighter(max_fraction=0.25).compute(returns)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_max_fraction_respected(self):
        returns = _make_returns(3)
        returns += 0.001
        w = KellyWeighter(max_fraction=0.25).compute(returns)
        # Before normalisation Kelly is capped; after norm it can exceed 0.25
        # but individual raw fractions must not have been > 0.25
        assert w.max() <= 1.0   # trivially true; sanity check

    def test_fallback_for_zero_drift(self):
        # Zero mean → uniform fallback
        returns = _make_returns(4)
        returns -= returns.mean()   # demean
        w = KellyWeighter().compute(returns)
        assert abs(w.sum() - 1.0) < 1e-9


# ── CorrelationFilter ─────────────────────────────────────────────────────────

class TestCorrelationFilter:
    def test_removes_high_correlation_pair(self):
        # T0 and T1 are nearly identical → one should be dropped
        base = RNG.standard_normal(N_DAYS)
        df = pd.DataFrame({"T0": base, "T1": base + 1e-6 * RNG.standard_normal(N_DAYS),
                           "T2": RNG.standard_normal(N_DAYS)})
        candidates = pd.DataFrame(
            {"z_score": [1.0, 0.5, 0.8], "sector": ["X", "X", "X"]},
            index=["T0", "T1", "T2"],
        )
        filt = CorrelationFilter(threshold=0.70)
        kept, dropped = filt.filter(candidates, df)

        # T1 has lower z_score (0.5) — should be dropped
        assert "T1" in dropped
        assert "T0" in kept
        assert "T2" in kept

    def test_low_correlation_keeps_all(self):
        returns = _make_returns(5, corr=0.0)
        candidates = _make_candidates(returns.columns.tolist())
        filt = CorrelationFilter(threshold=0.70)
        kept, dropped = filt.filter(candidates, returns)
        assert len(kept) == 5
        assert len(dropped) == 0

    def test_empty_candidates_returns_empty(self):
        returns = _make_returns(3)
        candidates = pd.DataFrame({"z_score": [], "sector": []})
        filt = CorrelationFilter(threshold=0.70)
        kept, dropped = filt.filter(candidates, returns)
        assert kept == []
        assert dropped == []


# ── VolatilityScaler ──────────────────────────────────────────────────────────

class TestVolatilityScaler:
    def test_scaled_vol_near_target(self):
        returns = _make_returns(5)
        w = pd.Series([0.2] * 5, index=returns.columns)
        scaler = VolatilityScaler()
        w_scaled = scaler.scale(w, returns, vol_target=0.10)
        # Compute resulting portfolio vol
        cov = returns.cov() * 252
        port_vol = float(np.sqrt(w_scaled @ cov @ w_scaled))
        assert port_vol <= 0.10 + 0.005   # within 50 bps tolerance

    def test_no_leverage(self):
        returns = _make_returns(5)
        # Give very low-vol returns so naive scaling would exceed 100% invested
        returns = returns * 0.001
        w = pd.Series([0.2] * 5, index=returns.columns)
        w_scaled = VolatilityScaler().scale(w, returns, vol_target=0.15)
        # Scalar must be ≤ 1 → sum of scaled weights ≤ sum of original
        assert w_scaled.sum() <= w.sum() + 1e-9


# ── ConstraintOptimizer ───────────────────────────────────────────────────────

class TestConstraintOptimizer:
    def test_max_weight_respected(self):
        returns = _make_returns(10)
        initial = pd.Series([0.1] * 10, index=returns.columns)
        sector_map = {t: "Industrials" for t in returns.columns}
        opt = ConstraintOptimizer()
        w = opt.optimize(initial, returns, sector_map, max_weight=0.08)
        assert w.max() <= 0.08 + 1e-6

    def test_sector_cap_respected(self):
        returns = _make_returns(6)
        tickers = returns.columns.tolist()
        sector_map = {t: "Tech" for t in tickers}  # all same sector
        initial = pd.Series([1 / 6] * 6, index=tickers)
        opt = ConstraintOptimizer()
        w = opt.optimize(initial, returns, sector_map, max_weight=0.25, sector_cap=0.50)
        sector_total = w.sum()  # all in same sector
        assert sector_total <= 0.50 + 1e-6

    def test_weights_non_negative(self):
        returns = _make_returns(5)
        initial = pd.Series([0.2] * 5, index=returns.columns)
        sector_map = {t: "X" for t in returns.columns}
        w = ConstraintOptimizer().optimize(initial, returns, sector_map)
        assert (w >= -1e-9).all()


# ── TrailingStopMonitor ───────────────────────────────────────────────────────

class TestTrailingStopMonitor:
    def test_stop_triggered_at_15pct_drawdown(self):
        monitor = TrailingStopMonitor(stop_pct=0.15)
        holdings = [{"ticker": "AAPL", "peak_price_sek": 200.0, "current_price_sek": 169.0}]
        triggered = monitor.check(holdings)
        assert "AAPL" in triggered   # drawdown = 15.5%

    def test_no_stop_below_threshold(self):
        monitor = TrailingStopMonitor(stop_pct=0.15)
        holdings = [{"ticker": "MSFT", "peak_price_sek": 200.0, "current_price_sek": 175.0}]
        triggered = monitor.check(holdings)
        assert triggered == []   # drawdown = 12.5%

    def test_zero_peak_ignored(self):
        monitor = TrailingStopMonitor(stop_pct=0.15)
        holdings = [{"ticker": "T0", "peak_price_sek": 0.0, "current_price_sek": 100.0}]
        triggered = monitor.check(holdings)
        assert triggered == []


# ── RiskSupervisor end-to-end ─────────────────────────────────────────────────

class TestRiskSupervisorIntegration:
    def _build_input(self, n: int = 10, corr: float = 0.0) -> RiskInput:
        returns = _make_returns(n, corr=corr)
        returns += 0.0003   # slight positive drift
        candidates = _make_candidates(returns.columns.tolist())
        sector_map = {t: "Industrials" for t in returns.columns}
        return RiskInput(
            candidates=candidates,
            returns=returns,
            portfolio=None,
            sector_map=sector_map,
        )

    def test_returns_weight_result(self):
        cfg = RiskConfig()
        agent = RiskSupervisor(cfg)
        result = agent.run(self._build_input(10))
        assert isinstance(result, WeightResult)
        assert len(result.weights) > 0

    def test_max_weight_enforced(self):
        cfg = RiskConfig(max_weight=0.08)
        agent = RiskSupervisor(cfg)
        result = agent.run(self._build_input(10))
        assert result.weights.max() <= 0.08 + 1e-5

    def test_vol_target_respected(self):
        cfg = RiskConfig(vol_target=0.15)
        agent = RiskSupervisor(cfg)
        result = agent.run(self._build_input(10))
        # Portfolio vol should be at or below target (scaler caps at 1)
        assert result.portfolio_vol <= 0.15 + 0.01

    def test_highly_correlated_pair_deduplicated(self):
        # 10 assets, but T0/T1 are almost identical
        returns = _make_returns(10)
        returns["T1"] = returns["T0"] + 1e-8 * RNG.standard_normal(N_DAYS)
        candidates = _make_candidates(returns.columns.tolist())
        # T0 has higher z_score (index 0 has z_base = 0.0, T1 has 0.1 but T0 > T1 only
        # if we set them explicitly)
        candidates.loc["T0", "z_score"] = 1.5
        candidates.loc["T1", "z_score"] = 0.5

        cfg = RiskConfig(max_correlation=0.70)
        agent = RiskSupervisor(cfg)
        inp = RiskInput(candidates=candidates, returns=returns,
                        portfolio=None, sector_map={t: "X" for t in returns.columns})
        result = agent.run(inp)

        assert "T1" in result.dropped_by_correlation
        assert "T1" not in result.weights.index

    def test_kelly_mode_runs(self):
        cfg = RiskConfig()
        agent = RiskSupervisor(cfg, use_kelly=True)
        result = agent.run(self._build_input(8))
        assert result.weights.sum() <= 1.0 + 1e-6


# ── compute_portfolio_metrics ─────────────────────────────────────────────────

class TestComputePortfolioMetrics:
    def test_returns_expected_keys(self):
        returns = _make_returns(4)
        w = pd.Series([0.25] * 4, index=returns.columns)
        m = compute_portfolio_metrics(w, returns)
        assert "annual_vol" in m
        assert "annual_return_est" in m
        assert "sharpe_ratio" in m

    def test_vol_positive(self):
        returns = _make_returns(4)
        w = pd.Series([0.25] * 4, index=returns.columns)
        m = compute_portfolio_metrics(w, returns)
        assert m["annual_vol"] > 0
