"""
Auditor_Agent

Stress-tests the proposed portfolio and writes a Markdown Vulnerability Report.

Stress scenario:
  A 20% broad market decline.  Each asset's contribution is scaled by
  its beta vs. the portfolio (proxy beta, computed from the covariance
  of each asset's returns with portfolio returns).

Historical metrics:
  • Maximum drawdown with date
  • Daily CVaR at 95% confidence

Report is written to output/<timestamp>_vulnerability_report.md.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.risk import TRADING_DAYS, compute_portfolio_metrics

logger = logging.getLogger(__name__)


class AuditorAgent:
    def __init__(self, market_drop_pct: float = 0.20):
        self.market_drop_pct = market_drop_pct

    def run(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        candidates: pd.DataFrame,
        output_dir: Path,
    ) -> str:
        """Returns the path of the written report."""
        w = weights.reindex(returns.columns).fillna(0.0)
        port_returns = (returns * w).sum(axis=1).dropna()

        # Historical drawdown
        cum = (1 + port_returns).cumprod()
        rolling_max = cum.cummax()
        dd = (cum - rolling_max) / rolling_max
        max_dd = float(dd.min())
        max_dd_date = str(dd.idxmin())[:10]

        # CVaR 95%
        threshold = port_returns.quantile(0.05)
        cvar_95 = float(port_returns[port_returns <= threshold].mean())

        # Proxy betas vs portfolio
        betas = self._proxy_betas(weights, returns)

        # Stressed losses per asset
        stress: dict[str, float] = {
            ticker: -self.market_drop_pct * betas.get(ticker, 1.0) * float(weights.get(ticker, 0))
            for ticker in weights.index
        }
        total_stress = sum(stress.values())

        metrics = compute_portfolio_metrics(weights, returns[weights.index])

        report = self._build_report(
            weights=weights,
            candidates=candidates,
            metrics=metrics,
            max_dd=max_dd,
            max_dd_date=max_dd_date,
            cvar_95=cvar_95,
            stress=stress,
            total_stress=total_stress,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{ts}_vulnerability_report.md"
        path.write_text(report, encoding="utf-8")
        logger.info("Vulnerability report → %s", path)
        return str(path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _proxy_betas(
        self, weights: pd.Series, returns: pd.DataFrame
    ) -> dict[str, float]:
        w = weights.reindex(returns.columns).fillna(0.0)
        port_ret = (returns * w).sum(axis=1)
        port_var = float(port_ret.var())
        if port_var <= 0:
            return {t: 1.0 for t in weights.index}
        return {
            ticker: float(returns[ticker].cov(port_ret) / port_var)
            for ticker in weights.index
            if ticker in returns.columns
        }

    def _build_report(
        self,
        weights: pd.Series,
        candidates: pd.DataFrame,
        metrics: dict,
        max_dd: float,
        max_dd_date: str,
        cvar_95: float,
        stress: dict[str, float],
        total_stress: float,
    ) -> str:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "# IRON-REBALANCER — Vulnerability Report",
            f"**Generated:** {now}",
            "",
            "## Portfolio Metrics",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Annualised Volatility | {metrics['annual_vol']:.1%} |",
            f"| Estimated Annual Return | {metrics['annual_return_est']:.1%} |",
            f"| Sharpe Ratio | {metrics['sharpe_ratio']:.2f} |",
            f"| Historical Max Drawdown | {max_dd:.1%} (peak on {max_dd_date}) |",
            f"| Daily CVaR (95%) | {cvar_95:.3%} |",
            "",
            f"## Stress Test: {self.market_drop_pct:.0%} Market Drop",
            f"**Total Estimated Portfolio Loss: {total_stress:.2%}**",
            "",
            "| Ticker | Weight | Beta (proxy) | Estimated Loss |",
            "|--------|--------|--------------|----------------|",
        ]

        for ticker in sorted(stress, key=lambda t: stress[t]):
            w = float(weights.get(ticker, 0))
            loss = stress[ticker]
            lines.append(f"| {ticker} | {w:.2%} | — | {loss:.2%} |")

        lines += ["", "## Risk Flags"]
        flags: list[str] = []

        if metrics["annual_vol"] > 0.18:
            flags.append(f"- ⚠️  Portfolio vol **{metrics['annual_vol']:.1%}** exceeds 18% ceiling")
        if total_stress < -0.15:
            flags.append(f"- ⚠️  Stress loss **{total_stress:.1%}** exceeds 15% tolerance")
        if abs(cvar_95) > 0.025:
            flags.append(f"- ⚠️  Daily CVaR **{cvar_95:.3%}** exceeds 2.5% threshold")
        if max_dd < -0.30:
            flags.append(f"- ⚠️  Historical max drawdown **{max_dd:.1%}** below −30%")

        if not flags:
            flags.append("- ✅  No critical risk flags detected.")

        lines += flags
        return "\n".join(lines) + "\n"
