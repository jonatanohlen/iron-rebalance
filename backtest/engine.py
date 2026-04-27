"""
Backtest Engine — Feature A
============================
Rolling weekly backtest of the IRON-REBALANCER momentum + risk pipeline.

Known limitation (documented):
  Fundamental filters (F-Score, ROIC) use CURRENT yfinance values — not
  point-in-time historical snapshots.  The backtest therefore measures the
  momentum + correlation + vol-target components of the strategy.  A full
  PIT fundamental backtest would require a paid fundamental data provider.

Algorithm per rebalancing step:
  1. Rank in-universe tickers by skip-1-month momentum (look-back to T-252d)
  2. Take top-N candidates
  3. Apply greedy correlation de-duplication
  4. Compute Inverse-Vol weights on trailing 63 days
  5. Clip to max_weight; scale to vol_target
  6. Compute realised return over the next 5 trading days
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from core.risk import (
    CorrelationFilter,
    InverseVolWeighter,
    RiskConfig,
    TrailingStopMonitor,
    VolatilityScaler,
)

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
REBAL_FREQ = 5          # every 5 trading days ≈ weekly


@dataclass
class BacktestConfig:
    start_date: str             # "YYYY-MM-DD"
    end_date: str               # "YYYY-MM-DD"
    top_n: int = 20             # candidates before corr filter
    max_weight: float = 0.08
    vol_target: float = 0.15
    max_correlation: float = 0.70
    momentum_long: int = 252    # 12M
    momentum_skip: int = 21     # 1M
    min_history: int = 273      # days needed before first rebal


@dataclass
class RebalStep:
    date: str
    weights: dict[str, float]
    candidates: list[str]
    dropped_by_corr: list[str]
    port_vol: float
    weekly_return: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    steps: list[RebalStep]
    equity_curve: pd.Series         # cumulative portfolio value, base 1.0
    benchmark_curve: pd.Series      # equal-weight benchmark
    metrics: dict[str, float]
    universe_tickers: list[str]

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "## Backtest Summary",
            f"Period      : {self.config.start_date} → {self.config.end_date}",
            f"Rebalances  : {len(self.steps)}",
            f"Annual Ret  : {m['annual_return']:.2%}",
            f"Annual Vol  : {m['annual_vol']:.2%}",
            f"Sharpe      : {m['sharpe']:.2f}",
            f"Max DD      : {m['max_drawdown']:.2%}",
            f"Calmar      : {m['calmar']:.2f}",
            f"Win Rate    : {m['win_rate']:.1%}",
            f"Avg Turnover: {m['avg_turnover']:.1%} / week",
            f"Bench Return: {m['benchmark_annual_return']:.2%}",
            f"Bench Sharpe: {m['benchmark_sharpe']:.2f}",
        ]
        return "\n".join(lines)


class BacktestEngine:
    def __init__(self, prices: pd.DataFrame, config: BacktestConfig, sector_map: dict[str, str]):
        self.prices = prices.copy()
        self.config = config
        self.sector_map = sector_map
        self._corr_filter = CorrelationFilter(
            threshold=config.max_correlation,
            lookback=63,
        )
        self._sizer = InverseVolWeighter(lookback=63)
        self._scaler = VolatilityScaler()

    def run(self) -> BacktestResult:
        cfg = self.config
        prices = self.prices

        log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
        dates = log_ret.index

        # Find first valid rebalancing index (need min_history bars of history)
        first_idx = cfg.min_history
        if first_idx >= len(dates):
            raise ValueError("Not enough history for backtest.")

        rebal_indices = list(range(first_idx, len(dates) - REBAL_FREQ, REBAL_FREQ))
        logger.info("Backtesting %d rebalances from %s to %s",
                    len(rebal_indices), dates[first_idx].date(), dates[-1].date())

        steps: list[RebalStep] = []
        port_values = [1.0]
        bench_values = [1.0]
        prev_weights: dict[str, float] = {}

        for idx in rebal_indices:
            step_date = dates[idx]

            # --- Momentum ranking (no look-ahead) ---
            candidates = self._rank_by_momentum(prices, idx)
            if not candidates:
                port_values.append(port_values[-1])
                bench_values.append(bench_values[-1])
                continue

            # Correlation filter
            ret_window = log_ret.iloc[max(0, idx - 63): idx]
            cand_df = pd.DataFrame(
                {"z_score": range(len(candidates), 0, -1)},
                index=candidates,
            )
            kept, dropped = self._corr_filter.filter(cand_df, ret_window)
            if not kept:
                kept = candidates[:1]

            # Inverse-vol weights
            ret_63 = log_ret[kept].iloc[max(0, idx - 63): idx]
            raw_w = self._sizer.compute(ret_63)
            raw_w = raw_w.clip(upper=cfg.max_weight)
            raw_w /= raw_w.sum()

            # Vol scaling
            scaled_w = self._scaler.scale(raw_w, ret_63, vol_target=cfg.vol_target)
            weights = scaled_w.to_dict()

            # --- Realised return next REBAL_FREQ days ---
            fwd = log_ret.iloc[idx: idx + REBAL_FREQ]
            port_ret = sum(
                w * fwd[t].sum() for t, w in weights.items() if t in fwd.columns
            )
            bench_tickers = [t for t in candidates if t in fwd.columns]
            bench_ret = fwd[bench_tickers].mean(axis=1).sum() if bench_tickers else 0.0

            port_values.append(port_values[-1] * np.exp(port_ret))
            bench_values.append(bench_values[-1] * np.exp(bench_ret))

            # Turnover
            turnover = sum(
                abs(weights.get(t, 0) - prev_weights.get(t, 0))
                for t in set(weights) | set(prev_weights)
            ) / 2
            prev_weights = weights

            # Portfolio vol estimate
            w_ser = pd.Series(weights)
            cov = ret_63[kept].cov() * TRADING_DAYS
            aligned = w_ser.reindex(cov.columns).fillna(0)
            port_vol = float(np.sqrt(max(float(aligned @ cov @ aligned), 1e-12)))

            steps.append(RebalStep(
                date=str(step_date.date()),
                weights=weights,
                candidates=candidates,
                dropped_by_corr=dropped,
                port_vol=port_vol,
                weekly_return=float(np.expm1(port_ret)),
            ))

        equity = pd.Series(port_values, index=[dates[first_idx]] + [dates[min(i + REBAL_FREQ, len(dates) - 1)] for i in rebal_indices[:len(port_values)-1]])
        benchmark = pd.Series(bench_values, index=equity.index)

        metrics = self._compute_metrics(equity, benchmark, steps)
        return BacktestResult(
            config=cfg,
            steps=steps,
            equity_curve=equity,
            benchmark_curve=benchmark,
            metrics=metrics,
            universe_tickers=list(prices.columns),
        )

    def _rank_by_momentum(self, prices: pd.DataFrame, idx: int) -> list[str]:
        cfg = self.config
        if idx < cfg.momentum_long:
            return []
        scores: dict[str, float] = {}
        for ticker in prices.columns:
            p = prices[ticker].iloc[: idx + 1].dropna()
            if len(p) < cfg.momentum_long:
                continue
            ret_long = p.iloc[-1] / p.iloc[-cfg.momentum_long] - 1
            ret_skip = p.iloc[-1] / p.iloc[-cfg.momentum_skip] - 1
            scores[ticker] = ret_long - ret_skip
        ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
        return ranked[: self.config.top_n]

    @staticmethod
    def _compute_metrics(
        equity: pd.Series, benchmark: pd.Series, steps: list[RebalStep]
    ) -> dict[str, float]:
        n = len(steps)
        if n == 0:
            return {}

        weekly_rets = equity.pct_change().dropna()
        bench_rets = benchmark.pct_change().dropna()

        ann = TRADING_DAYS / REBAL_FREQ  # annualisation factor for weekly data

        annual_return = float(equity.iloc[-1] ** (ann / n) - 1)
        annual_vol = float(weekly_rets.std() * np.sqrt(ann))
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = float(drawdown.min())
        calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0

        win_rate = float((pd.Series([s.weekly_return for s in steps]) > 0).mean())
        avg_turnover = float(np.mean([
            sum(abs(s.weights.get(t, 0) - (steps[i-1].weights.get(t, 0) if i > 0 else 0))
                for t in set(s.weights) | set(steps[i-1].weights if i > 0 else {})) / 2
            for i, s in enumerate(steps)
        ]))

        bench_annual = float(benchmark.iloc[-1] ** (ann / n) - 1)
        bench_vol = float(bench_rets.std() * np.sqrt(ann))
        bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0.0

        return {
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "win_rate": win_rate,
            "avg_turnover": avg_turnover,
            "benchmark_annual_return": bench_annual,
            "benchmark_sharpe": bench_sharpe,
        }
