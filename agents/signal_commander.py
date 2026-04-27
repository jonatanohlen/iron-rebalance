"""
Signal_Commander Agent

Diffs current portfolio.json against the new target weights and emits
a typed Action Table (BUY / SELL / TRIM / HOLD) with exact share counts
and SEK amounts.

Rules:
  • Trailing-stop tickers        → SELL  (tagged ⛔ STOP)
  • Ticker at 0 target, held now → SELL
  • Ticker not held, target > 0  → BUY
  • Target < current − threshold → TRIM
  • Target > current + threshold → BUY (top-up)
  • |drift| < threshold          → HOLD

Default threshold: 0.5 percentage points (avoids micro-trades).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from tabulate import tabulate

from core.portfolio import PortfolioState

logger = logging.getLogger(__name__)

Action = Literal["BUY", "SELL", "TRIM", "HOLD"]
_ACTION_PRIORITY: dict[Action, int] = {"SELL": 0, "TRIM": 1, "BUY": 2, "HOLD": 3}
_THRESHOLD = 0.005  # 0.5 pp drift triggers a trade


@dataclass
class Signal:
    ticker: str
    action: Action
    current_weight: float
    target_weight: float
    current_shares: float
    target_shares: float
    delta_shares: float
    delta_value_sek: float
    current_price_sek: float
    z_score: float
    sector: str
    trailing_stop: bool = False


class SignalCommander:
    def __init__(self, trade_threshold: float = _THRESHOLD):
        self.trade_threshold = trade_threshold

    def run(
        self,
        target_weights: pd.Series,
        portfolio: PortfolioState,
        candidates: pd.DataFrame,
        trailing_stop_tickers: list[str],
        output_dir: Path,
    ) -> list[Signal]:
        total = portfolio.total_value_sek
        current_weights = portfolio.as_weight_series()
        holdings_map = {h.ticker: h for h in portfolio.holdings}

        all_tickers = sorted(set(current_weights.index) | set(target_weights.index))
        signals: list[Signal] = []

        for ticker in all_tickers:
            cur_w = float(current_weights.get(ticker, 0.0))
            tgt_w = float(target_weights.get(ticker, 0.0))
            drift = tgt_w - cur_w

            h = holdings_map.get(ticker)
            price = h.current_price_sek if h else 0.0
            cur_shares = h.shares if h else 0.0
            z = float(candidates.loc[ticker, "z_score"]) if ticker in candidates.index else 0.0
            sector = candidates.loc[ticker, "sector"] if ticker in candidates.index else "Unknown"
            is_stop = ticker in trailing_stop_tickers

            if is_stop:
                action: Action = "SELL"
                tgt_w = 0.0
                delta_shares = -cur_shares
            elif tgt_w == 0.0 and cur_w > 0:
                action = "SELL"
                delta_shares = -cur_shares
            elif cur_w == 0.0 and tgt_w > 0:
                action = "BUY"
                delta_shares = (tgt_w * total / price) if price > 0 else 0.0
            elif drift > self.trade_threshold:
                action = "BUY"
                delta_shares = (drift * total / price) if price > 0 else 0.0
            elif drift < -self.trade_threshold:
                action = "TRIM"
                delta_shares = (drift * total / price) if price > 0 else 0.0
            else:
                action = "HOLD"
                delta_shares = 0.0

            signals.append(
                Signal(
                    ticker=ticker,
                    action=action,
                    current_weight=cur_w,
                    target_weight=tgt_w,
                    current_shares=cur_shares,
                    target_shares=cur_shares + delta_shares,
                    delta_shares=delta_shares,
                    delta_value_sek=delta_shares * price if price > 0 else 0.0,
                    current_price_sek=price,
                    z_score=z,
                    sector=sector,
                    trailing_stop=is_stop,
                )
            )

        signals.sort(
            key=lambda s: (_ACTION_PRIORITY[s.action], -abs(s.delta_value_sek))
        )

        self._write(signals, output_dir)
        return signals

    # ── Output ────────────────────────────────────────────────────────────────

    def _write(self, signals: list[Signal], output_dir: Path) -> None:
        rows = []
        for s in signals:
            label = f"{s.ticker}{'  ⛔ STOP' if s.trailing_stop else ''}"
            rows.append(
                [
                    label,
                    s.action,
                    f"{s.current_weight:.2%}",
                    f"{s.target_weight:.2%}",
                    f"{s.current_shares:.2f}",
                    f"{s.delta_shares:+.2f}",
                    f"{s.delta_value_sek:+,.0f}",
                    f"{s.z_score:.2f}",
                    s.sector,
                ]
            )

        headers = [
            "Ticker", "Action", "Cur Wt", "Tgt Wt",
            "Cur Shs", "Δ Shares", "Δ SEK", "Z-Score", "Sector",
        ]
        table = tabulate(rows, headers=headers, tablefmt="github")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{ts}_signals.md"
        path.write_text(
            f"# IRON-REBALANCER — Action Table\n"
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"{table}\n",
            encoding="utf-8",
        )
        logger.info("Action table → %s", path)
