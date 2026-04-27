"""
Portfolio state management.

Loads / saves portfolio.json and tracks per-position peak prices
for the trailing-stop calculation in TrailingStopMonitor.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Holding:
    ticker: str
    shares: float
    avg_cost_sek: float
    peak_price_sek: float
    current_price_sek: float = 0.0
    sector: str = "Unknown"

    @property
    def market_value_sek(self) -> float:
        return self.shares * self.current_price_sek

    @property
    def drawdown_from_peak(self) -> float:
        if self.peak_price_sek <= 0:
            return 0.0
        return (self.peak_price_sek - self.current_price_sek) / self.peak_price_sek


@dataclass
class PortfolioState:
    updated: str
    currency: str
    total_value_sek: float
    cash_sek: float
    holdings: list[Holding] = field(default_factory=list)

    def as_weight_series(self) -> pd.Series:
        total = sum(h.market_value_sek for h in self.holdings) + self.cash_sek
        if total <= 0:
            return pd.Series(dtype=float)
        return pd.Series({h.ticker: h.market_value_sek / total for h in self.holdings})

    def update_current_prices(self, latest_prices: pd.Series, usd_sek: float = 1.0) -> None:
        """
        Refresh current_price_sek for each holding and ratchet peak upward.
        For US tickers (no .ST suffix) prices are in USD — multiply by usd_sek.
        """
        for h in self.holdings:
            if h.ticker not in latest_prices.index:
                continue
            raw_price = float(latest_prices[h.ticker])
            h.current_price_sek = raw_price if h.ticker.endswith(".ST") else raw_price * usd_sek
            if h.current_price_sek > h.peak_price_sek:
                h.peak_price_sek = h.current_price_sek

        # Recalculate total portfolio value
        invested = sum(h.market_value_sek for h in self.holdings)
        self.total_value_sek = invested + self.cash_sek

    def save(self, path: Path) -> None:
        data = {
            "updated": str(date.today()),
            "currency": self.currency,
            "total_value_sek": round(self.total_value_sek, 2),
            "cash_sek": round(self.cash_sek, 2),
            "holdings": [
                {
                    "ticker": h.ticker,
                    "shares": h.shares,
                    "avg_cost_sek": round(h.avg_cost_sek, 4),
                    "peak_price_sek": round(h.peak_price_sek, 4),
                    "current_price_sek": round(h.current_price_sek, 4),
                    "sector": h.sector,
                }
                for h in self.holdings
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Portfolio saved → %s", path)


def load_portfolio(path: Path) -> PortfolioState:
    if not path.exists():
        raise FileNotFoundError(f"portfolio.json not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    holdings = [Holding(**h) for h in data.get("holdings", [])]
    return PortfolioState(
        updated=data["updated"],
        currency=data["currency"],
        total_value_sek=data["total_value_sek"],
        cash_sek=data["cash_sek"],
        holdings=holdings,
    )
