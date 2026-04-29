"""
Universe management.

Data loading priority:
  1. Local cache  — data/prices_close.csv + data/prices_volume.csv
                    (committed daily by the GitHub Actions workflow)
  2. yfinance     — direct download (works outside the sandboxed environment)

UniverseFilter then applies:
  - Data-gap filter  (max_gap_pct missing trading days)
  - Liquidity filter (20-day ADV >= min_adv_sek)

Returns clean prices and log-returns DataFrames ready for downstream agents.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

_FX_FALLBACK_USDSEK = 10.5


def get_usd_sek_rate() -> float:
    # Try the cached meta file first (always available in the Actions environment)
    meta_path = DATA_DIR / "prices_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            rate = float(meta["usd_sek"])
            logger.info("USD/SEK from cache: %.4f", rate)
            return rate
        except Exception:
            pass

    # Fall back to live yfinance
    try:
        import yfinance as yf
        hist = yf.Ticker("USDSEK=X").history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.warning("FX fetch failed: %s — using fallback %.2f", exc, _FX_FALLBACK_USDSEK)

    return _FX_FALLBACK_USDSEK


def _load_local_cache(
    lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (closes, volumes) from committed CSV files, or None if unavailable."""
    closes_path  = DATA_DIR / "prices_close.csv"
    volumes_path = DATA_DIR / "prices_volume.csv"

    if not closes_path.exists() or not volumes_path.exists():
        return None

    try:
        closes  = pd.read_csv(closes_path,  index_col=0, parse_dates=True)
        volumes = pd.read_csv(volumes_path, index_col=0, parse_dates=True)
    except Exception as exc:
        logger.warning("Failed to read local cache: %s", exc)
        return None

    # Trim to the requested lookback window (keep some extra buffer)
    cutoff = datetime.today() - timedelta(days=int(lookback_days * 1.6))
    closes  = closes[closes.index  >= cutoff]
    volumes = volumes[volumes.index >= cutoff]

    if closes.empty:
        logger.warning("Local cache is empty after trimming to lookback window")
        return None

    meta_path = DATA_DIR / "prices_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            logger.info(
                "Local cache: %d tickers, %d days, updated %s",
                len(closes.columns), len(closes), meta.get("updated_utc", "unknown"),
            )
        except Exception:
            pass

    return closes, volumes


def _download_yfinance(
    all_tickers: list[str],
    lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    end   = datetime.today()
    start = end - timedelta(days=int(lookback_days * 1.6))

    logger.info(
        "Downloading %d tickers via yfinance (%s → %s)",
        len(all_tickers), start.date(), end.date(),
    )

    raw = yf.download(
        all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned empty dataset — check network/API.")

    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"], raw["Volume"]

    ticker = all_tickers[0]
    return (
        raw[["Close"]].rename(columns={"Close": ticker}),
        raw[["Volume"]].rename(columns={"Volume": ticker}),
    )


class UniverseFilter:
    def __init__(
        self,
        se_tickers: list[str],
        us_tickers: list[str],
        min_adv_sek: float = 10_000_000.0,
        max_gap_pct: float = 0.05,
        lookback_days: int = 273,
    ):
        self.se_tickers   = se_tickers
        self.us_tickers   = us_tickers
        self.min_adv_sek  = min_adv_sek
        self.max_gap_pct  = max_gap_pct
        self.lookback_days = lookback_days

    def fetch_and_filter(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          prices      — adjusted close (rows=dates, cols=tickers)
          log_returns — daily log-returns (rows=dates, cols=tickers)
        """
        all_tickers = self.se_tickers + self.us_tickers

        # ── 1. Load price data ────────────────────────────────────────────────
        cached = _load_local_cache(self.lookback_days)
        if cached is not None:
            closes, volumes = cached
            logger.info("Using local cache (GitHub Actions data)")
        else:
            logger.info("Local cache not found — falling back to yfinance")
            closes, volumes = _download_yfinance(all_tickers, self.lookback_days)

        usd_sek = get_usd_sek_rate()

        # ── 2. Quality filters ────────────────────────────────────────────────
        passed: list[str] = []

        for ticker in all_tickers:
            if ticker not in closes.columns:
                logger.info("Drop %s: not in dataset", ticker)
                continue

            close = closes[ticker].dropna()
            vol   = volumes[ticker].dropna() if ticker in volumes.columns else pd.Series(dtype=float)

            if close.empty:
                logger.info("Drop %s: empty price series", ticker)
                continue

            gap = max(0, self.lookback_days - len(close)) / self.lookback_days
            if gap > self.max_gap_pct:
                logger.info("Drop %s: data gap %.1f%% > %.1f%%", ticker, gap * 100, self.max_gap_pct * 100)
                continue

            if not vol.empty:
                adv_20      = vol.tail(20).mean()
                price_local = close.tail(20).mean()
                is_se       = ticker.endswith(".ST")
                adv_sek     = adv_20 * price_local if is_se else adv_20 * price_local * usd_sek

                if adv_sek < self.min_adv_sek:
                    logger.info(
                        "Drop %s: ADV %.0f SEK < %.0f SEK minimum",
                        ticker, adv_sek, self.min_adv_sek,
                    )
                    continue

            passed.append(ticker)

        logger.info("Universe: %d / %d tickers passed filters", len(passed), len(all_tickers))

        if not passed:
            raise RuntimeError("All tickers dropped by universe filters.")

        prices      = closes[passed].dropna(how="all")
        log_returns = np.log(prices / prices.shift(1)).dropna(how="all")

        return prices, log_returns
