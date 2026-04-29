"""
Fetches OHLCV data for all universe tickers and saves to data/.

Run by GitHub Actions daily after market close, or manually:
    python tools/fetch_market_data.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    se_tickers: list[str] = cfg["universe"]["se_tickers"]
    us_tickers: list[str] = cfg["universe"]["us_tickers"]
    all_tickers = se_tickers + us_tickers

    lookback_days: int = cfg["risk"]["lookback_days"]
    end = datetime.today()
    # Extra buffer so momentum (12m skip-1m) always has enough history
    start = end - timedelta(days=int(lookback_days * 1.6))

    logger.info("Downloading %d tickers %s → %s", len(all_tickers), start.date(), end.date())

    raw = yf.download(
        all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        logger.error("yfinance returned empty dataset — aborting")
        sys.exit(1)

    closes: pd.DataFrame = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    volumes: pd.DataFrame = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]

    fetched = [t for t in all_tickers if t in closes.columns and closes[t].notna().any()]
    failed  = [t for t in all_tickers if t not in fetched]

    if failed:
        logger.warning("Failed to fetch %d tickers: %s", len(failed), failed)

    closes  = closes[fetched]
    volumes = volumes[fetched]

    closes.to_csv(DATA_DIR / "prices_close.csv")
    volumes.to_csv(DATA_DIR / "prices_volume.csv")
    logger.info("Saved prices_close.csv and prices_volume.csv (%d tickers, %d days)", len(fetched), len(closes))

    # USD/SEK rate
    usd_sek = 10.5
    try:
        fx = yf.Ticker("USDSEK=X").history(period="5d")
        if not fx.empty:
            usd_sek = float(fx["Close"].iloc[-1])
            logger.info("USD/SEK rate: %.4f", usd_sek)
    except Exception as exc:
        logger.warning("FX fetch failed: %s — using fallback %.2f", exc, usd_sek)

    meta = {
        "updated_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "tickers_fetched": fetched,
        "tickers_failed": failed,
        "usd_sek": usd_sek,
        "date_from": str(closes.index[0].date()),
        "date_to": str(closes.index[-1].date()),
        "trading_days": len(closes),
    }
    (DATA_DIR / "prices_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("Saved prices_meta.json — done.")


if __name__ == "__main__":
    main()
