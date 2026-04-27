"""
Portfolio Import CLI — Feature B
==================================
Populates portfolio.json from an Avanza CSV export or interactive prompts.

Avanza CSV export (Konto → Export → CSV) has these columns:
  Värdepapper, ISIN, Antal, Kurs, Marknadsvärde, Anskaffningsvärde

Usage:
  uv run python tools/import_portfolio.py --csv avanza_export.csv
  uv run python tools/import_portfolio.py --interactive
  uv run python tools/import_portfolio.py --show
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent.parent
PORTFOLIO_PATH = ROOT / "config" / "portfolio.json"
CONFIG_PATH = ROOT / "config" / "config.yaml"


def _load_sector_map() -> dict[str, str]:
    import yaml
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f).get("sector_map", {})


def _fetch_price_sek(ticker: str, usd_sek: float) -> float:
    try:
        hist = yf.Ticker(ticker).history(period="2d")
        if hist.empty:
            return 0.0
        price = float(hist["Close"].iloc[-1])
        return price if ticker.endswith(".ST") else price * usd_sek
    except Exception:
        return 0.0


def _get_usd_sek() -> float:
    try:
        hist = yf.Ticker("USDSEK=X").history(period="5d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else 10.5
    except Exception:
        return 10.5


def _build_holding(ticker: str, shares: float, avg_cost_sek: float, usd_sek: float, sector_map: dict) -> dict:
    price = _fetch_price_sek(ticker, usd_sek)
    return {
        "ticker": ticker.upper(),
        "shares": round(shares, 4),
        "avg_cost_sek": round(avg_cost_sek, 4),
        "peak_price_sek": round(max(price, avg_cost_sek), 4),
        "current_price_sek": round(price, 4),
        "sector": sector_map.get(ticker.upper(), "Unknown"),
    }


# ── Avanza CSV import ─────────────────────────────────────────────────────────

# Avanza CSV column name → normalized (handles Swedish locale)
_AVANZA_COL_MAP = {
    "värdepapper": "name",
    "isin": "isin",
    "antal": "shares",
    "kurs": "price_local",
    "marknadsvärde": "market_value",
    "anskaffningsvärde": "cost_basis",
}

# ISIN prefix → Yahoo Finance suffix
_ISIN_SUFFIX = {
    "SE": ".ST",  # Sweden
    "US": "",     # US — no suffix
    "GB": ".L",   # London (not traded on Avanza normally, but defensive)
}


def _isin_to_ticker_hint(isin: str) -> str:
    return _ISIN_SUFFIX.get(isin[:2].upper(), "")


def from_avanza_csv(csv_path: Path, cash_sek: float) -> dict:
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    col_map = {k: v for k, v in _AVANZA_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=col_map)

    for col in ("shares", "price_local", "cost_basis"):
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("\xa0", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

    sector_map = _load_sector_map()
    usd_sek = _get_usd_sek()
    holdings = []

    for _, row in df.iterrows():
        isin = str(row.get("isin", "")).strip()
        suffix = _isin_to_ticker_hint(isin)
        name_raw = str(row.get("name", "")).strip()

        ticker = input(
            f"  Ticker for '{name_raw}' (ISIN={isin}, suffix hint='{suffix}'): "
        ).strip().upper()
        if not ticker:
            print(f"  Skipping {name_raw}")
            continue

        shares = float(row.get("shares", 0) or 0)
        cost_basis = float(row.get("cost_basis", 0) or 0)
        avg_cost = cost_basis / shares if shares > 0 else 0.0

        holdings.append(_build_holding(ticker, shares, avg_cost, usd_sek, sector_map))
        print(f"  Added {ticker}: {shares:.2f} shares @ {avg_cost:.2f} SEK avg cost")

    invested = sum(h["shares"] * h["current_price_sek"] for h in holdings)
    return {
        "updated": str(date.today()),
        "currency": "SEK",
        "total_value_sek": round(invested + cash_sek, 2),
        "cash_sek": round(cash_sek, 2),
        "holdings": holdings,
    }


# ── Interactive mode ──────────────────────────────────────────────────────────

def from_interactive() -> dict:
    sector_map = _load_sector_map()
    usd_sek = _get_usd_sek()
    print(f"\nUSD/SEK rate: {usd_sek:.2f}")
    print("Enter holdings one at a time. Leave ticker blank to finish.\n")

    holdings = []
    while True:
        ticker = input("Ticker (e.g. AAPL or ERIC-B.ST): ").strip().upper()
        if not ticker:
            break
        try:
            shares = float(input(f"  Shares of {ticker}: "))
            avg_cost = float(input(f"  Average cost (SEK): "))
        except ValueError:
            print("  Invalid input, skipping.")
            continue
        h = _build_holding(ticker, shares, avg_cost, usd_sek, sector_map)
        holdings.append(h)
        print(f"  Current price: {h['current_price_sek']:.2f} SEK  |  Peak set to: {h['peak_price_sek']:.2f} SEK\n")

    try:
        cash = float(input("Cash balance (SEK): "))
    except ValueError:
        cash = 0.0

    invested = sum(h["shares"] * h["current_price_sek"] for h in holdings)
    return {
        "updated": str(date.today()),
        "currency": "SEK",
        "total_value_sek": round(invested + cash, 2),
        "cash_sek": round(cash, 2),
        "holdings": holdings,
    }


# ── Show current portfolio ────────────────────────────────────────────────────

def show_portfolio() -> None:
    if not PORTFOLIO_PATH.exists():
        print("No portfolio.json found.")
        return
    data = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    holdings = data.get("holdings", [])
    print(f"\nPortfolio as of {data['updated']}  |  Total: {data['total_value_sek']:,.0f} SEK  |  Cash: {data['cash_sek']:,.0f} SEK\n")
    if not holdings:
        print("  No holdings. Run with --interactive or --csv to import.")
        return
    print(f"  {'Ticker':<14} {'Shares':>8} {'Avg Cost':>10} {'Cur Price':>10} {'Mkt Val SEK':>12} {'Drawdown':>9} {'Sector'}")
    print("  " + "-" * 85)
    for h in sorted(holdings, key=lambda x: -x["shares"] * x["current_price_sek"]):
        mv = h["shares"] * h["current_price_sek"]
        dd = (h["peak_price_sek"] - h["current_price_sek"]) / h["peak_price_sek"] if h["peak_price_sek"] else 0
        flag = " ⛔" if dd >= 0.15 else ""
        print(f"  {h['ticker']:<14} {h['shares']:>8.2f} {h['avg_cost_sek']:>10.2f} {h['current_price_sek']:>10.2f} {mv:>12,.0f} {dd:>8.1%}{flag}  {h['sector']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IRON-REBALANCER portfolio importer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", metavar="FILE", help="Avanza CSV export path")
    group.add_argument("--interactive", action="store_true")
    group.add_argument("--show", action="store_true")
    parser.add_argument("--cash", type=float, default=0.0, help="Cash balance in SEK (used with --csv)")
    args = parser.parse_args()

    if args.show:
        show_portfolio()
        return

    if args.interactive:
        data = from_interactive()
    else:
        data = from_avanza_csv(Path(args.csv), args.cash)

    backup = PORTFOLIO_PATH.with_suffix(".json.bak")
    if PORTFOLIO_PATH.exists():
        backup.write_text(PORTFOLIO_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nBacked up existing portfolio to {backup.name}")

    PORTFOLIO_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(data['holdings'])} holdings to {PORTFOLIO_PATH}")
    print(f"Total value: {data['total_value_sek']:,.0f} SEK  |  Cash: {data['cash_sek']:,.0f} SEK")


if __name__ == "__main__":
    main()
