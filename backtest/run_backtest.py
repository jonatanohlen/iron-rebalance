"""
Backtest entry point — Feature A
Usage:
  uv run python -m backtest.run_backtest [--years 3] [--top-n 20] [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
CONFIG_PATH = ROOT / "config" / "config.yaml"


def main(args: argparse.Namespace) -> None:
    from core.universe import UniverseFilter
    from backtest.engine import BacktestConfig, BacktestEngine

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    sector_map: dict[str, str] = cfg.get("sector_map", {})
    universe_cfg = cfg["universe"]
    risk_cfg = cfg["risk"]

    end = datetime.today()
    start = end - timedelta(days=int(args.years * 365) + 90)   # extra buffer

    logger = logging.getLogger("backtest")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Downloading %d years of history …", args.years)
    filt = UniverseFilter(
        se_tickers=universe_cfg["se_tickers"],
        us_tickers=universe_cfg["us_tickers"],
        min_adv_sek=risk_cfg["min_adv_sek"],
        max_gap_pct=risk_cfg["max_data_gap_pct"],
        lookback_days=int(args.years * 252) + 90,
    )
    prices, _ = filt.fetch_and_filter()

    bt_cfg = BacktestConfig(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        top_n=args.top_n,
        max_weight=risk_cfg["max_single_weight"],
        vol_target=risk_cfg["vol_target"],
        max_correlation=risk_cfg["max_correlation"],
    )

    engine = BacktestEngine(prices=prices, config=bt_cfg, sector_map=sector_map)
    result = engine.run()

    print()
    print(result.summary())
    print()

    # Write equity curve CSV
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    curve_path = OUTPUT_DIR / f"{ts}_backtest_equity.csv"
    pd.DataFrame({
        "portfolio": result.equity_curve,
        "benchmark": result.benchmark_curve,
    }).to_csv(curve_path)

    # Write Markdown report
    report_path = OUTPUT_DIR / f"{ts}_backtest_report.md"
    steps_rows = "\n".join(
        f"| {s.date} | {len(s.weights)} | {s.port_vol:.1%} | {s.weekly_return:+.2%} | {', '.join(s.dropped_by_corr) or '—'} |"
        for s in result.steps[-12:]   # last 12 weeks in table
    )
    report = f"""# IRON-REBALANCER — Backtest Report
**Period:** {bt_cfg.start_date} → {bt_cfg.end_date}
**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

> ⚠️ Fundamental filters (F-Score / ROIC) use **current** values, not point-in-time.
> This backtest validates the **momentum + risk** components only.

{result.summary()}

## Last 12 Rebalancing Steps

| Date | Positions | Port Vol | Weekly Ret | Corr-Dropped |
|------|-----------|----------|------------|--------------|
{steps_rows}

## Equity Curve (CSV)
Saved to: `{curve_path.name}`
"""
    report_path.write_text(report, encoding="utf-8")
    logger.info("Backtest report → %s", report_path)
    logger.info("Equity curve   → %s", curve_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRON-REBALANCER Backtest")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--verbose", "-v", action="store_true")
    main(parser.parse_args())
