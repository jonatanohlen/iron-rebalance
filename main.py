"""
IRON-REBALANCER — Entry Point

Sequential deterministic pipeline:
  [1] DataArchitect     — fetch & validate universe
  [2] QuantResearcher   — alpha factor ranking (F-Score, ROIC, momentum)
  [3] RiskSupervisor    — correlation filter, position sizing, constraints
  [4] AuditorAgent      — stress test & vulnerability report
  [5] SignalCommander   — BUY / SELL / TRIM / HOLD action table

No LLM calls in the quantitative pipeline.
CrewAI is available for optional narrative wrapping via --crew flag.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Windows CP1252 consoles can't print box-drawing chars — force UTF-8 globally.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import yaml
from dotenv import load_dotenv

from agents.auditor_agent import AuditorAgent
from agents.data_architect import DataArchitect
from agents.quant_researcher import QuantResearcher
from agents.risk_supervisor import RiskInput, RiskSupervisor
from agents.signal_commander import SignalCommander
from core.factors import DataCoverageReport
from core.portfolio import PortfolioState, load_portfolio
from core.risk import RiskConfig
from core.universe import get_usd_sek_rate
from notifiers.email_notifier import EmailNotifier

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env", override=True)
OUTPUT_DIR = ROOT / "output"
CONFIG_PATH = ROOT / "config" / "config.yaml"
PORTFOLIO_PATH = ROOT / "config" / "portfolio.json"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_risk_config(raw: dict) -> RiskConfig:
    return RiskConfig(
        max_weight=raw["max_single_weight"],
        sector_cap=raw["sector_cap"],
        vol_target=raw["vol_target"],
        max_correlation=raw["max_correlation"],
        kelly_max_fraction=raw["kelly_max_fraction"],
        trailing_stop_pct=raw["trailing_stop_pct"],
        lookback_days=raw["lookback_days"],
        min_adv_sek=raw["min_adv_sek"],
        max_data_gap_pct=raw["max_data_gap_pct"],
        top_n_candidates=raw["top_n_candidates"],
    )


def main(args: argparse.Namespace) -> None:
    logger = logging.getLogger("iron-rebalancer")
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.crew:
        from crew import run_crew
        print(run_crew(verbose=args.verbose))
        return

    cfg = load_config(CONFIG_PATH)
    risk_config = build_risk_config(cfg["risk"])
    alpha_cfg = cfg["alpha"]
    sector_map: dict[str, str] = cfg.get("sector_map", {})
    universe_cfg = cfg["universe"]

    logger.info("════════════════════════════════════════")
    logger.info("  IRON-REBALANCER  v%s", cfg["system"]["version"])
    logger.info("════════════════════════════════════════")

    # ── [1] DataArchitect ─────────────────────────────────────────────────────
    logger.info("[1/5] DataArchitect — fetching universe …")
    data_agent = DataArchitect(
        se_tickers=universe_cfg["se_tickers"],
        us_tickers=universe_cfg["us_tickers"],
        min_adv_sek=risk_config.min_adv_sek,
        max_gap_pct=risk_config.max_data_gap_pct,
        lookback_days=risk_config.lookback_days + 30,
    )
    bundle = data_agent.run()

    # ── [2] QuantResearcher ───────────────────────────────────────────────────
    logger.info("[2/5] QuantResearcher — scoring & ranking …")
    researcher = QuantResearcher(
        f_score_min=alpha_cfg["f_score_min"],
        roic_min=alpha_cfg["roic_3yr_min"],
        top_n=risk_config.top_n_candidates,
    )
    coverage = DataCoverageReport()
    candidates = researcher.run(bundle, sector_map, coverage=coverage)

    if candidates.empty:
        logger.error("No candidates survived alpha filters — aborting.")
        logger.error("Coverage report:\n%s", coverage.to_markdown())
        sys.exit(1)

    # ── [3] RiskSupervisor ────────────────────────────────────────────────────
    logger.info("[3/5] RiskSupervisor — applying risk constraints …")

    portfolio: PortfolioState | None = None
    if PORTFOLIO_PATH.exists():
        portfolio = load_portfolio(PORTFOLIO_PATH)
        usd_sek = get_usd_sek_rate()
        portfolio.update_current_prices(bundle.latest_prices(), usd_sek=usd_sek)
    else:
        logger.warning("portfolio.json not found — treating portfolio as empty (all-cash).")
        portfolio = PortfolioState(
            updated="2026-04-27",
            currency="SEK",
            total_value_sek=500_000,
            cash_sek=500_000,
            holdings=[],
        )

    risk_agent = RiskSupervisor(config=risk_config, use_kelly=args.kelly)
    weight_result = risk_agent.run(
        RiskInput(
            candidates=candidates,
            returns=bundle.log_returns,
            portfolio=portfolio,
            sector_map=sector_map,
        )
    )

    # ── [4] AuditorAgent ──────────────────────────────────────────────────────
    logger.info("[4/5] AuditorAgent — stress testing …")
    auditor = AuditorAgent(market_drop_pct=0.20)
    report_path = auditor.run(
        weights=weight_result.weights,
        returns=bundle.log_returns,
        candidates=candidates,
        output_dir=OUTPUT_DIR,
    )

    # ── [5] SignalCommander ───────────────────────────────────────────────────
    logger.info("[5/5] SignalCommander — generating action signals …")
    commander = SignalCommander()
    signals = commander.run(
        target_weights=weight_result.weights,
        portfolio=portfolio,
        candidates=candidates,
        trailing_stop_tickers=weight_result.trailing_stop_triggered,
        output_dir=OUTPUT_DIR,
    )

    # Persist updated peak prices
    portfolio.save(PORTFOLIO_PATH)

    # ── Email notification (Feature C) ────────────────────────────────────────
    if args.notify:
        notifier = EmailNotifier()
        notifier.send_weekly_brief(
            signals=signals,
            weight_result=weight_result,
            report_path=report_path,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    buys  = sum(1 for s in signals if s.action == "BUY")
    sells = sum(1 for s in signals if s.action == "SELL")
    trims = sum(1 for s in signals if s.action == "TRIM")
    holds = sum(1 for s in signals if s.action == "HOLD")

    print()
    print("═" * 56)
    print("  IRON-REBALANCER — WEEKLY SIGNAL SUMMARY")
    print("═" * 56)
    print(f"  BUY   : {buys:>3} positions")
    print(f"  SELL  : {sells:>3} positions  (incl. {len(weight_result.trailing_stop_triggered)} stops)")
    print(f"  TRIM  : {trims:>3} positions")
    print(f"  HOLD  : {holds:>3} positions")
    print()
    print(f"  Portfolio vol target : {risk_config.vol_target:.0%}")
    print(f"  Realised portfolio vol: {weight_result.portfolio_vol:.1%}")
    print(f"  Corr-dropped         : {weight_result.dropped_by_correlation}")
    print(f"  Trailing stops hit   : {weight_result.trailing_stop_triggered}")
    print()
    print(f"  Vulnerability report : {report_path}")
    print(f"  Action table         : output/<timestamp>_signals.md")
    print("═" * 56)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRON-REBALANCER weekly signal generator")
    parser.add_argument("--kelly", action="store_true", help="Use Kelly sizing instead of Inverse-Vol")
    parser.add_argument("--notify", action="store_true", help="Send email notification (configure EMAIL_* in .env)")
    parser.add_argument("--crew", action="store_true", help="Run CrewAI research team (requires ANTHROPIC_API_KEY in .env)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parsed = parser.parse_args()
    setup_logging(parsed.verbose)
    main(parsed)
