"""
IRON-REBALANCER — CrewAI Research Team
========================================
Five LLM agents (Claude) that discuss and challenge each other's findings.

Architecture:
  - Every quantitative decision is made by deterministic Python tools
  - LLMs reason about tool outputs, flag concerns, and pass context forward
  - Sequential process: each agent's output becomes the next agent's context

Run:
  uv run python main.py --crew

Requires:
  ANTHROPIC_API_KEY in .env
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import yaml
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
CONFIG_PATH = ROOT / "config" / "config.yaml"
PORTFOLIO_PATH = ROOT / "config" / "portfolio.json"

logger = logging.getLogger(__name__)


# ── Shared pipeline state ─────────────────────────────────────────────────────
# Tools populate this as they run; each agent reads from it via its tools.

@dataclass
class _PipelineState:
    cfg: dict = field(default_factory=dict)
    bundle: object = None               # DataBundle
    candidates: object = None           # pd.DataFrame
    coverage: object = None             # DataCoverageReport
    weight_result: object = None        # WeightResult
    report_path: str = ""
    signals: list = field(default_factory=list)
    portfolio: object = None            # PortfolioState
    usd_sek: float = 10.5

_S = _PipelineState()


def _load_cfg() -> dict:
    if not _S.cfg:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            _S.cfg = yaml.safe_load(f)
    return _S.cfg


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool("Fetch and validate universe data")
def tool_fetch_data(note: str = "") -> str:
    """
    Downloads OHLCV data for all universe tickers, applies ADV and data-gap
    filters, computes RSI / MA / rolling vol indicators.
    Returns a quality report string.
    """
    from agents.data_architect import DataArchitect
    from core.universe import get_usd_sek_rate

    cfg = _load_cfg()
    risk = cfg["risk"]
    uni  = cfg["universe"]

    agent = DataArchitect(
        se_tickers=uni["se_tickers"],
        us_tickers=uni["us_tickers"],
        min_adv_sek=risk["min_adv_sek"],
        max_gap_pct=risk["max_data_gap_pct"],
        lookback_days=risk["lookback_days"] + 30,
    )
    bundle = agent.run()
    _S.bundle = bundle
    _S.usd_sek = get_usd_sek_rate()

    # Summary stats for the LLM to reason about
    vols = bundle.log_returns.std() * (252 ** 0.5)
    high_vol = vols[vols > 0.35].sort_values(ascending=False).head(5)
    low_liq  = []   # already filtered out by UniverseFilter

    lines = [
        f"Universe: {len(bundle.tickers)} tickers passed quality filters",
        f"Date range: {bundle.log_returns.index[0].date()} → {bundle.log_returns.index[-1].date()}",
        f"Trading days available: {len(bundle.log_returns)}",
        "",
        "High-volatility tickers (annualised vol > 35%):",
    ]
    if high_vol.empty:
        lines.append("  None — universe is within normal vol range")
    else:
        for t, v in high_vol.items():
            lines.append(f"  {t}: {v:.1%}")

    lines += [
        "",
        "SE tickers: " + ", ".join(t for t in bundle.tickers if t.endswith(".ST")),
        "US tickers: " + ", ".join(t for t in bundle.tickers if not t.endswith(".ST")),
    ]
    return "\n".join(lines)


@tool("Score and rank alpha candidates")
def tool_score_candidates(focus_note: str = "") -> str:
    """
    Runs Piotroski F-Score (>= 7), ROIC (> 15%), and skip-1-month momentum
    on all universe tickers. Returns ranked candidates with Z-scores and a
    data coverage breakdown.
    """
    from agents.quant_researcher import QuantResearcher
    from core.factors import DataCoverageReport

    if _S.bundle is None:
        return "ERROR: DataBundle not available. Run tool_fetch_data first."

    cfg = _load_cfg()
    alpha = cfg["alpha"]
    sector_map: dict = cfg.get("sector_map", {})

    coverage = DataCoverageReport()
    researcher = QuantResearcher(
        f_score_min=alpha["f_score_min"],
        roic_min=alpha["roic_3yr_min"],
        top_n=cfg["risk"]["top_n_candidates"],
    )
    candidates = researcher.run(_S.bundle, sector_map, coverage=coverage)
    _S.candidates = candidates
    _S.coverage = coverage

    if candidates is None or candidates.empty:
        return (
            "WARNING: No candidates survived alpha filters.\n"
            + coverage.to_markdown()
        )

    lines = [
        f"Candidates passed all filters: {len(candidates)}",
        "",
        "Coverage breakdown:",
        f"  Total universe      : {coverage.total}",
        f"  Passed F-Score      : {coverage.passed_f_score}",
        f"  Passed ROIC         : {coverage.passed_roic}",
        f"  Passed momentum     : {coverage.passed_momentum}",
        f"  No fundamentals     : {len(coverage.failed_no_fundamentals)} tickers",
        f"  F-Score < 5/9 data  : {len(coverage.failed_f_score_coverage)} tickers",
        f"  F-Score below thr.  : {len(coverage.failed_f_score_filter)} tickers",
        f"  ROIC data gap       : {len(coverage.failed_roic_coverage)} tickers",
        f"  ROIC below thr.     : {len(coverage.failed_roic_filter)} tickers",
        "",
        "Top 10 candidates by composite Z-score:",
    ]
    for ticker, row in candidates.head(10).iterrows():
        lines.append(
            f"  {ticker:<14}  z={row['z_score']:+.2f}  "
            f"F={int(row['f_score'])}  ROIC={row['roic']:.1%}  "
            f"Mom={row['momentum']:+.1%}  ({row['sector']})"
        )

    if coverage.failed_no_fundamentals:
        lines += [
            "",
            f"Tickers with NO fundamentals (yfinance gap — excluded, not failed):",
            "  " + ", ".join(coverage.failed_no_fundamentals),
        ]

    # Sector distribution of candidates
    sector_dist = candidates["sector"].value_counts()
    lines += ["", "Sector distribution of candidates:"]
    for sec, count in sector_dist.items():
        lines.append(f"  {sec}: {count}")

    return "\n".join(lines)


@tool("Apply risk constraints and compute target weights")
def tool_apply_risk(observations: str = "") -> str:
    """
    Runs correlation de-duplication, inverse-vol sizing, CVXPY constraint
    optimisation (8% max weight, 25% sector cap, 15% vol target), and
    trailing-stop enforcement. Returns weight table and risk metrics.
    """
    from agents.risk_supervisor import RiskInput, RiskSupervisor
    from core.portfolio import PortfolioState, load_portfolio
    from core.risk import RiskConfig

    if _S.candidates is None or _S.candidates.empty:
        return "ERROR: No candidates available. Run tool_score_candidates first."

    cfg = _load_cfg()
    risk_raw = cfg["risk"]
    sector_map: dict = cfg.get("sector_map", {})

    risk_config = RiskConfig(
        max_weight=risk_raw["max_single_weight"],
        sector_cap=risk_raw["sector_cap"],
        vol_target=risk_raw["vol_target"],
        max_correlation=risk_raw["max_correlation"],
        kelly_max_fraction=risk_raw["kelly_max_fraction"],
        trailing_stop_pct=risk_raw["trailing_stop_pct"],
        lookback_days=risk_raw["lookback_days"],
    )

    if PORTFOLIO_PATH.exists():
        portfolio = load_portfolio(PORTFOLIO_PATH)
        portfolio.update_current_prices(_S.bundle.latest_prices(), usd_sek=_S.usd_sek)
    else:
        portfolio = PortfolioState(
            updated=str(date.today()), currency="SEK",
            total_value_sek=500_000, cash_sek=500_000, holdings=[],
        )
    _S.portfolio = portfolio

    agent = RiskSupervisor(config=risk_config)
    wr = agent.run(RiskInput(
        candidates=_S.candidates,
        returns=_S.bundle.log_returns,
        portfolio=portfolio,
        sector_map=sector_map,
    ))
    _S.weight_result = wr

    lines = [
        f"Final positions: {len(wr.weights)}",
        f"Realised portfolio vol: {wr.portfolio_vol:.1%}  (target: {risk_config.vol_target:.0%})",
        "",
        "Correlation-dropped tickers:",
        "  " + (", ".join(wr.dropped_by_correlation) or "None"),
        "Trailing stops triggered:",
        "  " + (", ".join(wr.trailing_stop_triggered) or "None"),
        "",
        "Target weight table:",
        f"  {'Ticker':<14} {'Weight':>7}  Sector",
    ]
    for ticker, w in wr.weights.sort_values(ascending=False).items():
        sec = sector_map.get(ticker, "Unknown")
        lines.append(f"  {ticker:<14} {w:>6.2%}  {sec}")

    lines += ["", "Sector exposures:"]
    for sec, exp in sorted(wr.sector_exposures.items(), key=lambda x: -x[1]):
        lines.append(f"  {sec}: {exp:.1%}")

    return "\n".join(lines)


@tool("Run stress test and generate vulnerability report")
def tool_stress_test(risk_concerns: str = "") -> str:
    """
    Stress-tests portfolio against a 20% market drop using proxy betas.
    Computes historical max drawdown and daily CVaR (95%).
    Writes Markdown vulnerability report to output/.
    Returns risk summary.
    """
    from agents.auditor_agent import AuditorAgent

    if _S.weight_result is None:
        return "ERROR: Weight result not available. Run tool_apply_risk first."

    OUTPUT_DIR.mkdir(exist_ok=True)
    auditor = AuditorAgent(market_drop_pct=0.20)
    report_path = auditor.run(
        weights=_S.weight_result.weights,
        returns=_S.bundle.log_returns,
        candidates=_S.candidates,
        output_dir=OUTPUT_DIR,
    )
    _S.report_path = report_path

    # Read the report back for the LLM to reason about
    try:
        report_text = Path(report_path).read_text(encoding="utf-8")
    except Exception:
        report_text = f"Report written to {report_path}"

    return report_text


@tool("Generate final BUY/SELL/TRIM/HOLD action signals")
def tool_generate_signals(deployment_recommendation: str = "") -> str:
    """
    Diffs current portfolio.json against target weights and emits a typed
    action table. Writes signals Markdown to output/.
    Returns the action table as a string.
    """
    from agents.signal_commander import SignalCommander

    if _S.weight_result is None or _S.portfolio is None:
        return "ERROR: Weight result or portfolio not available."

    OUTPUT_DIR.mkdir(exist_ok=True)
    commander = SignalCommander()
    signals = commander.run(
        target_weights=_S.weight_result.weights,
        portfolio=_S.portfolio,
        candidates=_S.candidates,
        trailing_stop_tickers=_S.weight_result.trailing_stop_triggered,
        output_dir=OUTPUT_DIR,
    )
    _S.signals = signals
    _S.portfolio.save(PORTFOLIO_PATH)

    buys  = [s for s in signals if s.action == "BUY"]
    sells = [s for s in signals if s.action == "SELL"]
    trims = [s for s in signals if s.action == "TRIM"]
    holds = [s for s in signals if s.action == "HOLD"]

    lines = [
        f"Action summary: BUY={len(buys)}  SELL={len(sells)}  TRIM={len(trims)}  HOLD={len(holds)}",
        "",
        f"{'Ticker':<14} {'Action':<6} {'Cur Wt':>7} {'Tgt Wt':>7} {'Delta SEK':>12}  Sector",
    ]
    for s in signals:
        stop = " [STOP]" if s.trailing_stop else ""
        lines.append(
            f"{s.ticker + stop:<14} {s.action:<6} {s.current_weight:>6.2%} "
            f"{s.target_weight:>6.2%} {s.delta_value_sek:>+12,.0f}  {s.sector}"
        )

    return "\n".join(lines)


# ── Agent definitions ─────────────────────────────────────────────────────────

def build_crew(verbose: bool = True) -> Crew:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")

    llm = LLM(model="anthropic/claude-sonnet-4-6", temperature=0.2)

    data_architect = Agent(
        role="Data Architect",
        goal=(
            "Fetch clean, validated OHLCV data for all universe tickers. "
            "Flag any data quality issues — high volatility, missing tickers, "
            "liquidity concerns — so the research team is aware before scoring."
        ),
        backstory=(
            "You are a quantitative data engineer with deep knowledge of Swedish "
            "OMXS and US equity market microstructure. You have seen data pipelines "
            "fail catastrophically from survivorship bias and stale prices. "
            "You are rigorous, terse, and always flag anything unusual."
        ),
        tools=[tool_fetch_data],
        llm=llm,
        verbose=verbose,
    )

    quant_researcher = Agent(
        role="Quant Researcher",
        goal=(
            "Score the universe using Piotroski F-Score, ROIC, and skip-1-month "
            "momentum. Identify the top candidates and clearly explain why tickers "
            "failed. Flag sector concentration risks for the Risk Supervisor."
        ),
        backstory=(
            "You are a fundamental and quantitative analyst. You know that yfinance "
            "fundamental data is incomplete for many Swedish stocks — you distinguish "
            "between 'failed the screen' and 'had no data'. You are honest about "
            "data limitations and never overstate conviction."
        ),
        tools=[tool_score_candidates],
        llm=llm,
        verbose=verbose,
    )

    risk_supervisor = Agent(
        role="Risk Supervisor",
        goal=(
            "Apply correlation de-duplication, position sizing, and hard constraints "
            "(8% max weight, 25% sector cap, 15% vol target). Explain every dropped "
            "ticker and every binding constraint. Flag residual risks for the Auditor."
        ),
        backstory=(
            "You are a portfolio risk manager who has lived through 2008, 2020, and "
            "the 2022 rate shock. You are paranoid about hidden correlation during "
            "stress events and always ask: what happens when everyone sells at once? "
            "You enforce rules without exception."
        ),
        tools=[tool_apply_risk],
        llm=llm,
        verbose=verbose,
    )

    auditor_agent = Agent(
        role="Risk Auditor",
        goal=(
            "Stress-test the proposed portfolio against a 20% market drop. "
            "Compute historical max drawdown and CVaR. Write a vulnerability report. "
            "Give the Signal Commander a clear GO / REVIEW NEEDED verdict with reasoning."
        ),
        backstory=(
            "You are an independent risk auditor. Your job is to find holes in the "
            "portfolio construction before paper capital is deployed. You have no "
            "allegiance to the Quant Researcher's picks — you only care about "
            "tail risk and drawdown. You are direct and unapologetic."
        ),
        tools=[tool_stress_test],
        llm=llm,
        verbose=verbose,
    )

    signal_commander = Agent(
        role="Signal Commander",
        goal=(
            "Translate the target weights into a precise BUY/SELL/TRIM/HOLD action "
            "table. Incorporate all concerns raised by the team. Issue a final "
            "deployment verdict and any tranche or sequencing recommendations."
        ),
        backstory=(
            "You are the execution strategist. You read every note from the research "
            "team and translate it into clean, actionable signals. You remember that "
            "this is a paper trading model — no real money is at stake — but you "
            "treat it as if it were real to build discipline."
        ),
        tools=[tool_generate_signals],
        llm=llm,
        verbose=verbose,
    )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    task_data = Task(
        description=(
            "Fetch and validate all universe data. "
            "Report on data quality, any tickers with unusual volatility, "
            "and confirm the universe is clean for scoring. "
            "Note anything the Quant Researcher should be aware of."
        ),
        expected_output=(
            "A data quality report covering: tickers fetched, date range, "
            "any high-volatility or data-gap flags, and a summary statement "
            "confirming the universe is ready for alpha screening."
        ),
        agent=data_architect,
    )

    task_research = Task(
        description=(
            "Score the full universe using Piotroski F-Score, ROIC, and momentum. "
            "Explain which tickers passed, which failed, and which had data gaps. "
            "Flag sector concentration concerns to the Risk Supervisor. "
            "Context from Data Architect: {task_data_output}"
        ),
        expected_output=(
            "A ranked list of top candidates with Z-scores, F-Score, ROIC, and momentum. "
            "A breakdown of why tickers failed (filter vs data gap). "
            "Any sector concentration warnings for the Risk Supervisor."
        ),
        agent=quant_researcher,
        context=[task_data],
    )

    task_risk = Task(
        description=(
            "Apply all risk constraints to the candidate list. "
            "Explain every correlation drop and why it makes intuitive sense. "
            "Show the final weight table and sector exposures. "
            "Flag anything unusual for the Auditor. "
            "Context from Quant Researcher: {task_research_output}"
        ),
        expected_output=(
            "Final weight table with positions, weights, and sectors. "
            "List of tickers dropped by correlation with explanation. "
            "Portfolio vol vs target. Sector exposure summary. "
            "Risk flags passed to the Auditor."
        ),
        agent=risk_supervisor,
        context=[task_data, task_research],
    )

    task_audit = Task(
        description=(
            "Stress-test the proposed portfolio. "
            "Compute the loss under a 20% market drop, historical max drawdown, "
            "and daily CVaR at 95%. "
            "Put the numbers in context — is this concerning for a paper portfolio? "
            "Issue a GO or REVIEW NEEDED verdict with clear reasoning. "
            "Context from Risk Supervisor: {task_risk_output}"
        ),
        expected_output=(
            "Stress loss estimate, max drawdown, CVaR figures. "
            "Contextualised assessment of each risk metric. "
            "Clear GO / REVIEW NEEDED verdict with reasoning. "
            "Any recommendations for the Signal Commander (e.g. tranche deployment)."
        ),
        agent=auditor_agent,
        context=[task_data, task_research, task_risk],
    )

    task_signals = Task(
        description=(
            "Generate the final BUY/SELL/TRIM/HOLD action table. "
            "Incorporate all concerns raised by the team. "
            "Issue the final weekly deployment verdict. "
            "Remember: this is a PAPER TRADING model — no real money. "
            "Context from Auditor: {task_audit_output}"
        ),
        expected_output=(
            "Complete action table with ticker, action, current weight, target weight, "
            "and delta SEK for each position. "
            "Any tranche or sequencing recommendations. "
            "One-line final verdict: DEPLOY (paper) or REVIEW NEEDED."
        ),
        agent=signal_commander,
        context=[task_data, task_research, task_risk, task_audit],
    )

    return Crew(
        agents=[data_architect, quant_researcher, risk_supervisor, auditor_agent, signal_commander],
        tasks=[task_data, task_research, task_risk, task_audit, task_signals],
        process=Process.sequential,
        verbose=verbose,
    )


def run_crew(verbose: bool = True) -> str:
    """Entry point called from main.py --crew flag."""
    crew = build_crew(verbose=verbose)
    result = crew.kickoff()
    return str(result)
