"""
Email Notifier — Feature C
============================
Sends the weekly briefing via SMTP after each pipeline run.

Configure in .env:
  EMAIL_SMTP_HOST=smtp.gmail.com
  EMAIL_SMTP_PORT=587
  EMAIL_USERNAME=you@gmail.com
  EMAIL_PASSWORD=your_app_password   # Gmail: generate an App Password
  EMAIL_TO=recipient@example.com     # comma-separated for multiple

For Gmail: Settings → Security → 2FA → App Passwords → generate one.
"""
from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from agents.signal_commander import Signal
from core.risk import WeightResult

logger = logging.getLogger(__name__)


class EmailNotifier:
    def __init__(
        self,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        to_addrs: list[str] | None = None,
    ):
        self.smtp_host = smtp_host or os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self.username = username or os.getenv("EMAIL_USERNAME", "")
        self.password = password or os.getenv("EMAIL_PASSWORD", "")
        raw_to = os.getenv("EMAIL_TO", "")
        self.to_addrs = to_addrs or [t.strip() for t in raw_to.split(",") if t.strip()]

    def is_configured(self) -> bool:
        return bool(self.username and self.password and self.to_addrs)

    def send_weekly_brief(
        self,
        signals: list[Signal],
        weight_result: WeightResult,
        report_path: str,
        run_date: str | None = None,
    ) -> bool:
        if not self.is_configured():
            logger.warning("Email not configured — skipping notification. Set EMAIL_* in .env")
            return False

        run_date = run_date or datetime.utcnow().strftime("%Y-%m-%d")
        subject = f"IRON-REBALANCER — Weekly Signals {run_date}"
        html = self._build_html(signals, weight_result, report_path, run_date)
        text = self._build_text(signals, weight_result, run_date)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = ", ".join(self.to_addrs)
        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.username, self.to_addrs, msg.as_string())
            logger.info("Weekly brief emailed to %s", self.to_addrs)
            return True
        except Exception as exc:
            logger.error("Email failed: %s", exc)
            return False

    def _build_text(
        self, signals: list[Signal], wr: WeightResult, run_date: str
    ) -> str:
        buys  = [s for s in signals if s.action == "BUY"]
        sells = [s for s in signals if s.action == "SELL"]
        trims = [s for s in signals if s.action == "TRIM"]
        holds = [s for s in signals if s.action == "HOLD"]

        top5 = sorted(signals, key=lambda s: -s.target_weight)[:5]

        lines = [
            f"IRON-REBALANCER — Weekly Signals {run_date}",
            "=" * 50,
            f"  BUY  : {len(buys)}   SELL: {len(sells)}   TRIM: {len(trims)}   HOLD: {len(holds)}",
            f"  Portfolio vol : {wr.portfolio_vol:.1%}",
            "",
            "Top 5 positions:",
        ]
        for s in top5:
            lines.append(f"  {s.ticker:<14} {s.target_weight:.2%}  ({s.sector})")

        if wr.trailing_stop_triggered:
            lines += ["", f"STOPS TRIGGERED: {', '.join(wr.trailing_stop_triggered)}"]
        if wr.dropped_by_correlation:
            lines += [f"Corr-dropped   : {', '.join(wr.dropped_by_correlation)}"]

        return "\n".join(lines)

    def _build_html(
        self, signals: list[Signal], wr: WeightResult, report_path: str, run_date: str
    ) -> str:
        buys  = [s for s in signals if s.action == "BUY"]
        sells = [s for s in signals if s.action == "SELL"]
        trims = [s for s in signals if s.action == "TRIM"]
        holds = [s for s in signals if s.action == "HOLD"]

        action_colors = {"BUY": "#27ae60", "SELL": "#e74c3c", "TRIM": "#f39c12", "HOLD": "#95a5a6"}
        rows = ""
        for s in sorted(signals, key=lambda x: ({"BUY":0,"SELL":1,"TRIM":2,"HOLD":3}[x.action], -x.target_weight)):
            color = action_colors[s.action]
            stop_badge = ' <span style="color:#e74c3c">⛔</span>' if s.trailing_stop else ""
            rows += (
                f"<tr>"
                f"<td><b>{s.ticker}</b>{stop_badge}</td>"
                f"<td style='color:{color};font-weight:bold'>{s.action}</td>"
                f"<td>{s.target_weight:.2%}</td>"
                f"<td>{s.delta_shares:+.2f}</td>"
                f"<td>{s.delta_value_sek:+,.0f}</td>"
                f"<td>{s.z_score:.2f}</td>"
                f"<td>{s.sector}</td>"
                f"</tr>"
            )

        stops_html = ""
        if wr.trailing_stop_triggered:
            stops_html = f'<p style="color:#e74c3c"><b>⛔ Trailing stops triggered:</b> {", ".join(wr.trailing_stop_triggered)}</p>'

        return f"""
<!DOCTYPE html><html><body style="font-family:monospace;background:#0d1117;color:#c9d1d9;padding:24px">
<h2 style="color:#58a6ff">IRON-REBALANCER — Weekly Signals</h2>
<p style="color:#8b949e">{run_date} UTC</p>

<table style="border-collapse:collapse;margin-bottom:16px">
  <tr>
    <td style="padding:8px 24px;background:#161b22;border-radius:6px;text-align:center">
      <div style="font-size:24px;color:#27ae60"><b>{len(buys)}</b></div><div>BUY</div>
    </td>
    <td style="padding:8px 24px;background:#161b22;border-radius:6px;text-align:center;margin-left:8px">
      <div style="font-size:24px;color:#e74c3c"><b>{len(sells)}</b></div><div>SELL</div>
    </td>
    <td style="padding:8px 24px;background:#161b22;border-radius:6px;text-align:center">
      <div style="font-size:24px;color:#f39c12"><b>{len(trims)}</b></div><div>TRIM</div>
    </td>
    <td style="padding:8px 24px;background:#161b22;border-radius:6px;text-align:center">
      <div style="font-size:24px;color:#8b949e"><b>{len(holds)}</b></div><div>HOLD</div>
    </td>
    <td style="padding:8px 24px;background:#161b22;border-radius:6px;text-align:center">
      <div style="font-size:24px;color:#58a6ff"><b>{wr.portfolio_vol:.1%}</b></div><div>Port Vol</div>
    </td>
  </tr>
</table>

{stops_html}

<table style="border-collapse:collapse;width:100%;background:#161b22;border-radius:6px">
  <thead>
    <tr style="color:#8b949e;border-bottom:1px solid #30363d">
      <th style="padding:8px 12px;text-align:left">Ticker</th>
      <th style="padding:8px 12px">Action</th>
      <th style="padding:8px 12px">Tgt Wt</th>
      <th style="padding:8px 12px">Delta Shs</th>
      <th style="padding:8px 12px">Delta SEK</th>
      <th style="padding:8px 12px">Z-Score</th>
      <th style="padding:8px 12px;text-align:left">Sector</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<p style="color:#8b949e;margin-top:16px;font-size:12px">
  Full vulnerability report: {report_path}
</p>
</body></html>
"""
