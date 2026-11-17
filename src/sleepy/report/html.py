"""Self contained HTML reports: one page per night, plus an index page."""

from __future__ import annotations

import base64
import csv
import json
from datetime import datetime
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 60rem; color: #222; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
 table {{ border-collapse: collapse; font-size: 0.85rem; }}
 th, td {{ border: 1px solid #ddd; padding: 0.25rem 0.6rem; text-align: right; }}
 th {{ background: #f4f4f4; text-align: left; }}
 img {{ max-width: 100%; border: 1px solid #ddd; }}
 .meta {{ color: #666; font-size: 0.85rem; }}
 .none {{ color: #999; }}
</style></head><body>
<h1>{title}</h1>
<p class="meta">{meta}</p>
{body}
</body></html>
"""


def _embed_png(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="overview">'


def _table(rows: list[dict], limit: int = 200) -> str:
    if not rows:
        return '<p class="none">No epochs.</p>'
    headers = list(rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{row.get(h, '')}</td>" for h in headers) + "</tr>"
        for row in rows[:limit]
    )
    head = "".join(f"<th>{h}</th>" for h in headers)
    note = f"<p class='meta'>showing {min(len(rows), limit)} of {len(rows)} epochs</p>" if len(rows) > limit else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}"


def render_night(night_dir: Path) -> Path:
    """Write ``night_dir/index.html`` from the files ``process`` produced."""
    summary_path = night_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    metrics_path = night_dir / "metrics.csv"
    rows = list(csv.DictReader(metrics_path.open())) if metrics_path.exists() else []

    parts = []
    overview = night_dir / "overview.png"
    if overview.exists():
        parts.append(_embed_png(overview))

    channel = summary.get("channel", {})
    night_summary = summary.get("summary", {})
    facts = {
        "source": summary.get("source", ""),
        "duration": f"{summary.get('duration_s', 0) / 60:.1f} min",
        "sample rate": f"{summary.get('sample_rate_hz', 0):.2f} Hz",
        "packet loss": summary.get("packet_loss", ""),
        "sessions": channel.get("sessions", ""),
        "time without data": f"{channel.get('gap_s', 0) / 60:.1f} min",
        "clipped samples": channel.get("clipped_samples", ""),
    }
    facts.update({k: v for k, v in night_summary.items()})
    parts.append(
        "<h2>Summary</h2><table>"
        + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in facts.items())
        + "</table>"
    )
    parts.append("<h2>Epochs</h2>" + _table(rows))

    page = PAGE.format(
        title=night_dir.name,
        meta=f"generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        body="".join(parts),
    )
    target = night_dir / "index.html"
    target.write_text(page)
    return target


def render_index(nights_dir: Path) -> Path:
    """Write an index page listing every processed night."""
    entries = []
    for night in sorted(p for p in nights_dir.iterdir() if p.is_dir()):
        summary_path = night / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        night_summary = summary.get("summary", {})
        entries.append(
            f"<tr><th><a href='{night.name}/index.html'>{night.name}</a></th>"
            f"<td>{summary.get('duration_s', 0) / 60:.1f} min</td>"
            f"<td>{night_summary.get('respiration_bpm_median', '-')}</td>"
            f"<td>{night_summary.get('heart_bpm_median', '-')}</td>"
            f"<td>{night_summary.get('epochs_present', '-')}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>night</th><th>duration</th>"
        "<th>breathing/min</th><th>heart/min</th><th>epochs in bed</th></tr></thead>"
        f"<tbody>{''.join(entries)}</tbody></table>"
        if entries
        else '<p class="none">No processed nights yet.</p>'
    )
    target = nights_dir / "index.html"
    target.write_text(
        PAGE.format(
            title="sleepy - nights",
            meta=f"generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            body=f"<h2>Nights</h2>{table}",
        )
    )
    return target
