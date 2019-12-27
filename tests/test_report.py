"""HTML report generation."""

import json
from pathlib import Path

import numpy as np

from sleepy.report import html
from sleepy.report.plots import plot_night


def _make_night(root: Path) -> Path:
    night = root / "2026-01-01_test"
    night.mkdir(parents=True)
    (night / "summary.json").write_text(
        json.dumps(
            {
                "source": "raw/test.bin",
                "sample_rate_hz": 139.95,
                "duration_s": 3600.0,
                "packet_loss": "0.10%",
                "channel": {"clipped_samples": 12},
                "summary": {
                    "epochs": 120,
                    "epochs_present": 100,
                    "respiration_bpm_median": 14.5,
                    "heart_bpm_median": 58.0,
                },
            }
        )
    )
    (night / "metrics.csv").write_text(
        "start_s,end_s,respiration_bpm\n0.0,30.0,14.5\n30.0,60.0,15.0\n"
    )
    plot_night(
        np.array([0.0, 30.0]),
        np.array([14.5, 15.0]),
        np.array([58.0, 59.0]),
        np.array([100.0, 120.0]),
        night / "overview.png",
    )
    return night


def test_night_page_embeds_plot_and_summary(tmp_path: Path):
    night = _make_night(tmp_path)
    page = html.render_night(night)
    content = page.read_text()
    assert page.exists()
    assert "data:image/png;base64," in content  # plot embedded, no external files
    assert "14.5" in content
    assert "raw/test.bin" in content


def test_index_lists_nights(tmp_path: Path):
    nights = tmp_path / "nights"
    nights.mkdir()
    night = _make_night(nights)
    index = html.render_index(nights)
    content = index.read_text()
    assert f"{night.name}/index.html" in content
    assert "breathing/min" in content
