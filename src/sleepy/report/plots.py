"""Matplotlib figures for quick inspection of a recording."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

from ..dsp.analyse import psd  # noqa: E402
from ..dsp.preprocess import Channel  # noqa: E402

BREATHING_BAND = (0.1, 0.6)
HEART_BAND = (0.8, 3.5)


def plot_overview(
    channel: Channel,
    path: Path | str,
    events: list[tuple[float, float, float]] | None = None,
    slow_channel: Channel | None = None,
    title: str = "",
) -> Path:
    """Time series + spectrum of one channel, with load events shaded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    times = channel.times - channel.times[0] if channel.times.size else channel.times
    values = channel.values
    stride = max(int(values.size // 200_000), 1)

    figure, axes = plt.subplots(3 if slow_channel is not None else 2, 1, figsize=(12, 9), constrained_layout=True)
    axis = axes[0]
    axis.plot(times[::stride], values[::stride], linewidth=0.4)
    axis.set_ylabel("raw counts")
    axis.set_title(title or f"{channel.name} - raw signal")
    axis.grid(alpha=0.3)
    for start, end, _ in events or []:
        axis.axvspan(start, end, color="orange", alpha=0.25)
    if stride > 1:
        axis.set_xlabel(f"seconds (every {stride}th sample shown)")

    axis = axes[1]
    frequencies, power = psd(channel)
    if frequencies.size:
        axis.semilogx(frequencies[1:], power[1:], linewidth=0.8)
        for name, (low, high) in (("breathing", BREATHING_BAND), ("heart", HEART_BAND)):
            axis.axvspan(low, high, alpha=0.15, label=f"{name} {low}-{high} Hz")
        axis.legend(fontsize="small")
        axis.set_xlabel("Hz")
        axis.set_ylabel("PSD")
        axis.set_title("spectrum")
        axis.grid(alpha=0.3)

    if slow_channel is not None and slow_channel.times.size:
        axis = axes[2]
        axis.plot(slow_channel.times - channel.times[0], slow_channel.values, linewidth=0.8)
        axis.set_ylabel("raw counts")
        axis.set_xlabel("seconds")
        axis.set_title(f"{slow_channel.name} (slow channel)")
        axis.grid(alpha=0.3)

    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def plot_night(
    times: np.ndarray,
    respiration_bpm: np.ndarray,
    heart_bpm: np.ndarray,
    movement: np.ndarray,
    path: Path | str,
    title: str = "night overview",
) -> Path:
    """Hypnogram style overview of a processed night."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(times, respiration_bpm, linewidth=1.0)
    axes[0].set_ylabel("breaths/min")
    axes[0].grid(alpha=0.3)
    axes[1].plot(times, heart_bpm, linewidth=1.0)
    axes[1].set_ylabel("beats/min")
    axes[1].grid(alpha=0.3)
    axes[2].plot(times, movement, linewidth=1.0)
    axes[2].set_ylabel("movement")
    axes[2].set_xlabel("seconds since start")
    axes[2].grid(alpha=0.3)
    axes[0].set_title(title)
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path
