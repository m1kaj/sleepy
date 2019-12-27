"""Signal analysis helpers: smoothing, load event detection, spectra."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from .preprocess import Channel


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average that ignores NaN (windows that are all NaN give NaN)."""
    array = np.asarray(values, dtype=float)
    window = max(int(window), 1)
    valid = np.isfinite(array)
    filled = np.where(valid, array, 0.0)
    kernel = np.ones(window)
    total = np.convolve(filled, kernel, mode="same")
    count = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def rolling_nanmedian(values: np.ndarray, window: int, block: int | None = None) -> np.ndarray:
    """Moving median that ignores NaN, fast enough for a whole night.

    A per sample rolling median over a 10 minute window would cost hours on an
    8 hour recording, so we first reduce the signal with a per ``block`` median,
    run the rolling median on that short series, and interpolate back to sample
    resolution. For a slowly moving baseline the result is indistinguishable
    from the exact one, and it runs in milliseconds.
    """
    array = np.asarray(values, dtype=float)
    window = max(int(window), 1)
    if array.size == 0:
        return array
    if window == 1 or array.size <= window:
        return np.full(array.size, np.nanmedian(array))

    block = block or max(window // 20, 1)
    n_blocks = array.size // block
    if n_blocks < 8:  # short input: do it exactly
        padded = np.pad(array, (window // 2, window - 1 - window // 2), constant_values=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.array([np.nanmedian(padded[i : i + window]) for i in range(array.size)])

    reduced = np.nanmedian(array[: n_blocks * block].reshape(n_blocks, block), axis=1)
    reduced_window = max(window // block, 1)
    padded = np.pad(
        reduced,
        (reduced_window // 2, reduced_window - 1 - reduced_window // 2),
        constant_values=np.nan,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        smooth = np.nanmedian(sliding_window_view(padded, reduced_window), axis=1)
    centers = (np.arange(smooth.size) + 0.5) * block
    return np.interp(np.arange(array.size), centers, smooth)


def detrend(channel: Channel, window_s: float = 10.0) -> np.ndarray:
    """Remove the slow baseline (body weight, posture drift) from a channel."""
    if channel.values.size == 0:
        return channel.values
    window = max(int(window_s * channel.sample_rate_hz), 1)
    if window >= channel.values.size // 2:
        return channel.values - np.nanmedian(channel.values)
    return channel.values - rolling_nanmedian(channel.values, window)


def robust_scale(values: np.ndarray) -> float:
    """Noise scale estimate: 1.4826 x MAD, falling back to std."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    mad = np.median(np.abs(finite - np.median(finite)))
    scale = 1.4826 * mad
    return float(scale if scale > 0 else finite.std() or 1.0)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end index pairs (end exclusive) of each run of True in ``mask``."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts, ends))


def load_events(
    channel: Channel,
    sigma: float = 4.0,
    min_duration_s: float = 1.0,
    smooth_s: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Find intervals where the signal departs from its baseline.

    Returns ``(start_s, end_s, mean_deviation)`` tuples, times relative to the
    first sample. Used to spot somebody getting onto the sensor, and to validate
    the setup with a deliberate press.
    """
    if channel.values.size == 0:
        return []
    baseline_removed = channel.values - np.nanmedian(channel.values)
    scale = robust_scale(baseline_removed)
    smoothed = rolling_nanmean(baseline_removed, int(smooth_s * channel.sample_rate_hz))
    mask = np.abs(smoothed) > sigma * scale
    min_length = int(min_duration_s * channel.sample_rate_hz)
    events = []
    for start, end in contiguous_runs(np.nan_to_num(mask).astype(bool)):
        if end - start < min_length:
            continue
        events.append(
            (
                float(channel.times[start] - channel.times[0]),
                float(channel.times[end - 1] - channel.times[0]),
                float(np.nanmean(smoothed[start:end])),
            )
        )
    return events


def psd(channel: Channel, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density of a channel (NaN gaps are interpolated away)."""
    values = channel.values
    finite = np.isfinite(values)
    if finite.sum() < 8:
        return np.array([]), np.array([])
    filled = values.copy()
    if not finite.all():
        filled[~finite] = np.interp(
            np.flatnonzero(~finite), np.flatnonzero(finite), values[finite]
        )
    if nperseg is None:
        nperseg = int(min(len(filled), 8 * channel.sample_rate_hz))
    nperseg = int(min(max(nperseg, 16), len(filled)))
    frequencies, power = signal.welch(
        filled, fs=channel.sample_rate_hz, nperseg=nperseg, detrend="constant"
    )
    return frequencies, power


def band_power(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    """Total power in a frequency band, as a fraction of the total power."""
    if frequencies.size == 0:
        return float("nan")
    band = (frequencies >= low) & (frequencies < high)
    total = power[(frequencies > 0) & (frequencies < frequencies.max())].sum()
    if total <= 0:
        return float("nan")
    return float(power[band].sum() / total)


def dominant_frequency(
    frequencies: np.ndarray, power: np.ndarray, low: float, high: float
) -> tuple[float, float]:
    """Peak frequency inside a band, returned as ``(hz, power)``."""
    band = (frequencies >= low) & (frequencies < high)
    if not band.any():
        return float("nan"), float("nan")
    index = np.argmax(power[band])
    return float(frequencies[band][index]), float(power[band][index])
