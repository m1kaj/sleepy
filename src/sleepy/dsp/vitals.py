"""Breathing and heart rate estimation from the 140 Hz force channel.

The strip measures the mechanical force a sleeping person exerts on the bed.
On top of the (huge, slow) weight baseline that signal contains:

* respiration: chest movement, roughly 0.1-0.6 Hz (6-36 breaths/min),
* heart beat: ballistocardiogram, roughly 0.8-3.5 Hz (48-210 bpm),
* movement: broadband transients that drown both.

Everything works on short windows and reports a confidence value, because a
real night is mostly "we cannot tell" while the sleeper moves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

from .analyse import robust_scale, rolling_nanmedian

BREATHING_BAND_HZ = (0.1, 0.6)
HEART_BAND_HZ = (0.8, 3.5)

#: Shortest in-bed stretch we are willing to estimate a rate from. Below 20 s
#: the breathing band has only a couple of cycles in it and the frequency
#: resolution (1/T) is worse than 3 breaths per minute.
MIN_RESPIRATION_S = 20.0
MIN_HEART_S = 8.0


@dataclass
class Epoch:
    """One analysis window of the night."""

    start_s: float
    end_s: float
    load: float
    movement: float
    respiration_bpm: float | None = None
    respiration_confidence: float = 0.0
    heart_bpm: float | None = None
    heart_confidence: float = 0.0
    present: bool = False
    usable: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def bandpass(values: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero phase Butterworth bandpass, tolerant of NaN gaps."""
    finite = np.isfinite(values)
    if finite.sum() < 8 or fs <= 0:
        return np.full_like(values, np.nan)
    filled = values.copy()
    if not finite.all():
        filled[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), values[finite])
    nyquist = fs / 2.0
    low = max(low / nyquist, 1e-4)
    high = min(high / nyquist, 0.99)
    if low >= high:
        return np.full_like(values, np.nan)
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, filled)


def _peak_frequency(values: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[float, float]:
    """Dominant frequency inside ``band`` as ``(hz, peak_power)``."""
    windowed = values * np.hanning(len(values))
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(windowed), 1.0 / fs)
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    if not mask.any():
        return float("nan"), float("nan")
    band_power = spectrum[mask]
    band_freq = frequencies[mask]
    index = int(np.argmax(band_power))
    peak = band_freq[index]
    # Parabolic interpolation around the peak for sub-bin resolution.
    if 0 < index < len(band_power) - 1:
        left, mid, right = band_power[index - 1 : index + 2]
        denominator = left - 2 * mid + right
        if denominator != 0:
            peak += 0.5 * (left - right) / denominator * (band_freq[1] - band_freq[0])
    return float(peak), float(band_power[index])


def estimate_rate(
    values: np.ndarray,
    fs: float,
    band: tuple[float, float],
) -> tuple[float | None, float]:
    """Rate in cycles per minute plus a 0-1 confidence, for one window.

    Confidence is how far the spectral peak stands out above the rest of the
    band, so a flat, noisy window scores near zero.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 16:
        return None, 0.0
    # A reconnect leaves a NaN gap. Interpolating across it would silently
    # squeeze or stretch the rhythm, so refuse windows that straddle a big gap.
    if finite.size / values.size < 0.95:
        return None, 0.0
    detrended = values - finite.mean()
    if np.allclose(detrended[np.isfinite(detrended)], 0):
        return None, 0.0
    filtered = bandpass(detrended, fs, *band)
    if not np.isfinite(filtered).all() or np.nanstd(filtered) <= 0:
        return None, 0.0

    peak_hz, peak_power = _peak_frequency(filtered, fs, band)
    if not np.isfinite(peak_hz) or peak_hz <= 0:
        return None, 0.0

    spectrum = np.abs(np.fft.rfft(filtered * np.hanning(len(filtered))))
    frequencies = np.fft.rfftfreq(len(filtered), 1.0 / fs)
    in_band = spectrum[(frequencies >= band[0]) & (frequencies <= band[1])]
    median_power = float(np.median(in_band)) if in_band.size else 0.0
    if median_power <= 0:
        return None, 0.0
    # 1x = no peak at all, 10x = unmistakable rhythm.
    confidence = float(np.clip((peak_power / median_power - 2.0) / 8.0, 0.0, 1.0))
    return peak_hz * 60.0, confidence


def movement_index(values: np.ndarray, fs: float) -> float:
    """Movement proxy: robust spread of the signal in counts (scale aware)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(robust_scale(finite - np.median(finite)))


#: Level the sensor reports when the bed is empty, observed on our unit: it
#: sits close to the bottom of the int16 range (-32768). Used when a recording
#: never shows an empty bed to compare against.
DEFAULT_EMPTY_LEVEL = -30_000.0


def presence_threshold(
    load_track: np.ndarray,
    min_separation: float = 10_000.0,
    bins: int = 64,
) -> float | None:
    """Split the load track into "empty" and "occupied" (Otsu's method).

    A person in bed moves the DC level by tens of thousands of counts, so the
    load track is clearly bimodal on a real night and clearly unimodal when
    nobody ever gets in. Returns the level that separates the two, or ``None``
    when the two modes are not far enough apart to mean anything (nobody in
    bed, or only sensor drift).
    """
    values = load_track[np.isfinite(load_track)]
    if values.size < bins:
        return None
    counts, edges = np.histogram(values, bins=bins)
    if counts.sum() == 0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_low = np.cumsum(counts)
    weight_high = weight_low[-1] - weight_low
    sum_low = np.cumsum(counts * centers)
    total = sum_low[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_low = sum_low / np.maximum(weight_low, 1)
        mean_high = (total - sum_low) / np.maximum(weight_high, 1)
        variance = weight_low * weight_high * (mean_low - mean_high) ** 2
    if not np.isfinite(variance).any():
        return None
    split = int(np.nanargmax(variance))
    lower_mean = float(mean_low[split])
    upper_mean = float(mean_high[split])
    if not (np.isfinite(lower_mean) and np.isfinite(upper_mean)):
        return None
    if upper_mean - lower_mean < min_separation:
        # Only one mode: either nobody ever got in, or the person was already in
        # bed when recording started. Fall back to the level this sensor reports
        # for an empty bed, so an occupied recording still reads as occupied.
        lowest = float(np.nanpercentile(values, 2))
        return min(DEFAULT_EMPTY_LEVEL, lowest) + min_separation
    return float(centers[split])


def quietest_window(
    values: np.ndarray,
    fs: float,
    span_s: float,
    window_s: float,
    step_s: float = 5.0,
) -> np.ndarray:
    """Pick the calmest ``window_s`` inside a longer ``span_s`` of signal.

    Real nights are full of tossing and turning; the estimate from the calmest
    stretch beats the estimate from the whole epoch.
    """
    window_s = min(window_s, span_s)
    window = int(window_s * fs)
    step = max(int(step_s * fs), 1)
    best_index = 0
    best_score = float("inf")
    fallback = 0
    for start in range(0, max(int(span_s * fs) - window, 1), step):
        candidate = values[start : start + window]
        if np.isfinite(candidate).mean() < 0.95:
            continue
        fallback = start
        score = movement_index(candidate, fs)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_index = start
    return values[best_index : best_index + window] if best_score < float("inf") else values[fallback:]


def _occupied_span(
    values: np.ndarray, occupied: np.ndarray, end: int, max_samples: int
) -> np.ndarray:
    """The stretch of signal ending at ``end`` in which the bed was occupied.

    Without this, the "calmest window" search happily picks the quiet empty bed
    instead of the slightly noisier breathing signal.
    """
    start = max(end - 1, 0)
    while start > 0 and end - start < max_samples and occupied[start - 1]:
        start -= 1
    return values[start:end]


def process_channel(
    times: np.ndarray,
    values: np.ndarray,
    fs: float,
    epoch_s: float = 30.0,
    respiration_window_s: float = 60.0,
    heart_window_s: float = 20.0,
    load_window_s: float = 30.0,
    movement_limit: float | None = None,
) -> list[Epoch]:
    """Slide over a channel and estimate breathing and heart rate per epoch."""
    epochs: list[Epoch] = []
    step = int(epoch_s * fs)
    span = int(times.size - step + 1) if times.size > step else 0
    if span <= 0:
        return epochs

    # Slow "load" track: how much weight is on the sensor right now. A person
    # moves the DC level by tens of thousands of counts (empty bed sits near
    # -32000, an occupied one near +32000), so a one minute median is plenty to
    # separate the two states while still reacting within a minute.
    load_window = max(int(load_window_s * fs), 1)
    if values.size > 2 * load_window:
        load_track = rolling_nanmedian(values, load_window)
    else:
        load_track = np.full(values.size, float(np.nanmedian(values)))
    threshold = presence_threshold(load_track)
    occupied = np.isfinite(load_track) & (load_track > threshold)

    movements = [movement_index(values[a : a + step], fs) for a in range(0, span, step)]
    if movement_limit is None and movements:
        # Movement is judged against the quietest parts of the night.
        movement_limit = float(np.nanmedian(movements) * 3.0 + 1.0)
    movement_limit = movement_limit or float("inf")

    for index, offset in enumerate(range(0, span, step)):
        epoch_values = values[offset : offset + step]
        if not np.isfinite(epoch_values).any():
            continue
        movement = movements[index]
        load = float(np.nanmedian(load_track[offset : offset + step]))
        present = bool(threshold is not None and np.isfinite(load) and load > threshold)

        # Only estimate vitals from signal where somebody is actually in bed:
        # walk back from the end of the epoch while the load stays high, then
        # use the calmest window inside that stretch.
        respiration_span = _occupied_span(values, occupied, offset + step, int(respiration_window_s * fs))
        if respiration_span.size / fs >= MIN_RESPIRATION_S:
            respiration_bpm, respiration_confidence = estimate_rate(
                quietest_window(respiration_span, fs, respiration_span.size / fs, 30.0),
                fs,
                BREATHING_BAND_HZ,
            )
        else:
            respiration_bpm, respiration_confidence = None, 0.0

        heart_span = _occupied_span(values, occupied, offset + step, int(heart_window_s * fs))
        if heart_span.size / fs >= MIN_HEART_S:
            heart_bpm, heart_confidence = estimate_rate(
                quietest_window(heart_span, fs, heart_span.size / fs, min(heart_window_s, 20.0), 5.0),
                fs,
                HEART_BAND_HZ,
            )
        else:
            heart_bpm, heart_confidence = None, 0.0
        quiet = bool(np.isfinite(movement) and movement <= movement_limit)
        epochs.append(
            Epoch(
                start_s=round(offset / fs, 3),
                end_s=round((offset + step) / fs, 3),
                load=round(load, 1) if np.isfinite(load) else float("nan"),
                movement=round(movement, 2) if np.isfinite(movement) else float("nan"),
                respiration_bpm=None if respiration_bpm is None else round(respiration_bpm, 2),
                respiration_confidence=round(respiration_confidence, 3),
                heart_bpm=None if heart_bpm is None else round(heart_bpm, 2),
                heart_confidence=round(heart_confidence, 3),
                present=present,
                usable=bool(quiet and present),
            )
        )
    return epochs


def summarise(epochs: list[Epoch]) -> dict:
    """Whole night summary from the epoch table."""
    if not epochs:
        return {}
    usable = [e for e in epochs if e.usable]
    present = [e for e in epochs if e.present]
    respiration = [e.respiration_bpm for e in usable if e.respiration_bpm and e.respiration_confidence >= 0.25]
    heart = [e.heart_bpm for e in usable if e.heart_bpm and e.heart_confidence >= 0.25]
    epoch_length = epochs[0].end_s - epochs[0].start_s
    return {
        "epochs": len(epochs),
        "epochs_present": len(present),
        "epochs_usable": len(usable),
        "in_bed_s": round(len(present) * epoch_length, 1),
        "respiration_bpm_mean": round(float(np.mean(respiration)), 2) if respiration else None,
        "respiration_bpm_median": round(float(np.median(respiration)), 2) if respiration else None,
        "heart_bpm_mean": round(float(np.mean(heart)), 2) if heart else None,
        "heart_bpm_median": round(float(np.median(heart)), 2) if heart else None,
        "movement_median": round(float(np.nanmedian([e.movement for e in epochs])), 2),
    }
