"""Breathing and heart rate estimation on synthetic signals."""

import numpy as np
import pytest

from sleepy.dsp.vitals import (
    BREATHING_BAND_HZ,
    HEART_BAND_HZ,
    estimate_rate,
    movement_index,
    presence_threshold,
    process_channel,
    quietest_window,
)

FS = 140.0


def _sine(hz: float, seconds: float = 60.0, amplitude: float = 50.0, noise: float = 5.0) -> np.ndarray:
    times = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(42)
    return amplitude * np.sin(2 * np.pi * hz * times) + noise * rng.standard_normal(times.size)


def test_breathing_rate_is_recovered():
    signal = _sine(0.25)  # 15 breaths per minute
    bpm, confidence = estimate_rate(signal, FS, BREATHING_BAND_HZ)
    assert bpm is not None
    assert bpm == pytest.approx(15.0, abs=1.0)
    assert confidence > 0.5


def test_heart_rate_is_recovered():
    signal = _sine(1.2)  # 72 beats per minute
    bpm, confidence = estimate_rate(signal, FS, HEART_BAND_HZ)
    assert bpm is not None
    assert bpm == pytest.approx(72.0, abs=2.0)
    assert confidence > 0.5


def test_flat_noise_has_no_confidence():
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(int(60 * FS))
    bpm, confidence = estimate_rate(noise, FS, BREATHING_BAND_HZ)
    assert confidence < 0.5


def test_process_channel_produces_epochs():
    """A whole night in bed: the DC level is high and stays high."""
    times = np.arange(int(300 * FS)) / FS
    signal = _sine(0.25, seconds=300) + 30_000  # occupied the whole time
    epochs = process_channel(times, signal, FS, epoch_s=30.0)
    assert len(epochs) == 10
    assert all(epoch.movement >= 0 for epoch in epochs)
    breathing = [e.respiration_bpm for e in epochs if e.respiration_bpm]
    assert breathing, "expected at least one breathing estimate"
    assert np.median(breathing) == pytest.approx(15.0, abs=1.5)


def test_presence_is_detected_when_the_dc_level_jumps():
    """Empty bed sits near -32000 counts, an occupied bed near +32000."""
    times = np.arange(int(600 * FS)) / FS
    signal = np.full(times.size, -32_000.0)
    rng = np.random.default_rng(1)
    signal += rng.standard_normal(times.size) * 50
    signal[int(300 * FS) :] += 64_000.0  # somebody gets in at t = 300 s

    epochs = process_channel(times, signal, FS, epoch_s=30.0)
    before = [e for e in epochs if e.start_s < 240]
    after = [e for e in epochs if e.start_s > 360]
    assert not any(e.present for e in before), "empty bed must not read as present"
    assert all(e.present for e in after), "occupied bed must read as present"


def test_no_presence_when_nobody_gets_in():
    """An empty bed that only drifts must not produce a presence threshold."""
    times = np.arange(int(600 * FS)) / FS
    signal = np.full(times.size, -32_000.0)
    rng = np.random.default_rng(5)
    signal += rng.standard_normal(times.size) * 50
    signal += np.linspace(0, 800, signal.size)  # slow drift, no body

    epochs = process_channel(times, signal, FS, epoch_s=30.0)
    assert not any(e.present for e in epochs)


def test_presence_threshold_separates_empty_from_occupied():
    empty = np.full(10_000, -32_000.0) + np.random.default_rng(2).standard_normal(10_000) * 40
    # A bed that stays empty: the threshold must sit above everything we saw.
    assert presence_threshold(empty) > empty.max()

    two_modes = np.concatenate([np.full(5_000, -32_000.0), np.full(5_000, 30_000.0)])
    threshold = presence_threshold(two_modes)
    assert -32_000 < threshold < 30_000

    # Somebody already in bed when recording starts: still recognised.
    occupied = np.full(10_000, 30_000.0)
    assert presence_threshold(occupied) < 30_000


def test_quietest_window_prefers_the_calm_part():
    calm = _sine(0.25, seconds=20, amplitude=40, noise=2)
    noisy = np.random.default_rng(3).standard_normal(20 * int(FS)) * 5_000
    window = np.concatenate([noisy, calm])
    chosen = quietest_window(window, FS, span_s=40.0, window_s=20.0, step_s=10.0)
    assert chosen.size == int(20 * FS)
    assert movement_index(chosen, FS) < movement_index(noisy, FS)
