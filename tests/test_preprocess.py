"""Decoding notifications into a uniform, gap aware signal."""

import struct

import numpy as np
import pytest

from sleepy.dsp.preprocess import decode
from sleepy.protocol.constants import Handle

SAMPLES_PER_PACKET = 9
RATE = 140.0


def _packets(count: int, lost_at: set[int] | None = None) -> list[tuple[float, bytes]]:
    """Synthetic notifications: one packet every 9 samples worth of time."""
    lost_at = lost_at or set()
    interval = SAMPLES_PER_PACKET / RATE
    records = []
    timestamp = 1_700_000_000.0
    sequence = 0
    for index in range(count):
        if index in lost_at:
            sequence = (sequence + 2) % 256  # one packet went missing
        else:
            sequence = (sequence + 1) % 256
        timestamp += interval
        samples = [1000 + index * SAMPLES_PER_PACKET + offset for offset in range(SAMPLES_PER_PACKET)]
        payload = bytes([sequence]) + struct.pack(f"<{SAMPLES_PER_PACKET}h", *samples)
        records.append((timestamp, payload))
    return records


def test_sample_rate_is_recovered():
    channel = decode(_packets(200), int(Handle.PCM1_DATA), RATE, SAMPLES_PER_PACKET, "test")
    assert channel.sample_rate_hz == pytest.approx(140.0, abs=0.5)
    assert channel.packets == 200
    assert channel.lost_packets == 0
    assert channel.values.size == 200 * SAMPLES_PER_PACKET


def test_samples_land_in_order():
    channel = decode(_packets(50), int(Handle.PCM1_DATA), RATE, SAMPLES_PER_PACKET, "test")
    finite = channel.values[np.isfinite(channel.values)]
    assert finite[0] == 1000
    assert finite[-1] == 1000 + 50 * SAMPLES_PER_PACKET - 1


def test_lost_packets_leave_nan_gaps():
    channel = decode(_packets(50, lost_at={10}), int(Handle.PCM1_DATA), RATE, SAMPLES_PER_PACKET, "test")
    assert channel.lost_packets == 1
    assert channel.loss_ratio > 0
    assert np.isnan(channel.values).sum() == SAMPLES_PER_PACKET


def test_sequence_wrap_is_handled():
    channel = decode(_packets(400), int(Handle.PCM1_DATA), RATE, SAMPLES_PER_PACKET, "test")
    assert channel.lost_packets == 0  # the counter wrapped many times
    assert channel.sample_rate_hz == pytest.approx(140.0, abs=0.5)


def test_clipped_samples_are_dropped():
    records = _packets(20)
    # Rail the samples of the first packet: they must not pollute the signal.
    records[0] = (records[0][0], bytes([records[0][1][0]]) + b"\xff\x7f" * SAMPLES_PER_PACKET)
    channel = decode(records, int(Handle.PCM1_DATA), RATE, SAMPLES_PER_PACKET, "test")
    assert channel.clipped == SAMPLES_PER_PACKET
    assert np.isnan(channel.values[:SAMPLES_PER_PACKET]).all()
    assert np.isfinite(channel.values[SAMPLES_PER_PACKET:]).all()
    assert channel.stats()["clipped_samples"] == SAMPLES_PER_PACKET


def test_reconnect_resets_the_sequence_without_squeezing_the_timeline():
    """A reconnect used to drag the least squares fit off by 20 %."""
    first = _packets(100)
    gap_s = 30.0
    second = [(timestamp + gap_s, payload) for timestamp, payload in _packets(100)]
    channel = decode(first + second, int(Handle.PCM1_DATA), RATE, SAMPLES_PER_PACKET, "test")

    assert channel.sessions == 2
    assert channel.sample_rate_hz == pytest.approx(140.0, abs=0.5)
    assert channel.gap_s > gap_s - 100 * SAMPLES_PER_PACKET / RATE
    assert channel.packets == 200
    # Both sessions are decoded in full; the reconnect only leaves NaNs.
    assert np.isfinite(channel.values).sum() == 200 * SAMPLES_PER_PACKET
    assert channel.values.size > 200 * SAMPLES_PER_PACKET
    finite = channel.values[np.isfinite(channel.values)]
    assert finite[0] == 1000
    assert finite[-1] == 1000 + 100 * SAMPLES_PER_PACKET - 1  # second session restarts


def test_duplicate_notifications_are_dropped():
    """Old recorders added their callback once per reconnect, duplicating packets."""
    records = _packets(100)
    duplicated = [record for record in records for _ in range(3)]
    channel = decode(duplicated, int(Handle.PCM1_DATA), RATE, SAMPLES_PER_PACKET, "test")
    assert channel.packets == 100
    assert channel.values.size == 100 * SAMPLES_PER_PACKET
    assert channel.sample_rate_hz == pytest.approx(140.0, abs=0.5)
    assert np.isfinite(channel.values).all()
