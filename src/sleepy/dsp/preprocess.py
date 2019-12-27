"""Turn raw notifications into a uniformly sampled, gap aware signal.

Notifications arrive in bursts at BLE connection events, so the arrival
timestamps are jittery. We therefore

1. unwrap the per packet sequence counter to get a packet index that keeps
   counting across the 0x00-0xff wrap,
2. least-squares fit arrival time against that index, which both removes the
   BLE jitter and corrects the sensor/host clock skew,
3. rebuild a uniform sample grid at the measured sample interval, leaving NaN
   wherever packets were lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..protocol.packets import parse


@dataclass
class Channel:
    """A decoded, uniformly sampled sensor channel."""

    name: str
    sample_rate_hz: float
    times: np.ndarray
    values: np.ndarray
    packets: int = 0
    lost_packets: int = 0
    clipped: int = 0
    sample_rate_nominal_hz: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return float(self.times[-1] - self.times[0])

    @property
    def valid(self) -> np.ndarray:
        return self.values[np.isfinite(self.values)]

    @property
    def loss_ratio(self) -> float:
        total = self.packets + self.lost_packets
        return self.lost_packets / total if total else 0.0

    def stats(self) -> dict:
        values = self.valid
        if values.size == 0:
            return {"n": 0}
        return {
            "samples": int(values.size),
            "duration_s": round(self.duration_s, 2),
            "sample_rate_hz": round(self.sample_rate_hz, 3),
            "packet_loss": f"{self.loss_ratio:.2%}",
            "clipped_samples": self.clipped,
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": round(float(values.mean()), 1),
            "std": round(float(values.std()), 2),
        }


#: Samples at or beyond this magnitude are treated as clipped (int16 rails).
CLIP_LIMIT = 32760


def decode(
    records: list[tuple[float, bytes]],
    handle: int,
    sample_rate_hz: float,
    samples_per_packet: int,
    name: str = "channel",
    clip_limit: int = CLIP_LIMIT,
) -> Channel:
    """Decode notifications of one PCM channel into a :class:`Channel`.

    Args:
        records: ``(timestamp, payload)`` pairs, as stored in a raw log.
        handle: handle the notifications arrived on (used for parsing).
        sample_rate_hz: nominal rate, from the sensor's own description.
        samples_per_packet: samples carried by one notification.
        name: label for the channel.
    """
    sequences = []
    arrival = []
    blocks: list[tuple[int, ...]] = []
    for timestamp, payload in records:
        packet = parse(payload, handle)
        if not packet.is_pcm or packet.sequence is None:
            continue
        sequences.append(packet.sequence)
        arrival.append(timestamp)
        blocks.append(packet.samples)

    channel = Channel(
        name=name,
        sample_rate_hz=float(sample_rate_hz),
        times=np.array([], dtype=float),
        values=np.array([], dtype=float),
        sample_rate_nominal_hz=float(sample_rate_hz),
    )
    if len(blocks) < 2:
        channel.notes.append("fewer than 2 packets, nothing to decode")
        return channel

    seq = np.asarray(sequences, dtype=np.int64)
    arrival = np.asarray(arrival, dtype=float)
    # Unwrap the rolling 8 bit counter into a monotonically increasing index.
    deltas = np.diff(seq) % 256
    index = np.concatenate([[0], np.cumsum(deltas)])
    channel.packets = len(index)
    channel.lost_packets = int((deltas[deltas > 1] - 1).sum())

    # Least squares fit: arrival_time = seconds_per_block * index + offset
    design = np.vstack([index, np.ones_like(index, dtype=float)]).T
    (seconds_per_block, offset), *_ = np.linalg.lstsq(design, arrival, rcond=None)
    if seconds_per_block <= 0:
        channel.notes.append("non-positive sample interval, falling back to nominal rate")
        seconds_per_block = samples_per_packet / sample_rate_hz
        offset = arrival[0]

    interval = seconds_per_block / samples_per_packet
    channel.sample_rate_hz = 1.0 / interval
    packet_times = offset + seconds_per_block * index

    first_time = packet_times[0] - (samples_per_packet - 1) * interval
    last_time = packet_times[-1]
    length = int(round((last_time - first_time) / interval)) + 1
    values = np.full(length, np.nan, dtype=float)

    samples = np.asarray(blocks, dtype=float)
    for row, packet_time in enumerate(packet_times):
        start = int(round((packet_time - (samples_per_packet - 1) * interval - first_time) / interval))
        start = max(start, 0)
        stop = min(start + samples.shape[1], length)
        values[start:stop] = samples[row, : stop - start]

    # Samples at the rails carry no information: the sensor's DC level with a
    # person in bed sits close to +32767, so large movements clip. Dropping them
    # keeps spectra and detrending honest.
    clipped = np.abs(values) >= clip_limit
    channel.clipped = int(clipped.sum())
    values[clipped] = np.nan

    channel.times = first_time + np.arange(length) * interval
    channel.values = values
    return channel
