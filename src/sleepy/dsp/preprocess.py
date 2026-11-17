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
    sessions: int = 1
    gap_s: float = 0.0

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
            "sessions": self.sessions,
            "gap_s": round(self.gap_s, 2),
            "clipped_samples": self.clipped,
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": round(float(values.mean()), 1),
            "std": round(float(values.std()), 2),
        }


#: Samples at or beyond this magnitude are treated as clipped (int16 rails).
CLIP_LIMIT = 32760

#: A silence longer than this starts a new recording session. The sensor sends
#: 9 samples every ~64 ms, so anything longer than a second is a stalled link or
#: a reconnect, not normal BLE jitter. The timing fit must not run across such
#: a gap: a reconnect resets the sequence counter and the wall clock moves on.
SESSION_GAP_S = 1.0

#: A sequence jump larger than this within one session means the stream
#: restarted (64 packets are >4 s of samples, far more than BLE can drop in the
#: sub-second gap between two consecutive notifications).
SEQUENCE_RESTART = 64

#: Duplicate notifications written back to back by an older recorder are not
#: new samples. Repeats are adjacent in the log, within microseconds.
DUPLICATE_WINDOW_S = 0.02


def _collect_blocks(
    records: list[tuple[float, bytes]], handle: int
) -> list[tuple[int, float, tuple[int, ...]]]:
    """Parse records into ``(sequence, timestamp, samples)`` blocks.

    Exact duplicates are dropped: older versions of the recorder registered the
    callback once per reconnect, so every notification was logged two, three,
    ... times. They are adjacent in the log and within a few hundred
    microseconds of each other, which is what the duplicate window catches.
    """
    blocks: list[tuple[int, float, tuple[int, ...]]] = []
    previous_payload: bytes | None = None
    previous_timestamp = 0.0
    for timestamp, payload in records:
        packet = parse(payload, handle)
        if not packet.is_pcm or packet.sequence is None:
            continue
        if (
            payload == previous_payload
            and abs(timestamp - previous_timestamp) < DUPLICATE_WINDOW_S
        ):
            continue
        blocks.append((packet.sequence, timestamp, packet.samples))
        previous_payload = payload
        previous_timestamp = timestamp
    return blocks


def _split_sessions(
    blocks: list[tuple[int, float, tuple[int, ...]]]
) -> tuple[list[list[tuple[int, float, tuple[int, ...]]]], int]:
    """Split blocks where the link was lost, returning the segments and loss.

    A new segment starts on a wall clock jump (either direction), an unchanged
    sequence number, or a sequence jump too large to be packet loss.
    """
    segments: list[list[tuple[int, float, tuple[int, ...]]]] = []
    current: list[tuple[int, float, tuple[int, ...]]] = []
    boundary_lost = 0
    for block in blocks:
        if current:
            sequence, timestamp, _ = block
            previous_sequence, previous_timestamp, _ = current[-1]
            delta_s = timestamp - previous_timestamp
            delta_index = (sequence - previous_sequence) % 256
            if (
                abs(delta_s) > SESSION_GAP_S
                or delta_index == 0
                or delta_index > SEQUENCE_RESTART
            ):
                segments.append(current)
                current = []
                # A short stall in an otherwise continuous stream is still
                # packet loss; count it. A multi-second gap is a reconnect and
                # its "lost" sequence numbers tell us nothing useful.
                if 0 < delta_index <= SEQUENCE_RESTART and 0 < delta_s < 5.0:
                    boundary_lost += delta_index - 1
        current.append(block)
    if current:
        segments.append(current)
    return segments, boundary_lost


def decode(
    records: list[tuple[float, bytes]],
    handle: int,
    sample_rate_hz: float,
    samples_per_packet: int,
    name: str = "channel",
    clip_limit: int = CLIP_LIMIT,
) -> Channel:
    """Decode notifications of one PCM channel into a :class:`Channel`.

    Records are split into sessions wherever the BLE link was lost. Each
    session gets its own timing fit; the output grid spans the real wall clock
    time, so a reconnect leaves a NaN gap instead of squeezing the whole night.

    Args:
        records: ``(timestamp, payload)`` pairs, as stored in a raw log.
        handle: handle the notifications arrived on (used for parsing).
        sample_rate_hz: nominal rate, from the sensor's own description.
        samples_per_packet: samples carried by one notification.
        name: label for the channel.
    """
    blocks = _collect_blocks(records, handle)

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

    segments, boundary_lost = _split_sessions(blocks)
    channel.sessions = len(segments)
    channel.packets = len(blocks)
    channel.gap_s = sum(
        max(0.0, next_segment[0][1] - segment[-1][1])
        for segment, next_segment in zip(segments, segments[1:])
    )

    # Estimate the sample interval from complete sessions. The plain least
    # squares fit used to run across reconnects and was dragged off by the
    # wall clock jumps (a real 15 minute capture decoded as 115 Hz instead of
    # 140 Hz). A ratio of the first and last packet of each session cannot be
    # fooled that way.
    nominal_block = samples_per_packet / sample_rate_hz
    ratios = []
    unwrapped: list[np.ndarray] = []
    for segment in segments:
        sequence = np.asarray([block[0] for block in segment], dtype=np.int64)
        deltas = np.diff(sequence) % 256
        index = np.concatenate([[0], np.cumsum(deltas)])
        unwrapped.append(index)
        if len(segment) >= 4 and index[-1] > 0:
            ratios.append((segment[-1][1] - segment[0][1]) / index[-1])
    seconds_per_block = float(np.median(ratios)) if ratios else nominal_block
    if (
        not np.isfinite(seconds_per_block)
        or not 0.7 * nominal_block <= seconds_per_block <= 1.3 * nominal_block
    ):
        channel.notes.append(
            f"timing fit {seconds_per_block:.6f} s/packet is implausible; "
            f"using the nominal {nominal_block:.6f} s/packet"
        )
        seconds_per_block = nominal_block

    interval = seconds_per_block / samples_per_packet
    channel.sample_rate_hz = 1.0 / interval

    lost_packets = boundary_lost
    packet_times: list[np.ndarray] = []
    for segment, index in zip(segments, unwrapped):
        sequence = np.asarray([block[0] for block in segment], dtype=np.int64)
        deltas = np.diff(sequence) % 256
        lost_packets += int((deltas[deltas > 1] - 1).sum())
        arrival = np.asarray([block[1] for block in segment], dtype=float)
        offset = float(np.median(arrival - seconds_per_block * index))
        packet_times.append(offset + seconds_per_block * index)
    channel.lost_packets = lost_packets

    first_time = min(
        times[0] - (samples_per_packet - 1) * interval for times in packet_times
    )
    last_time = max(times[-1] for times in packet_times)
    length = int(round((last_time - first_time) / interval)) + 1
    values = np.full(length, np.nan, dtype=float)

    for segment, times in zip(segments, packet_times):
        for (_, _, samples), packet_time in zip(segment, times):
            start = int(
                round(
                    (packet_time - (samples_per_packet - 1) * interval - first_time)
                    / interval
                )
            )
            start = max(start, 0)
            stop = min(start + len(samples), length)
            values[start:stop] = samples[: stop - start]

    # Samples at the rails carry no information: the sensor's DC level with a
    # person in bed sits close to +32767, so large movements clip. Dropping them
    # keeps spectra and detrending honest.
    clipped = np.abs(values) >= clip_limit
    channel.clipped = int(clipped.sum())
    values[clipped] = np.nan

    channel.times = first_time + np.arange(length) * interval
    channel.values = values
    return channel
