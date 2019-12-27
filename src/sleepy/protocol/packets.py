"""Parsing of notification payloads from the Beddit V3 sensor.

Two generations of packet layout exist; both are supported here.

The InMoodForLife teardown (BLE Beddit, 2017) saw::

    80 00 a6 78 9f 78 ...   magic 0x80, sequence, 9 x int16 little endian

Our unit (Beddit V3 Sensor, SW 1.1/4a2417b) has **no** magic byte. The first
byte is the rolling sequence counter and the rest are int16 samples::

    00 d2 7a 97 7d 87 86 21 79 e0 81 50 85 ec 78 55 85 50 82   (19 B, PCM ch1)
    ^^ sequence
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 9 x int16 little endian

    00 e8 24                                                    (3 B, PCM ch2)
    ^^ sequence, then 1 x int16

Temperature and humidity are the standard Bluetooth SIG Environmental Sensing
values in units of 0.01 deg C / 0.01 %.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .constants import Handle

#: Payload lengths seen on our unit.
PCM1_PACKET_LEN = 19
PCM2_PACKET_LEN = 3

#: The other known generation: 0x80 magic + sequence + samples (20 bytes).
LEGACY_PACKET_LEN = 20
LEGACY_MAGIC = 0x80


@dataclass(frozen=True)
class Packet:
    """One decoded notification."""

    raw: bytes
    handle: int | None = None
    kind: str = "unknown"
    sequence: int | None = None
    samples: tuple[int, ...] = ()
    value: float | None = None

    @property
    def is_pcm(self) -> bool:
        return self.kind == "pcm"


def parse(data: bytes, handle: int | None = None) -> Packet:
    """Decode a notification payload, using the handle to pick the kind."""
    if not data:
        return Packet(raw=data, handle=handle, kind="empty")

    if handle == Handle.TEMPERATURE and len(data) >= 2:
        return Packet(
            raw=data,
            handle=handle,
            kind="temperature",
            value=struct.unpack("<h", data[:2])[0] / 100.0,
        )
    if handle == Handle.HUMIDITY and len(data) >= 2:
        return Packet(
            raw=data,
            handle=handle,
            kind="humidity",
            value=struct.unpack("<H", data[:2])[0] / 100.0,
        )
    if handle == Handle.FLAGS:
        return Packet(raw=data, handle=handle, kind="flags", value=data[0])

    # The legacy layout is unambiguous only by length: 20 bytes that start with
    # the 0x80 magic. We must not key on the magic alone, because on our unit
    # the first byte is a rolling sequence counter that legitimately hits 0x80.
    if len(data) == LEGACY_PACKET_LEN and data[0] == LEGACY_MAGIC:
        return _parse_pcm(data[2:], handle, sequence=data[1])

    if handle in (Handle.PCM1_DATA, Handle.PCM2_DATA) or len(data) in (
        PCM1_PACKET_LEN,
        PCM2_PACKET_LEN,
    ):
        return _parse_pcm(data, handle)
    return Packet(raw=data, handle=handle, kind="unknown")


def _parse_pcm(data: bytes, handle: int | None, sequence: int | None = None) -> Packet:
    if not data:
        return Packet(raw=data, handle=handle, kind="pcm")
    if sequence is None:
        sequence = data[0]
        payload = data[1:]
    else:
        payload = data
    usable = len(payload) - (len(payload) % 2)
    samples = struct.unpack(f"<{usable // 2}h", payload[:usable]) if usable else ()
    return Packet(
        raw=data,
        handle=handle,
        kind="pcm",
        sequence=sequence,
        samples=tuple(samples),
    )


def sequence_gap(previous: int | None, current: int | None) -> int:
    """How many packets were lost between two sequence numbers (0 if contiguous)."""
    if previous is None or current is None:
        return 0
    return (current - previous - 1) % 256


def describe(data: bytes, handle: int | None = None) -> str:
    """One line, human readable description of a payload."""
    packet = parse(data, handle)
    if packet.kind == "pcm":
        return f"seq={packet.sequence:3d} n={len(packet.samples)} samples={list(packet.samples)}"
    if packet.kind in ("temperature", "humidity"):
        unit = "C" if packet.kind == "temperature" else "%"
        return f"{packet.kind}={packet.value:.2f}{unit}"
    if packet.kind == "flags":
        return f"flags=0x{data[0]:02x}"
    if packet.kind == "empty":
        return "(empty payload)"
    return f"unknown ({len(data)} B): {data.hex()}"
