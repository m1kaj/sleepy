"""Decoding of the sensor's self description.

Each PCM data channel has a companion "description" characteristic whose
payload is a ``uint16`` sample rate in 1/100 Hz followed by an ASCII tag::

    b0 36 "u2force"  ->  0x36b0 = 14000  ->  140.00 Hz force channel
    f4 01 "u2force"  ->  0x01f4 =   500  ->    5.00 Hz force channel
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelDescription:
    """What the sensor tells us about one of its PCM channels."""

    sample_rate_hz: float
    tag: str

    def __str__(self) -> str:
        return f"{self.sample_rate_hz:.2f} Hz {self.tag!r}"


def parse_channel_description(data: bytes) -> ChannelDescription:
    """Decode a PCM channel description characteristic value."""
    if len(data) < 2:
        return ChannelDescription(sample_rate_hz=float("nan"), tag="")
    rate = struct.unpack("<H", data[:2])[0] / 100.0
    tag = data[2:].decode("ascii", errors="replace").strip("\x00")
    return ChannelDescription(sample_rate_hz=rate, tag=tag)
