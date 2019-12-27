"""Append-only binary log for raw BLE notifications.

Every record is a small TLV::

    u64  timestamp, nanoseconds since the epoch (time.time_ns())
    u16  ATT handle the notification arrived on
    u8   payload length
    bytes payload

The file starts with the ASCII magic ``SLEEPY1`` followed by a newline so the
format can be identified and versioned later on.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from pathlib import Path

FILE_MAGIC = b"SLEEPY1\n"
_RECORD_HEADER = struct.Struct("<QHB")


@dataclass(frozen=True)
class RawRecord:
    timestamp_ns: int
    handle: int
    payload: bytes

    @property
    def timestamp(self) -> float:
        return self.timestamp_ns / 1e9


class RawLogWriter:
    """Buffered writer for raw notification records."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("wb")
        self._fh.write(FILE_MAGIC)
        self.records = 0

    def write(self, handle: int, payload: bytes, timestamp_ns: int | None = None) -> None:
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        self._fh.write(_RECORD_HEADER.pack(timestamp_ns, handle, len(payload)))
        self._fh.write(payload)
        self.records += 1

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self.flush()
        self._fh.close()

    def __enter__(self) -> RawLogWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_raw_log(path: Path | str) -> list[RawRecord]:
    """Read back a raw log written by :class:`RawLogWriter`."""
    data = Path(path).read_bytes()
    if not data.startswith(FILE_MAGIC):
        raise ValueError(f"{path} is not a sleepy raw log (bad magic)")
    offset = len(FILE_MAGIC)
    records: list[RawRecord] = []
    while offset + _RECORD_HEADER.size <= len(data):
        timestamp_ns, handle, length = _RECORD_HEADER.unpack_from(data, offset)
        offset += _RECORD_HEADER.size
        payload = data[offset : offset + length]
        if len(payload) != length:
            break
        offset += length
        records.append(RawRecord(timestamp_ns, handle, payload))
    return records


def raw_log_summary(path: Path | str) -> dict:
    """Cheap statistics over a raw log: per handle counts, rates, sizes."""
    records = read_raw_log(path)
    per_handle: dict[int, dict] = {}
    for record in records:
        entry = per_handle.setdefault(
            record.handle,
            {"count": 0, "first_ns": record.timestamp_ns, "last_ns": record.timestamp_ns,
             "lengths": set(), "bytes": 0},
        )
        entry["count"] += 1
        entry["last_ns"] = record.timestamp_ns
        entry["lengths"].add(len(record.payload))
        entry["bytes"] += len(record.payload)

    for entry in per_handle.values():
        duration = (entry["last_ns"] - entry["first_ns"]) / 1e9
        entry["duration_s"] = round(duration, 3)
        entry["rate_hz"] = round(entry["count"] / duration, 2) if duration > 0 else None
        entry["lengths"] = sorted(entry["lengths"])
        entry.pop("first_ns")
        entry.pop("last_ns")
    return {"records": len(records), "handles": per_handle}
