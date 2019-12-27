"""Raw log round trip tests."""

from pathlib import Path

from sleepy.protocol.constants import Handle
from sleepy.store import rawlog


def test_write_read_round_trip(tmp_path: Path):
    path = tmp_path / "capture.bin"
    with rawlog.RawLogWriter(path) as writer:
        writer.write(int(Handle.PCM1_DATA), bytes(19))
        writer.write(int(Handle.PCM2_DATA), bytes.fromhex("00e824"), timestamp_ns=1_700_000_000_000_000_000)

    records = rawlog.read_raw_log(path)
    assert len(records) == 2
    assert records[0].handle == int(Handle.PCM1_DATA)
    assert records[0].payload == bytes(19)
    assert records[1].payload == bytes.fromhex("00e824")
    assert records[1].timestamp == 1_700_000_000.0


def test_summary_counts_per_handle(tmp_path: Path):
    path = tmp_path / "capture.bin"
    with rawlog.RawLogWriter(path) as writer:
        for second in range(4):
            timestamp = 1_700_000_000_000_000_000 + second * 1_000_000_000
            writer.write(int(Handle.PCM1_DATA), bytes(19), timestamp_ns=timestamp)

    summary = rawlog.raw_log_summary(path)
    assert summary["records"] == 4
    entry = summary["handles"][int(Handle.PCM1_DATA)]
    assert entry["count"] == 4
    assert entry["duration_s"] == 3.0
    assert entry["rate_hz"] == round(4 / 3, 2)


def test_bad_magic_is_rejected(tmp_path: Path):
    path = tmp_path / "not_a_log.bin"
    path.write_bytes(b"something else")
    try:
        rawlog.read_raw_log(path)
    except ValueError as error:
        assert "not a sleepy raw log" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")
