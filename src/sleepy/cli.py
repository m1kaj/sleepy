"""Command line interface for sleepy."""

from __future__ import annotations

import asyncio
import contextlib
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from .dsp.analyse import band_power, dominant_frequency, load_events, psd
from .dsp.preprocess import decode
from .dsp.vitals import process_channel, summarise
from .protocol.constants import (
    CONTROL_START,
    PCM1_SAMPLES_PER_PACKET,
    PCM2_SAMPLES_PER_PACKET,
    Handle,
)
from .protocol.packets import describe, parse
from .report import html
from .report.plots import plot_night, plot_overview
from .store import rawlog
from .transport import scanner as scanning
from .transport.session import DEFAULT_COMMAND_VALUE, BedditSession

DEFAULT_COMMAND = f"0x{int(Handle.CONTROL_POINT):02x}:{CONTROL_START.hex()}"

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
LOGGER = logging.getLogger("sleepy")

DATA_DIR = Path("data")
GATT_DIR = DATA_DIR / "gatt"
RAW_DIR = DATA_DIR / "raw"
DEVICE_FILE = DATA_DIR / "device.json"


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _remember_device(address: str, name: str | None) -> None:
    """Store the last sensor we talked to, so later commands need no --address."""
    DEVICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEVICE_FILE.write_text(json.dumps({"address": address, "name": name}, indent=2))


def _known_address() -> str | None:
    if DEVICE_FILE.exists():
        try:
            return json.loads(DEVICE_FILE.read_text()).get("address")
        except json.JSONDecodeError:
            return None
    return None


def _resolve_address(address: str | None) -> str | None:
    return address or _known_address()


def _parse_command_spec(spec: str) -> tuple[int, bytes]:
    """Parse ``HANDLE:VALUE`` where value is hex (``01``) or ASCII (``INFO``)."""
    handle_text, _, value_text = spec.partition(":")
    handle = int(handle_text, 0)
    if not value_text:
        return handle, DEFAULT_COMMAND_VALUE
    return handle, _parse_value(value_text)


@app.command()
def scan(
    timeout: float = typer.Option(10.0, "--timeout", "-t", help="Scan duration in seconds."),
    show_all: bool = typer.Option(False, "--all", "-a", help="Show every device, not only Beddit."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Scan for BLE devices and report Beddit sensors."""
    _setup_logging(verbose)
    devices = asyncio.run(scanning.scan(timeout=timeout))
    shown = devices if show_all else [d for d in devices if d.is_beddit]
    if not shown:
        typer.echo("No Beddit sensor found. Is it powered?")
        raise typer.Exit(code=1)
    for device in shown:
        marker = "*" if device.is_beddit else " "
        typer.echo(f"{marker} {device}")
        if device.is_beddit or show_all:
            typer.echo(f"    uuids={device.service_uuids} mfg={device.manufacturer_data}")
    for device in shown:
        if device.is_beddit:
            _remember_device(device.address, device.name)
            typer.echo(f"\nSaved {device.address} as the default sensor.")
            break


@app.command()
def gatt(
    address: Optional[str] = typer.Option(None, "--address", "-a", help="Sensor MAC address."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Where to write the JSON dump."),
    no_read: bool = typer.Option(False, "--no-read", help="Skip reading characteristic values."),
    timeout: float = typer.Option(30.0, "--timeout", help="Connection timeout in seconds."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Connect and dump every service, characteristic, descriptor and value."""
    _setup_logging(verbose)

    async def run() -> dict:
        session = BedditSession(_resolve_address(address), timeout=timeout)
        await session.connect()
        try:
            gatt_map = await session.gatt_map(read_values=not no_read)
        finally:
            await session.disconnect()
        return gatt_map

    gatt_map = asyncio.run(run())
    _remember_device(gatt_map["address"], gatt_map.get("name"))

    for service in gatt_map["services"]:
        typer.echo(f"\nservice {service['uuid']}  handle=0x{service['handle']:04x}  {service['description']}")
        for char in service["characteristics"]:
            props = ",".join(char["properties"])
            line = f"  char  0x{char['handle']:04x} {char['uuid']} [{props}] {char['description']}"
            typer.echo(line)
            if "value_hex" in char:
                typer.echo(f"        value={char['value_hex']}  ascii={char['value_ascii']!r}")
            if "read_error" in char:
                typer.echo(f"        read failed: {char['read_error']}")
            for desc in char["descriptors"]:
                typer.echo(
                    f"      desc 0x{desc['handle']:04x} {desc['uuid']} {desc['description']}"
                    + (f" value={desc['value_hex']}" if "value_hex" in desc else "")
                )

    out = out or GATT_DIR / f"gatt_{gatt_map['address'].replace(':', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gatt_map, indent=2))
    typer.echo(f"\nGATT map written to {out}")


@app.command()
def listen(
    address: Optional[str] = typer.Option(None, "--address", "-a", help="Sensor MAC address."),
    seconds: float = typer.Option(20.0, "--seconds", "-s", help="How long to listen."),
    handle: Optional[list[int]] = typer.Option(
        None, "--handle", "-H", help="Only enable notifications on this handle (repeatable)."
    ),
    each: bool = typer.Option(
        False, "--each", help="Enable one notifiable characteristic at a time, to tell channels apart."
    ),
    command: Optional[str] = typer.Option(
        None, "--command", "-c", help="Command to start streaming, as HANDLE:VALUE (e.g. 0x0b:01)."
    ),
    no_command: bool = typer.Option(False, "--no-command", help="Do not write any start command."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Raw log path."),
    no_save: bool = typer.Option(False, "--no-save", help="Do not write a raw log."),
    show: int = typer.Option(8, "--show", "-n", help="How many payloads to print per handle."),
    timeout: float = typer.Option(30.0, "--timeout", help="Connection timeout in seconds."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Enable notifications, start the stream and dump what the sensor sends."""
    _setup_logging(verbose)
    command_handle, command_value = _parse_command_spec(command or DEFAULT_COMMAND)

    async def run() -> tuple[BedditSession, Path | None]:
        session = BedditSession(_resolve_address(address), timeout=timeout)
        await session.connect()
        log = None if no_save else rawlog.RawLogWriter(
            out or RAW_DIR / f"raw_{session.address.replace(':', '')}_{_stamp()}.bin"
        )
        shown: dict[int, int] = {}

        def on_data(handle: int, payload: bytes, timestamp: float) -> None:
            if log is not None:
                log.write(handle, payload)
            printed = shown.get(handle, 0)
            if printed < show:
                shown[handle] = printed + 1
                typer.echo(f"0x{handle:04x} {describe(payload, handle)}")

        session.add_callback(on_data)

        targets = session.notify_characteristics() if not handle else [
            session.characteristic_by_handle(h) for h in handle
        ]
        if not targets:
            typer.echo("Sensor exposes no notifiable characteristic - nothing to listen to.")
        elif each:
            for char in targets:
                typer.echo(f"\n--- listening on 0x{char.handle:04x} ({char.uuid}) only ---")
                await session.enable_notifications([char.handle])
                if not no_command:
                    await _try_command(session, command_handle, command_value)
                await asyncio.sleep(seconds)
                await session.client.stop_notify(char.handle)
        else:
            await session.enable_notifications([c.handle for c in targets])
            if not no_command:
                await _try_command(session, command_handle, command_value)
            typer.echo(f"Listening for {seconds:.0f} s...")
            await asyncio.sleep(seconds)

        if log is not None:
            log.close()
        await session.disconnect()
        return session, log.path if log is not None else None

    session, path = asyncio.run(run())
    if session.address:
        _remember_device(session.address, None)

    typer.echo("\n=== Summary ===")
    summary = session.stats.summary()
    for handle, entry in summary.items():
        typer.echo(
            f"0x{handle:04x}: {entry['count']} packets, {entry['rate_hz']} pkt/s, "
            f"lengths={entry['payload_lengths']}, {entry['duration_s']} s"
        )
        data_packets = [p for p in session.stats.last_payloads[handle] if parse(p, handle).is_pcm]
        if data_packets:
            samples_per_packet = len(parse(data_packets[0], handle).samples)
            rate = entry["rate_hz"]
            hint = f"  -> looks like sensor data: {samples_per_packet} int16 samples/packet"
            if rate:
                hint += f", ~{rate * samples_per_packet:.0f} samples/s"
            typer.echo(hint)
    if path is not None:
        typer.echo(f"\nRaw log: {path}")


def _parse_value(text: str) -> bytes:
    """Accept hex (``01``, ``0x01``) or ASCII (``INFO``) command payloads."""
    if text.lower().startswith("0x"):
        return bytes.fromhex(text[2:])
    try:
        return bytes.fromhex(text)
    except ValueError:
        return text.encode("ascii")


@app.command()
def info(
    address: Optional[str] = typer.Option(None, "--address", "-a"),
    timeout: float = typer.Option(30.0, "--timeout"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show what the sensor says about itself: identity, channels, settings."""
    _setup_logging(verbose)

    async def run() -> dict:
        session = BedditSession(_resolve_address(address), timeout=timeout)
        await session.connect()
        try:
            return await session.sensor_info()
        finally:
            await session.disconnect()

    sensor_info = asyncio.run(run())
    width = max(len(key) for key in sensor_info)
    for key, value in sensor_info.items():
        typer.echo(f"{key:<{width}} : {value}")


@app.command("probe-cmd")
def probe_cmd(
    handle: str = typer.Option("0x0a", "--handle", "-H", help="Characteristic handle to write to."),
    values: str = typer.Option(
        "01,01000000,0100000000000000,02,ffff,0000",
        "--values",
        help="Comma separated payloads to try; hex or ASCII.",
    ),
    wait: float = typer.Option(4.0, "--wait", "-w", help="Seconds to watch for data after each write."),
    address: Optional[str] = typer.Option(None, "--address", "-a"),
    show: int = typer.Option(3, "--show", "-n", help="Payloads to print per positive result."),
    reread: bool = typer.Option(True, "--reread/--no-reread", help="Read the handle back after each write."),
    timeout: float = typer.Option(30.0, "--timeout"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Try candidate command payloads on a handle and report which ones start data."""
    _setup_logging(verbose)
    candidates = [v.strip() for v in values.split(",") if v.strip()]
    target = int(handle, 0)

    async def run() -> None:
        session = BedditSession(_resolve_address(address), timeout=timeout)
        await session.connect()
        printed: dict[int, int] = {}

        def on_data(char_handle: int, payload: bytes, timestamp: float) -> None:
            printed[char_handle] = printed.get(char_handle, 0)
            if printed[char_handle] < show:
                printed[char_handle] += 1
                typer.echo(f"      0x{char_handle:04x} {describe(payload, char_handle)}")

        session.add_callback(on_data)
        enabled = await session.enable_notifications()
        typer.echo(f"Notifications enabled on {[hex(h) for h in enabled]}")
        try:
            for candidate in candidates:
                payload = _parse_value(candidate)
                before = dict(session.stats.counts)
                typer.echo(f"\nwrite 0x{target:02x} <- {payload.hex()}  ({candidate})")
                try:
                    await session.write_command(target, payload)
                except Exception as error:  # noqa: BLE001
                    typer.echo(f"  FAILED: {type(error).__name__}: {error}")
                    continue
                if reread:
                    try:
                        value = bytes(await session.client.read_gatt_char(target))
                        typer.echo(f"  read back: {value.hex()}")
                    except Exception as error:  # noqa: BLE001
                        typer.echo(f"  read back failed: {type(error).__name__}: {error}")
                await asyncio.sleep(wait)
                delta = {h: c - before.get(h, 0) for h, c in session.stats.counts.items() if c - before.get(h, 0)}
                if delta:
                    typer.echo(f"  >>> DATA: { {hex(h): n for h, n in delta.items()} }")
                    for char_handle in delta:
                        for payload_seen in session.stats.last_payloads[char_handle][-show:]:
                            typer.echo(f"      0x{char_handle:04x} {describe(payload_seen, char_handle)}")
                else:
                    typer.echo("  no data")
        finally:
            await session.disconnect()

    asyncio.run(run())


async def _try_command(session: BedditSession, handle: int, value: bytes) -> None:
    try:
        await session.write_command(handle, value)
    except Exception as error:  # noqa: BLE001 - probing: a bad guess must not abort
        typer.echo(f"  command 0x{handle:02x}<-{value.hex()} failed: {type(error).__name__}: {error}")


def _stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


@app.command("log-info")
def log_info(
    path: Path = typer.Argument(..., help="A raw log written by 'listen'."),
    show: int = typer.Option(5, "--show", "-n", help="How many records to print."),
) -> None:
    """Print statistics about a captured raw log."""
    summary = rawlog.raw_log_summary(path)
    typer.echo(f"{path}: {summary['records']} records")
    for handle, entry in summary["handles"].items():
        typer.echo(
            f"0x{handle:04x}: {entry['count']} packets, {entry['rate_hz']} pkt/s, "
            f"lengths={entry['lengths']}, {entry['bytes']} B, {entry['duration_s']} s"
        )
    for record in rawlog.read_raw_log(path)[:show]:
        typer.echo(
            f"  t={record.timestamp:.3f} 0x{record.handle:04x} {describe(record.payload, record.handle)}"
        )


@app.command()
def analyse(
    path: Path = typer.Argument(..., help="Raw log written by 'listen' or 'record'."),
    plot: Optional[Path] = typer.Option(None, "--plot", "-p", help="Write an overview PNG here."),
    pcm1_rate: float = typer.Option(140.0, help="PCM channel 1 sample rate, Hz."),
    pcm2_rate: float = typer.Option(5.0, help="PCM channel 2 sample rate, Hz."),
    seconds: int = typer.Option(20, "--seconds", "-s", help="Per-second table rows to print."),
) -> None:
    """Decode a raw log and report what is in it."""
    records = rawlog.read_raw_log(path)
    by_handle: dict[int, list[tuple[float, bytes]]] = {}
    for record in records:
        by_handle.setdefault(record.handle, []).append((record.timestamp, record.payload))

    typer.echo(f"{path}: {len(records)} notifications")
    for handle, items in sorted(by_handle.items()):
        lengths = sorted({len(p) for _, p in items})
        typer.echo(f"  0x{handle:04x}: {len(items)} packets, payload lengths {lengths}")

    environment = []
    for record in records:
        packet = parse(record.payload, record.handle)
        if packet.kind in ("temperature", "humidity"):
            environment.append((record.timestamp, packet.kind, packet.value))
        if packet.kind == "flags":
            environment.append((record.timestamp, "flags", packet.value))
    for timestamp, kind, value in environment:
        typer.echo(f"  {kind:<11} {value}")

    pcm1_records = by_handle.get(int(Handle.PCM1_DATA), [])
    pcm2_records = by_handle.get(int(Handle.PCM2_DATA), [])
    if not pcm1_records:
        typer.echo("\nNo PCM channel 1 data in this log.")
        raise typer.Exit(code=1)

    channel = decode(pcm1_records, int(Handle.PCM1_DATA), pcm1_rate, PCM1_SAMPLES_PER_PACKET, "pcm1 140 Hz force")
    typer.echo(f"\nPCM1 {channel.stats()}")
    slow = (
        decode(pcm2_records, int(Handle.PCM2_DATA), pcm2_rate, PCM2_SAMPLES_PER_PACKET, "pcm2 5 Hz")
        if pcm2_records
        else None
    )
    if slow is not None:
        typer.echo(f"PCM2 {slow.stats()}")

    events = load_events(channel)
    if events:
        typer.echo("\nLoad events (signal leaving its baseline):")
        for start, end, deviation in events:
            typer.echo(f"  {start:7.1f} - {end:7.1f} s  ({end - start:.1f} s, mean deviation {deviation:+.0f})")
    else:
        typer.echo("\nNo load events: the signal stayed at its baseline.")

    frequencies, power = psd(channel)
    if frequencies.size:
        for name, band in (("breathing", (0.1, 0.6)), ("heart", (0.8, 3.5)), ("movement", (0.01, 0.1))):
            peak, peak_power = dominant_frequency(frequencies, power, *band)
            typer.echo(
                f"  {name:<9} band {band[0]}-{band[1]} Hz: "
                f"{band_power(frequencies, power, *band):6.2%} of power, peak at {peak:.3f} Hz"
            )

    if seconds and channel.values.size:
        typer.echo(f"\nPer second, first {seconds} s (relative to the first sample):")
        typer.echo("     t     mean      std      min      max")
        rate = int(round(channel.sample_rate_hz))
        for second in range(min(seconds, int(channel.duration_s))):
            chunk = channel.values[second * rate : (second + 1) * rate]
            finite = chunk[np.isfinite(chunk)]
            if finite.size == 0:
                continue
            typer.echo(
                f"  {second:4d} {finite.mean():9.1f} {finite.std():8.2f} "
                f"{finite.min():9.0f} {finite.max():9.0f}"
            )

    if plot:
        written = plot_overview(channel, plot, events=events, slow_channel=slow, title=str(path))
        typer.echo(f"\nPlot written to {written}")


@app.command()
def record(
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Raw log path."),
    address: Optional[str] = typer.Option(None, "--address", "-a"),
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Stop after N seconds."),
    stall_timeout: float = typer.Option(
        20.0, "--stall-timeout", help="Re-arm the sensor after this many seconds without data."
    ),
    flush_every: float = typer.Option(5.0, "--flush-every", help="Flush the log every N seconds."),
    timeout: float = typer.Option(30.0, "--timeout"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Record overnight: keep the stream alive, re-arming and reconnecting as needed."""
    _setup_logging(verbose)

    async def run() -> dict:
        session = BedditSession(_resolve_address(address), timeout=timeout)
        await session.connect()
        info = await session.sensor_info()
        log = rawlog.RawLogWriter(
            out or RAW_DIR / f"night_{session.address.replace(':', '')}_{_stamp()}.bin"
        )
        meta = {
            "address": session.address,
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sensor": info,
            "raw_log": str(log.path),
        }
        log.path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

        state = {"last_packet": time.monotonic(), "last_flush": time.monotonic(), "stalls": 0}

        def on_data(handle: int, payload: bytes, timestamp: float) -> None:
            log.write(handle, payload)
            state["last_packet"] = time.monotonic()

        session.add_callback(on_data)
        await session.enable_notifications()
        await session.start_stream()

        started = time.monotonic()
        try:
            while True:
                await asyncio.sleep(1.0)
                now = time.monotonic()
                if duration is not None and now - started >= duration:
                    break
                if now - state["last_flush"] >= flush_every:
                    log.flush()
                    state["last_flush"] = now
                if now - state["last_packet"] >= stall_timeout:
                    state["stalls"] += 1
                    LOGGER.warning(
                        "No data for %.0f s (stall %d); re-arming the sensor",
                        now - state["last_packet"],
                        state["stalls"],
                    )
                    try:
                        await session.start_stream()
                    except Exception as error:  # noqa: BLE001
                        LOGGER.warning("Re-arm failed: %s", error)
                    state["last_packet"] = now
                    if state["stalls"] % 3 == 0:
                        LOGGER.warning("Reconnecting...")
                        try:
                            await session.disconnect()
                        except Exception:  # noqa: BLE001
                            pass
                        await session.connect()
                        session.add_callback(on_data)
                        await session.enable_notifications()
                        await session.start_stream()
                        state["last_packet"] = time.monotonic()
        finally:
            LOGGER.info("Stopping; %d packets written to %s", log.records, log.path)
            log.close()
            with contextlib.suppress(Exception):
                await session.stop_stream()
            await session.disconnect()
        return {"packets": log.records, "path": str(log.path), "stalls": state["stalls"]}

    try:
        result = asyncio.run(run())
    except KeyboardInterrupt:
        typer.echo("\nStopped by user.")
        return
    typer.echo(f"Recorded {result['packets']} packets to {result['path']} ({result['stalls']} stalls)")


@app.command()
def process(
    path: Path = typer.Argument(..., help="Raw log written by 'record'."),
    out_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Where to write results."),
    epoch_s: float = typer.Option(30.0, "--epoch", help="Epoch length in seconds."),
    pcm1_rate: float = typer.Option(140.0, help="PCM channel 1 sample rate, Hz."),
) -> None:
    """Decode a recording and estimate breathing and heart rate per epoch."""
    records = rawlog.read_raw_log(path)
    by_handle: dict[int, list[tuple[float, bytes]]] = {}
    for record in records:
        by_handle.setdefault(record.handle, []).append((record.timestamp, record.payload))

    channel = decode(
        by_handle.get(int(Handle.PCM1_DATA), []),
        int(Handle.PCM1_DATA),
        pcm1_rate,
        PCM1_SAMPLES_PER_PACKET,
        "pcm1",
    )
    if channel.values.size == 0:
        typer.echo("No PCM1 data in this log.")
        raise typer.Exit(code=1)

    epochs = process_channel(channel.times, channel.values, channel.sample_rate_hz, epoch_s=epoch_s)
    summary = summarise(epochs)
    out_dir = out_dir or DATA_DIR / "nights" / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    table = [e.as_dict() for e in epochs]
    csv_path = out_dir / "metrics.csv"
    if table:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "source": str(path),
                "sample_rate_hz": round(channel.sample_rate_hz, 3),
                "duration_s": round(channel.duration_s, 1),
                "packet_loss": f"{channel.loss_ratio:.3%}",
                "channel": channel.stats(),
                "summary": summary,
            },
            indent=2,
        )
    )
    plot_night(
        np.array([e.start_s for e in epochs]),
        np.array([e.respiration_bpm if e.respiration_bpm is not None else np.nan for e in epochs]),
        np.array([e.heart_bpm if e.heart_bpm is not None else np.nan for e in epochs]),
        np.array([e.movement for e in epochs]),
        out_dir / "overview.png",
        title=f"{path.name}: breathing, heart rate and movement",
    )

    typer.echo(
        f"Decoded {channel.values.size} samples at {channel.sample_rate_hz:.2f} Hz "
        f"({channel.duration_s:.0f} s, packet loss {channel.loss_ratio:.2%})"
    )
    for key, value in summary.items():
        typer.echo(f"  {key:<24} {value}")
    typer.echo(f"\nWrote {csv_path}\nWrote {out_dir / 'summary.json'}\nWrote {out_dir / 'overview.png'}")


@app.command()
def report(
    night: Optional[Path] = typer.Argument(
        None, help="Processed night directory; omit to rebuild every night and the index."
    ),
    nights_dir: Path = typer.Option(DATA_DIR / "nights", "--dir", help="Where processed nights live."),
) -> None:
    """Build self contained HTML reports (no server needed)."""
    targets = [night] if night else [p for p in nights_dir.iterdir() if p.is_dir()]
    built = []
    for target in targets:
        if not (target / "summary.json").exists():
            typer.echo(f"skipping {target}: no summary.json (run 'process' first)")
            continue
        built.append(html.render_night(target))
    index = html.render_index(nights_dir)
    for path in built:
        typer.echo(f"Wrote {path}")
    typer.echo(f"Wrote {index}")


if __name__ == "__main__":
    app()
