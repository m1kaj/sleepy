"""Connection handling for the Beddit sensor.

Wraps :class:`bleak.BleakClient` with the bits this project needs: a GATT map
dump (services, characteristics, descriptors, handles and readable values),
notification enabling and the vendor "start measuring" command.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from bleak import BleakClient, BleakScanner

from ..protocol.channels import parse_channel_description
from ..protocol.constants import CONTROL_START, CONTROL_STOP, UUID, Handle
from .scanner import find_beddit

LOGGER = logging.getLogger(__name__)

#: The "start measuring" command goes to the vendor Control Point. The legacy
#: code guessed 0x000b, which is actually a descriptor of the Control Point at
#: 0x000a, so its start command never reached the sensor.
DEFAULT_COMMAND_HANDLE = int(Handle.CONTROL_POINT)

#: Value written to the control point to start streaming.
DEFAULT_COMMAND_VALUE = CONTROL_START

NotifyCallback = Callable[[int, bytes, float], None]
DisconnectCallback = Callable[[], None]


@dataclass
class NotificationStats:
    """Counters kept while notifications arrive."""

    counts: dict[int, int] = field(default_factory=dict)
    lengths: dict[int, set[int]] = field(default_factory=dict)
    first_ts: dict[int, float] = field(default_factory=dict)
    last_ts: dict[int, float] = field(default_factory=dict)
    last_payloads: dict[int, list[bytes]] = field(default_factory=dict)

    def record(self, handle: int, payload: bytes, timestamp: float, keep: int = 5) -> None:
        self.counts[handle] = self.counts.get(handle, 0) + 1
        self.lengths.setdefault(handle, set()).add(len(payload))
        self.first_ts.setdefault(handle, timestamp)
        self.last_ts[handle] = timestamp
        payloads = self.last_payloads.setdefault(handle, [])
        payloads.append(payload)
        del payloads[:-keep]

    def rate(self, handle: int) -> float | None:
        duration = self.duration(handle)
        return self.counts[handle] / duration if duration else None

    def duration(self, handle: int) -> float:
        if handle not in self.first_ts:
            return 0.0
        return self.last_ts[handle] - self.first_ts[handle]

    def summary(self) -> dict[int, dict]:
        return {
            handle: {
                "count": count,
                "rate_hz": round(rate, 2) if (rate := self.rate(handle)) else None,
                "duration_s": round(self.duration(handle), 3),
                "payload_lengths": sorted(self.lengths[handle]),
            }
            for handle, count in sorted(self.counts.items())
        }


class BedditSession:
    """A BLE connection to one Beddit sensor."""

    def __init__(
        self,
        address: str | None = None,
        adapter: str | None = None,
        timeout: float = 30.0,
        disconnected_callback: DisconnectCallback | None = None,
    ) -> None:
        self.address = address
        self.adapter = adapter
        self.timeout = timeout
        self.client: BleakClient | None = None
        self.stats = NotificationStats()
        self._callbacks: list[NotifyCallback] = []
        self._disconnected_callback = disconnected_callback
        self._notified: set[int] = set()
        self._closing = False
        self.connections = 0
        #: Set as soon as BlueZ reports the link down. The recorder waits on it
        #: instead of discovering the loss through a 20 s notification timeout.
        self.disconnected = asyncio.Event()
        #: Cached BlueZ device. Reusing it lets us reconnect without scanning
        #: (the sensor stops advertising while it is connected).
        self.device = None

    @property
    def is_connected(self) -> bool:
        return self.client is not None and self.client.is_connected

    # -- connection ---------------------------------------------------------

    async def connect(self, scan_timeout: float = 10.0, retries: int = 3) -> str:
        """Connect to the sensor, scanning for it first.

        BlueZ forgets devices it has not bonded, so we look the device up
        immediately before connecting and retry a couple of times; connecting
        straight to a stale D-Bus path fails with "device not found".

        On a reconnect the cached :class:`BLEDevice` is reused, so BlueZ can
        page the sensor directly instead of waiting for it to advertise again.
        If that fails (the D-Bus object is gone), we fall back to a scan.
        """
        if self.address is None:
            found = await find_beddit(timeout=scan_timeout)
            self.address = found.address
            self.device = found.ble_device
            LOGGER.info("Using discovered sensor %s (%s)", found.name, found.address)

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            device = self.device
            if device is None:
                device = await BleakScanner.find_device_by_address(
                    self.address, timeout=scan_timeout
                )
            if device is None:
                last_error = RuntimeError(f"{self.address} not found by scanner")
                LOGGER.warning("Attempt %d/%d: %s", attempt, retries, last_error)
                await asyncio.sleep(1.0)
                continue
            LOGGER.info("Connecting to %s (attempt %d/%d)", self.address, attempt, retries)
            client = BleakClient(
                device,
                timeout=self.timeout,
                disconnected_callback=self._on_disconnect,
            )
            try:
                await client.connect()
            except Exception as error:  # noqa: BLE001 - retried below
                last_error = error
                LOGGER.warning("Attempt %d/%d failed: %s", attempt, retries, error)
                # A cached BlueZ path can disappear; scan again next time.
                self.device = None
                await asyncio.sleep(1.0)
                continue
            self.client = client
            self.device = device
            self._notified.clear()
            self.disconnected.clear()
            self.connections += 1
            LOGGER.info("Connected; MTU = %s", client.mtu_size)
            return self.address

        raise RuntimeError(f"Could not connect to {self.address}: {last_error}")

    def _on_disconnect(self, _client: object | None = None) -> None:
        """Called by bleak (from the event loop) when the link goes down."""
        if not self._closing:
            LOGGER.warning("Sensor disconnected")
        self.disconnected.set()
        if self._disconnected_callback is not None:
            try:
                self._disconnected_callback()
            except Exception:  # noqa: BLE001 - a callback must not kill the loop
                LOGGER.exception("Disconnect callback failed")

    async def disconnect(self) -> None:
        """Close the link, ignoring errors from a link that already died.

        ``bleak.BleakClient.disconnect()`` must be called even when the client
        already reports ``is_connected == False``.  bleak only releases its
        private D-Bus connection in ``disconnect()``; if an unexpected link
        loss short-circuits this, the old client's ``StartNotify`` subscriptions
        stay alive in BlueZ.  The next connection then receives every
        notification twice, which is exactly the 2x packet rate seen after a
        reconnect in the 15 minute capture.
        """
        self._closing = True
        client, self.client = self.client, None
        if client is None:
            self._notified.clear()
            self._closing = False
            return
        try:
            if client.is_connected and client.services is not None:
                # Best effort: explicitly remove subscriptions while the link
                # is still up.  A failure here must not prevent the bus release.
                for handle in list(self._notified):
                    characteristic = client.services.get_characteristic(handle)
                    if characteristic is None:
                        continue
                    try:
                        await client.stop_notify(characteristic)
                    except Exception:  # noqa: BLE001 - link may already be gone
                        pass
            # This must run unconditionally: bleak releases its private D-Bus
            # connection here.  Skipping it when is_connected is already False
            # leaves the old StartNotify subscriptions alive in BlueZ.
            await client.disconnect()
            LOGGER.info("Disconnected")
        except Exception as error:  # noqa: BLE001 - shutting down
            LOGGER.debug("Disconnect failed: %s", error)
        finally:
            self._notified.clear()
            self._closing = False

    async def __aenter__(self) -> BedditSession:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()

    def _require_client(self) -> BleakClient:
        if self.client is None or not self.client.is_connected:
            raise RuntimeError("Not connected to the sensor")
        return self.client

    # -- discovery ----------------------------------------------------------

    @property
    def characteristics(self):
        return list(self._require_client().services.characteristics.values())

    def characteristic_by_handle(self, handle: int):
        for char in self.characteristics:
            if char.handle == handle:
                return char
        raise KeyError(f"No characteristic with handle 0x{handle:04x}")

    def notify_characteristics(self) -> list:
        return [
            c
            for c in self.characteristics
            if "notify" in c.properties or "indicate" in c.properties
        ]

    def writable_characteristics(self) -> list:
        return [
            c
            for c in self.characteristics
            if "write" in c.properties or "write-without-response" in c.properties
        ]

    async def gatt_map(self, read_values: bool = True) -> dict:
        """Dump the full GATT tree, optionally reading every readable value."""
        client = self._require_client()
        services = []
        for service in client.services.services.values():
            characteristics = []
            for char in service.characteristics:
                entry = {
                    "uuid": char.uuid,
                    "handle": char.handle,
                    "properties": list(char.properties),
                    "description": char.description,
                }
                if read_values and "read" in char.properties:
                    try:
                        value = bytes(await client.read_gatt_char(char))
                        entry["value_hex"] = value.hex()
                        entry["value_ascii"] = _to_ascii(value)
                    except Exception as error:  # noqa: BLE001 - probing: report anything
                        entry["read_error"] = f"{type(error).__name__}: {error}"
                descriptors = []
                for desc in char.descriptors:
                    desc_entry = {
                        "uuid": desc.uuid,
                        "handle": desc.handle,
                        "description": desc.description,
                    }
                    if read_values:
                        try:
                            value = bytes(await client.read_gatt_descriptor(desc))
                            desc_entry["value_hex"] = value.hex()
                        except Exception as error:  # noqa: BLE001
                            desc_entry["read_error"] = f"{type(error).__name__}: {error}"
                    descriptors.append(desc_entry)
                entry["descriptors"] = descriptors
                characteristics.append(entry)
            services.append(
                {
                    "uuid": service.uuid,
                    "handle": service.handle,
                    "description": service.description,
                    "characteristics": characteristics,
                }
            )

        return {
            "address": self.address,
            "name": getattr(client, "name", None),
            "mtu": client.mtu_size,
            "dumped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "services": services,
        }

    # -- data flow ----------------------------------------------------------

    def add_callback(self, callback: NotifyCallback) -> None:
        """Register a callback invoked as ``callback(handle, payload, timestamp)``.

        Callbacks survive reconnects and are de-duplicated by identity: older
        recorders added the same callback again after every reconnect, so each
        notification was written two, three, ... times to the raw log.
        """
        if any(existing is callback for existing in self._callbacks):
            return
        self._callbacks.append(callback)

    def remove_callback(self, callback: NotifyCallback) -> None:
        self._callbacks = [existing for existing in self._callbacks if existing is not callback]

    def _dispatch(self, char, data: bytearray) -> None:
        timestamp = time.time()
        handle = char.handle
        payload = bytes(data)
        self.stats.record(handle, payload, timestamp)
        for callback in self._callbacks:
            callback(handle, payload, timestamp)

    async def enable_notifications(self, handles: list[int] | None = None) -> list[int]:
        """Enable notifications on notifiable characteristics (all, or a subset)."""
        client = self._require_client()
        targets = self.notify_characteristics()
        if handles:
            targets = [c for c in targets if c.handle in handles]
        enabled = []
        for char in targets:
            if char.handle in self._notified:
                enabled.append(char.handle)
                continue
            await client.start_notify(char, self._dispatch)
            self._notified.add(char.handle)
            enabled.append(char.handle)
            LOGGER.info("Notifications enabled on 0x%04x (%s)", char.handle, char.uuid)
        return enabled

    async def write_command(
        self,
        handle: int = DEFAULT_COMMAND_HANDLE,
        value: bytes = DEFAULT_COMMAND_VALUE,
        response: bool = True,
    ) -> None:
        """Write the vendor command that starts the measurement stream."""
        client = self._require_client()
        LOGGER.info("Writing %s to handle 0x%04x", value.hex(), handle)
        await client.write_gatt_char(handle, value, response=response)

    def characteristic_by_uuid(self, uuid: str):
        for char in self.characteristics:
            if char.uuid.lower() == uuid.lower():
                return char
        raise KeyError(f"No characteristic with UUID {uuid}")

    async def start_stream(self) -> None:
        """Start the measurement stream (write 0x01 to the control point)."""
        await self.write_command(self._handle(UUID.CONTROL_POINT), CONTROL_START)

    async def stop_stream(self) -> None:
        """Stop the measurement stream (write 0x00 to the control point)."""
        await self.write_command(self._handle(UUID.CONTROL_POINT), CONTROL_STOP)

    def _handle(self, uuid: str, default: Handle | None = None) -> int:
        """Resolve a vendor UUID to a handle, falling back to our known map."""
        try:
            return self.characteristic_by_uuid(uuid).handle
        except KeyError:
            if default is None:
                raise
            return int(default)

    async def sensor_info(self) -> dict:
        """Read the device information, channel descriptions and current settings."""
        client = self._require_client()

        async def read(uuid: str, default: Handle) -> str | None:
            try:
                value = bytes(await client.read_gatt_char(self._handle(uuid, default)))
            except Exception as error:  # noqa: BLE001 - probing
                return f"<{type(error).__name__}: {error}>"
            try:
                return value.decode("ascii")
            except UnicodeDecodeError:
                return value.hex()

        async def read_raw(uuid: str, default: Handle) -> str | None:
            try:
                value = bytes(await client.read_gatt_char(self._handle(uuid, default)))
            except Exception as error:  # noqa: BLE001
                return f"<{type(error).__name__}: {error}>"
            return value.hex()

        descriptions = {}
        for name, uuid, default in (
            ("pcm1", UUID.PCM1_DESCRIPTION, Handle.PCM1_DESCRIPTION),
            ("pcm2", UUID.PCM2_DESCRIPTION, Handle.PCM2_DESCRIPTION),
        ):
            try:
                value = bytes(await client.read_gatt_char(self._handle(uuid, default)))
                descriptions[name] = str(parse_channel_description(value))
                descriptions[f"{name}_rate_hz"] = parse_channel_description(value).sample_rate_hz
            except Exception as error:  # noqa: BLE001
                descriptions[name] = f"<{type(error).__name__}: {error}>"

        return {
            "address": self.address,
            "mtu": client.mtu_size,
            "manufacturer": await read("00002a29-0000-1000-8000-00805f9b34fb", Handle.MANUFACTURER_NAME),
            "model": await read("00002a24-0000-1000-8000-00805f9b34fb", Handle.MODEL_NUMBER),
            "serial": await read("00002a25-0000-1000-8000-00805f9b34fb", Handle.SERIAL_NUMBER),
            "hardware_revision": await read("00002a27-0000-1000-8000-00805f9b34fb", Handle.HARDWARE_REVISION),
            "firmware_revision": await read("00002a26-0000-1000-8000-00805f9b34fb", Handle.FIRMWARE_REVISION),
            "software_revision": await read("00002a28-0000-1000-8000-00805f9b34fb", Handle.SOFTWARE_REVISION),
            "control_point": await read_raw(UUID.CONTROL_POINT, Handle.CONTROL_POINT),
            "flags": await read_raw(UUID.FLAGS, Handle.FLAGS),
            "touch_base_line": await read_raw(UUID.TOUCH_BASE_LINE, Handle.TOUCH_BASE_LINE),
            "touch_threshold": await read_raw(UUID.TOUCH_THRESHOLD, Handle.TOUCH_THRESHOLD),
            "movement_threshold": await read_raw(UUID.MOVEMENT_THRESHOLD, Handle.MOVEMENT_THRESHOLD),
            "movement_period": await read_raw(UUID.MOVEMENT_PERIOD, Handle.MOVEMENT_PERIOD),
            **descriptions,
        }


def _to_ascii(value: bytes) -> str:
    return "".join(chr(b) if 32 <= b < 127 else "." for b in value)
