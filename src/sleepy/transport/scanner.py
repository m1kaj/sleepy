"""BLE scanning: locate the Beddit sensor on the air."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from bleak import BleakScanner

LOGGER = logging.getLogger(__name__)

#: Beddit sensors advertise their short name, e.g. "Beddit 3912".
BEDDIT_NAME_PREFIX = "Beddit"


@dataclass
class DiscoveredDevice:
    """One device seen in a scan."""

    address: str
    name: str
    rssi: int | None = None
    service_uuids: list[str] = field(default_factory=list)
    manufacturer_data: dict[str, str] = field(default_factory=dict)
    service_data: dict[str, str] = field(default_factory=dict)
    is_beddit: bool = False

    def __str__(self) -> str:
        rssi = "-" if self.rssi is None else f"{self.rssi} dBm"
        return f"{self.name or '(no name)':<20} {self.address}  rssi={rssi}"


async def scan(
    timeout: float = 10.0,
    name_filter: str | None = None,
) -> list[DiscoveredDevice]:
    """Scan for ``timeout`` seconds and return every device found.

    Args:
        timeout: how long to keep scanning, in seconds.
        name_filter: if given, only devices whose name starts with this
            string are returned.
    """
    LOGGER.info("Scanning for %.1f s...", timeout)
    results = await BleakScanner.discover(timeout=timeout, return_adv=True)

    devices: list[DiscoveredDevice] = []
    for device, adv in results.values():
        name = adv.local_name or device.name or ""
        devices.append(
            DiscoveredDevice(
                address=device.address,
                name=name,
                rssi=adv.rssi,
                service_uuids=list(adv.service_uuids),
                manufacturer_data={
                    hex(key): value.hex() for key, value in adv.manufacturer_data.items()
                },
                service_data={key: value.hex() for key, value in adv.service_data.items()},
                is_beddit=name.startswith(BEDDIT_NAME_PREFIX),
            )
        )

    if name_filter:
        devices = [d for d in devices if d.name.startswith(name_filter)]

    devices.sort(key=lambda d: (not d.is_beddit, d.name))
    LOGGER.info("Scan found %d device(s)", len(devices))
    return devices


async def find_beddit(timeout: float = 10.0) -> DiscoveredDevice:
    """Scan and return the first Beddit sensor found.

    Raises:
        RuntimeError: if no Beddit sensor was seen.
    """
    devices = await scan(timeout=timeout, name_filter=BEDDIT_NAME_PREFIX)
    if not devices:
        raise RuntimeError(
            f"No device advertising a name starting with {BEDDIT_NAME_PREFIX!r} was found. "
            "Is the sensor powered?"
        )
    if len(devices) > 1:
        LOGGER.warning("Found %d Beddit devices, using %s", len(devices), devices[0].address)
    return devices[0]
