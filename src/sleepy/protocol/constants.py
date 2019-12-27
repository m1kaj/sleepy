"""Constants for the Beddit V3 sensor, as discovered from our own unit.

Discovered with ``sleepy gatt`` on sensor ``Beddit 3912`` (MAC C0:53:22:98:1B:B8):

* Device Information: manufacturer ``Beddit``, model ``V3 Sensor``,
  hardware ``3.0``, software ``1.1/4a2417b``.
* Vendor service ``e6807d20-...`` with a "Control Point" characteristic and two
  "PCM channel" data characteristics. Each PCM channel has a description
  characteristic whose payload is ``uint16 sample_rate_in_centihz`` followed by
  an ASCII tag, e.g. ``b0 36 "u2force"`` -> 0x36b0 = 14000 = 140.00 Hz.

Handles are stable for one unit but may differ between firmware versions, so
code should prefer looking characteristics up by UUID.
"""

from __future__ import annotations

from enum import IntEnum

#: Vendor specific service that carries the sensor data.
VENDOR_SERVICE_UUID = "e6807d20-b90a-11e5-a837-0800200c9a66"

ENVIRONMENTAL_SENSING_UUID = "0000181a-0000-1000-8000-00805f9b34fb"
DEVICE_INFORMATION_UUID = "0000180a-0000-1000-8000-00805f9b34fb"


class UUID(str):
    """Namespace of the vendor characteristic UUIDs (all share the base)."""

    BASE = "e6807d20-b90a-11e5-a837-0800200c9a66"
    CONTROL_POINT = "e6807d21-b90a-11e5-a837-0800200c9a66"
    FLAGS = "e6807d22-b90a-11e5-a837-0800200c9a66"
    TOUCH_FACTORY_CALIB = "e6807d23-b90a-11e5-a837-0800200c9a66"
    TOUCH_BASE_LINE = "e6807d24-b90a-11e5-a837-0800200c9a66"
    TOUCH_THRESHOLD = "e6807d25-b90a-11e5-a837-0800200c9a66"
    MOVEMENT_THRESHOLD = "e6807d26-b90a-11e5-a837-0800200c9a66"
    MOVEMENT_PERIOD = "e6807d27-b90a-11e5-a837-0800200c9a66"
    PCM1_DATA = "e6807e20-b90a-11e5-a837-0800200c9a66"
    PCM1_DESCRIPTION = "e6807e21-b90a-11e5-a837-0800200c9a66"
    PCM2_DATA = "e6807e24-b90a-11e5-a837-0800200c9a66"
    PCM2_DESCRIPTION = "e6807e25-b90a-11e5-a837-0800200c9a66"


class Handle(IntEnum):
    """Attribute handles on our unit, kept for reference and quick experiments."""

    DEVICE_NAME = 0x0002
    CONTROL_POINT = 0x000A
    FLAGS = 0x000D
    TOUCH_FACTORY_CALIB = 0x0011
    TOUCH_BASE_LINE = 0x0014
    TOUCH_THRESHOLD = 0x0018
    MOVEMENT_THRESHOLD = 0x001B
    MOVEMENT_PERIOD = 0x001E
    PCM1_DATA = 0x0021
    PCM1_DESCRIPTION = 0x0025
    PCM2_DATA = 0x0028
    PCM2_DESCRIPTION = 0x002C
    TEMPERATURE = 0x0030
    HUMIDITY = 0x0034
    MANUFACTURER_NAME = 0x0039
    MODEL_NUMBER = 0x003B
    SERIAL_NUMBER = 0x003D
    HARDWARE_REVISION = 0x003F
    FIRMWARE_REVISION = 0x0041
    SOFTWARE_REVISION = 0x0043


#: Write this to the control point to start streaming.
CONTROL_START = b"\x01"

#: Value we believe stops streaming (0x02 was rejected by the device).
CONTROL_STOP = b"\x00"

#: Descriptions read from the device (rate in 1/100 Hz + ASCII tag).
PCM1_SAMPLE_RATE_HZ = 140.0
PCM2_SAMPLE_RATE_HZ = 5.0

#: Samples carried in one notification of each channel.
PCM1_SAMPLES_PER_PACKET = 9
PCM2_SAMPLES_PER_PACKET = 1
