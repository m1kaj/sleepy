"""
Receives data from a Beddit sleep sensor.

Tested with a single Beddit B3 device - not guaranteed to work with other sensors!

Performs a Bluetooth scan looking for Beddit sensor. Connects to sensor and
starts to receive data notifications. Data is written to file on disk.
Keeps receiving until there has not been new data notifications during last
30 seconds.
"""

import time
import sys
import logging
from datetime import datetime
from bluepy.btle import Scanner, DefaultDelegate, Peripheral, BTLEException


VALUE_LOCAL_NAME = 0x9      # advertising data index for device name
VALUE_PRESSURE_DATA = 0x22  # data notification handle


class BedditDelegate(DefaultDelegate):
    """Handles scanner events and asynchronous messages from connected
    peripheral. Created from DefaultDelegate base class.
    """
    def __init__(self, data_file):
        DefaultDelegate.__init__(self)
        self.data_file = data_file

    def handleDiscovery(self, device, isNewDev, isNewData):
        """New device is discovered while a Scanner object is active"""
        if isNewDev or isNewData:
            name = device.getValueText(VALUE_LOCAL_NAME)
            logging.info(f"Discovered {name}, MAC: {device.addr}, RSSI: {device.rssi}")

    def handleNotification(self, cHandle, data):
        """Writes data from notification to file, with timestamp"""
        if cHandle == VALUE_PRESSURE_DATA:
            self.data_file.write(f"{datetime.now().timestamp()} data.hex()")


class BedditWriter:
    """Connects to Beddit sensor and starts receiving data and writing
    it to a file with timestamp. Input argument is data file name.
    """
    def __init__(self, data_file):
        self.data_file = data_file
        self.sensor_device = None
        self.sensor_peripheral = None

    def scan(self):
        """Performs a default 10 second scan. Selects the first found device that has
        a name starting with 'Beddit'. Returns True if scan was
        successful.
        """
        scanner = Scanner().withDelegate(BedditDelegate(self.data_file))
        results = scanner.scan()
        beddits = [r for r in results if r.getValueText(VALUE_LOCAL_NAME).startswith("Beddit")]
        if len(beddits) < 1:
            logging.error("Scan did not find a device with name that starts with 'Beddit'")
            return False
        elif len(beddits) > 1:
            logging.warning("Scan found {len(beddits)} devices:")
            for device in beddits:
                logging.warning("{device.getValueTExt(VALUE_LOCAL_NAME)}")
            logging.warning("Selecting the first device on list!")
        else:
            logging.info(f"Scan found device {beddits[0].getValueTExt(VALUE_LOCAL_NAME)}")
        self.sensor_device = beddits[0]
        return True

    def receive_until_disconnect(self):
        """Starts receiving data and writing it to file until data notifications
        from sensor stop coming-
        """
        self._get_peripheral()
        self._write_magic_bytes()

    def _get_peripheral(self):
        """Register delegate for notifications.
        Return True if successful.
        """
        self.sensor_peripheral = Peripheral(self.sensor_device)
        self.sensor_peripheral.withDelegate(DataDelegate(self.data_file))

    def _write_magic_bytes(self):
        """Writes characteristics to the device in order to start receiving data from it.

        The bytes written were found by taking a BT packet log from Beddit app in Android phone.
        This method was shown to be working in
        https://medium.com/inmoodforlife/beddit-reverse-engineering-c4bdca8e07b8
        It appears there may be differences based on Beddit sensor version.
        These values work for my sensor.
        """
        # We don't bother setting constants, just go with these random byte values.
        # Some day we should look up the values in BTLE specification.
        try:
            characteristics = self.sensor_peripheral.getCharacteristics()
            self.sensor_peripheral.writeCharacteristic(0x10, bytes([0x1, 0x0]), withResponse=True)
            time.sleep(0.1)
            self.sensor_peripheral.writeCharacteristic(0x24, bytes([0x1, 0x0]), withResponse=True)
            time.sleep(0.1)
            self.sensor_peripheral.writeCharacteristic(0x2b, bytes([0x1, 0x0]), withResponse=True)
            time.sleep(0.1)
            for c in characteristics:
                if c.getHandle == 0x0b:
                    c.write(bytes([0x01]))
            logging.info("Finished writing magic bytes; now hope and pray!")
        except BTLEException as e:
            logging.error(f"Caught BTLEException {e.code}: {e.message}")
            sys.exit(e.code)
