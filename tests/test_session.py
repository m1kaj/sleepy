"""Session callback bookkeeping (no BLE hardware needed)."""

import asyncio

from sleepy.transport.session import BedditSession


class _Characteristic:
    handle = 0x0021


def test_add_callback_is_idempotent():
    """Adding the same callback after a reconnect must not duplicate packets."""
    session = BedditSession("C0:53:22:98:1B:B8")
    received = []

    def callback(handle, payload, timestamp):
        received.append((handle, payload))

    session.add_callback(callback)
    session.add_callback(callback)
    session._dispatch(_Characteristic(), bytearray(b"abc"))

    assert received == [(0x0021, b"abc")]
    assert session.stats.counts[0x0021] == 1


def test_remove_callback():
    session = BedditSession("C0:53:22:98:1B:B8")
    received = []

    def callback(handle, payload, timestamp):
        received.append(handle)

    session.add_callback(callback)
    session.remove_callback(callback)
    session._dispatch(_Characteristic(), bytearray(b"abc"))

    assert received == []


def test_disconnect_callback_accepts_the_bleak_client_argument():
    """bleak passes the client as a positional argument via functools.partial."""
    called = []
    session = BedditSession("C0:53:22:98:1B:B8", disconnected_callback=lambda: called.append(True))
    session._on_disconnect(object())

    assert session.disconnected.is_set()
    assert called == [True]


class _FakeClient:
    def __init__(self, connected: bool, services=None):
        self.is_connected = connected
        self.services = services
        self.calls = []

    async def stop_notify(self, characteristic):
        self.calls.append(("stop_notify", characteristic.handle))

    async def disconnect(self):
        self.calls.append(("disconnect",))


class _FakeServices:
    def __init__(self, *characteristics):
        self._characteristics = {c.handle: c for c in characteristics}

    def get_characteristic(self, handle):
        return self._characteristics.get(handle)


def test_disconnect_releases_the_bus_even_when_already_disconnected():
    """Skipping client.disconnect() leaves StartNotify alive and doubles packets."""
    session = BedditSession("C0:53:22:98:1B:B8")
    client = _FakeClient(connected=False)
    session.client = client

    asyncio.run(session.disconnect())

    assert ("disconnect",) in client.calls
    assert session.client is None


def test_disconnect_stops_notifications_before_releasing_the_bus():
    session = BedditSession("C0:53:22:98:1B:B8")
    characteristic = _Characteristic()
    client = _FakeClient(connected=True, services=_FakeServices(characteristic))
    session.client = client
    session._notified.add(characteristic.handle)

    asyncio.run(session.disconnect())

    assert client.calls == [("stop_notify", characteristic.handle), ("disconnect",)]
