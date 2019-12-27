"""Packet decoding tests, using payloads captured from the real sensor."""

from sleepy.protocol.constants import Handle
from sleepy.protocol.packets import describe, parse, sequence_gap

OUR_PACKET = bytes.fromhex("00d27a977d87862179e0815085ec7855855082")
ARTICLE_PACKET = bytes.fromhex("8000a6789f78a178b078ad78b778ce78d178e278")
SLOW_PACKET = bytes.fromhex("00e824")


def test_our_packet_has_no_magic_and_nine_samples():
    packet = parse(OUR_PACKET, Handle.PCM1_DATA)
    assert packet.kind == "pcm"
    assert packet.sequence == 0
    assert len(packet.samples) == 9
    assert packet.samples[0] == 0x7AD2
    assert packet.samples[1] == 0x7D97


def test_article_packet_format_is_also_supported():
    packet = parse(ARTICLE_PACKET, Handle.PCM1_DATA)
    assert packet.kind == "pcm"
    assert packet.sequence == 0
    assert packet.samples[0] == 0x78A6
    assert len(packet.samples) == 9


def test_slow_channel_packet():
    packet = parse(SLOW_PACKET, Handle.PCM2_DATA)
    assert packet.sequence == 0
    assert packet.samples == (0x24E8,)


def test_temperature_and_humidity():
    assert parse(bytes.fromhex("5c0a"), Handle.TEMPERATURE).value == 26.52
    assert parse(bytes.fromhex("7f0d"), Handle.HUMIDITY).value == 34.55


def test_flags():
    assert parse(bytes.fromhex("02"), Handle.FLAGS).value == 2


def test_sequence_gap():
    assert sequence_gap(0, 1) == 0
    assert sequence_gap(0, 3) == 2
    assert sequence_gap(255, 1) == 1  # wraps around
    assert sequence_gap(None, 5) == 0


def test_describe_mentions_samples():
    assert "seq=  0" in describe(OUR_PACKET, Handle.PCM1_DATA)
    assert "temperature=27.24C" == describe(bytes.fromhex("a40a"), Handle.TEMPERATURE)
