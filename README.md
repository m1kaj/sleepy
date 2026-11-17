# sleepy

Collect and analyse raw data from a **Beddit B3 / V3 Sensor** sleep
tracker over Bluetooth LE.

## History up until now
Years ago I got Beddit sleep sensor. It is a strip that goes on the bed,
under the person. It looks like this:</br>
![Beddit B3](docs/images/sensor.jpg)

Very soon after this Apple bought the company. From what I could see, the
Android app was no longer maintained.</br>I found a
[web page called InMoodForLife that showed how to receive data from the sensor.](https://medium.com/inmoodforlife/beddit-reverse-engineering-c4bdca8e07b8)

I managed to capture Bluetooth packets using Android phone developer mode.
And soon had a command that started measuremement with my version of the tracker.
But I did not know how to process the raw data. Samples arrived in batches.
I likely got both ordering and sample rate wrong, and my filtering
did not produce sensible results. Thus, as I often do, I left the
project lie on the corner of a hard drive. Abandoned it like Apple abandoned
Android users.

Recently I wanted to see what will 20€ credit and a Chinese LLM do.
Having never run an agent, I made a separate VLAN in home network,
installed clean Debian in an old laptop, and placed it alone in the VLAN to
run the agent. It has access to internet and Bluetooth. I gave it a description
what I had tried, told it to tell me when to power the sensor on/off, 
and figure out what is needed.

Python code and following sections of this README is written by LLM.
The result of one session.</br>A short 5 minute test gave plausible values for
breathing and heart rate. Longer run failed with connection error early on.
So there is still work to do.

*(2026-11-17: the reconnect, duplicate-packet and decoder bugs behind that
failure are fixed; see "Reconnects are normal" below and `PLAN.md` 11-16.)*

**IMPORTANT** Do not run this without checking the included dependencies
first! There are some libraries that I am not familiar with and have not checked yet.
Some likely for prettifying CLI stuff. And Windows packages for no reason.


## What we know about the sensor

* Identity (Device Information service): manufacturer `Beddit`, model
  `V3 Sensor`, hardware `3.0`, software `1.1/4a2417b`.
* Two data channels, and the sensor **tells you their sample rates**:
  * PCM channel 1: `140.00 Hz` force (`u2force`) - breathing and heartbeat
  * PCM channel 2: `5.00 Hz` force/touch (`u2touch`) - slow load/presence
* Plus standard Temperature and Humidity notifications.
* Start streaming: enable notifications, then write `0x01` to the vendor
  **Control Point** characteristic.

## Install

```sh
uv sync            # creates .venv with bleak, numpy, scipy, matplotlib, typer
```

Bluetooth needs a working BlueZ setup and membership of the `bluetooth` group.

## Usage

```sh
uv run sleepy scan                 # find the sensor (remembers its address)
uv run sleepy info                 # what the sensor says about itself
uv run sleepy gatt                 # dump every service/characteristic/value
uv run sleepy listen -s 30         # short capture, prints decoded packets
uv run sleepy probe-cmd -H 0x0a -w 4 --values 01,00   # experiment with commands
uv run sleepy record -d 480 -o data/raw/test.bin      # long capture, reconnects automatically
uv run sleepy record                                 # overnight, PCM1+PCM2 (like the phone app)
uv run sleepy record --notify all                    # also record flags/temperature/humidity
uv run sleepy analyse data/raw/test.bin --plot out.png
uv run sleepy process data/raw/test.bin               # breathing + heart rate
uv run sleepy report                                  # self contained HTML
```

Output of `process` lands in `data/nights/<name>/`: `metrics.csv` (per epoch
breathing rate, heart rate, movement, load), `summary.json`, `overview.png`.
`report` turns those into a single HTML page per night plus an index.

A recording also gets `<name>.events.jsonl` next to the raw log. Every
connect/disconnect is written there with a timestamp and the length of the
resulting data gap, so a long night can be audited without guessing.

## Reconnects are normal

The sensor's BLE link can drop after a few minutes. That happens to the
original phone app as well (its decompiled session code treats "Connection
closed unexpectedly" as a normal event and retries). `record` therefore
watches for the disconnect, reconnects immediately, re-enables notifications,
re-sends the single `0x01` start byte and keeps appending to the same raw log.
It does **not** send periodic keepalive commands, because the official app
does not send any either.

`process` splits the log wherever a reconnect happened, fits each session
separately and leaves NaN across the gap. The summary reports the number of
sessions and the total time without data.

Each reconnect explicitly calls `stop_notify()` and then `client.disconnect()`
even when `is_connected` already reads `False`. This releases bleak's private
D-Bus connection. Skipping it leaves the old `StartNotify` subscriptions alive
in BlueZ, and every packet after the reconnect is delivered **twice** (exactly
2x the normal 15.56 pkt/s). The raw log would otherwise contain adjacent
byte-identical duplicates; `process` also removes them as a safety net.

If disconnects are still frequent, check the Linux LE supervision timeout.
The kernel default is only 420 ms, so a short firmware stall kills the link.
BlueZ can raise it in `/etc/bluetooth/main.conf`:

```ini
[LE]
MinConnectionInterval=24           # 30 ms
MaxConnectionInterval=40           # 50 ms
ConnectionLatency=0
ConnectionSupervisionTimeout=2000  # 20 s
```

Then `sudo systemctl restart bluetooth`. See `PLAN.md` section 16 for the
evidence and the `hcitool lecup` alternative. There is also a small helper:

```sh
./scripts/tune-bluez-le.sh                 # dry run
sudo ./scripts/tune-bluez-le.sh --apply    # back up main.conf, set [LE], restart bluetooth
```

## How the signal is processed

1. Decode notifications into a uniform 140 Hz series. The per packet sequence
   counter detects loss; reconnect boundaries split the stream into sessions
   and the timing fit is done per session, so a dropped link becomes a NaN gap
   instead of shifting the whole night.
2. Drop clipped samples (the DC level with a person in bed sits near the rail).
3. Rolling medians separate the slow "load" (in bed?) from posture drift.
4. Breathing: 0.1-0.6 Hz bandpass + 60 s FFT peak, with a confidence score.
5. Heart rate: 0.8-3.5 Hz ballistocardiogram, 20 s windows - harder, always
   reported with confidence.
6. Movement epochs are marked unusable instead of being silently trusted.

## Status

Working: discovery, streaming, automatic reconnect (without duplicate
packets), recording, session-aware decoding, breathing/heart rate estimation,
HTML reports.
Tests: `uv run pytest`.
