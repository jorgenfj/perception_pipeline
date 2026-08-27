# PTP time sync (stereo rig)

Topology for a two-FLIR-camera stereo rig: the Jetson is PTP grandmaster,
both cameras are slaves.

```
        Jetson (grandmaster)
         /              \
   camera L          camera R
   (slave)            (slave)
```

## Why the Jetson, not a camera, is master

Stereo left/right sync accuracy only requires both cameras converging on the
*same* master -- it does not by itself require that master to be the Jetson.
The Jetson is the better choice anyway because it is already the thing
timestamping everything else in the fusion stack that isn't PTP-aware (IMU,
CAN/serial sensors, ROS/middleware message stamps, logs). Making it master
means the cameras lock directly onto the same timeline the rest of the stack
already runs on, instead of introducing a third clock domain to reconcile.

Don't leave this to Best Master Clock (BMC) election -- two cameras
advertising decent clock quality could in principle out-negotiate the Jetson.
Force it with a low `priority1` in the Jetson's `ptp4l` config (lower wins),
and set the camera's PTP mode to slave-preferring if the node map exposes one.

The one case where the Jetson should be slave instead: an external
GPS-disciplined grandmaster on the network, needed if timestamps must
correlate to real UTC across multiple rigs or log sessions. Nothing about a
single rig calls for that.

## Which mode applies to your hardware

Check before doing anything else:

```bash
ethtool -T <iface>
```

No `PTP Hardware Clock` line (just `software-transmit`/`software-receive`/
`software-system-clock`) means no PHC -- use **software master mode** below.
A `PTP Hardware Clock: <N>` line and a `/dev/ptp<N>` device means the NIC has
one -- use **hardware master mode**.

The current NX devkit's onboard GigE (`r8168` driver, Realtek PCIe NIC) has no
PHC. The AGX board is expected to expose one via the SoC's native
controller instead -- re-run the check above once you're on it.

## Common: `ptp4l.conf`

Both modes share one config file, forcing the Jetson to win BMC:

```ini
# /etc/ptp4l-master.conf
[global]
priority1     0
priority2     0
domainNumber  0
```

## Software master mode (no PHC)

```bash
sudo apt install linuxptp

# -S: software timestamping -- there is no PHC to timestamp with hardware,
# so ptp4l disciplines CLOCK_REALTIME directly. No phc2sys step: there is no
# separate NIC clock to copy to/from.
sudo ptp4l -i <iface> -S -f /etc/ptp4l-master.conf -m
```

Expect tens of microseconds to low-single-digit milliseconds of sync
accuracy, depending on interrupt/scheduling load.

## Hardware master mode (PHC present)

```bash
sudo apt install linuxptp

# Seed the PHC from the system clock, then keep steering it. -w waits for
# ptp4l to reach sync before stepping; run this continuously (systemd unit),
# not as a one-shot.
sudo phc2sys -s CLOCK_REALTIME -c <iface> -O 0 -w -m &

# No -S here: hardware timestamping is the default when a PHC exists. The PHC
# is what becomes the PTP master the cameras hear and sync to.
sudo ptp4l -i <iface> -f /etc/ptp4l-master.conf -m
```

Run both as systemd services for anything beyond a bench test -- a manual
`&`-backgrounded process won't survive a reboot or a crash.

## Camera-side config

Add to `features` in `camera.yaml` / `acquire.yaml` (`applyFeature` in
`spinnaker/src/spinnaker_source.cpp` sets any writable node by name, so no
code change is needed if the camera exposes this node):

```yaml
    # PTP slave sync -- see spinnaker/README.md for host-side setup.
    - GevIEEE1588: "true"
```

## Verifying it actually locked

Three independent places to check -- host side, wire protocol, camera side.
They can disagree (e.g. host thinks it's GM but the camera never hears it),
which is itself diagnostic.

**Host: is the Jetson actually grandmaster?**

```bash
sudo pmc -u -b 0 'GET PARENT_DATA_SET'
```

GM identity should equal the Jetson's own clock ID. Run continuously via the
systemd unit's own log instead of a one-shot check:

```bash
journalctl -u ptp4l.service -f      # "master offset" lines, once a slave exists
journalctl -u phc2sys.service -f    # hardware mode only
```

`ptp4l -m` (already in the commands above) logs to stdout/journal either way;
`-m` is what makes it verbose enough to show BMC decisions and offset updates
rather than just errors.

**Camera: is it actually locked, per the app itself?**

`SpinnakerSource::ptp_status()` / `ptp_offset_ns()`
(`spinnaker/include/spinnaker_source.hpp`) read `GevIEEE1588Status` /
`GevIEEE1588OffsetFromMaster` straight off the node map -- no SpinView needed.
`acquire` prints this automatically when `GevIEEE1588` is enabled in config:
one line at startup (confirms the feature applied; BMC needs a few `Announce`
intervals to settle, so don't expect `Slave` immediately here) and again
every 60 frames in the running report line, e.g.:

```
ptp: status=Listening (see spinnaker/README.md if this doesn't reach Slave)
...
t=...  | ptp Slave offset=482ns
```

`status` stuck at `Listening` past the first few report lines means the
camera never heard an `Announce` from the Jetson -- check both are on the
same L2 broadcast domain and that any switch between them isn't a PTP-aware
one silently intercepting/altering the traffic. `offset` climbing instead of
staying small and flat means it's nominally locked but the servo isn't
converging -- check for other PTP masters on the segment fighting the BMC
election, or a `priority1` that isn't actually as low as intended.

## Scheduling captures without GPIO: Action Commands

Two ways to trigger a camera over the network instead of a hardware line,
both built on `System::SendActionCommand`:

- **Unscheduled** (`actionTime=0`): fires "now" on every matching camera, one
  broadcast packet. Simultaneity is bounded by however consistently that
  packet lands on each camera's NIC -- fine for a rough sync, no PTP required.
- **Scheduled** (`actionTime` = a future PTP-epoch nanosecond timestamp,
  same units as `GetTimeStamp()`): each camera's own firmware fires at that
  instant off its own PTP-disciplined clock, independent of network jitter in
  delivering the command. This is what a stereo rig wants, and it needs PTP
  locked first -- an unlocked camera acks `NO_REF_TIME` for any scheduled
  command.

### Why one-shot `AcquisitionStart` is not enough

`TriggerSelector=AcquisitionStart` sends one scheduled action that aligns the
*start* of `AcquisitionMode=Continuous` on every camera to the same instant.
That part works: both eyes begin within the precision PTP gives you.

They do not stay there. Once running, `AcquisitionFrameRate` is generated
from the **sensor clock domain** -- the camera's own crystal -- not from the
PTP-disciplined clock. PTP corrects `GetTimeStamp()`; it does not steer the
oscillator that decides when the next exposure starts. So two cameras both
told "7.5 Hz" are really running at 7.5 Hz ± their own crystal error, and the
phase between them walks at the difference. A few ppm is milliseconds of skew
over a minute, and it accumulates without bound.

The timestamps do not show this honestly either: both are PTP-corrected, so a
pair whose shutters have drifted apart can still report plausible-looking
stamps. Watch `max_skew` in stereo pairing, not the stamps alone.

**`TriggerSelector=FrameStart` + one scheduled Action Command per frame is
what actually holds two shutters together over a long run.** Every exposure is
pinned to a PTP instant, so the sensor crystal stops mattering -- it only has
to be stable enough within a single frame period, not across the run. The
costs are real and worth knowing: host-loop timing jitter on each command, and
a bound from `ActionQueueSize` (check it's nonzero on your camera model). They
buy bounded skew, which the one-shot cannot give you.

One-shot `AcquisitionStart` remains the right choice for a short run, a single
camera, or when the pairing tolerance is loose enough that the drift never
reaches it.

#### Per-frame triggering in `acquire`

Set `action_sync.per_frame: true` in `app/config/acquire.yaml`:

```yaml
action_sync:
  enabled: true
  per_frame: true
  trigger_hz: 7.5
  trigger_lead_ms: 100.0
```

`acquire` then appends the trigger features to every camera itself --
`TriggerSelector=FrameStart`, `TriggerSource=Action0`, `TriggerMode=On`, the
action keys, and `AcquisitionFrameRateEnable=false` -- after whatever
`camera.features` said, so the yaml above wins and `AcquisitionFrameRate`
stops mattering. `trigger_hz` is the capture rate.

A host thread then sends one scheduled command per frame. Unlike the one-shot
path, PTP not being locked at startup is a warning rather than a fatal error:
the loop keeps running and starts working when the cameras reach `Slave`.

The run prints a `trig` tally every 60 frames and once at the end:

```
trig sent=1204 ok=1204 missing_ack=0 not_ok=0
```

`sent` should track delivered frames one-to-one. `missing_ack` counts commands
where fewer cameras answered than exist -- lost on the wire, or the ack lost
the race with image traffic on the return path. `not_ok` counts commands that
landed and were refused, `OVERFLOW` among them.

Watch `OVERFLOW` when raising the rate: commands in flight is roughly
`trigger_lead_ms / (1000 / trigger_hz)`, so 100 ms of lead at 60 Hz keeps six
of them queued on a camera whose `ActionQueueSize` may be smaller. Bring
`trigger_lead_ms` down as the rate goes up. `tools/ptp_trigger_test` sweeps
rates and prints the same counters if you want the ceiling before committing a
config to it.

### Verifying it end-to-end: `acquire_action_sync_test.yaml`

`app/config/acquire_action_sync_test.yaml` runs `acquire` with
`TriggerSelector=AcquisitionStart`/`TriggerSource=Action0` and an
`action_sync:` section that makes `acquire` do the whole loop itself:

```bash
./build/bin/acquire config/acquire_action_sync_test.yaml
```

It schedules one Action Command for the next whole second (plus
`lead_time_ms` of margin), prints the per-camera ack (`OK`, or one of the
`ActionCommandStatus` failure codes above), then checks the first
`check_frames` delivered timestamps against that schedule -- first frame's
proximity to the scheduled instant (offset by `expected_start_offset_ms`),
and frame-to-frame spacing against `1/expected_hz` -- and prints a
`PASS`/`FAIL` verdict:

```
action_sync: scheduled AcquisitionStart for t=1787053290000000000ns (1371ms from now)
action_sync: ack from 34.0.0.10: OK
...
action_sync: PASS -- start offset 7.987 ms, expected 8.000 ms, deviation -0.013 ms
(tolerance 2.00 ms); frame spacing deviation from 1/expected_hz min/mean/max
0.001/0.001/0.002 ms over 60 frames
t=... | ... | fps actual=7.5000 expected=7.5000 spacing dev min/mean/max -0.002/0.001/0.003 ms
```

That `PASS`/`FAIL` verdict is a one-shot check of the first `check_frames`
frames -- it only tells you the scheduled start landed correctly. The `fps`
field on the regular report line is the ongoing version of the same check:
it keeps running the whole session, reset every 60 frames like the rest of
the health line, so a rate that's fine at the start but drifts or degrades
later still shows up.

`start offset` is not compared to zero -- `GetTimeStamp()`'s latch point
relative to the trigger instant is a sensor convention this app can't know in
advance. On the camera this was measured on, it latches at end-of-exposure:
the raw offset came back ~7.99ms against `ExposureTime: 8000.0` (8ms), which
is why `expected_start_offset_ms: 8.0` is set in the yaml -- `deviation` is
what's actually checked against `tolerance_ms`. Re-measure and update that
value if `ExposureTime` changes or you swap camera models; don't just widen
`tolerance_ms` to paper over it, that hides a real regression in the same
place it hides the expected constant.

Frame *spacing* has no such excuse and needs no config -- it should come back
close to zero regardless of camera model, since once started it's governed
purely by the camera's own free-running timer, nothing external. Run this
before adding the second camera; a single camera alone is a complete test of
"did the scheduled trigger land where it should."

## Seeing it work: `stereo_view`

`stereo_view` is the CUDA-free stereo viewer and recorder. It opens both
cameras, pairs their frames by camera timestamp, shows the two halves side by
side with the pair's skew under them, and can write everything both cameras
delivered to disk. Nothing in it links CUDA, so it runs on a machine with no
GPU -- and `--play` needs neither a GPU nor the Spinnaker SDK, which is how a
recording gets looked at on a laptop.

```bash
./build/bin/stereo_view                       # both cameras, live
./build/bin/stereo_view --record              # ... and record from the start
./build/bin/stereo_view --play recordings/recording-2026-08-21T13-22-04
./build/bin/stereo_view --info recordings/recording-2026-08-21T13-22-04
```

In the window: `space` pauses, `r` toggles recording, `q` quits. Config is
`tools/stereo_view/config/stereo.yaml`; see the comments in it for the pairing
tolerance and the recording settings.

This is the tool that answers "is the rig synced" directly, and it is worth
running before the GPU pipeline is worth starting:

```
stereo paired=1204 unpaired=0/2 dropped=0/0 max_skew=41us pairs/s=59.9
```

`paired` climbing with `unpaired` flat is a synced rig. `max_skew` is the worst
gap inside any pair -- microseconds once PTP is locked and Scheduled Action
Commands are armed. **`paired=0` is the expected result with PTP unlocked**:
free-running cameras share no epoch, so their timestamps are unrelated counters
and nothing can pair. Check `ptp` in the startup line before suspecting the
tolerance.

### What the recording is

Every frame each camera delivered, per stream, plus a 32-byte index record per
frame carrying both the camera timestamp and the host receive time. No pairing
is stored -- pairing is a merge over the camera timestamps done at read time,
so a six-month-old recording can be re-paired at a different tolerance to
answer a question you did not know you had. `--info` does exactly that, and
prints the drift of `host_recv - timestamp` across the file, which is a
measurement of whether PTP was actually disciplining rather than a claim that
it was.

Frames are copied out of the camera buffer into a bounded staging ring, so a
disk that cannot keep up drops the recorder's own frames and counts them
(`rec drops=`) rather than throttling acquisition. That matters: a recorder
that backs up into the camera would change the timing of the run it is
recording. See `recording_plan.md`.

## Gotchas

- **TAI vs UTC.** PTP's epoch is TAI, not UTC. Expect camera timestamps to
  sit ~37s off `CLOCK_REALTIME` (the current leap-second count) unless
  something in the chain accounts for it. Don't confuse this constant offset
  with a sync failure.
- **Host stamps are `CLOCK_REALTIME`, and must stay that way.**
  `LatencyProbe::host_now_ns()` and `host_now_ns()` in
  `capture/include/frame_sink.hpp` are both `system_clock`, deliberately: a
  monotonic clock shares no epoch with a PTP-disciplined camera, so
  `host_recv_ns - timestamp_ns` would stop being transport latency and start
  being a meaningless difference of two unrelated counters. Two *different*
  host clocks would be just as bad -- they would disagree by microseconds and
  make the same subtraction lie. If either is ever switched to `steady_clock`,
  the recording index and the latency probe both stop meaning anything.
