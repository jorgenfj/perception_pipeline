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

Run that check on *every* port a camera is on, and compare the clock numbers:
two ports on the same card often carry two independent PHCs, which needs the
setup in **Two cameras on two NIC ports** below rather than the single-interface
recipes that follow.

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

## Two cameras on two NIC ports: one master, not two

A rig with each camera on its own NIC port is the case the topology diagram
above quietly hides. Run the obvious thing --

```bash
sudo ptp4l -i enP5p1s0f0 -f /etc/ptp4l-master.conf -m
sudo ptp4l -i enP5p1s0f1 -f /etc/ptp4l-master.conf -m
```

-- and you have **two grandmasters, not one**. Each instance disciplines its own
port's PHC, `priority1 0` makes each win its own segment, and nothing ties the
two PHCs together. Each camera then locks perfectly to a different free-running
oscillator. Both report `Slave` with a sub-microsecond offset, truthfully, and
their timestamps still share no epoch. Pairing finds nothing.

Check whether the ports actually have separate clocks before assuming either
way:

```bash
$ ethtool -T enP5p1s0f0 | grep "PTP Hardware Clock"
PTP Hardware Clock: 0
$ ethtool -T enP5p1s0f1 | grep "PTP Hardware Clock"
PTP Hardware Clock: 1
```

Different numbers mean different `/dev/ptp*` devices, which means two epochs
unless something joins them. One `ptp4l` across both ports is the fix, and it is
what `boundary_clock_jbod 1` in the shared config is for -- that option does
nothing whatsoever when each instance is handed a single interface:

```bash
sudo ptp4l -i enP5p1s0f0 -i enP5p1s0f1 -f /etc/ptp4l-master.conf -m &
sudo phc2sys -a -r -r -m
```

`ptp4l` alone is not enough here, and this is the part that is easy to get
wrong. In JBOD mode it disciplines the other PHCs to the PHC of whichever port
is in `SLAVE` state -- and a pure grandmaster has no slave port, so there is no
reference and each PHC keeps its own time. `phc2sys` is what gives them a common
one: `-a` follows `ptp4l`'s port states, the first `-r` puts `CLOCK_REALTIME`
into the set being synchronised, and the second lets it be the *source* when no
port is slave, which is exactly the grandmaster case. Both PHCs then track the
host clock, and therefore each other.

Working, the two agree to tens of nanoseconds:

```
phc2sys[...]: enP5p1s0f0 sys offset  16329 s2 freq +207438 delay 2176
phc2sys[...]: enP5p1s0f1 sys offset  16302 s2 freq +207420 delay 2176
```

Read the two `sys offset` columns against *each other*, not against zero. Their
difference is what the cameras inherit; the common part cancels.

Do not add `phc2sys -s /dev/ptp0 -c /dev/ptp1 -O 0` on top of that. Chaining one
PHC to the other looks reasonable and conflicts twice: two servos writing
`/dev/ptp1`, and a `/dev/ptp0` that no longer bears any relation to
`CLOCK_REALTIME` -- which breaks scheduled Action Commands, because the host
builds their target from its own clock. See **TAI vs UTC** in the gotchas.

The simplest way out of all of this is one NIC port and a switch. Two cameras at
7.5 Hz is ~187 Mbps, comfortable on a gigabit port, and one port means one PHC
and one grandmaster by construction.

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

#### Per-frame triggering in `stereo_view`

`stereo_view` has the same mechanism behind a flag rather than a config key:

```bash
./build/bin/stereo_view --capture-hz 3.5 tools/stereo_view/config/stereo_sync.yaml
```

It appends the same trigger features *after* whatever `camera.features` said --
the action keys, `AcquisitionFrameRateEnable=false`,
`TriggerSelector=FrameStart`, `TriggerSource=Action0`, `TriggerMode=On` -- so
the flag wins over the yaml and `AcquisitionFrameRate` stops mattering. It also
forces `action_sync.enabled = false`, because per-frame triggering and the
one-shot `AcquisitionStart` are mutually exclusive: in this mode the cameras
expose only when the loop tells them to.

**`action_sync.per_frame: true` in the yaml does nothing here.**
`load_action_sync_config` parses it and `acquire` honours it, but `stereo_view`
branches only on `--capture-hz`. `trigger_lead_ms` is ignored the same way --
that loop hardcodes 100 ms of lead. Setting either key and seeing no change is
this, not a broken rig.

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

`stereo_view` is the CUDA-free stereo viewer. It opens both cameras, pairs
their frames by camera timestamp, and shows the two halves side by side with the
pair's skew under them. Nothing in it links CUDA, so it runs on a machine with
no GPU.

```bash
./build/bin/stereo_view                       # both cameras, live
./build/bin/stereo_view --sync-start          # ... with a scheduled start, and check it
./build/bin/stereo_view --capture-hz 3.5      # ... every frame PTP-scheduled
./build/bin/stereo_view --no-display          # pair and report headless
```

In the window: `space` pauses, `q` quits. Config is
`tools/stereo_view/config/stereo.yaml`; see the comments in it for the pairing
tolerance.

Recording and replay are `acquire`'s, on MCAP: `acquire --record` writes one,
and a `-DPERCEPTION_SOURCE=recording` build replays it through the whole
pipeline with no cameras attached.

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

### `paired=0` while `ptp` reads `Slave/Slave`

`paired=0` with PTP unlocked is the easy case. The confusing one is `paired=0`
with *both* cameras reporting `Slave`. The per-stream rates and the skew in the
report line are the instrument, but read them carefully -- the obvious reading
is wrong.

A run that would not pair:

```
stream 0 (left)  23494258: 38 frames, 3.60 Hz
  host_recv - timestamp: -36985.481ms -> -36986.115ms (drift -0.635ms)
stream 1 (right) 23494259: 38 frames, 3.60 Hz
  host_recv - timestamp: -36873.087ms -> -36873.464ms (drift -0.377ms)
  pairing at 500us: paired=0 unpaired=38/38
```

and the same rig once it was fixed:

```
stream 0 (left)  23494258: 219 frames, 3.52 Hz
  host_recv - timestamp: -36985.944ms -> -36985.868ms (drift 0.076ms)
stream 1 (right) 23494259: 219 frames, 3.52 Hz
  host_recv - timestamp: -36873.400ms -> -36873.246ms (drift 0.154ms)
  pairing at 500us: paired=219 unpaired=0/0 max_skew=5.8us
```

The ~112 ms difference between the two streams is **the same in both**, working
and broken alike. It is not a clock offset, and reading it as one sends you
after the wrong thing. `host_recv - timestamp` is transport latency plus any
clock error, so differencing the two streams gives transport *asymmetry* plus
the inter-camera clock gap -- and when the two cameras sit on links of different
speed, the transport term buries everything else. A 1440x1080 BayerRG8 frame is
12.44 Mbit, which serialises in 124 ms at 100 Mbps and 12 ms at gigabit: 112 ms
of difference, which is what both recordings show to within half a millisecond.
Equal-speed links make this term vanish and the same subtraction becomes
meaningful again.

What each number is actually good for:

- **The constant, ~-37 s on both streams.** TAI vs UTC, and correct. See the
  gotcha below.
- **The difference between the streams.** Transport asymmetry first, clock gap
  second. Predict the transport part from frame size and link speed before
  attributing any of it to the clocks.
- **The difference between the two drifts.** This one is clean, because a
  constant transport asymmetry differentiates away. 0.26 ms over 10.5 s above,
  i.e. ~25 ppm, against 0.08 ms over 62 s -- ~1 ppm -- once fixed. Two cameras
  genuinely locked to one master hold sub-ppm against each other; tens of ppm
  means they are not on a common master, or have not finished converging.
- **The `pairing at Nus:` line itself.** The most direct answer available: it
  re-pairs the recording at a tolerance you choose, so widening it until pairs
  appear measures the inter-camera timestamp gap instead of inferring it.

`Slave` means "this camera is following a master". It does not mean "this camera
agrees with the other one", and both are true of two cameras following two
different masters.

Give convergence real time before blaming anything else. A camera closes a large
offset by slewing, not stepping, so a rig just repointed at a new master needs
minutes, or a power cycle to force a fresh acquisition. Watch the `ptp` offsets
settle and *stay* settled -- one that is still walking means the master
underneath it is still moving too.

### What the recording is

An MCAP, written by `acquire` under the `ros2` profile, so `ros2 bag play` and
Foxglove open it directly. One `sensor_msgs/Image` topic per camera role, each
message carrying its own geometry and the camera's timestamp; `/disparity`
alongside them when an engine is running. No pairing is stored -- pairing is a
merge over the camera timestamps done at read time, so an old recording can be
re-paired at a different tolerance to answer a question you did not know you
had.

Frames are copied out of the camera buffer into the tap ring and encoded on the
recorder's own thread, so a disk that cannot keep up drops the recorder's own
messages and counts them (`drops=`) rather than throttling acquisition. That
matters: a recorder that backs up into the camera would change the timing of the
run it is recording.

`McapReplaySource` reads one image topic back out and feeds it through a
`FrameSink` at the pacing it was recorded with, which is what makes a
`-DPERCEPTION_SOURCE=recording` build a full pipeline run with no rig.

## Gotchas

- **TAI vs UTC, and which host clock a camera-facing number belongs in.**
  PTP's epoch is TAI, not UTC, so camera timestamps come off the wire ~37s (the
  current leap-second count) ahead of `CLOCK_REALTIME`.

  In `acquire` that offset is removed **once**, at the pipeline boundary:
  `AcquireSource::epoch_offset_ns()` reports it (read once at construction, so a
  leap second cannot step a running timeline) and `RingFrameSink::commit()`
  subtracts it before the probe, the tap or the ingress ring sees the frame.
  Everything downstream is therefore UTC and lines up with any sensor stamped
  `host_now_ns()`, and the MCAP says `timestamp_epoch: UTC` with
  `epoch_offset_ns` recorded so the raw PTP value is recoverable. A recording
  source reports 0 -- its stamps are already on the host epoch, and rebasing
  them again would be this same bug mirrored.

  The tools do **not** rebase: `stereo_view`, `spin_acquire` and
  `ptp_trigger_test` take frames straight from `SpinnakerSource`, so their
  numbers are raw TAI. Expect the ~37s constant in `host_recv - timestamp`
  there -- and *not* in `acquire` -- and don't read either as a sync failure.

  It is not merely cosmetic. A camera evaluates a **Scheduled Action Command's
  instant against its own PTP clock**, so a target built from `host_now_ns()`
  lands ~37 s in that camera's past. Every command then acks `ACTION_LATE` and
  the camera fires it immediately on receipt instead -- so the rig looks
  triggered, delivers frames at exactly the requested rate, and is not
  scheduled at all. The signature is `trig fail` equal to `trig sent`, every
  interval, while frames keep arriving.

  Use `perception::ptp_now_ns()` (`capture/include/frame_sink.hpp`, which is
  `clock_gettime(CLOCK_TAI)`) for anything a camera will compare against its own
  clock: Action Command targets, and `ActionSyncChecker::target_ns`, which is
  checked against delivered `GetTimeStamp()` values.

  `target_ns` is the one number that lives in both worlds, and it is why
  `SpinnakerAcquireSource::finish_arming()` exists: the value on the wire stays
  TAI because the camera evaluates it, while the checker's copy is rebased to
  UTC because the frames it will be compared against arrived through
  `RingFrameSink`. Arm a new trigger path without going through that function
  and the verdict reads 37 seconds late while every frame is on time. Keep the *whole* trigger
  loop in that timebase -- the target and the sleep alike -- because mixing the
  two epochs inside one loop is the same 37-second error wearing a different
  hat.

  `ptp_timebase_ready()` is false when the kernel holds no TAI offset, i.e.
  nothing has disciplined the clock yet; `ptp_now_ns()` is then a silent alias
  for `host_now_ns()` and the bug is quietly back. The trigger paths warn on
  that rather than mis-scheduling in silence.

- **Link speed is a bandwidth problem before it is a PTP problem.** A port that
  came up at 100 Mbps barely moves PTP: the delay-request mechanism measures the
  longer path and subtracts it, leaving only PHY asymmetry, which is tens to
  hundreds of nanoseconds. What it does do is run out of room. 1440x1080
  BayerRG8 is 12.44 Mbit a frame, so 3.5 Hz is 43.5 Mbps and fits, while 7.5 Hz
  is 93.3 Mbps and does not. Expect incomplete frames, not skew.

  Check `cat /sys/class/net/*/speed` before suspecting the clock -- and check
  *advertised* modes, not just supported ones. A NetworkManager profile carrying
  `802-3-ethernet.speed: 100` produces a link indistinguishable from a bad
  cable:

  ```
  Supported link modes:   10baseT/Half ... 1000baseT/Full
  Advertised link modes:  100baseT/Full        <- pinned, not negotiated
  ```

  Clear it with `nmcli con mod <name> 802-3-ethernet.speed 0
  802-3-ethernet.duplex ""`. Note also that a jumbo frame is a much larger
  latency quantum down there: 9000 bytes serialises in 72 us at 1 Gbps and
  720 us at 100 Mbps, which is real PTP jitter on a link also carrying image
  data.
- **Host stamps are `CLOCK_REALTIME`, and must stay that way.**
  `LatencyProbe::host_now_ns()` and `host_now_ns()` in
  `capture/include/frame_sink.hpp` are both `system_clock`, deliberately: a
  monotonic clock shares no epoch with a PTP-disciplined camera, so
  `host_recv_ns - timestamp_ns` would stop being transport latency and start
  being a meaningless difference of two unrelated counters. Two *different*
  host clocks would be just as bad -- they would disagree by microseconds and
  make the same subtraction lie. If either is ever switched to `steady_clock`,
  the recording index and the latency probe both stop meaning anything.

  This is not in tension with `ptp_now_ns()` above -- the two answer different
  questions. Host-side wall clock work (frame arrival, latency against
  `host_recv_ns`) is UTC and stays `system_clock`; a number the *camera* will
  evaluate is TAI. Picking by "which clock is the host on" rather than "who
  reads this number" is what produced the 37-second bug in the first place.

  `LatencyProbe` needs neither rule: it estimates the host/camera offset as a
  rolling minimum and applies it, so it reads correctly whether it is handed
  raw TAI stamps or rebased ones. What changes is the number it converges on --
  transport latency in `acquire`, that plus ~37s in the tools.
