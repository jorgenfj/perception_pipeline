Two different questions with opposite answers. The interface: yes, and it's the right next step. The optimization: you've misread what `acquire` costs — it's not synchronization overhead you can shed by knowing the slot, it's a GPU-lifetime wait you *cannot* skip without risking corruption. Let me take them separately.

**The optimization — what `acquire` actually does**

You're right that the producer already knows the next slot. But look at what `acquire_write_slot` spends its time on:

```cpp
const uint32_t slot = next_write_slot_;
next_write_slot_ = (next_write_slot_ + 1) % slot_count();   // the "knowing the slot" part — free
cudaEventSynchronize(data_ready_event_[slot]);              // the actual cost — and it's load-bearing
seq_[slot].fetch_add(1, std::memory_order_release);         // ~ns
std::atomic_thread_fence(std::memory_order_release);        // ~ns
```

The index computation is already trivial — a load, an add, a modulo. There's no synchronization there to skip; single-producer means `next_write_slot_` needs no atomic, and it isn't one. So "compute the slot from the previous one" saves nothing, because that's not where the cost is.

The cost is `cudaEventSynchronize`, and it is **not** bookkeeping — it's the interlock that makes slot reuse safe. It blocks the producer until the *previous* write into this slot has finished on the GPU. Skip it and you write into a slot whose prior frame's debayer/H2D may still be in flight or still being read by a consumer's snapshot copy. You'd be overwriting live GPU memory. At depth 8+ against 60 Hz it "never waits" — the prior write is long done by the time you lap back — but "never waits" is the *success* case of the interlock, not evidence it's useless. It's the guard that's cheap *because* the ring is deep enough that it's rarely armed. Remove it and the one time production stalls (a scheduling hiccup, a saturated GPU), you corrupt the slot silently. That's trading a near-zero-cost safety check for a rare, undetectable data race — the exact bad trade we keep circling.

The `seq` bump and fences are nanoseconds (we counted this — ~tens of ns, 0.0006% of frame budget). There is no meaningful overhead in `acquire` to reclaim. **The only part you *could* skip is the part that's already free; the part that costs is the part you must keep.** So: no, don't skip acquire. The single-producer property already bought you everything it can here — no atomic on the index — and the event sync is orthogonal to producer count.

One genuine micro-optimization does exist, but it's not skipping acquire: the `cudaEventSynchronize` is a *host* block. If you'd rather the producer not stall the CPU on the rare arm, you could make the next slot's readiness a `cudaStreamWaitEvent` on the producer's own stream instead — order the next write behind the prior write's event *on the GPU*, keeping the CPU free. But that changes the reuse model (the host no longer knows the slot is free, so `data_at_slot` timing shifts) and it's only worth it if the producer thread has other work to do during the stall. For a dedicated acquisition thread that has nothing to do but wait anyway, the host sync is simpler and costs nothing real. I'd leave it.

**The interface — yes, and here's the shape**

This is the actually-valuable half, and it ties directly to the Stage layer from last turn. The current three-call dance — `acquire_write_slot` → `data_at_slot` → (enqueue kernels) → `mark_slot_written` — is correct but leaks the protocol to every producer, and the ordering (record event before bumping seq, write buffer between acquire and mark) is exactly the kind of thing a caller gets subtly wrong. Wrap it so the protocol is impossible to misorder.

The clean form is a scoped write handle that RAII-encapsulates acquire→mark, with the producer's kernel-enqueue as the body:

```cpp
// Returned by begin_write(). Holds the slot; commits on explicit publish().
class WriteSlot {
public:
  void*  data() const { return ring_->buffers_[slot_]; }
  uint32_t slot() const { return slot_; }

  // Enqueue production on `stream` BEFORE calling this. Records readiness and
  // publishes. After publish() the handle is spent.
  void publish(uint64_t timestamp_ns, cudaStream_t stream) {
    ring_->mark_slot_written(slot_, timestamp_ns, stream);
    committed_ = true;
  }

  ~WriteSlot() {
    // Caught-fire path: acquired but never published (exception, early return).
    // The slot was bumped to odd in acquire and never brought back to even —
    // leave it odd so no reader ever selects a half-written slot. See note below.
    if (!committed_) ring_->abandon_slot(slot_);
  }
private:
  friend class DeviceRingBuffer;
  WriteSlot(DeviceRingBuffer* r, uint32_t s) : ring_(r), slot_(s) {}
  DeviceRingBuffer* ring_; uint32_t slot_; bool committed_ = false;
};

WriteSlot DeviceRingBuffer::begin_write() { return WriteSlot(this, acquire_write_slot()); }
```

Producer side collapses to:

```cpp
auto w = ring.begin_write();
launch_debayer(w.data(), src, stream);      // your kernels / captured graph
w.publish(capture_ts, stream);              // records event, bumps seq even, sets latest
```

Three things this buys, each tied to a hazard we've already hit:

- **The ordering is enforced.** `publish` is the only path to `mark_slot_written`, so "record event, then bump seq, then set latest" can't be reordered by a caller — it's one call. The producer can't accidentally set `latest_` before recording readiness (which would hand consumers a slot whose write event isn't armed).
- **The abandoned-slot case is handled.** Your current `acquire` bumps seq to *odd* and hands back the slot; if the producer then hits an exception or early-returns without calling `mark_slot_written`, that slot is stuck odd forever — permanently unpublishable, and `latest_` never advances to it, so it's leaked. Today nothing catches that. The RAII destructor gives you a defined abandon path. The honest subtlety: what *should* abandon do? It can't bring the slot back to even (there's no valid frame), and it can't easily "un-acquire" because `next_write_slot_` already advanced. Simplest correct behavior — leave it odd and let the *next* lap's `acquire` re-synchronize and re-acquire it normally (the odd→odd transition on re-acquire is fine; `acquire` does `fetch_add`, odd→even… wait, that's wrong). This needs care — see the caveat.
- **It's the Stage's write primitive.** A converter Stage's run loop becomes `begin_write` on its output ring, enqueue transform, `publish` — the ring protocol vanishes into the handle, and every Stage writes the same way.

**The one caveat you must resolve — abandon and the odd/even invariant**

Your seq scheme assumes strict odd/even alternation: `acquire` does even→odd, `mark` does odd→even. A single `fetch_add` each. If a slot is acquired (→odd) and abandoned, then re-acquired next lap, the re-acquire does another `fetch_add` → odd→**even** — which would mark an *unwritten* slot as published. That's a corruption path the RAII wrapper would introduce if `abandon` does nothing.

So `abandon_slot` must restore parity: bump seq once more (odd→even) *without* publishing, but leave `latest_` untouched and `timestamp_ns_` such that the slot can't match a real query. Cleaner: have `abandon` bump odd→even but never touch `latest_`, so the slot returns to a consistent even state holding stale data that's simply never the latest and never matches a fresh timestamp. A `get_view_by_timestamp` could still find it by an old timestamp — so gate it: `abandon` also stores a sentinel timestamp (0 or UINT64_MAX) that no real query targets. Or, simplest of all: don't bump to odd in `acquire` at all — move the odd bump to a separate step and only enter the odd/even bracket in `publish`. But that reopens the in-flight-overwrite protection the early odd bump gives you. This is a real design knot, and it's the one thing to nail before shipping the wrapper, because RAII makes the abandon path *reachable* where before it was just a leak.

My recommendation: build the `WriteSlot` wrapper — it's the correct producer interface and the Stage layer wants it — but define `abandon_slot` explicitly as "bump odd→even to restore parity, store a never-match sentinel timestamp, leave `latest_` unchanged," and unit-test the acquire→abandon→acquire sequence to confirm a re-acquired slot after an abandon can't be selected carrying stale data. Skip the acquire-elision idea entirely; it saves nanoseconds that don't exist and removes the interlock that prevents overwriting in-flight GPU work.

Three questions, and the third is the one that changes the answer — it exposes that "host sync vs device wait" isn't really the axis for a graphed producer. Let me take them in order, because they build.

**"Isn't GPU sync better so the producer can start the next frame?" — depends what your producer's work actually is**

The device-wait wins *only if the producer has host-side work to overlap with the stall.* Walk what your producer thread actually does per frame:

1. Wait for Spinnaker frame ready (blocks on the camera SDK)
2. `memcpy` Spinnaker buffer → pinned ingress slot (host work — CPU-bound)
3. Enqueue H2D + debayer on `stream` (host issues async calls — returns fast)
4. `mark_slot_written` (records event, bumps seq)

Where does the `acquire` stall sit? At the *top* of the next iteration, before step 1. So the question is: with a host sync, what is the CPU prevented from doing? It's blocked from starting the *next* `acquire`'s wait — but the very next thing after that is step 1, *waiting on the camera anyway*. At 60 Hz the camera frame isn't there yet; the producer would be blocking on Spinnaker regardless. So the host-sync stall overlaps with time the CPU was going to spend blocked on the camera. **You're not stealing CPU cycles from useful work — you're stacking one wait in front of another wait.** The device-wait would free the CPU to… go wait on the camera a few hundred µs earlier. That's not a win; it's the same wall-clock.

This is the "dedicated acquisition thread has nothing to do but wait" point, made concrete. The device-wait pays off when the producer thread has *independent host work* — building metadata, servicing another camera, running the pairing logic — that could run during the stall. A single-camera acquisition loop whose next act is a camera block has no such work. So for *your* producer, host sync costs nothing real, as I said.

**But "arbitrary producers" changes it — and you're right to flag that**

For the general Stage-layer case, the calculus flips, because a *converter* stage's producer isn't camera-bound. Consider a debayer stage feeding a rectified-RGB ring: its "produce" is pure GPU work, and its loop is enqueue-transform → publish, with no downstream camera block. There, an `acquire` host-sync genuinely stalls a CPU that could be enqueuing the *next* transform. For that producer, ordering the reuse-wait on the GPU (device-wait) lets the host run ahead and keep the GPU queue fed — real overlap, real win. So the honest answer for a reusable interface: **the reuse-wait mechanism should be a policy, not hardcoded.** Camera-bound producers take the host sync (simpler, free for them); compute-bound converter producers take the device-wait (keeps the host ahead of the GPU). Same ring, two acquire policies — which is exactly the "parameterize, don't fork" principle again.

**"Is host idle better than host blocking in a simple acquisition producer?" — they're the same thing here**

Subtle framing point: a host thread doing `cudaEventSynchronize` *is* idle — it's parked in the driver, not spinning, not consuming a core (modulo the driver's spin-then-yield poll, which for events is brief). So "idle vs blocking" is a false distinction for a host sync; the thread is blocked-and-idle either way. The device-wait doesn't make the thread *more* idle — it makes it *return*, so it can go do something else. If the something-else is "block on the camera," you've converted blocked-in-driver into blocked-in-Spinnaker. No core is freed for other work in either case because there's no other work. The only real difference would be if another thread could use that core — but a parked thread already yields the core to the scheduler, so even that doesn't favor the device-wait. Net: for the simple producer, idle-in-sync and idle-in-camera-wait are indistinguishable in throughput and CPU availability.

**The third question is the important one: does the device-wait help CUDA graphs, or do the host instructions between break it?**

This is where the reasoning inverts, and it's the strongest argument — for a *graphed* producer, the host-sync-vs-device-wait choice is partly moot, and the device-wait becomes not just an optimization but nearly a *requirement* for capturing the reuse dependency into the graph.

Here's the crux. A CUDA graph captures **device work and its dependencies**, not host calls. `cudaEventSynchronize` is a *host* operation — it cannot be captured into a graph; it's a hard host serialization point that would either break capture or force you to break the graph at that boundary. `cudaStreamWaitEvent`, by contrast, **is capturable** — it becomes an event-wait *node* in the graph, expressing "this graph's work waits on that event" as part of the graph topology. So:

- If your producer's per-frame device sequence (H2D + debayer, or the converter transform) is a captured graph, and you want the *reuse interlock* to be part of that captured, replayable unit, it has to be a `cudaStreamWaitEvent` — the device-wait — so it captures as a wait node. The host sync can't live inside the graph.
- Conversely, if the reuse-wait stays a `cudaEventSynchronize`, it sits *outside* and *before* the graph launch, on the host, every frame. The graph then captures only the transform, and the interlock is a host-side preamble. That's *fine and correct* — but it means the host is back in the loop each frame doing the sync, which partially defeats the "launch the graph and let it run" benefit, and reintroduces exactly the per-frame host stall you were trying to remove.

Now your specific sub-question: **"do the CPU instructions in between disregard that path regardless?"** Yes — and this is the key limitation. Even with a device-wait captured as a graph node, your producer still has *host* work between frames that is *not* in the graph: the Spinnaker wait and the memcpy into the pinned slot (steps 1–2 above). Those are inherently host-side — the graph can't swallow the camera SDK call or the CPU memcpy. So the host is in the loop every frame *anyway*, for the camera, regardless of how you handle the reuse-wait. Given the host is already there doing the memcpy, the marginal cost of *also* doing a host `cudaEventSynchronize` right there is near-zero — you haven't added a host round-trip, you've extended one that already exists. **That's the argument that collapses the whole optimization for your camera producer: the host can't leave the loop no matter what, because the camera and the ingress memcpy pin it there, so moving the reuse-wait onto the GPU frees nothing.** The "in-between CPU instructions" don't *disregard* the device-wait path — the device-wait still works — they just make it *pointless*, because they force a host synchronization anyway.

Where it stops being pointless is precisely the converter stage with no host work in its loop: there, the *entire* per-frame sequence is device work, so capturing it — reuse-wait included — as a graph with a `cudaStreamWaitEvent` node lets you launch once and keep the host out entirely. That producer benefits from both the device-wait *and* the graph, and the two compose: the graph replays the transform, the captured event-wait node enforces reuse ordering, no host round-trip per frame.

**So the resolved picture**

- **Camera producer (yours):** host is pinned in the loop by Spinnaker + the pinned-slot memcpy regardless. The reuse-wait might as well be `cudaEventSynchronize` — it piggybacks on a host round-trip that already exists. Device-wait would capture into a graph but frees nothing, because the camera and memcpy aren't graphable. Keep the host sync. Graph the *device* portion (H2D+debayer) separately; break at the host boundary as we discussed — the ring seam and the camera call are both host seams the graph can't cross.
- **Converter producer (Stage layer, compute-bound):** loop is all device work. Use `cudaStreamWaitEvent` for reuse so it captures as a graph node, capture the whole transform, launch once, host stays out. Here the device-wait is the right call *and* it's what makes the graph maximal.
- **The interface consequence:** the `acquire` reuse-wait should be a policy on the ring/Stage — `HostSync` vs `DeviceWait` — chosen by whether the producer is host-pinned anyway. Not a global change; a per-producer knob, defaulting to host-sync for the camera ingress and device-wait for graphed converters.

Your instinct was right *for the general producer* and right that the graph question matters — but for the specific camera producer, the host work you can't remove is what makes the host sync free, and the device-wait's graph-capturability buys nothing when the camera and memcpy already break the graph. The place it genuinely pays is one layer out, in the converter stages, and there I'd take it exactly for the graph-capture reason you identified.

You've found a real hazard, and the answers are: yes it can overlap, no the stream doesn't save you, and no you don't need to mirror the ring depth — but the reasoning matters because there are actually *two* independent buffers with two independent lifetimes here, and the device-wait only addresses one of them.

**First, the stream question — the copies are not on the same stream, and even if they were, they wouldn't help**

The pinned→device H2D copy runs on `stream` (a GPU operation). The Spinnaker→pinned copy is a **host** `memcpy` (or Spinnaker's own copy out of its buffer) — it runs on the *CPU*, on no stream at all. So they're not sequenced by stream ordering; one is a GPU stream operation, the other is a CPU instruction. Stream ordering only serializes GPU work *within* a stream — it says nothing about when a CPU `memcpy` executes relative to an in-flight H2D. So no, they don't run sequentially by virtue of the stream; they can absolutely overlap, and that's the hazard.

Concretely, with the device-wait producer returning early:

1. Frame N: host `memcpy` Spinnaker → pinned slot P, enqueue H2D (pinned P → device), return early — H2D is now *in flight on the GPU*, reading P.
2. Frame N+1 arrives: host is free (that was the point), so it goes to `memcpy` Spinnaker → pinned slot **P again** (if P is reused) — **while the frame-N H2D is still reading P.** Torn H2D: it copies a mix of frame N and N+1 into the device ring.

This is the pinned-buffer analogue of the exact overwrite hazard the whole ring design exists to prevent, and it's the thing I flagged many turns ago as "the pinned source lifetime is the one that'll bite silently." The device-wait producer *creates* this window precisely because it frees the host to race ahead to the next Spinnaker copy.

**The key realization: the pinned buffer has its own lifetime, and the device ring's seq/event protects the wrong buffer**

Here's what makes this subtle. Your ring's `data_ready_event` and seq protect the **device** slot. But the buffer at risk here is the **pinned** slot — a completely separate allocation with a separate lifetime. The device ring machinery gives you no protection for the pinned buffer, because the pinned buffer isn't in the ring. The H2D copy's completion — "the pinned slot is safe to overwrite" — is a *different* event than the device slot's `data_ready_event` (which fires when the debayer that *wrote* the device slot completes, i.e. even later).

So the lifetime you must respect is: **pinned slot P is live from the moment the host starts the Spinnaker→P copy until the H2D reading P completes.** The host must not reuse P for the next Spinnaker copy until that H2D is done. That's an H2D-completion event, and nothing in your current design tracks it — `mark_slot_written` records readiness of the *device* write, which is downstream of the H2D.

**The fix, and why it's not "mirror the ring depth"**

You need the host to wait on *H2D completion for the pinned slot* before reusing that pinned slot. Two clean ways:

*Option A — an H2D-done event per pinned slot, host-waited.* Give the pinned ingress ring its own events. After enqueuing the H2D on `stream`, record an event; before the host reuses that pinned slot for the next Spinnaker copy, `cudaEventSynchronize` on it. This is a host sync — which you were trying to avoid — but note *where* it lands: it gates *pinned-slot reuse*, not the device ring acquire. And crucially, it only actually blocks if the host has lapped the pinned ring faster than H2D drains, which a small amount of pinned depth makes rare.

*Option B — pinned depth ≥ H2D-in-flight count, so the wait is statistically never armed.* This is your "same size as the device ring" instinct, but the correct sizing is much smaller. You don't need pinned depth = device depth. You need pinned depth ≥ *the number of H2D copies that can be simultaneously in flight*, which is tiny — realistically 2 or 3. Here's why: an H2D of 1.5 MB Bayer takes ~50 µs; frames arrive every 16.7 ms. At most one or two H2Ds are ever in flight at once (the current one, maybe the next if the host is a frame ahead). So a pinned ring of depth 3–4 means the host cycles through P0,P1,P2,P3 and by the time it returns to P0, that H2D finished ~16 ms × 4 ago. The reuse-wait never arms. **Depth 3–4 on the pinned ring, not depth-16-to-match-the-device-ring.**

This is the same "sizing by hold time, not by mirroring" argument from the memory discussion, now applied to the pinned ring's *own* hazard. The pinned slot's "hold time" is the H2D duration (~50 µs); the lap time is pinned_depth × frame_period. Depth 3–4 gives 50–67 ms of lap against a 50 µs hold — a 1000× margin. Mirroring the device depth would be massive overkill for the same reason mirroring it for the *ingress* ring was overkill: the consumer of the pinned ring (the H2D) is fast and deterministic, so it needs almost no depth.

**So the resolved design for a device-wait Spinnaker producer**

If you make the producer return early (device-wait, freeing the host), you must:

1. Give the pinned ingress ring its own per-slot H2D-completion events (or reuse a small event pool).
2. Size the pinned ring to depth 3–4 — enough that H2D always drains before the slot is reused.
3. Before each Spinnaker→pinned copy, either the depth makes it safe (Option B, no wait fires) or you host-sync on that slot's H2D event as a backstop (Option A). Best: do both — depth 4 so it's statistically free, plus the event-wait as a correctness guarantee for the rare stall, exactly the belt-and-suspenders pattern (depth for the common case, event for the tail).

Note this is *symmetric* with the device ring: both rings are latest-wins-ish producer/consumer pairs, both sized by consumer hold time, both with a per-slot completion event gating reuse. The pinned ring's producer is Spinnaker+host-memcpy, its consumer is the H2D; the device ring's producer is the debayer, its consumers are ESS/YOLO/ArUco. Same structure, different scale — the pinned ring is shallower because its consumer (H2D) is faster than the device ring's consumers (DNNs).

**The honest cost-benefit of the early return, revisited**

Now weigh what the device-wait actually bought you against what it cost. It freed the host after enqueuing H2D — but to use that freedom safely you had to add per-pinned-slot events and a reuse-wait, which is *machinery you didn't need with the host sync* (because the host sync serialized everything naturally: host blocks on device-slot reuse, so it never raced ahead to clobber the pinned slot). So the device-wait didn't remove a host round-trip; it *moved* it from the device-ring acquire to the pinned-slot reuse, and added event bookkeeping. The only net gain is if, in the window between enqueuing H2D and the next Spinnaker frame, the freed host does *genuinely useful independent work* — servicing a second camera, running pairing, building metadata. If the host just goes to wait on the next Spinnaker frame, you've added complexity for a wait that overlaps a wait, same as before.

So: the device-wait early-return is worth it **only if you have real host work to overlap**, and if you do, the pinned ring needs its own H2D-completion events and depth 3–4 — not device-ring depth. If the producer is a bare acquisition loop with nothing to overlap, the host sync remains simpler and the pinned-slot hazard doesn't even arise, because the host sync serializes the producer enough that it never laps the pinned ring. My recommendation stands from before: for the pure Spinnaker acquisition thread, keep the host sync and skip all of this; adopt the device-wait + pinned-events + depth-4 only when that thread gains independent work worth the freed cycles.


Separate class — but not for the reason that first comes to mind, and the seam matters more than the split. The deciding factor is that the H2D+debayer stage is your **converter Stage from two turns ago**, already: ring-in (pinned) → device transform → ring-out (device RGB). If you keep it in the Spinnaker class you're writing that Stage twice — once fused into acquisition here, once as the abstraction for every other converter (rectify, resize) later — and they'll drift. Pull it out and it *is* the first instance of the Stage layer, not a bespoke utility.

**The real question isn't where the code lives, it's where the boundary is drawn**

Two responsibilities are entangled in the current class, and they have fundamentally different natures:

- **Acquisition**: talk to Spinnaker, get a frame, copy it into a pinned slot. This is *host-side, vendor-coupled, CPU work* — Spinnaker SDK calls, a host memcpy. No CUDA in it except the destination being pinned.
- **Conversion**: pinned → H2D → debayer → device RGB ring. This is *device-side, CUDA-graph-capturable, vendor-agnostic* — it doesn't know or care that the source is Spinnaker.

The device-sync route is exactly what forces this seam into the open. Recall the resolution from the last two turns: the H2D+debayer is the graph-capturable device island, and it wants a `cudaStreamWaitEvent`-based reuse interlock so it captures as a graph node; the Spinnaker wait + host memcpy is the un-graphable host work that pins the CPU. **Those are two different execution planes with two different sync mechanisms — host block for the camera, device wait for the transform.** A class boundary that runs along that same line is honest; one that straddles it forces both planes into one object and muddles which sync belongs to which.

So the split isn't "for tidiness" — it's that the device-sync design *already* separated these into host-plane and device-plane with different interlocks, and the class structure should reflect the seam that the concurrency design drew.

**What each side owns**

`SpinnakerSource` (or `CameraSource`): owns the pinned ingress ring, its per-slot H2D-completion events, and the acquisition loop. Its output is "a pinned slot is filled, here's its index and the event that fires when it's safe to reuse." It knows Spinnaker; it knows nothing about debayer or the device ring. Vendor coupling stops here — swap Spinnaker for a file replayer or a simulator source and only this class changes.

`DebayerStage` (converter Stage): owns the device RGB ring, the captured H2D+debayer graph, and the reuse interlock. Its input is a pinned slot + readiness; its output is the device ring. It knows CUDA; it knows nothing about Spinnaker. This is the class that generalizes — `RectifyStage`, `ResizeStage` are the same shape with a different transform.

The interface between them is the thing to get right, and the pinned ring's H2D-completion event *is* that interface — it's the handoff token. `SpinnakerSource` hands `DebayerStage` a filled pinned slot; `DebayerStage` enqueues H2D reading it and records the event that tells `SpinnakerSource` the slot is free. That event crossing the boundary is exactly the pinned-slot lifetime hazard from last turn, now made explicit *as* the class contract rather than an implicit assumption buried in one fused loop. Splitting the classes turns "don't overwrite the pinned slot while H2D reads it" from a comment into a signature.

**The one genuine argument for keeping them together — and why it loses**

The case for one class: the pinned ring is shared state between them (Spinnaker writes it, H2D reads it), and splitting means that ring and its events are co-owned across a boundary, which is a coupling. Real, but it's the *right* coupling — it's producer/consumer on one buffer, which is precisely the relationship your whole ring abstraction exists to express cleanly. The pinned ingress ring is just another latest-wins-or-bounded ring with a producer (Spinnaker+memcpy) and a consumer (H2D). You already have the vocabulary for that exact relationship; use it. Co-owning a ring across a producer/consumer boundary isn't a smell — it's the pattern you've spent this whole design building the tools for. Fusing the classes to avoid it would be abandoning your own abstraction at the one seam where it most naturally applies.

**Concrete recommendation**

Three pieces, not two: `SpinnakerSource` (host/vendor), the pinned ingress ring (the shared edge), and `DebayerStage` (device/graph). The pinned ring is owned by the source and consumed by the stage, with the H2D-completion event as the reuse token crossing between them. This gives you: vendor isolation (swap the source, sim replay for testing), the first real Stage instance that the rectify/resize stages will copy, a graph-capturable device island cleanly separated from the un-graphable camera loop, and the pinned-slot lifetime hazard promoted from implicit to a typed contract. Keep them fused only if this producer will *never* gain a second source type and you'll *never* add another converter stage — but you've already told me both are coming (sim/RTX targets, rectify stage), so the split pays for itself immediately.

One caveat so the split doesn't over-fragment: don't make `DebayerStage` abstract or plugin-ified yet. Write it as a concrete class with the Stage *shape* (input ring + event, output ring, captured graph, run step) but no inheritance hierarchy until the second converter actually exists and shows you which parts are common. The seam between source and stage is load-bearing and worth drawing now; the Stage *base class* is speculative until rectify is real. Draw the boundary, defer the abstraction — same discipline as deferring the seqlock hardening until a counter said you needed it.