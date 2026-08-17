Explain how the buffer structure should look like. I think it makes sense to keep it modular in a way so that the pipeline can be used with and without stereo. Also the aruco and yolo part should also be able to be turned on/off. Only on startup though from a config/compiled, not during runtime. I want to make it so that the input can be an arbitrary source. Either the frames from the spinnaker API, frames from a recording, or frames from the simulator. I think it makes sense to design a CPP struct to define the message format. I like using Asio. Is that a good fit for handling async incoming images? Should the images come in into host pinned memory so that they can be async copied into GPU allocated memory? Should the buffers live in CPU or GPU memory.

This is where the modularity actually falls out of one decision: model everything downstream as **independent consumers subscribing to per-camera rings**. Once frames land in a per-camera ring, stereo, ArUco, and YOLO are just separate readers — each pulls the frame it needs and does its own thing. Enabling or disabling a feature becomes "construct that consumer, or don't," and the buffer structure never changes. Let me build the whole thing around that and answer your specific questions along the way.

## The three memory tiers (answers CPU-vs-GPU and pinned)

Design for *discrete* memory even though prod is Jetson — that's the portable superset, and on Orin the extra copies are just cheap unified-memory touches while on the RTX box they're real but fast. Three pools:

1. **Host-pinned ingress ring** (per camera) — raw Bayer8 from the source. Pinned (`cudaMallocHost`) so the H2D copy is async and DMA-driven. This answers Q7: yes, frames should land in pinned host memory so they can be `cudaMemcpyAsync`'d to the device.
2. **Device working ring** (per camera) — the debayered RGB, plus derived device buffers (YOLO tensor, ESS input pair). This is the main fan-out surface, GPU-resident, shared to consumers by pointer.
3. **Host-pinned output** — the D2H target for the ArUco branch, since ArUco runs on the CPU.

So the answer to Q8: ingress and the ArUco output live in host-pinned memory; everything the GPU produces and the DNNs consume lives in device memory. Debayer once on the GPU, keep RGB on the GPU, and only the CPU-bound branch (ArUco) pays a copy back.

One Spinnaker caveat on the pinned question: Spinnaker allocates its *own* acquisition buffers, and you can't force it to allocate into your pinned pool. Two clean options — pre-register Spinnaker's buffer pool once at startup with `cudaHostRegister` (its buffers are stable when you fix the count, so register-once works), or `memcpy` each frame from Spinnaker's pageable buffer into a pinned ingress slot (~a few µs for 1.5 MB) and upload from there. The recording and simulator sources you control, so allocate pinned directly.

## The message struct

Make the message a lightweight **handle into a pool slot**, not an owning object. Allocating/freeing a buffer per frame is exactly the latency and fragmentation you want to avoid; the pool owns the memory for the program's life and the message just describes a slot.

```cpp
enum class PixelFormat : uint8_t { Bayer8_RGGB, RGB8, RGBA8, GRAY8 };
enum class MemSpace    : uint8_t { HostPinned, Device };

struct FrameMeta {
    uint64_t    capture_time_ns = 0; // PTP capture time — the ESKF anchor
    uint64_t    sequence_id     = 0; // action-command seq → L/R pairing key #TODO make this timestamp sequence based.
    uint32_t    camera_id       = 0;
    uint32_t    width           = 0;
    uint32_t    height          = 0;
    uint32_t    stride_bytes    = 0; // row pitch, may exceed width*bpp
    PixelFormat format          = PixelFormat::Bayer8_RGGB;
};

// Copyable, owns nothing. Consumers wait on `ready` before touching `ptr`.
struct FrameView {
    FrameMeta   meta;
    void*       ptr        = nullptr; // host or device address (see space)
    MemSpace    space      = MemSpace::Device;
    cudaEvent_t ready      = nullptr; // signaled when the producing op finishes
    uint32_t    slot       = 0;       // index in the owning pool
    uint64_t    generation = 0;       // bumped on reuse → overwrite detection
};
```

Two things earn their place here. `capture_time_ns` travels with the frame all the way to the detectors' outputs, so your ESKF times fused detections by capture instant, not arrival — essential once the three paths have different latencies. And `generation`: because this is latest-wins with drop, a slot can be recycled under a slow consumer; after finishing, the consumer checks `pool.generation(slot) == view.generation` and knows whether it lost the race (if it keeps losing, deepen the ring). That's a cheap ABA guard that makes the lock-free hand-off robust.

## The pool/ring

```cpp
template <MemSpace Space>
class FrameRing {
public:
    FrameRing(std::size_t n_slots, std::size_t bytes_per_slot);

    FrameView acquire();                  // producer: next writable slot (round-robin)
    void      publish(const FrameView&);  // record readiness event + atomically set latest
    FrameView latest() const;             // consumer: newest published slot
    bool      still_valid(const FrameView&) const; // generation check

private:
    std::vector<Buffer>      slots_;       // pinned or device allocations
    std::vector<cudaEvent_t> events_;
    std::vector<uint64_t>    generation_;
    std::atomic<uint32_t>    latest_{~0u}; // published-latest index (lock-free)
};
```

The only synchronization primitives in the whole system are the per-slot CUDA event (producer→consumer GPU ordering) and that one atomic latest-index per ring (the latest-wins hand-off). Ring depth 6–8 as discussed — the safety margin against your slowest consumer (~15 ms ESS at a 16.6 ms period). Eight RGB slots × 4.45 MiB × 2 cameras ≈ 71 MiB, nothing on Orin.

## Arbitrary source

An abstract source that delivers a pinned host frame and nothing more; the pipeline behind it never knows which source it is:

```cpp
class IFrameSource {
public:
    virtual ~IFrameSource() = default;
    virtual bool start() = 0;
    virtual void stop()  = 0;

    // Push model: the source calls this on its own thread when a frame lands,
    // handing a filled host-pinned FrameView. The sink must return promptly
    // (upload + publish) so the source can recycle its buffer.
    using Sink = std::function<void(const FrameView&)>;
    virtual void set_sink(Sink) = 0;
};
```

`SpinnakerSource`, `RecordingSource`, `SimulatorSource` each implement it. Push fits all three: Spinnaker has an image-event callback on its acquisition thread; the recorder runs a thread that reads and rate-limits to real time (replaying the original PTP timestamps so pairing behaves identically to live); the simulator delivers on receipt. A factory picks the concrete type at startup from config — that's inherently a runtime choice (you don't know at compile time whether you'll run live or replay), and a virtual call per frame is free at these rates.

## Toggling stereo / ArUco / YOLO at startup

You want these decided at init and fixed for the run — so read a config, then conditionally *construct* consumers:

```cpp
struct PipelineConfig {
    SourceKind source;         // Spinnaker | Recording | Simulator
    bool enable_stereo = true;
    bool enable_aruco  = true;
    bool enable_yolo   = true;
    int  ring_depth    = 8;
};
```

At startup you always build the per-camera ingest+debayer producers, then: if `enable_stereo`, construct a `StereoConsumer` holding refs to both rings (it matches pairs by `sequence_id` and reads the newest matched pair); if `enable_aruco`, an `ArucoConsumer` on the reference ring (latest → D2H → CPU); if `enable_yolo`, a `YoloConsumer` on the reference ring (latest → resize/normalize → TensorRT). Disabled means never constructed — no runtime branch in the hot path, exactly the "startup-only" behavior you asked for. Stereo-off just drops that one consumer; the per-camera rings still feed the mono paths, so with/without stereo is the same code.

For the heavy optional dependencies, wrap the *compilation* too — `#ifdef ENABLE_YOLO` around the YOLO/TensorRT translation units — so that when YOLO is off you don't link TensorRT or ship the engine at all. That's the compile-time half of "config or compiled": runtime flags choose the graph shape, compile flags strip unused deps and shrink the binary.

## Is Asio a good fit?

Qualified yes, and it helps to be precise about *what* it's doing. Asio's real strength is async **socket** I/O, and in the data path that only matters for sources that are actually networked. Your live-camera path doesn't touch it — Spinnaker owns its own acquisition threading and hands you frames via callback, so Asio isn't in the GigE path at all. Where it genuinely shines is a **simulator that streams frames over TCP/UDP**: that's textbook Asio, and worth using there. Beyond that it's a clean CPU-side executor — an `io_context` as the event loop, `post()` to move work between stages, **strands** to serialize per-camera ring writes without explicit mutexes, and timers for frame-timeout watchdogs (useful for detecting a dropped camera).

The caveat that matters: **don't try to drive CUDA through Asio.** CUDA streams and events own GPU ordering, and there's no clean bridge that lets Asio wait on stream completion as if it were a file descriptor. Your GPU consumers have to block or poll on CUDA events (`cudaEventSynchronize` / `cudaEventQuery`), and a blocking CUDA wait inside an Asio handler starves the Asio thread pool. So the coherent threading model is: Asio (or a small set of plain threads) for ingestion and the CPU-side ArUco stage; a **dedicated thread per GPU consumer** that waits on that consumer's stream events; and CUDA streams doing the actual GPU async. If your sources are all local (Spinnaker + local file + local sim over shared memory), honestly a handful of `std::thread`s plus lock-free rings is simpler and gives more predictable real-time behavior than Asio's machinery — the fan-out itself needs neither. Reach for Asio when a source is networked or when you specifically want its strand/timer/executor model; skip it if everything's local.

## The shape end to end

Per camera: source thread → pinned ingress slot → `cudaMemcpyAsync` H2D → debayer kernel → device RGB ring slot → record event → publish latest → release source buffer. Then, independently: the stereo consumer takes the newest matched pair off both rings; the ArUco and YOLO consumers each take the reference camera's newest slot (not the paired one — freshness, not pairing, as we settled). Every output carries its source frame's `capture_time_ns` for the estimator. The whole thing is one process, three memory tiers, per-slot events plus one atomic per ring, and consumers you switch on at startup — which is precisely the regime where plain CUDA plus this small amount of structure beats pulling in a framework.