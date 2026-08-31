# The components

## `host_ingress_ring.hpp/.cpp`
Pinned (`cudaMallocHost`) staging between the acquisition thread and the GPU,
with a per-slot `h2d_done` event. The pinned Host allocated memory is what allows us to outsource the copy from host->device to the DMA. Two fill modes:

## `device_ring_buffer.hpp/.cpp`
The device-side ring: N `cudaMalloc`'d slots, the seq/readers state
[LEASES.md](LEASES.md) describes, and per-slot events. Optionally creates a `cudaTextureObject_t` per slot
(`DeviceRingTextureDesc`) so consumers can sample through the Texture Mapping Unit (TMU).

`ReuseWait` picks how the producer waits for its own previous write to the slot:
`HostSync` blocks the host on `cudaEventSynchronize`; `DeviceWait` only orders
the stream. `WritePolicy` picks the slot: `RoundRobin` (one candidate, always
blocking) or `ScanForFree` (scan, and `try_acquire_write` becomes legal).


- **`Staged`** — we own the slot order: `acquire()` → `host_ptr()` → `commit()`,
  with `abandon()` for a frame that never arrived. `acquire()` synchronises on
  that slot's `h2d_done` before handing it back, which is what stops a refill
  from racing an in-flight upload.
- **`External`** — a vendor DMA (Spinnaker's `SetUserBuffers`) writes into the
  slots and picks which one. `buffers()` exposes the pointer array,
  `slot_of()` maps an address back, `commit_external()` accepts any slot in any
  order, and `slot_consumed()` tells the transport when its handle may go back
  to the pool. The `SlotState` enum exists only for this mode: in `Staged` the
  acquired/released counters already say where a slot is.

## `upload_stage.hpp/.cpp`
The host→device boundary. Pops a staged frame, H2D-copies it, optionally runs a
transform (see `device_transform` below), and publishes to a device ring. Owns one non-blocking stream and,
when transforming, a small scratch ring so the copy and the kernel are
decoupled. `release()`s the pinned slot the moment the copy is *enqueued*.

## `device_transform.hpp/.cpp`
The interface the domain implements: `output_desc()`, optional `prepare()`, and
`enqueue(src, in_desc, dst, out_desc, stream)`.


## `transform_stage.hpp/.cpp`
The device→device hop: input ring → transform → output ring, one stream, no
scratch (the input slot is already device memory in the right place) and no
backpressure (latest-wins input; nothing to pop, nothing to release).

## `ring_pair_consumer.hpp/.cpp`
Pairs two device rings by timestamp: lease the newest from the reference ring,
find the partner in the other within `tolerance_ns`, hand **both leases plus a
stream already ordered behind both `data_ready_event`s** to a callback.


