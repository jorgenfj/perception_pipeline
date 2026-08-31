# pipeline/

Rings, leases, and the stages between them. This is the transport layer: it
moves frames from host memory to the GPU and between GPU buffers, and it owns
the rules that keep a buffer from being rewritten while something is still
reading it.


- **[COMPONENTS.md](COMPONENTS.md)** -- what each file in here is, and how they
  compose into a frame path.

## Rules worth not relearning

1. Never use the default stream. The APIs throw if you try.
2. One `consumer_id` per consumer, unique per ring, `< max_consumers`.
3. Wait on `data_ready_event()` before reading a leased slot.
4. `drop_hold()` when the work is *enqueued*, not when it completes -- and only
   from the stream the work went on.
   two rings.
5. Ring depth must exceed the number of simultaneous holders, or
   `acquire_write` throws.
6. Use either `step()` or `start()`, never both
