#pragma once

#include <cstddef>
#include <cstdint>

namespace perception {

// Where a camera puts frames, expressed without naming what is on the other
// side. The source hands the sink its own pinned buffers to fill, so nothing
// here mentions CUDA and this subproject builds with no CUDA toolkit present.
//
// The lifetime contract is the whole point of the interface: a slot handed to
// commit() is off limits to the transport until consumed() says otherwise, and
// the source enforces that by holding the vendor handle for exactly that long.
class FrameSink {
 public:
  static constexpr uint32_t kNoSlot = ~0u;

  virtual ~FrameSink() = default;

  virtual uint32_t slot_count() const = 0;
  virtual std::size_t slot_bytes() const = 0;

  // The slot pointers, contiguous, for handing to a transport that fills memory
  // it did not allocate. Must stay valid for the sink's lifetime.
  virtual void* const* buffers() const = 0;

  // Which slot an address belongs to, or kNoSlot if it is not one of these
  // buffers -- which means the transport ignored them and filled its own.
  virtual uint32_t slot_of(const void* ptr) const = 0;

  // Hand over a slot the transport filled itself, in whatever order it chose.
  virtual void commit(uint32_t slot, uint64_t timestamp_ns, std::size_t bytes) = 0;

  // True once the reader is finished and the slot may be refilled. False until
  // then, which is what stops a camera overwriting a frame still in use.
  virtual bool consumed(uint32_t slot) = 0;
};

}  // namespace perception
