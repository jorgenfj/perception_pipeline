#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace perception {

/**
 * @brief The read side of CdrWriter: just enough CDR to get this project's own
 * messages back off a recording.
 *
 * Same two rules, and they bite the same way. Every primitive is aligned to its
 * own size measured from the payload start AFTER the 4-byte encapsulation
 * header, and a fixed-size array carries no length prefix where a sequence
 * carries a uint32 one.
 *
 * Bounds-checked rather than trusting the file: a truncated or foreign message
 * makes ok() false and every subsequent read return zero, so a caller checks
 * once at the end instead of after every field. Reading past the end is not
 * undefined here -- it is a message the replay source skips and counts.
 *
 * tests/cdr_reader.hpp is a SEPARATE implementation, deliberately: the recorder
 * tests exist to catch an alignment bug in CdrWriter, and a reader sharing this
 * one's assumptions would agree with a broken writer perfectly. This one is for
 * production reads; that one is for checking the writer.
 */
class CdrReader {
 public:
  CdrReader(const void* data, std::size_t size)
      : data_(static_cast<const unsigned char*>(data)), size_(size) {
    // The encapsulation header. Anything shorter is not a CDR message at all.
    if (size_ < 4) {
      ok_ = false;
      return;
    }
    at_ = 4;
  }

  uint8_t u8() { return read<uint8_t>(1); }
  bool b() { return u8() != 0; }
  int32_t i32() { return read<int32_t>(4); }
  uint32_t u32() { return read<uint32_t>(4); }
  int64_t i64() { return read<int64_t>(8); }
  uint64_t u64() { return read<uint64_t>(8); }
  float f32() { return read<float>(4); }
  double f64() { return read<double>(8); }

  /**
   * @brief A string: length including the terminator, then the bytes.
   *
   * A zero length is malformed rather than empty -- the terminator is always
   * counted -- so it fails the read instead of yielding "".
   */
  std::string str() {
    const uint32_t len = u32();
    if (!ok_ || len == 0 || at_ + len > size_) {
      ok_ = false;
      return {};
    }
    std::string out(reinterpret_cast<const char*>(data_ + at_), len - 1);
    at_ += len;
    return out;
  }

  /**
   * @brief A uint8 sequence, borrowed rather than copied.
   *
   * The pixels of a 1.5MB frame, so this hands back a pointer into the message
   * the reader already holds. Valid only while that message is alive -- which
   * for a replay is exactly long enough to memcpy it into the sink's slot.
   */
  bool bytes(const unsigned char*& out, std::size_t& count) {
    const uint32_t n = u32();
    if (!ok_ || at_ + n > size_) {
      ok_ = false;
      return false;
    }
    out = data_ + at_;
    count = n;
    at_ += n;
    return true;
  }

  /** @brief False once any read ran past the end or hit a malformed field. */
  bool ok() const { return ok_; }

  /** @brief True when every byte was consumed -- what catches a field read at the wrong offset. */
  bool exhausted() const { return ok_ && at_ == size_; }

  std::size_t offset() const { return at_; }

 private:
  // Alignment is measured from the payload start, i.e. AFTER the encapsulation
  // header -- the same kOrigin CdrWriter aligns against. Aligning to the buffer
  // start instead puts every 8-byte field four bytes out.
  template <typename T>
  T read(std::size_t align) {
    const std::size_t aligned = kOrigin + (at_ - kOrigin + align - 1) / align * align;
    if (!ok_ || aligned + sizeof(T) > size_) {
      ok_ = false;
      return T{};
    }
    T value{};
    std::memcpy(&value, data_ + aligned, sizeof(T));
    at_ = aligned + sizeof(T);
    return value;
  }

  static constexpr std::size_t kOrigin = 4;  // the encapsulation header

  const unsigned char* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t at_ = 0;
  bool ok_ = true;
};

}  // namespace perception
