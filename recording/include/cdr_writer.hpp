#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace perception {

/**
 * @brief Just enough CDR to serialise the ROS 2 messages this project writes.
 *
 * Two rules, both silently corrupting when broken. Every primitive is aligned
 * to its own size, measured from the payload start AFTER the 4-byte
 * encapsulation header. And a fixed-size array (float64[9]) carries no length
 * prefix where a sequence (float64[], uint8[]) carries a uint32 one -- a single
 * character apart in the schema, so they are separate methods here.
 */
class CdrWriter {
 public:
  CdrWriter() { clear(); }

  void u8(uint8_t v) { raw(&v, 1, 1); }
  void b(bool v) { u8(v ? 1 : 0); }
  void i32(int32_t v) { raw(&v, 4, 4); }
  void u32(uint32_t v) { raw(&v, 4, 4); }
  void i64(int64_t v) { raw(&v, 8, 8); }
  void u64(uint64_t v) { raw(&v, 8, 8); }
  void f32(float v) { raw(&v, 4, 4); }
  void f64(double v) { raw(&v, 8, 8); }

  /**
   * @brief Write a string: length including the terminator, then the bytes.
   * @param v string_view, not const std::string&, so a literal frame_id written
   *        per sample costs no allocation.
   */
  void str(std::string_view v) {
    u32(static_cast<uint32_t>(v.size() + 1));
    if (!v.empty()) buffer_.insert(buffer_.end(), v.begin(), v.end());
    buffer_.push_back(0);
  }

  /**
   * @brief Write a fixed-size array: no length prefix.
   *
   * One align() covers the run because an element's size is its alignment,
   * which is also why there is no generic array<T>.
   */
  void f64_array(const double* v, std::size_t count) { run(v, count, sizeof(double), 8); }
  void f32_array(const float* v, std::size_t count) { run(v, count, sizeof(float), 4); }

  /** @brief Write a sequence: a uint32 count, then the elements. */
  void f64_seq(const double* v, std::size_t count) {
    u32(static_cast<uint32_t>(count));
    f64_array(v, count);
  }
  void f32_seq(const float* v, std::size_t count) {
    u32(static_cast<uint32_t>(count));
    f32_array(v, count);
  }

  /** @brief bool[]: one octet each, so no alignment to get wrong. */
  void b_seq(const bool* v, std::size_t count) {
    u32(static_cast<uint32_t>(count));
    for (std::size_t i = 0; i < count; ++i) buffer_.push_back(v[i] ? 1 : 0);
  }

  /** @brief uint8[]: a length then the bytes. */
  void bytes(const void* data, std::size_t count) {
    u32(static_cast<uint32_t>(count));
    const auto* p = static_cast<const unsigned char*>(data);
    buffer_.insert(buffer_.end(), p, p + count);
  }

  void reserve(std::size_t extra) { buffer_.reserve(buffer_.size() + extra); }

  /** @brief Empty the message, keeping capacity. What recycling a pooled buffer is. */
  void clear() {
    static constexpr unsigned char kEncapsulation[4] = {0x00, 0x01, 0x00, 0x00};
    buffer_.clear();
    buffer_.insert(buffer_.end(), kEncapsulation, kEncapsulation + 4);
  }

  const std::vector<unsigned char>& data() const { return buffer_; }
  std::size_t size() const { return buffer_.size(); }

  /** Watched across an encode: a capacity that moved means the buffer was too small. */
  std::size_t capacity() const { return buffer_.capacity(); }

 private:
  static constexpr std::size_t kOrigin = 4;  // the encapsulation header

  void align(std::size_t to) {
    while ((buffer_.size() - kOrigin) % to != 0) buffer_.push_back(0);
  }

  void raw(const void* v, std::size_t size, std::size_t alignment) {
    align(alignment);
    const auto* p = static_cast<const unsigned char*>(v);
    buffer_.insert(buffer_.end(), p, p + size);
  }

  void run(const void* v, std::size_t count, std::size_t element, std::size_t alignment) {
    if (count == 0) return;  // an empty sequence is a bare count, with no padding after it
    align(alignment);
    const auto* p = static_cast<const unsigned char*>(v);
    buffer_.insert(buffer_.end(), p, p + count * element);
  }

  std::vector<unsigned char> buffer_;
};

}  // namespace perception
