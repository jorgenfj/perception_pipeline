#pragma once

// The mirror of CdrWriter, and deliberately a separate implementation: a reader
// that shared the writer's alignment bug would agree with it perfectly. The
// fixed-array and sequence readers are separate for the same reason.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace test {

class CdrReader {
 public:
  CdrReader(const unsigned char* data, std::size_t size) : data_(data), size_(size), at_(4) {}
  CdrReader(const std::byte* data, std::size_t size)
      : CdrReader(reinterpret_cast<const unsigned char*>(data), size) {}

  uint8_t u8() { return read<uint8_t>(1); }
  bool b() { return u8() != 0; }
  int32_t i32() { return read<int32_t>(4); }
  uint32_t u32() { return read<uint32_t>(4); }
  int64_t i64() { return read<int64_t>(8); }
  uint64_t u64() { return read<uint64_t>(8); }
  float f32() { return read<float>(4); }
  double f64() { return read<double>(8); }

  std::string str() {
    const uint32_t len = u32();
    if (len == 0 || at_ + len > size_) return {};
    std::string out(reinterpret_cast<const char*>(data_ + at_), len - 1);  // drop the terminator
    at_ += len;
    return out;
  }

  // Fixed size: the count comes from the schema, not the stream.
  std::vector<double> f64_array(std::size_t count) { return run<double>(count, 8); }
  std::vector<float> f32_array(std::size_t count) { return run<float>(count, 4); }

  // Sequences: the count is on the wire, ahead of the elements.
  std::vector<double> f64_seq() { return run<double>(u32(), 8); }
  std::vector<float> f32_seq() { return run<float>(u32(), 4); }

  std::vector<bool> b_seq() {
    const uint32_t count = u32();
    std::vector<bool> out;
    if (at_ + count > size_) return out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) out.push_back(data_[at_ + i] != 0);
    at_ += count;
    return out;
  }

  std::vector<unsigned char> bytes() {
    const uint32_t count = u32();
    std::vector<unsigned char> out;
    if (at_ + count > size_) return out;
    out.assign(data_ + at_, data_ + at_ + count);
    at_ += count;
    return out;
  }

  // What catches a field at the wrong offset: every other check can pass on a
  // stream that is one pad byte long.
  bool exhausted() const { return at_ == size_; }
  std::size_t offset() const { return at_; }

 private:
  template <typename T>
  T read(std::size_t alignment) {
    while ((at_ - 4) % alignment != 0) ++at_;
    T v{};
    if (at_ + sizeof(T) > size_) return v;
    std::memcpy(&v, data_ + at_, sizeof(T));
    at_ += sizeof(T);
    return v;
  }

  template <typename T>
  std::vector<T> run(std::size_t count, std::size_t alignment) {
    std::vector<T> out;
    if (count == 0) return out;  // an empty sequence has nothing after it to align
    while ((at_ - 4) % alignment != 0) ++at_;
    if (at_ + count * sizeof(T) > size_) return out;
    out.resize(count);
    std::memcpy(out.data(), data_ + at_, count * sizeof(T));
    at_ += count * sizeof(T);
    return out;
  }

  const unsigned char* data_;
  std::size_t size_;
  std::size_t at_;
};

}  // namespace test
