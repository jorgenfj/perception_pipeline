// Encoder round trip, with no file, no mcap and no threads.
//
// mcap_recorder_test proves a reader can open the file; this proves the bytes in
// it are right. An alignment bug does not fail loudly -- it decodes into
// plausible nonsense -- so it is worth catching without a filesystem.

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "cdr_reader.hpp"
#include "cdr_writer.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

test::CdrReader reader_for(const perception::CdrWriter& cdr) {
  return test::CdrReader(cdr.data().data(), cdr.size());
}

// Bit patterns that a byte swap or a four-byte shift cannot survive. A round
// number like 1.0 has six zero bytes and agrees with far too many bugs.
constexpr uint64_t kOddU64 = 0x0123456789ABCDEFull;
constexpr int64_t kOddI64 = -0x0123456789ABCDEFll;
constexpr uint32_t kOddU32 = 0xDEADBEEFu;
constexpr int32_t kOddI32 = -123456789;

void primitives() {
  std::printf("primitives round trip\n");

  perception::CdrWriter cdr;
  cdr.u8(0xA5);
  cdr.b(true);
  cdr.i32(kOddI32);
  cdr.u32(kOddU32);
  cdr.i64(kOddI64);
  cdr.u64(kOddU64);
  cdr.f32(std::numeric_limits<float>::denorm_min());
  cdr.f64(-0.0);

  test::CdrReader r = reader_for(cdr);
  check(r.u8() == 0xA5, "uint8");
  check(r.b(), "bool");
  check(r.i32() == kOddI32, "int32, negative");
  check(r.u32() == kOddU32, "uint32");
  check(r.i64() == kOddI64, "int64, negative");
  check(r.u64() == kOddU64, "uint64, every byte distinct");
  check(r.f32() == std::numeric_limits<float>::denorm_min(), "float32 denormal");

  // -0.0 is the one double whose only set bit is the sign: a writer that lost
  // it would still compare equal to 0.0.
  const double zero = r.f64();
  check(zero == 0.0 && std::signbit(zero), "float64 negative zero, sign preserved");

  check(r.exhausted(), "no field left unread");
}

// The strongest check available. An independent reader still agrees with a
// writer that is wrong by a consistent offset; a literal does not.
void byte_layout() {
  std::printf("byte layout, against a literal\n");

  perception::CdrWriter cdr;
  cdr.u8(0xAB);
  cdr.f64(1.0);

  const std::vector<unsigned char> expected = {
      0x00, 0x01, 0x00, 0x00,                          // encapsulation: CDR_LE
      0xAB,                                            // the uint8
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,        // 7 pad, to align the double
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F,  // 1.0, little endian
  };

  check(cdr.data() == expected, "uint8 then float64 is exactly the 20 bytes CDR says");
}

void alignment() {
  std::printf("alignment after strings\n");

  // A string leaves the payload at an arbitrary offset. Each of these must
  // still land its double at payload offset 8.
  for (const std::string& name : {std::string(), std::string("a"), std::string("ab"),
                                  std::string("abc"), std::string("abcd")}) {
    perception::CdrWriter cdr;
    cdr.str(name);
    cdr.f64(2.5);

    test::CdrReader r = reader_for(cdr);
    const std::string back = r.str();
    const double value = r.f64();
    check(back == name && value == 2.5 && r.exhausted(),
          "str(\"" + name + "\") then float64 round trips and consumes the stream");
  }

  // uint32 then a covariance: four bytes of pad the reader has to skip.
  const double nine[9] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
  perception::CdrWriter cdr;
  cdr.u32(7);
  cdr.f64_array(nine, 9);
  check(cdr.size() == 4 + 4 + 4 + 72, "uint32 then float64[9] pads to 84 bytes");

  test::CdrReader r = reader_for(cdr);
  check(r.u32() == 7, "the uint32 ahead of the array");
  check(r.f64_array(9) == std::vector<double>(nine, nine + 9), "float64[9] after the pad");
  check(r.exhausted(), "no pad left over");
}

// The difference that costs a whole field rather than a shift.
void fixed_versus_sequence() {
  std::printf("fixed arrays carry no length, sequences do\n");

  const double nine[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  const float four[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  perception::CdrWriter fixed;
  fixed.f64_array(nine, 9);
  check(fixed.size() == 4 + 72, "float64[9] is 72 bytes and no prefix");

  perception::CdrWriter seq;
  seq.f32_seq(four, 4);
  check(seq.size() == 4 + 4 + 16, "float32[] is a uint32 count then 16 bytes");

  test::CdrReader rf = reader_for(fixed);
  check(rf.f64_array(9) == std::vector<double>(nine, nine + 9), "the fixed array reads back");
  check(rf.exhausted(), "fixed array consumed exactly");

  test::CdrReader rs = reader_for(seq);
  check(rs.f32_seq() == std::vector<float>(four, four + 4), "the sequence reads back");
  check(rs.exhausted(), "sequence consumed exactly");

  // An empty sequence is a bare count. Padding after it is padding no reader
  // skips, so there must not be any.
  perception::CdrWriter empty;
  empty.u8(1);
  empty.f64_seq(nullptr, 0);
  check(empty.size() == 4 + 1 + 3 + 4, "an empty float64[] pads for its count and stops");

  const bool flags[5] = {true, false, true, true, false};
  perception::CdrWriter bools;
  bools.b_seq(flags, 5);
  check(bools.size() == 4 + 4 + 5, "bool[] is one octet each, no alignment");
  test::CdrReader rb = reader_for(bools);
  check(rb.b_seq() == std::vector<bool>(flags, flags + 5), "bool[] reads back");
  check(rb.exhausted(), "bool sequence consumed exactly");
}

// The pool invariant: a clear() that released capacity would make every
// recycled buffer reallocate on the pushing thread.
void clear_keeps_capacity() {
  std::printf("clear() empties the message and keeps the buffer\n");

  perception::CdrWriter cdr;
  cdr.reserve(1u << 20);
  const std::size_t reserved = cdr.capacity();
  check(reserved >= (1u << 20), "reserve() took the capacity");

  const std::vector<unsigned char> payload(4096, 0x5A);
  cdr.bytes(payload.data(), payload.size());
  check(cdr.size() > 4, "something was written");

  cdr.clear();
  check(cdr.size() == 4, "cleared back to the encapsulation header alone");
  check(cdr.capacity() == reserved, "capacity survived the clear");

  // And the cleared writer still produces a correct message.
  cdr.str("again");
  test::CdrReader r = reader_for(cdr);
  check(r.str() == "again" && r.exhausted(), "a cleared writer encodes from the start");
  check(cdr.capacity() == reserved, "and still did not reallocate");
}

}  // namespace

int main() {
  primitives();
  byte_layout();
  alignment();
  fixed_versus_sequence();
  clear_keeps_capacity();

  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
