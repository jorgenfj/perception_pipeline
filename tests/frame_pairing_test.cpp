// Behaviour tests for the offline pairing merge. Pure integers -- no GPU, no
// camera, no files.
#include "frame_pairing.hpp"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

using perception::first_timestamp_regression;
using perception::pair_by_timestamp;
using perception::PairResult;

// 60 Hz, in nanoseconds. Half a period is 8'333'333ns.
constexpr uint64_t kPeriod = 16'666'667;
constexpr uint64_t kTol = 500'000;  // 500us, comfortably inside the bound

std::vector<uint64_t> ramp(std::size_t count, int64_t offset_ns = 0) {
  std::vector<uint64_t> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    out.push_back(static_cast<uint64_t>(static_cast<int64_t>(i * kPeriod) + offset_ns));
  }
  return out;
}

// Every input frame must be paired or unpaired exactly once, on every case.
void check_conservation(const PairResult& r, const std::vector<uint64_t>& a,
                        const std::vector<uint64_t>& b, const char* what) {
  const bool counts_ok = r.stats.accounts_for(a.size(), b.size());
  const bool vectors_ok = r.pairs.size() == r.stats.paired &&
                          r.unpaired[0].size() == r.stats.unpaired[0] &&
                          r.unpaired[1].size() == r.stats.unpaired[1];
  check(counts_ok && vectors_ok, what);
}

void test_exact_and_skew() {
  const std::vector<uint64_t> a = ramp(5);

  PairResult r = pair_by_timestamp(a, a, kTol);
  check(r.stats.paired == 5 && r.stats.unpaired[0] == 0 && r.stats.unpaired[1] == 0,
        "identical streams pair completely");
  check(r.stats.max_abs_skew_ns == 0, "identical streams report zero skew");
  check_conservation(r, a, a, "identical streams account for every frame");

  const std::vector<uint64_t> late = ramp(5, 300'000);
  r = pair_by_timestamp(a, late, kTol);
  check(r.stats.paired == 5, "a constant 300us skew still pairs");
  check(r.pairs[0].skew_ns == 300'000, "skew is signed, other minus reference");
  check(r.stats.max_abs_skew_ns == 300'000, "max skew is reported");

  const std::vector<uint64_t> early = ramp(5, -300'000);
  r = pair_by_timestamp(a, early, kTol);
  check(r.stats.paired == 5 && r.pairs[0].skew_ns == -300'000,
        "a negative skew pairs and keeps its sign");
}

void test_tolerance_boundary() {
  const std::vector<uint64_t> a = ramp(5);

  PairResult r = pair_by_timestamp(a, ramp(5, static_cast<int64_t>(kTol)), kTol);
  check(r.stats.paired == 5, "skew exactly at the tolerance pairs (inclusive)");

  r = pair_by_timestamp(a, ramp(5, static_cast<int64_t>(kTol) + 1), kTol);
  check(r.stats.paired == 0 && r.stats.unpaired[0] == 5 && r.stats.unpaired[1] == 5,
        "one nanosecond over the tolerance pairs nothing");
  check_conservation(r, a, a, "and still accounts for every frame");

  r = pair_by_timestamp(a, ramp(5, -static_cast<int64_t>(kTol)), kTol);
  check(r.stats.paired == 5, "the boundary is symmetric on the negative side");
}

void test_dropouts() {
  const std::vector<uint64_t> a = ramp(5);
  const std::vector<uint64_t> b = {0, 3 * kPeriod, 4 * kPeriod};

  PairResult r = pair_by_timestamp(a, b, kTol);
  check(r.stats.paired == 3 && r.unpaired[0] == std::vector<std::size_t>{1, 2} &&
            r.stats.unpaired[1] == 0,
        "a dropout in b leaves exactly those a-frames unpaired");
  check(r.pairs[1].index[0] == 3 && r.pairs[1].index[1] == 1,
        "pairs carry the index into each input, not just the timestamp");
  check_conservation(r, a, b, "a dropout accounts for every frame");

  r = pair_by_timestamp(b, a, kTol);
  check(r.stats.paired == 3 && r.stats.unpaired[0] == 0 &&
            r.unpaired[1] == std::vector<std::size_t>{1, 2},
        "the mirrored case is symmetric");

  const std::vector<uint64_t> c = {0, 1 * kPeriod, 3 * kPeriod, 4 * kPeriod, 6 * kPeriod};
  const std::vector<uint64_t> d = {0, 2 * kPeriod, 3 * kPeriod, 5 * kPeriod, 6 * kPeriod};
  r = pair_by_timestamp(c, d, kTol);
  check(r.stats.paired == 3 && r.unpaired[0] == std::vector<std::size_t>{1, 3} &&
            r.unpaired[1] == std::vector<std::size_t>{1, 3},
        "interleaved dropouts pair only the frames both streams have");
  check_conservation(r, c, d, "interleaved dropouts account for every frame");
}

void test_degenerate() {
  const std::vector<uint64_t> empty;
  const std::vector<uint64_t> a = ramp(3);

  PairResult r = pair_by_timestamp(empty, empty, kTol);
  check(r.stats.paired == 0 && r.stats.unpaired[0] == 0 && r.stats.unpaired[1] == 0,
        "two empty streams pair nothing and report nothing");

  r = pair_by_timestamp(a, empty, kTol);
  check(r.stats.paired == 0 && r.stats.unpaired[0] == 3,
        "an empty partner leaves every frame unpaired");
  check_conservation(r, a, empty, "an empty partner accounts for every frame");

  r = pair_by_timestamp(empty, a, kTol);
  check(r.stats.unpaired[1] == 3, "and the same the other way round");

  // One stream ending early: the tail has to be drained, which is the loop
  // exit rather than the loop body, and is the easy half to forget.
  const std::vector<uint64_t> shorter = ramp(2);
  r = pair_by_timestamp(a, shorter, kTol);
  check(r.stats.paired == 2 && r.unpaired[0] == std::vector<std::size_t>{2},
        "a short partner drains the reference tail");
  check_conservation(r, a, shorter, "a short partner accounts for every frame");
}

void test_monotonicity() {
  const std::vector<uint64_t> good = ramp(5);
  check(first_timestamp_regression(good.data(), good.size()) == good.size(),
        "a rising stream reports no regression");

  std::vector<uint64_t> reset = ramp(5);
  reset[3] = 10;  // a free-running camera clock restarting mid-stream
  check(first_timestamp_regression(reset.data(), reset.size()) == 3,
        "a clock reset is reported at the frame it happens");

  const std::vector<uint64_t> flat(4, 7);
  check(first_timestamp_regression(flat.data(), flat.size()) == flat.size(),
        "equal timestamps are not a regression");

  check(first_timestamp_regression(nullptr, 0) == 0, "an empty stream is trivially monotonic");
}

}  // namespace

int main() {
  test_exact_and_skew();
  test_tolerance_boundary();
  test_dropouts();
  test_degenerate();
  test_monotonicity();

  std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
