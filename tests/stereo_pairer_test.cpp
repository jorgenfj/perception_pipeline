// Behaviour tests for the live host-side matcher. No GPU, no camera, no files.
//
// The property that matters most is the last one: what the live matcher pairs
// must be a subset of what the offline merge pairs over the same frames. Live
// can lose a pair -- a partner that never arrived, a queue that overran -- but
// it may never invent one the offline merge would not agree with, because then
// the recording and the live run would disagree about what happened.
#include "stereo_pairer.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "frame_pairing.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

using perception::pair_by_timestamp;
using perception::PairResult;
using perception::StereoPairer;

constexpr uint64_t kPeriod = 16'666'667;  // 60 Hz
constexpr uint64_t kTol = 500'000;

// One byte of payload is enough: nothing here looks at pixels, and a frame that
// carries its own index is easier to assert on than one that carries an image.
void push(StereoPairer& pairer, uint32_t stream, uint64_t timestamp_ns, unsigned char tag) {
  pairer.push(stream, timestamp_ns, timestamp_ns + 1'000'000, static_cast<uint32_t>(tag), &tag, 1);
}

StereoPairer::Config config(uint32_t queue = 8, int hold_ms = 40) {
  StereoPairer::Config c;
  c.tolerance_ns = kTol;
  c.frame_period_ns = kPeriod;
  c.queue_frames = queue;
  c.hold = std::chrono::milliseconds(hold_ms);
  return c;
}

void test_pairs_in_order() {
  StereoPairer pairer(config());
  for (unsigned char i = 0; i < 4; ++i) {
    push(pairer, 0, i * kPeriod, i);
    push(pairer, 1, i * kPeriod + 200'000, i);
  }

  StereoPairer::Pair pair;
  int count = 0;
  while (pairer.try_pop(pair)) {
    check(pair.complete(), "a frame and its partner come out as one pair");
    check(pair.skew_ns == 200'000, "skew is signed, other minus reference");
    check(*pair.data[0] == count && *pair.data[1] == count, "the payload is the frame's own");
    check(pair.pair_id == static_cast<uint64_t>(count), "pair ids are consecutive from zero");
    ++count;
  }
  check(count == 4, "every complete pair is emitted");
  check(pairer.paired() == 4 && pairer.unpaired(0) == 0 && pairer.unpaired(1) == 0,
        "the counters agree with what came out");
  check(pairer.max_abs_skew_ns() == 200'000, "worst skew is remembered");
}

void test_tolerance_is_enforced() {
  // The bound is strictly under half the period, and 16'666'667 is odd, so
  // 8'333'333 is genuinely inside it and 8'333'334 is the first value at which
  // a frame midway between two partners would match both.
  const auto refused = [](uint64_t tolerance_ns) {
    StereoPairer::Config c = config();
    c.tolerance_ns = tolerance_ns;
    try {
      StereoPairer pairer(c);
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };

  check(!refused(kPeriod / 2), "the largest tolerance under half a period is allowed");
  check(refused(kPeriod / 2 + 1), "one nanosecond past it is refused, as offline");
  check(refused(kPeriod), "and so is anything larger");

  StereoPairer pairer(config());
  push(pairer, 0, 10 * kPeriod, 0);
  push(pairer, 1, 10 * kPeriod + kTol + 1, 1);  // one nanosecond outside
  StereoPairer::Pair pair;
  // Neither can pair with the other, and the older one is disposed of first.
  check(pairer.try_pop(pair) == false || !pair.complete(),
        "a partner one nanosecond outside the tolerance is not a partner");
}

void test_unpairable_frame_is_surfaced_not_dropped() {
  StereoPairer pairer(config());
  // Stream 0 runs; stream 1 misses frame 1 entirely.
  push(pairer, 0, 0, 0);
  push(pairer, 0, kPeriod, 1);
  push(pairer, 0, 2 * kPeriod, 2);
  push(pairer, 1, 0, 0);
  push(pairer, 1, 2 * kPeriod, 2);

  StereoPairer::Pair pair;
  check(pairer.try_pop(pair) && pair.complete() && *pair.data[0] == 0, "frame 0 pairs");

  // Frame 1 can never pair -- stream 1 has already got past that instant -- but
  // it is still handed out, as a single. Discarding it is what left an unsynced
  // rig drawing one camera and a black rectangle.
  check(pairer.try_pop(pair) && !pair.complete() && pair.have[0] && *pair.data[0] == 1,
        "the unpairable frame comes out as a single");
  check(pairer.unpaired(0) == 1, "and is counted as unpaired");

  check(pairer.try_pop(pair) && pair.complete() && *pair.data[0] == 2, "then frame 2 pairs");
  check(pairer.try_pop(pair) == false, "nothing is left over");
}

// Every frame pushed comes out exactly once, paired or single -- the invariant
// PairStats::accounts_for() checks offline, now also true live. This is what
// guarantees both halves of the window keep updating however badly the rig is
// synced.
void test_every_frame_is_accounted_for() {
  StereoPairer pairer(config(64));

  // Two clocks with no common epoch, which is what an unsynced rig produces:
  // stream 1 is a full second away, so nothing can ever pair and stream 1's
  // head is always the older one.
  constexpr uint64_t kOffset = 1'000'000'000;
  for (unsigned char i = 0; i < 8; ++i) {
    push(pairer, 0, kOffset + i * kPeriod, i);
    push(pairer, 1, i * kPeriod, i);
  }

  uint32_t seen[2] = {0, 0};
  StereoPairer::Pair pair;
  while (pairer.try_pop(pair)) {
    check(!pair.complete(), "two clocks a second apart never pair");
    for (uint32_t s = 0; s < 2; ++s) {
      if (pair.have[s]) ++seen[s];
    }
  }

  check(seen[1] == 8, "every frame of the older stream is still surfaced");
  check(pairer.unpaired(1) == 8, "and counted");
  // Stream 0's frames are still queued waiting out the hold, which is correct:
  // their partner could still arrive. The point is that stream 1's were not
  // silently thrown away while stream 0 monopolised the display.
  check(seen[0] + 8 >= seen[1], "neither stream is starved of the display");
}

void test_one_camera_stopped() {
  StereoPairer pairer(config(8, 5));
  push(pairer, 0, 0, 7);

  StereoPairer::Pair pair;
  check(!pairer.try_pop(pair), "a lone frame waits for its partner first");

  std::this_thread::sleep_for(std::chrono::milliseconds(12));
  check(pairer.try_pop(pair), "past the hold, a lone frame is released rather than lost");
  check(pair.have[0] && !pair.have[1], "it is released as a single, not as a pair");
  check(*pair.data[0] == 7, "and it is the frame that was waiting");
  check(pairer.unpaired(0) == 1, "counted as unpaired");
}

void test_queue_overrun_drops_the_oldest() {
  StereoPairer pairer(config(2));
  for (unsigned char i = 0; i < 4; ++i) push(pairer, 0, i * kPeriod, i);
  check(pairer.overrun(0) == 2, "a full queue drops, and counts what it dropped");

  // The two survivors are the newest (frames 2 and 3), so the partner that
  // turns up next still finds something to pair with. Frame 2 comes out first
  // as a single -- it is older than anything stream 1 has left -- and then
  // frame 3 pairs.
  push(pairer, 1, 3 * kPeriod, 3);
  StereoPairer::Pair pair;
  check(pairer.try_pop(pair) && !pair.complete() && *pair.data[0] == 2,
        "the older survivor is surfaced as a single");
  check(pairer.try_pop(pair) && pair.complete() && *pair.data[0] == 3,
        "the newest frames are the ones kept");
}

void test_data_survives_until_the_next_pop() {
  StereoPairer pairer(config());
  push(pairer, 0, 0, 11);
  push(pairer, 1, 0, 22);
  push(pairer, 0, kPeriod, 33);
  push(pairer, 1, kPeriod, 44);

  StereoPairer::Pair first;
  check(pairer.try_pop(first), "first pair comes out");
  const unsigned char* held = first.data[0];
  check(*held == 11, "its buffer reads back what was pushed");

  StereoPairer::Pair second;
  check(pairer.try_pop(second), "second pair comes out");
  check(*second.data[0] == 33, "and carries its own frame, not the previous one");
}

// The cross-check from sync_plan.md: live pairs must be a subset of offline
// pairs over the same frames.
void test_live_pairs_are_a_subset_of_offline() {
  // A stream with jitter, a dropout on each side, and a skew that wanders --
  // enough for the two algorithms to disagree about yield if they are going to.
  std::vector<uint64_t> a, b;
  for (int i = 0; i < 40; ++i) {
    const uint64_t base = static_cast<uint64_t>(i) * kPeriod;
    if (i != 11) a.push_back(base + static_cast<uint64_t>((i * 7919) % 40'000));
    if (i != 23) b.push_back(base + 120'000 + static_cast<uint64_t>((i * 104729) % 40'000));
  }

  const PairResult offline = pair_by_timestamp(a, b, kTol);

  StereoPairer pairer(config(64));
  std::size_t i = 0, j = 0;
  // Interleaved in timestamp order, i.e. the order the host would have seen
  // them. The queue is deep enough that nothing is lost to an overrun, so any
  // difference in yield is the algorithm's and not the harness's.
  while (i < a.size() || j < b.size()) {
    if (j >= b.size() || (i < a.size() && a[i] <= b[j])) {
      push(pairer, 0, a[i], static_cast<unsigned char>(i));
      ++i;
    } else {
      push(pairer, 1, b[j], static_cast<unsigned char>(j));
      ++j;
    }
  }

  std::vector<std::pair<uint64_t, uint64_t>> live;
  StereoPairer::Pair pair;
  while (pairer.try_pop(pair)) {
    if (pair.complete()) live.emplace_back(pair.timestamp_ns[0], pair.timestamp_ns[1]);
  }

  bool subset = true;
  for (const auto& [ts_a, ts_b] : live) {
    bool found = false;
    for (const perception::FramePair& offline_pair : offline.pairs) {
      if (offline_pair.timestamp_ns[0] == ts_a && offline_pair.timestamp_ns[1] == ts_b) {
        found = true;
        break;
      }
    }
    if (!found) subset = false;
  }

  check(!live.empty(), "the live matcher paired something to compare");
  check(subset, "every live pair is a pair the offline merge also makes");
  check(live.size() == offline.stats.paired,
        "with a deep enough queue and every frame fed, live yields what offline does");
}

}  // namespace

int main() {
  test_pairs_in_order();
  test_tolerance_is_enforced();
  test_unpairable_frame_is_surfaced_not_dropped();
  test_every_frame_is_accounted_for();
  test_one_camera_stopped();
  test_queue_overrun_drops_the_oldest();
  test_data_survives_until_the_next_pop();
  test_live_pairs_are_a_subset_of_offline();

  std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
