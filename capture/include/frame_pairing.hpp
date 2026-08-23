#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace perception {

struct FramePair {
  uint64_t pair_id = 0;
  uint64_t timestamp_ns[2] = {0, 0};
  std::size_t index[2] = {0, 0};
  int64_t skew_ns = 0;  // timestamp_ns[1] - timestamp_ns[0]
};

struct PairStats {
  uint64_t paired = 0;
  uint64_t unpaired[2] = {0, 0};
  int64_t max_abs_skew_ns = 0;

  // Every input frame is either paired or unpaired, exactly once. The merge
  // has no other way to dispose of one, so this is the invariant that catches
  // a dropped or double-counted frame regardless of which branch did it.
  bool accounts_for(std::size_t count_a, std::size_t count_b) const {
    return 2 * paired + unpaired[0] + unpaired[1] == count_a + count_b;
  }
};

struct PairResult {
  std::vector<FramePair> pairs;
  std::vector<std::size_t> unpaired[2];  // indices into the inputs
  PairStats stats;
};

// The tolerance must be strictly under half the frame period.
inline void require_pair_tolerance(uint64_t tolerance_ns, uint64_t min_period_ns) {
  if (min_period_ns == 0 || tolerance_ns >= min_period_ns ||
      tolerance_ns >= min_period_ns - tolerance_ns) {
    throw std::runtime_error(
        "pair tolerance " + std::to_string(tolerance_ns) +
        "ns must be under half the frame period (" + std::to_string(min_period_ns) +
        "ns); at or above that, two different frames qualify as the same instant");
  }
}

// Index of the first timestamp that is smaller than the one before it, or
// `count` if the array is non-decreasing.
//
// A regression is not a corrupt file, it is a diagnosis: the camera clock was
// free-running and reset (typically across a reconnect). Cross-camera stamps
// share no epoch in that case, so pairing them was never going to mean
// anything -- better to say so than to emit plausible nonsense.
inline std::size_t first_timestamp_regression(const uint64_t* timestamps, std::size_t count) {
  for (std::size_t i = 1; i < count; ++i) {
    if (timestamps[i] < timestamps[i - 1]) return i;
  }
  return count;
}

// The merge. Both arrays must be non-decreasing; check them first if they come
// from a file. `tolerance_ns` should have been through require_pair_tolerance().
inline PairResult pair_by_timestamp(const uint64_t* a, std::size_t count_a, const uint64_t* b,
                                    std::size_t count_b, uint64_t tolerance_ns) {
  PairResult result;
  std::size_t i = 0;
  std::size_t j = 0;

  while (i < count_a && j < count_b) {
    // Signed: these are wall-clock nanoseconds under PTP, so the difference is
    // small even though the operands are enormous.
    const int64_t d = static_cast<int64_t>(a[i]) - static_cast<int64_t>(b[j]);
    const uint64_t distance = d < 0 ? static_cast<uint64_t>(-d) : static_cast<uint64_t>(d);

    if (distance <= tolerance_ns) {
      FramePair pair;
      pair.pair_id = result.stats.paired;
      pair.timestamp_ns[0] = a[i];
      pair.timestamp_ns[1] = b[j];
      pair.index[0] = i;
      pair.index[1] = j;
      pair.skew_ns = -d;  // b - a

      const int64_t abs_skew = pair.skew_ns < 0 ? -pair.skew_ns : pair.skew_ns;
      if (abs_skew > result.stats.max_abs_skew_ns) result.stats.max_abs_skew_ns = abs_skew;

      result.pairs.push_back(pair);
      ++result.stats.paired;
      ++i;
      ++j;
    } else if (d < 0) {
      // a[i] is older than everything b has left, so its partner cannot appear.
      result.unpaired[0].push_back(i);
      ++result.stats.unpaired[0];
      ++i;
    } else {
      result.unpaired[1].push_back(j);
      ++result.stats.unpaired[1];
      ++j;
    }
  }

  // Whatever is left over ran out of partners.
  for (; i < count_a; ++i) {
    result.unpaired[0].push_back(i);
    ++result.stats.unpaired[0];
  }
  for (; j < count_b; ++j) {
    result.unpaired[1].push_back(j);
    ++result.stats.unpaired[1];
  }

  return result;
}

inline PairResult pair_by_timestamp(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b,
                                    uint64_t tolerance_ns) {
  return pair_by_timestamp(a.data(), a.size(), b.data(), b.size(), tolerance_ns);
}

}  // namespace perception
