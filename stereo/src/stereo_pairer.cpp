#include "stereo_pairer.hpp"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "frame_pairing.hpp"

namespace perception {

StereoPairer::StereoPairer(const Config& config) : config_(config) {
  if (config_.queue_frames == 0) throw std::runtime_error("stereo: queue_frames must be positive");
  // The same check the offline merge makes, in the same place it would matter:
  // before a single frame has been accepted.
  if (config_.frame_period_ns != 0) {
    require_pair_tolerance(config_.tolerance_ns, config_.frame_period_ns);
  }
  overrun_[0].store(0);
  overrun_[1].store(0);
}

bool StereoPairer::push(uint32_t stream, uint64_t timestamp_ns, uint64_t host_recv_ns,
                        uint32_t frame_id, const void* data, std::size_t bytes) {
  if (stream > 1) throw std::runtime_error("stereo: bad stream id");

  std::lock_guard<std::mutex> lock(mutex_[stream]);

  bool dropped = false;
  if (queue_[stream].size() >= config_.queue_frames) {
    // Oldest, not newest: the head is the frame whose partner has had the
    // longest to turn up, so if anything here is stale it is that one.
    recycled_[stream].push_back(std::move(queue_[stream].front().data));
    queue_[stream].pop_front();
    overrun_[stream].fetch_add(1, std::memory_order_relaxed);
    dropped = true;
  }

  Entry entry;
  if (!recycled_[stream].empty()) {
    entry.data = std::move(recycled_[stream].back());
    recycled_[stream].pop_back();
  }
  entry.data.resize(bytes);
  std::memcpy(entry.data.data(), data, bytes);
  entry.timestamp_ns = timestamp_ns;
  entry.host_recv_ns = host_recv_ns;
  entry.frame_id = frame_id;
  entry.bytes = bytes;
  entry.arrived = std::chrono::steady_clock::now();

  queue_[stream].push_back(std::move(entry));
  return !dropped;
}

void StereoPairer::hold_entry(uint32_t stream, Entry&& entry, Pair& out) {
  // Whatever the caller was looking at last time goes back on the free list
  // now, which is also the moment its data pointer stops being valid.
  if (!held_[stream].data.empty()) {
    recycled_[stream].push_back(std::move(held_[stream].data));
  }
  held_[stream] = std::move(entry);

  out.have[stream] = true;
  out.timestamp_ns[stream] = held_[stream].timestamp_ns;
  out.host_recv_ns[stream] = held_[stream].host_recv_ns;
  out.frame_id[stream] = held_[stream].frame_id;
  out.data[stream] = held_[stream].data.data();
  out.bytes[stream] = held_[stream].bytes;
}

bool StereoPairer::try_pop(Pair& out) {
  out = Pair{};

  std::scoped_lock lock(mutex_[0], mutex_[1]);

  while (!queue_[0].empty() && !queue_[1].empty()) {
    // Signed: wall-clock nanoseconds under PTP, so the difference is small even
    // though the operands are enormous.
    const int64_t d = static_cast<int64_t>(queue_[0].front().timestamp_ns) -
                      static_cast<int64_t>(queue_[1].front().timestamp_ns);
    const uint64_t distance = d < 0 ? static_cast<uint64_t>(-d) : static_cast<uint64_t>(d);

    if (distance <= config_.tolerance_ns) {
      Entry a = std::move(queue_[0].front());
      Entry b = std::move(queue_[1].front());
      queue_[0].pop_front();
      queue_[1].pop_front();

      out.pair_id = next_pair_id_++;
      out.skew_ns = -d;  // b - a
      hold_entry(0, std::move(a), out);
      hold_entry(1, std::move(b), out);

      const int64_t abs_skew = out.skew_ns < 0 ? -out.skew_ns : out.skew_ns;
      if (abs_skew > max_abs_skew_ns_) max_abs_skew_ns_ = abs_skew;
      ++paired_;
      return true;
    }

    // The older head can never pair: the other stream's queue is in timestamp
    // order, so everything still to come on that side is newer still. This is
    // the same disposal the offline merge makes, and it is available live for
    // the same reason -- it does not depend on seeing the future, only on the
    // other stream having already got past this instant.
    //
    // It is emitted as a single rather than discarded, so every frame pushed
    // comes out exactly once, paired or not -- the invariant
    // PairStats::accounts_for() checks offline.
    //
    // Emitting matters most in the case this tool exists for. With PTP
    // unlocked the two clocks share no epoch, so one stream's head is *always*
    // the older one and every one of its frames lands here; discarding them
    // meant an unsynced rig drew one camera and left the other half of the
    // window black -- precisely when you need to see both.
    const uint32_t stale = d < 0 ? 0u : 1u;
    Entry entry = std::move(queue_[stale].front());
    queue_[stale].pop_front();
    out.pair_id = next_pair_id_++;
    hold_entry(stale, std::move(entry), out);
    ++unpaired_[stale];
    return true;
  }

  // One side has nothing queued. Everything left is waiting for a partner that
  // may still be in flight, so it is held -- up to Config::hold, after which a
  // camera that has stopped is not allowed to blank the other one.
  const auto now = std::chrono::steady_clock::now();
  for (uint32_t s = 0; s < 2; ++s) {
    if (queue_[s].empty()) continue;
    if (now - queue_[s].front().arrived < config_.hold) continue;

    Entry entry = std::move(queue_[s].front());
    queue_[s].pop_front();
    out.pair_id = next_pair_id_++;
    hold_entry(s, std::move(entry), out);
    ++unpaired_[s];
    return true;
  }

  return false;
}

std::string StereoPairer::health_line() const {
  char buffer[192];
  std::snprintf(buffer, sizeof(buffer),
                "stereo paired=%lu unpaired=%lu/%lu dropped=%lu/%lu max_skew=%.1fus",
                static_cast<unsigned long>(paired_), static_cast<unsigned long>(unpaired_[0]),
                static_cast<unsigned long>(unpaired_[1]), static_cast<unsigned long>(overrun(0)),
                static_cast<unsigned long>(overrun(1)),
                static_cast<double>(max_abs_skew_ns_) * 1e-3);
  return buffer;
}

}  // namespace perception
