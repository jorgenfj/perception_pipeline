// The tap ring: one copy, many consumers, latest wins.
//
// The slot accounting is what this is really testing. A frame is unavailable
// while anyone holds it OR while it is the newest, and those two overlap --
// which is easy to get right by inspection and wrong in practice, because the
// symptom is a producer that quietly stops finding slots hours into a run.

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "host_frame_ring.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

constexpr std::size_t kBytes = 64;

std::vector<unsigned char> frame_of(unsigned char seed) {
  return std::vector<unsigned char>(kBytes, seed);
}

bool holds_value(const std::shared_ptr<const perception::HostFrame>& frame, unsigned char seed) {
  if (!frame) return false;
  const auto* p = static_cast<const unsigned char*>(frame->data);
  for (std::size_t i = 0; i < kBytes; ++i) {
    if (p[i] != seed) return false;
  }
  return true;
}

void round_trip() {
  std::printf("a frame published is the frame acquired\n");

  perception::HostFrameRing ring(3, kBytes, 8, 8);
  const uint32_t reader = ring.add_consumer("reader");

  check(ring.publish(frame_of(7).data(), kBytes, 1000), "published");
  const auto frame = ring.acquire_latest(reader);
  check(holds_value(frame, 7), "the bytes survived the copy");
  check(frame->timestamp_ns == 1000, "and so did the stamp");
  check(frame->sequence == 1, "sequence starts at one");
  check(ring.skipped(reader) == 0, "nothing skipped");
}

void latest_wins() {
  std::printf("a consumer that looks late gets the newest, and counts the rest\n");

  perception::HostFrameRing ring(4, kBytes, 8, 8);
  const uint32_t reader = ring.add_consumer("reader");

  for (unsigned char i = 1; i <= 3; ++i) {
    check(ring.publish(frame_of(i).data(), kBytes, 1000 + i), "publish " + std::to_string(i));
  }

  const auto frame = ring.acquire_latest(reader);
  check(holds_value(frame, 3), "got the third, not the first");
  check(ring.skipped(reader) == 2, "and counted the two it stepped over");
}

// The property the whole design exists for.
void a_slow_consumer_pins_one_slot() {
  std::printf("a consumer that never lets go costs exactly one slot\n");

  perception::HostFrameRing ring(4, kBytes, 8, 8);
  const uint32_t slow = ring.add_consumer("slow");

  ring.publish(frame_of(1).data(), kBytes, 1000);
  const auto stuck = ring.acquire_latest(slow);  // held for the rest of this test
  check(holds_value(stuck, 1), "the slow consumer took frame 1");

  // Far more frames than there are slots. With one slot pinned and one holding
  // the newest, two remain to cycle through -- so every publish must succeed.
  bool all_published = true;
  for (int i = 0; i < 200; ++i) {
    if (!ring.publish(frame_of(2).data(), kBytes, 2000 + i)) all_published = false;
  }
  check(all_published, "200 publishes, none dropped, while one slot stayed pinned");
  check(ring.drops() == 0, "the producer never ran out");
  check(holds_value(stuck, 1), "and the pinned frame was never overwritten");
}

void the_producer_drops_when_every_slot_is_pinned() {
  std::printf("with every slot pinned the producer drops rather than waits\n");

  // Two slots, two consumers: one pinned by each, none left to write into.
  perception::HostFrameRing ring(2, kBytes, 8, 8);
  const uint32_t a = ring.add_consumer("a");
  const uint32_t b = ring.add_consumer("b");

  ring.publish(frame_of(1).data(), kBytes, 1000);
  const auto held_a = ring.acquire_latest(a);
  ring.publish(frame_of(2).data(), kBytes, 2000);
  const auto held_b = ring.acquire_latest(b);

  const bool accepted = ring.publish(frame_of(3).data(), kBytes, 3000);
  check(!accepted, "publish refused");
  check(ring.drops() == 1, "and counted it");
  check(holds_value(held_a, 1) && holds_value(held_b, 2), "neither held frame was clobbered");
}

void slots_come_back() {
  std::printf("a released frame frees its slot\n");

  perception::HostFrameRing ring(2, kBytes, 8, 8);
  const uint32_t reader = ring.add_consumer("reader");

  ring.publish(frame_of(1).data(), kBytes, 1000);
  {
    const auto frame = ring.acquire_latest(reader);
    check(ring.in_use() == 1, "one slot unavailable while it is both newest and held");
  }
  check(ring.in_use() == 1, "still one: it is no longer held but it is still the newest");

  check(ring.publish(frame_of(2).data(), kBytes, 2000), "the other slot took the next frame");
  check(ring.in_use() == 1, "and the one it replaced went free");
}

void stop_wakes_a_waiter() {
  std::printf("stop() releases a consumer waiting for a frame\n");

  perception::HostFrameRing ring(2, kBytes, 8, 8);
  const uint32_t reader = ring.add_consumer("reader");

  std::shared_ptr<const perception::HostFrame> got;
  bool returned = false;
  std::thread waiter([&] {
    got = ring.acquire_latest(reader);  // nothing published yet, so this blocks
    returned = true;
  });

  ring.stop();
  waiter.join();
  check(returned && got == nullptr, "the waiter returned null rather than hanging");
}

void two_consumers_share_one_copy() {
  std::printf("both consumers see the same frame, from one publish\n");

  perception::HostFrameRing ring(4, kBytes, 8, 8);
  const uint32_t a = ring.add_consumer("a");
  const uint32_t b = ring.add_consumer("b");

  ring.publish(frame_of(9).data(), kBytes, 1000);
  const auto from_a = ring.acquire_latest(a);
  const auto from_b = ring.acquire_latest(b);

  check(holds_value(from_a, 9) && holds_value(from_b, 9), "both got the bytes");
  check(from_a->data == from_b->data, "and it is literally the same buffer, not a second copy");
  check(ring.published() == 1, "one publish served both");
}

void a_short_frame_reports_its_own_length() {
  // What the recorder writes into the MCAP, so a slot-sized answer here is a
  // message with trailing garbage in it.
  std::printf("a frame carries the length that was published, not the slot's\n");

  perception::HostFrameRing ring(3, kBytes, 8, 8);
  const uint32_t reader = ring.add_consumer("reader");

  ring.publish(frame_of(3).data(), kBytes / 2, 1000);
  const auto partial = ring.acquire_latest(reader);
  check(partial->bytes == kBytes / 2, "the short frame says so");
  check(ring.frame_bytes() == kBytes, "and the ring still reports the capacity");

  // The next full frame must not be capped by the short one before it, which is
  // what happens if the capacity is read back off a frame.
  ring.publish(frame_of(4).data(), kBytes, 2000);
  const auto full = ring.acquire_latest(reader);
  check(holds_value(full, 4) && full->bytes == kBytes, "a full frame still fits after it");
}

}  // namespace

int main() {
  round_trip();
  latest_wins();
  a_slow_consumer_pins_one_slot();
  the_producer_drops_when_every_slot_is_pinned();
  slots_come_back();
  stop_wakes_a_waiter();
  two_consumers_share_one_copy();
  a_short_frame_reports_its_own_length();

  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
