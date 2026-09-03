// The pull adapter over a push sink: newest frame wins, and the one it displaced
// goes back to the producer immediately.
//
// The releasing is what this is really testing. A holder that keeps a reference
// to the frame it just replaced costs the producer a pool slot per consumer for
// no reason, and the symptom is a download stage that drops with a pool that
// looks half empty.

#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include "latest_frame.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

// Stands in for a pool slot: what matters here is the refcount, not the pixels.
std::shared_ptr<perception::HostFrame> frame_of(uint64_t sequence) {
  auto frame = std::make_shared<perception::HostFrame>();
  frame->sequence = sequence;
  frame->timestamp_ns = sequence * 1000;
  return frame;
}

void round_trip() {
  std::printf("a frame offered is the frame acquired\n");

  perception::LatestFrame latest("consumer");
  const auto sink = latest.sink();

  const auto produced = frame_of(1);
  sink(produced);

  const auto got = latest.acquire_latest();
  check(got != nullptr && got->sequence == 1, "the consumer got it");
  check(latest.offered() == 1 && latest.skipped() == 0, "one offered, none skipped");
}

void latest_wins_and_counts_the_rest() {
  std::printf("a consumer that looks late gets the newest, and counts what it missed\n");

  perception::LatestFrame latest("consumer");
  const auto sink = latest.sink();

  for (uint64_t i = 1; i <= 3; ++i) sink(frame_of(i));

  const auto got = latest.acquire_latest();
  check(got != nullptr && got->sequence == 3, "the newest, not the oldest");
  check(latest.skipped() == 2, "and the two it stepped over are counted");
  check(latest.take_latest() == nullptr, "nothing left behind it");
}

void the_displaced_frame_goes_back_at_once() {
  std::printf("displacing a frame releases it, rather than pinning a second slot\n");

  perception::LatestFrame latest("consumer");
  const auto sink = latest.sink();

  const auto first = frame_of(1);
  sink(first);
  check(first.use_count() == 2, "held: the producer's copy and the holder's");

  sink(frame_of(2));
  check(first.use_count() == 1, "and dropped the moment a newer one arrived");

  // The consumer's own frame is its to keep; the holder must not still have it.
  const auto held = latest.acquire_latest();
  check(held.use_count() == 1, "what the consumer took is held by the consumer alone");
}

void stop_wakes_a_waiter() {
  std::printf("stop() releases a consumer waiting for a frame\n");

  perception::LatestFrame latest("consumer");

  perception::LatestFrame::Frame got;
  bool returned = false;
  std::thread waiter([&] {
    got = latest.acquire_latest();  // nothing offered yet, so this blocks
    returned = true;
  });

  latest.stop();
  waiter.join();
  check(returned && got == nullptr, "the waiter returned null rather than hanging");
}

void stop_releases_a_frame_nobody_took() {
  std::printf("stop() hands back the frame it was holding\n");

  perception::LatestFrame latest("consumer");
  const auto sink = latest.sink();

  const auto stranded = frame_of(1);
  sink(stranded);
  check(stranded.use_count() == 2, "held by the holder");

  latest.stop();
  check(stranded.use_count() == 1, "and released, rather than pinned until the holder dies");
}

void a_frame_offered_after_stop_is_dropped() {
  std::printf("the producer may outlive the consumer, and its frames are not kept\n");

  perception::LatestFrame latest("consumer");
  const auto sink = latest.sink();
  latest.stop();

  const auto late = frame_of(1);
  sink(late);
  check(late.use_count() == 1, "the late frame went straight back");
  check(latest.acquire_latest() == nullptr, "and the consumer stays stopped");
}

void take_latest_does_not_block() {
  std::printf("take_latest() answers immediately, empty or not\n");

  perception::LatestFrame latest("viewer");
  const auto sink = latest.sink();

  check(latest.take_latest() == nullptr, "null when nothing has arrived");
  sink(frame_of(7));
  const auto got = latest.take_latest();
  check(got != nullptr && got->sequence == 7, "the frame when one has");
  check(latest.take_latest() == nullptr, "and null again once taken");
}

}  // namespace

int main() {
  round_trip();
  latest_wins_and_counts_the_rest();
  the_displaced_frame_goes_back_at_once();
  stop_wakes_a_waiter();
  stop_releases_a_frame_nobody_took();
  a_frame_offered_after_stop_is_dropped();
  take_latest_does_not_block();

  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
