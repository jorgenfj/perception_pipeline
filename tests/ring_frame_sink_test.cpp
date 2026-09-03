// The camera/pipeline seam, and the one thing it does that is not plumbing:
// rebasing a PTP camera's TAI stamp onto the host's UTC epoch.
//
// Worth its own test because every way of getting this wrong is silent. Convert
// twice and the timeline is 74s off; convert nowhere and it is 37s off; convert
// after the probe or the tap and two consumers of the same frame disagree by 37s
// -- none of which shows up as a failure, only as a recording nobody can line up
// against anything else.

#include <chrono>
#include <cstdio>
#include <string>

#include "host_frame_ring.hpp"
#include "host_ingress_ring.hpp"
#include "ring_frame_sink.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

constexpr std::size_t kBytes = 256;
constexpr uint64_t kCameraStamp = 1788281225865213416ull;  // a real PTP/TAI stamp from the rig
constexpr int64_t kTaiOffsetNs = 37'000'000'000ll;

perception::FrameMeta meta_at(uint64_t timestamp_ns) {
  perception::FrameMeta meta;
  meta.timestamp_ns = timestamp_ns;
  meta.bytes = kBytes;
  return meta;
}

bool pop_stamp(perception::HostIngressRing& ring, uint64_t& out) {
  perception::HostIngressRing::Staged staged;
  if (!ring.pop(staged, std::chrono::milliseconds(500))) return false;
  out = staged.timestamp_ns;
  return true;
}

void a_camera_stamp_is_rebased_once() {
  std::printf("a TAI stamp reaches the pipeline as UTC\n");

  perception::HostIngressRing ring(2, kBytes, 0, perception::FillMode::External);
  perception::RingFrameSink sink(ring, nullptr, kTaiOffsetNs);

  sink.commit(0, meta_at(kCameraStamp));

  uint64_t got = 0;
  check(pop_stamp(ring, got), "the frame came through");
  check(got == kCameraStamp - static_cast<uint64_t>(kTaiOffsetNs),
        "and its stamp moved by exactly the offset, once");
}

void a_source_already_on_the_host_epoch_is_left_alone() {
  std::printf("a zero offset is a pass-through, so replay is not shifted again\n");

  perception::HostIngressRing ring(2, kBytes, 0, perception::FillMode::External);
  perception::RingFrameSink sink(ring, nullptr, 0);

  sink.commit(0, meta_at(kCameraStamp));

  uint64_t got = 0;
  check(pop_stamp(ring, got), "the frame came through");
  check(got == kCameraStamp, "with the stamp it arrived with");
}

void the_tap_and_the_pipeline_agree() {
  std::printf("both sides of commit() see the same rebased stamp\n");

  perception::HostIngressRing ring(2, kBytes, 0, perception::FillMode::External);
  perception::HostFrameRing tap(3, kBytes, 4, 4);
  const uint32_t consumer = tap.add_consumer("consumer");

  perception::RingFrameSink sink(ring, nullptr, kTaiOffsetNs);
  sink.set_tap(&tap);

  sink.commit(0, meta_at(kCameraStamp));

  uint64_t ingress = 0;
  const bool popped = pop_stamp(ring, ingress);
  const auto tapped = tap.acquire_latest(consumer);

  check(popped && tapped != nullptr, "both paths got the frame");
  if (popped && tapped) {
    check(tapped->timestamp_ns == ingress,
          "and neither is 37s from the other -- one rebase, before the fan-out");
    check(ingress == kCameraStamp - static_cast<uint64_t>(kTaiOffsetNs), "on the host epoch");
  }
}

}  // namespace

int main() {
  try {
    a_camera_stamp_is_rebased_once();
    a_source_already_on_the_host_epoch_is_left_alone();
    the_tap_and_the_pipeline_agree();
  } catch (const std::exception& e) {
    std::printf("  [FAIL] threw: %s\n", e.what());
    ++g_failures;
  }

  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
