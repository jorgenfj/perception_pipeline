#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// The on-disk shape of a recording, and nothing else. No CUDA, no Spinnaker,
// no GL -- a reader on a laptop with none of those installed can still open a
// file written on the rig.
//
// The format is specified in recording_plan.md; the two rules that matter most
// are worth repeating where the structs are, because they are what stops this
// drifting into something a reader has to branch on:
//
//   1. A recording records what each camera delivered. It does not record what
//      the recorder thought about it -- no capture mode, no pair tolerance, no
//      pair_id. Pairing is a read-time merge over the camera timestamps, see
//      frame_pairing.hpp.
//   2. Streams are independent. Per-stream .idx/.dat files, and no
//      cross-stream relationship expressed in the byte layout at all.

namespace perception {

// Payloads are padded up to this in the .dat, which buys page-aligned reads now
// and the option of O_DIRECT later. 0.08% overhead at 1440x1080 Bayer8.
constexpr std::size_t kRecordAlign = 4096;

// Bumped only when a reader that does not know about the change would
// misinterpret an existing field. Adding a manifest key does not qualify.
constexpr uint32_t kRecordingVersion = 1;

// One frame's entry in camN.idx. Fixed size, little-endian, so frame i sits at
// byte offset i * 32 and seeking is a shift.
#pragma pack(push, 1)
struct IndexRecord {
  // The camera's own clock, verbatim, never rewritten. The only field pairing
  // reads.
  uint64_t timestamp_ns = 0;

  // CLOCK_REALTIME when the host took delivery. Diagnostic only: with PTP
  // locked, host_recv_ns - timestamp_ns is transport latency outright, and its
  // drift across a recording is what proves whether PTP was actually
  // disciplining. See recording_plan.md, "Two stamps, not one".
  uint64_t host_recv_ns = 0;

  // Byte offset into camN.dat. Stored rather than derived from
  // i * record_stride_bytes -- equal today, but it is what keeps a truncated
  // recording readable, and what would have to exist the moment frames stop
  // being fixed-size.
  uint64_t offset = 0;

  // Payload length. Explicit for the same reason.
  uint32_t bytes = 0;

  // The camera's own frame counter. A gap here means the frame never reached
  // the host (packet loss, pool starvation); a frame missing from the index
  // with frame_id contiguous around it means the recorder dropped it. Storing
  // it is what makes those two distinguishable. Restarts at a reconnect, which
  // is also how a reconnect seam is found.
  uint32_t frame_id = 0;
};
#pragma pack(pop)
static_assert(sizeof(IndexRecord) == 32, "IndexRecord must stay 32 bytes: it is a file format");

// One camera's geometry and file names, as the manifest carries them.
struct StreamInfo {
  uint32_t id = 0;
  std::string role;  // "left" / "right" -- a label for humans, never branched on
  std::string serial;

  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t stride_bytes = 0;
  std::size_t frame_bytes = 0;  // what commit() reported, i.e. PayloadSize

  // frame_bytes rounded up to kRecordAlign: the spacing of payloads in the
  // .dat. Written out so a reader need not re-derive the padding rule.
  std::size_t record_stride_bytes = 0;

  // GenICam symbolic name ("BayerRG8"), not a pipeline enum, so playback
  // reconstructs the geometry through the same mapping the live path uses.
  std::string pixel_format;

  uint64_t frames = 0;
  std::string data;   // camN.dat, relative to the recording directory
  std::string index;  // camN.idx
};

struct RecordingManifest {
  uint32_t version = kRecordingVersion;
  std::string created_utc;

  // Single shared time origin for the whole recording: the smallest first
  // timestamp across every stream. Playback pacing derives from this one number
  // and never from a per-stream first frame, so a stream that started late
  // replays late.
  uint64_t epoch_ns = 0;

  // What the node map said at startup, verbatim. Provenance, not a claim that
  // PTP was working -- that is recovered from the index by fitting
  // host_recv_ns - timestamp_ns over time.
  std::string ptp_status_at_start;

  std::vector<StreamInfo> streams;

  // Verbatim copy of the camera features that produced this recording, so the
  // file is self-describing six months later. Read by humans and bug reports,
  // never by the pairing code.
  std::vector<std::pair<std::string, std::string>> camera_features;
};

constexpr std::size_t round_up(std::size_t value, std::size_t to) {
  return ((value + to - 1) / to) * to;
}

// "recording-2026-08-21T13-22-04". Colons are legal on Linux but make a
// directory name awkward to type and illegal elsewhere, so the time separators
// are dashes.
std::string recording_dir_name(uint64_t wall_ns);

// ISO-8601 UTC, for the manifest's created_utc.
std::string iso8601_utc(uint64_t wall_ns);

}  // namespace perception
