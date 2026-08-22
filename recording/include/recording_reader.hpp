#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "frame_pairing.hpp"
#include "recording_format.hpp"

namespace perception {

// Opens a recording directory: manifest, both indices, and payloads on demand.
//
// The whole index is read into memory at open -- 32 bytes a frame, so an hour
// of 60 Hz stereo is 14 MB -- which is what makes the timing of a recording
// analysable without touching a single pixel.
class RecordingReader {
 public:
  // Throws if the directory, the manifest or an index is missing or malformed.
  explicit RecordingReader(const std::string& directory);
  ~RecordingReader();

  RecordingReader(const RecordingReader&) = delete;
  RecordingReader& operator=(const RecordingReader&) = delete;

  const RecordingManifest& manifest() const { return manifest_; }
  std::size_t stream_count() const { return manifest_.streams.size(); }
  const StreamInfo& stream(std::size_t s) const { return manifest_.streams.at(s); }

  // The index, in file order. Frame i of stream s is index(s)[i].
  const std::vector<IndexRecord>& index(std::size_t s) const { return indices_.at(s); }

  // Camera timestamps of one stream, which is all the pairing merge needs.
  const std::vector<uint64_t>& timestamps(std::size_t s) const { return timestamps_.at(s); }

  // Read frame `i` of stream `s` into `dst`, which must hold at least
  // stream(s).frame_bytes. Throws on a short read, which is how a truncated
  // .dat announces itself.
  void read_frame(std::size_t s, std::size_t i, void* dst) const;

  // The pairing, computed here rather than read from the file -- there is
  // nothing in the file to read it from, on purpose. Two streams only: pairing
  // more than that is a different problem and this tool does not have it.
  //
  // Throws if either stream's timestamps regress, which means the camera clock
  // was free-running and reset, and cross-camera stamps then share no epoch.
  PairResult pair(uint64_t tolerance_ns) const;

  // Whether frame i of stream s belongs to a reconnect seam: frame_id jumped
  // backwards while host_recv_ns kept increasing. Nothing in the manifest says
  // so, because the index already does.
  bool is_reconnect_seam(std::size_t s, std::size_t i) const;

  // Human-readable summary of what the file contains -- geometry, duration,
  // gaps, and the PTP verdict recovered from host_recv_ns - timestamp_ns.
  std::string info(uint64_t tolerance_ns) const;

 private:
  std::string directory_;
  RecordingManifest manifest_;
  std::vector<std::vector<IndexRecord>> indices_;
  std::vector<std::vector<uint64_t>> timestamps_;
  std::vector<int> data_fds_;
};

// Parse a manifest.yaml. Exposed separately so a tool can inspect a recording's
// geometry without opening its payloads.
RecordingManifest read_manifest(const std::string& path);

}  // namespace perception
