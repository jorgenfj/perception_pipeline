#include "spinnaker_source.hpp"

#include <SpinGenApi/SpinnakerGenApi.h>

#include <cstdarg>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace perception {
namespace {

using namespace Spinnaker::GenApi;

#ifdef PERCEPTION_TRACE_POOL
[[gnu::format(printf, 1, 2)]] void trace_pool(const char* fmt, ...) {
  std::va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
}
#else
[[gnu::format(printf, 1, 2)]] void trace_pool(const char*, ...) {}
#endif

// USB3 packet size. Buffers not rounded up to it tear on USB3 cameras; on GigE
// it is harmless padding.
constexpr std::size_t kUsbPacketSize = 1024;

// Spinnaker::ImageStatus ranges -1 (UNKNOWN_ERROR) .. 14
// (GENDC_PART_DATA_INVALID); +1 maps it into a dense 0-based index.
std::size_t incompleteStatusIndex(Spinnaker::ImageStatus status) {
  return static_cast<std::size_t>(static_cast<int>(status) + 1);
}

// Short, greppable names -- what actually distinguishes real network packet
// loss from a CRC failure or a host-side resource problem. See
// https://www.teledynevisionsolutions.com/support/support-center/troubleshooting/iis/troubleshooting-image-consistency-errors/
const char* incompleteStatusName(Spinnaker::ImageStatus status) {
  switch (status) {
    case Spinnaker::SPINNAKER_IMAGE_STATUS_CRC_CHECK_FAILED:
      return "crc";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_DATA_OVERFLOW:
      return "data_overflow";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_MISSING_PACKETS:
      return "missing_packets";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_LEADER_BUFFER_SIZE_INCONSISTENT:
      return "leader_size_inconsistent";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_TRAILER_BUFFER_SIZE_INCONSISTENT:
      return "trailer_size_inconsistent";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_PACKETID_INCONSISTENT:
      return "packetid_inconsistent";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_MISSING_LEADER:
      return "missing_leader";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_MISSING_TRAILER:
      return "missing_trailer";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_DATA_INCOMPLETE:
      return "data_incomplete";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_INFO_INCONSISTENT:
      return "info_inconsistent";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_CHUNK_DATA_INVALID:
      return "chunk_data_invalid";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_NO_SYSTEM_RESOURCES:
      return "no_system_resources";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_GENDC_DATA_INVALID:
      return "gendc_data_invalid";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_GENDC_PART_DATA_INVALID:
      return "gendc_part_data_invalid";
    case Spinnaker::SPINNAKER_IMAGE_STATUS_NO_ERROR:
      return "no_error";
    default:
      return "unknown_error";
  }
}

void selectEnum(INodeMap& nodes, const char* node, const char* entry) {
  const CEnumerationPtr selector = nodes.GetNode(node);
  if (!IsWritable(selector)) {
    throw std::runtime_error(std::string(node) + " is not writable");
  }
  const CEnumEntryPtr value = selector->GetEntryByName(entry);
  if (!IsReadable(value)) {
    throw std::runtime_error(std::string(node) + " has no entry " + entry);
  }
  selector->SetIntValue(value->GetValue());
}

bool parseBool(const std::string& text, const std::string& node) {
  if (text == "true" || text == "True" || text == "1") return true;
  if (text == "false" || text == "False" || text == "0") return false;
  throw std::runtime_error(node + ": expected a boolean, got '" + text + "'");
}

// One setter for every feature, typed by the node itself rather than by the
// config. That is what lets the config name any node the camera exposes
// without this file knowing anything about it.
void applyFeature(INodeMap& nodes, const std::string& name, const std::string& value) {
  const CNodePtr node = nodes.GetNode(name.c_str());
  if (!node.IsValid()) throw std::runtime_error(name + ": no such node on this camera");
  if (!IsWritable(node)) throw std::runtime_error(name + ": not writable in this state");

  switch (node->GetPrincipalInterfaceType()) {
    case intfIInteger:
      CIntegerPtr(node)->SetValue(std::stoll(value));
      return;
    case intfIFloat:
      CFloatPtr(node)->SetValue(std::stod(value));
      return;
    case intfIBoolean:
      CBooleanPtr(node)->SetValue(parseBool(value, name));
      return;
    case intfIString:
      CStringPtr(node)->SetValue(value.c_str());
      return;
    case intfIEnumeration: {
      const CEnumerationPtr selector(node);
      const CEnumEntryPtr entry = selector->GetEntryByName(value.c_str());
      if (!IsReadable(entry)) {
        throw std::runtime_error(name + ": has no entry '" + value + "'");
      }
      selector->SetIntValue(entry->GetValue());
      return;
    }
    case intfICommand:
      // Value ignored: a command node has none, it just fires.
      CCommandPtr(node)->Execute();
      return;
    default:
      throw std::runtime_error(name + ": node type cannot be set from config");
  }
}

void applyFeatures(INodeMap& nodes, const FeatureList& features, const char* what) {
  for (const auto& [name, value] : features) {
    try {
      applyFeature(nodes, name, value);
    } catch (const Spinnaker::Exception& e) {
      throw std::runtime_error(std::string(what) + ": " + name + ": " + e.what());
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string(what) + ": " + e.what());
    }
  }
}

std::string currentEnum(INodeMap& nodes, const char* name) {
  const CEnumerationPtr selector = nodes.GetNode(name);
  if (!IsReadable(selector)) return {};
  const CEnumEntryPtr entry = selector->GetCurrentEntry();
  return IsReadable(entry) ? std::string(entry->GetSymbolic().c_str()) : std::string();
}

int64_t readInt(INodeMap& nodes, const char* node) {
  const CIntegerPtr value = nodes.GetNode(node);
  if (!IsReadable(value)) {
    throw std::runtime_error(std::string(node) + " is not readable");
  }
  return value->GetValue();
}

}  // namespace

Spinnaker::CameraPtr SpinnakerSource::select(Spinnaker::CameraList& cameras,
                                             const std::string& serial) {
  if (cameras.GetSize() == 0) throw std::runtime_error("no cameras detected");
  if (serial.empty()) return cameras.GetByIndex(0);

  Spinnaker::CameraPtr camera = cameras.GetBySerial(serial);
  if (!camera.IsValid()) {
    throw std::runtime_error("no camera with serial " + serial);
  }
  return camera;
}

SpinnakerSource::SpinnakerSource(Spinnaker::CameraPtr camera, CameraConfig config)
    : camera_(camera), config_(std::move(config)) {
  camera_->Init();

  INodeMap& nodes = camera_->GetNodeMap();
  INodeMap& stream = camera_->GetTLStreamNodeMap();

  applyFeatures(nodes, config_.features, "camera.features");
  applyFeatures(stream, config_.stream_features, "camera.stream_features");

  // Forced, not configurable: user buffers are only honoured with a manually
  // set count. On Auto the engine sizes its own pool and ignores what we hand
  // it, which silently drops the whole zero-copy path.
  selectEnum(stream, "StreamBufferCountMode", "Manual");

  // The overwrite modes keep one buffer in hand to swap, so they need three
  // where the plain queueing modes need two.
  const std::string handling = currentEnum(stream, "StreamBufferHandlingMode");
  min_slots_ = (handling == "NewestOnly" || handling == "OldestFirstOverwrite") ? 3 : 2;

  geometry_.width = static_cast<uint32_t>(readInt(nodes, "Width"));
  geometry_.height = static_cast<uint32_t>(readInt(nodes, "Height"));
  geometry_.pixel_format = currentEnum(nodes, "PixelFormat");
  geometry_.frame_bytes = static_cast<std::size_t>(readInt(nodes, "PayloadSize"));

  // PayloadSize is the authority on transfer size, so the row pitch follows
  // from it rather than from width * bpp -- the camera may pad rows.
  geometry_.stride_bytes =
      geometry_.height ? static_cast<uint32_t>(geometry_.frame_bytes / geometry_.height) : 0;

  geometry_.buffer_bytes =
      ((geometry_.frame_bytes + kUsbPacketSize - 1) / kUsbPacketSize) * kUsbPacketSize;
}

SpinnakerSource::~SpinnakerSource() {
  stop();
  if (camera_ && camera_->IsInitialized()) camera_->DeInit();
}

std::string SpinnakerSource::ptp_status() {
  INodeMap& nodes = camera_->GetNodeMap();
  return currentEnum(nodes, "GevIEEE1588Status");
}

bool SpinnakerSource::ptp_offset_ns(int64_t& out) {
  INodeMap& nodes = camera_->GetNodeMap();
  const CIntegerPtr value = nodes.GetNode("GevIEEE1588OffsetFromMaster");
  if (!IsReadable(value)) return false;
  out = value->GetValue();
  return true;
}

std::string SpinnakerSource::incomplete_breakdown() const {
  std::ostringstream out;
  bool first = true;
  for (int status = -1; status <= 14; ++status) {
    const uint64_t count = incomplete_by_status_[incompleteStatusIndex(
                                static_cast<Spinnaker::ImageStatus>(status))]
                               .load(std::memory_order_relaxed);
    if (!count) continue;
    if (!first) out << ' ';
    first = false;
    out << incompleteStatusName(static_cast<Spinnaker::ImageStatus>(status)) << '=' << count;
  }
  return out.str();
}

void SpinnakerSource::bind_buffers(FrameSink& sink) {
  if (sink.slot_bytes() < geometry_.buffer_bytes) {
    throw std::runtime_error("SpinnakerSource: sink slots are smaller than buffer_bytes");
  }
  if (sink.slot_count() < min_slots_) {
    throw std::runtime_error("SpinnakerSource: this stream mode needs at least " +
                             std::to_string(min_slots_) + " user buffers; the sink has " +
                             std::to_string(sink.slot_count()));
  }

  camera_->SetBufferOwnership(Spinnaker::SPINNAKER_BUFFER_OWNERSHIP_USER);
  // Non-contiguous overload: the sink's slots are separate allocations, not one
  // block.
  camera_->SetUserBuffers(const_cast<void**>(sink.buffers()), sink.slot_count(),
                          sink.slot_bytes());
}

void SpinnakerSource::start(FrameSink& sink) {
  if (running_.exchange(true)) return;
  held_.assign(sink.slot_count(), nullptr);
  held_since_.assign(sink.slot_count(), std::chrono::steady_clock::time_point{});
  held_now_.store(0, std::memory_order_relaxed);
  held_peak_.store(0, std::memory_order_relaxed);
  was_starved_ = false;
  thread_ = std::thread(&SpinnakerSource::run, this, std::ref(sink));
}

void SpinnakerSource::stop() {
  if (!running_.exchange(false)) return;
  if (thread_.joinable()) thread_.join();
}

void SpinnakerSource::reclaim(FrameSink& sink) {
  for (uint32_t slot = 0; slot < held_.size(); ++slot) {
    if (held_[slot] && sink.consumed(slot)) {
      const auto held_us = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - held_since_[slot])
                               .count();
      held_[slot]->Release();
      held_[slot] = nullptr;
      held_now_.fetch_sub(1, std::memory_order_relaxed);

      reclaimed_.fetch_add(1, std::memory_order_relaxed);
      hold_sum_us_.fetch_add(static_cast<uint64_t>(held_us), std::memory_order_relaxed);
      uint64_t peak = hold_max_us_.load(std::memory_order_relaxed);
      while (static_cast<uint64_t>(held_us) > peak &&
             !hold_max_us_.compare_exchange_weak(peak, static_cast<uint64_t>(held_us),
                                                 std::memory_order_relaxed)) {
      }
    }
  }
}

void SpinnakerSource::release_held() {
  for (Spinnaker::ImagePtr& image : held_) {
    if (!image) continue;
    try {
      image->Release();
    } catch (const Spinnaker::Exception&) {
      // Teardown; the pool is about to go with the stream anyway.
    }
    image = nullptr;
  }
  held_now_.store(0, std::memory_order_relaxed);
}

void SpinnakerSource::run(FrameSink& sink) {
  try {
    bind_buffers(sink);
    camera_->BeginAcquisition();
  } catch (const std::exception& e) {
    // SPINNAKER_ERR_NOT_IMPLEMENTED lands here when the transport rejects this
    // stream mode with user-owned buffers, which is the one combination the
    // zero-copy path depends on.
    std::fprintf(stderr, "BeginAcquisition failed: %s\n", e.what());
    running_.store(false);
    return;
  }

  while (running_.load(std::memory_order_relaxed)) {
    try {
      reclaim(sink);

      const uint32_t held = held_now_.load(std::memory_order_relaxed);
      uint32_t peak = held_peak_.load(std::memory_order_relaxed);
      while (held > peak &&
             !held_peak_.compare_exchange_weak(peak, held, std::memory_order_relaxed)) {
      }
      if (held >= held_.size()) {
        starved_.fetch_add(1, std::memory_order_relaxed);
        if (!was_starved_) {
          was_starved_ = true;
          trace_pool("camera pool starved: all %zu user buffers held, "
                     "nothing free for the transport (waiting up to %llu ms)\n",
                     held_.size(), static_cast<unsigned long long>(config_.timeout_ms));
        }
      } else if (was_starved_) {
        was_starved_ = false;
        trace_pool("camera pool recovered: %u/%zu buffers held\n", held, held_.size());
      }

      Spinnaker::ImagePtr image = camera_->GetNextImage(config_.timeout_ms);

      if (image->IsIncomplete()) {
        const Spinnaker::ImageStatus status = image->GetImageStatus();
        incomplete_.fetch_add(1, std::memory_order_relaxed);
        incomplete_by_status_[incompleteStatusIndex(status)].fetch_add(
            1, std::memory_order_relaxed);
        trace_pool("incomplete frame: status=%s (%d) frame_id=%llu held=%u/%zu\n",
                   incompleteStatusName(status), static_cast<int>(status),
                   static_cast<unsigned long long>(image->GetFrameID()),
                   held_now_.load(std::memory_order_relaxed), held_.size());
        image->Release();
        continue;
      }

      const uint32_t slot = sink.slot_of(image->GetData());
      if (slot == FrameSink::kNoSlot) {
        foreign_.fetch_add(1, std::memory_order_relaxed);
        image->Release();
        continue;
      }

      // Held, not released: the reader works on this buffer directly, so
      // handing it back now would let the camera overwrite a frame still in
      // flight. The sink tells us when it has retired.
      if (!held_[slot]) held_now_.fetch_add(1, std::memory_order_relaxed);
      held_[slot] = image;
      held_since_[slot] = std::chrono::steady_clock::now();
      sink.commit(slot, image->GetTimeStamp(), geometry_.frame_bytes);
      delivered_.fetch_add(1, std::memory_order_relaxed);
    } catch (const Spinnaker::Exception& e) {
      // A timeout is both an idle camera and a pool held empty by a backed-up
      // reader; neither is fatal, and reclaim() runs again on the next pass.
      if (e.GetError() == Spinnaker::SPINNAKER_ERR_TIMEOUT) {
        timeouts_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      std::fprintf(stderr, "acquisition stopped: %s\n", e.what());
      break;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "acquisition stopped: %s\n", e.what());
      break;
    }
  }

  running_.store(false);
  camera_->EndAcquisition();
  // After EndAcquisition, so nothing can be handed back into a live pool.
  release_held();
}

}  // namespace perception
