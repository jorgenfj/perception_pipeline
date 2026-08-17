#include "spinnaker_source.hpp"

#include <SpinGenApi/SpinnakerGenApi.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace perception {
namespace {

using namespace Spinnaker::GenApi;

// USB3 packet size. Buffers not rounded up to it tear on USB3 cameras; on GigE
// it is harmless padding.
constexpr std::size_t kUsbPacketSize = 1024;

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
  thread_ = std::thread(&SpinnakerSource::run, this, std::ref(sink));
}

void SpinnakerSource::stop() {
  if (!running_.exchange(false)) return;
  if (thread_.joinable()) thread_.join();
}

void SpinnakerSource::reclaim(FrameSink& sink) {
  for (uint32_t slot = 0; slot < held_.size(); ++slot) {
    if (held_[slot] && sink.consumed(slot)) {
      held_[slot]->Release();
      held_[slot] = nullptr;
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

      Spinnaker::ImagePtr image = camera_->GetNextImage(config_.timeout_ms);

      if (image->IsIncomplete()) {
        incomplete_.fetch_add(1, std::memory_order_relaxed);
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
      held_[slot] = image;
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
