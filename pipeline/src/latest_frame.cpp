#include "latest_frame.hpp"

#include <cstdio>
#include <utility>

namespace perception {

LatestFrame::LatestFrame(std::string name) : name_(std::move(name)) {}

LatestFrame::Sink LatestFrame::sink() {
  return [this](const Frame& frame) {
    // Outside the lock, so the pool slot goes back on the producer's thread
    // rather than under this holder's mutex.
    Frame displaced;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      // The producer may still be running after stop(); dropping the frame is
      // the only sensible answer, since nobody is left to take it.
      if (!running_) return;
      ++offered_;
      if (latest_) ++skipped_;
      displaced = std::move(latest_);
      latest_ = frame;
    }
    frame_ready_.notify_one();
  };
}

LatestFrame::Frame LatestFrame::acquire_latest() {
  std::unique_lock<std::mutex> lock(mutex_);
  frame_ready_.wait(lock, [this] { return !running_ || latest_ != nullptr; });
  if (!running_) return nullptr;
  return std::move(latest_);
}

LatestFrame::Frame LatestFrame::take_latest() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!running_) return nullptr;
  return std::move(latest_);
}

void LatestFrame::stop() {
  // Released outside the lock, and released at all: a frame nobody will take is
  // a pool slot the producer could be refilling.
  Frame dropped;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) return;
    running_ = false;
    dropped = std::move(latest_);
  }
  frame_ready_.notify_all();
}

uint64_t LatestFrame::offered() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return offered_;
}

uint64_t LatestFrame::skipped() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return skipped_;
}

std::string LatestFrame::health_line() const {
  uint64_t offered = 0;
  uint64_t skipped = 0;
  bool holding = false;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    offered = offered_;
    skipped = skipped_;
    holding = latest_ != nullptr;
  }

  char line[192];
  std::snprintf(line, sizeof(line), "%s: %llu offered, %llu skipped, holding %u", name_.c_str(),
                static_cast<unsigned long long>(offered),
                static_cast<unsigned long long>(skipped), holding ? 1u : 0u);
  return line;
}

}  // namespace perception
