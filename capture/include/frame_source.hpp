#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "camera_config.hpp"
#include "frame_sink.hpp"

namespace perception {

// Whatever is putting frames into a FrameSink, named without saying what it is:
// a camera (spinnaker/spinnaker_source.hpp) or a recording played back at its
// original pacing (recording/recording_source.hpp). The composing application
// picks one at configure time -- see -DPERCEPTION_SOURCE and app/src/source_*.cpp.
//
// Nothing here is per-frame virtual. start() hands the sink over once and the
// source drives its own thread from there.
class FrameSource {
 public:
  virtual ~FrameSource() = default;

  // Readable before start(): the sink is sized from geometry().buffer_bytes.
  virtual const CameraGeometry& geometry() const = 0;

  // Fewest slots the source can work with. A source that holds a frame until
  // the reader retires it needs at least two, or it stalls on itself.
  virtual uint32_t min_slot_count() const = 0;

  // `sink` must outlive the source and have slots of at least
  // geometry().buffer_bytes.
  virtual void start(FrameSink& sink) = 0;
  virtual void stop() = 0;

  // True once no further frame can ever arrive: a camera that gave up, or a
  // recording played to its end. What the run loop waits on -- it must stop
  // waiting for a publish that is not coming, and it does not care which.
  virtual bool finished() const = 0;

  // Of those, the ones that are faults. A recording reaching its end is
  // finished() but not failed(), which is exit 0 rather than exit 1.
  virtual bool failed() const = 0;
  virtual const std::string& failure() const = 0;

  // Invoked once, on the source's own thread, immediately after finished() is
  // set, so the owner can kick whatever it is parked on -- e.g.
  // DeviceRingBuffer::wake_all(). Set before start(); throws are swallowed.
  virtual void set_finished_callback(std::function<void()> cb) = 0;

  virtual uint64_t delivered() const = 0;

  // Reporting is formatted by the source rather than read field by field: the
  // interesting counters are exactly the ones the two do not share.
  // "incomplete" is meaningless for a file and "late" is meaningless for a
  // camera, so a common struct would be mostly zeroes and a lie about which.

  // Counters for the periodic report line, e.g.
  // "incomplete=0 foreign=0 timeouts=0". No leading or trailing space.
  virtual std::string counters() const { return {}; }

  // Appended after the counters when something needs explaining. Empty in the
  // normal case.
  virtual std::string notes() const { return {}; }

  // Printed only under PERCEPTION_TRACE_POOL. `depth` is the sink's slot
  // count, which the source does not otherwise know.
  virtual std::string pool_line(uint32_t depth) const {
    (void)depth;
    return {};
  }

  // GevIEEE1588Status ("Slave" once locked), or empty for a source with no
  // clock of its own. A live register read on the camera path, so call it on
  // the report line and not per frame.
  virtual std::string ptp_status() { return {}; }
  virtual bool ptp_offset_ns(int64_t& out) {
    (void)out;
    return false;
  }
};

}  // namespace perception
