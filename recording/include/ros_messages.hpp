#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <perception/utils/message_types.hpp>

#include "cdr_reader.hpp"
#include "cdr_writer.hpp"
#include "mcap_player.hpp"
#include "mcap_recorder.hpp"
#include "ros_schemas.hpp"

namespace perception::ros_msg {

// Puts the plain structs in message_types.hpp on the wire, and declares the topics
// that carry them. The one file a new message type touches: a payload struct
// there, a MessageTraits specialisation here and an encode_payload() overload.
//
// Schema text comes from the generated ros_schemas.hpp, so nothing here retypes
// a message definition. Standard ROS types wherever one fits, custom messages
// only where none does: a custom schema is readable anywhere, but `ros2 bag
// play` cannot republish it without it built and sourced.

/** What a message type must declare to be recordable. Undefined so the error names the type. */
template <typename M>
struct MessageTraits;

/** For a message whose size is fixed and whose payload needs nothing from the topic. */
struct NoContext {};

/**
 * @brief A topic, typed by what goes on it.
 *
 * Typed rather than a bare TopicId because pushing the wrong message would
 * otherwise write plausible bytes under the wrong schema.
 */
template <typename M>
struct Topic {
  McapRecorder::TopicId id{};
  typename MessageTraits<M>::Context context{};
};

/** A traits specialisation and an encode() overload -- so missing one is a sentence, not a page. */
template <typename M>
concept Recordable =
    requires(CdrWriter& cdr, const Topic<M>& topic, const M& message,
             const typename MessageTraits<M>::Context& context) {
      { message.header.stamp_ns } -> std::convertible_to<uint64_t>;
      { MessageTraits<M>::type } -> std::convertible_to<std::string_view>;
      { MessageTraits<M>::schema_text } -> std::convertible_to<std::string_view>;
      { MessageTraits<M>::max_bytes(context) } -> std::convertible_to<std::size_t>;
      encode(cdr, topic, message);
    };

/** @brief std_msgs/Header: the stamp split into sec and nanosec, then the frame. */
void write_header(CdrWriter& cdr, uint64_t stamp_ns, std::string_view frame_id);

inline void write_header(CdrWriter& cdr, const Header& header) {
  write_header(cdr, header.stamp_ns, header.frame_id);
}

// Every message is a header then a payload, so that is written once here and
// each type supplies only its own fields. DisparityView is the exception and
// overrides this, because its nested Image carries a second header.
template <typename Payload>
void encode(CdrWriter& cdr, const Topic<Message<Payload>>& topic, const Message<Payload>& message) {
  write_header(cdr, message.header);
  encode_payload(cdr, message.data, topic.context);
}

/**
 * @brief Declare a topic and get the typed handle to write to.
 *
 * Type name, schema and byte budget come from the traits, so no call site
 * states them.
 *
 * @param rate_hz Sizes the queue only.
 * @param context Topic-constant fields, for the types that have any.
 * @param max_message_bytes Overrides the type's default reservation; 0 takes it.
 * @throws std::runtime_error as McapRecorder::add_topic does.
 */
template <Recordable M>
Topic<M> add_topic(McapRecorder& recorder, std::string name, double rate_hz,
                   typename MessageTraits<M>::Context context = {},
                   std::size_t max_message_bytes = 0) {
  using Traits = MessageTraits<M>;

  McapRecorder::Topic declared;
  declared.name = std::move(name);
  declared.type = std::string(Traits::type);
  declared.schema = std::string(Traits::schema_text);
  declared.rate_hz = rate_hz;
  declared.max_message_bytes =
      max_message_bytes > 0 ? max_message_bytes : Traits::max_bytes(context);

  Topic<M> topic;
  topic.id = recorder.add_topic(declared);
  topic.context = std::move(context);
  return topic;
}

/**
 * @brief Encode `message` and hand it to the recorder, on this thread.
 * @return False if dropped, which recorder.drops(topic.id) counts.
 */
template <Recordable M>
bool write(McapRecorder& recorder, const Topic<M>& topic, const M& message) {
  return recorder.push(topic.id, message.header.stamp_ns,
                       [&](CdrWriter& cdr) { encode(cdr, topic, message); });
}

/**
 * @brief std_msgs/Header, back off the wire: sec and nanosec rejoined, then the frame.
 */
void read_header(CdrReader& cdr, Header& out);

void decode_payload(CdrReader& cdr, Imu& imu);
void decode_payload(CdrReader& cdr, MagneticField& field);
void decode_payload(CdrReader& cdr, FluidPressure& pressure);

/**
 * @brief Decode a replayed message into its struct.
 *
 * @return False if the bytes are not this message, which is a message to count
 *         and skip rather than a run to end. A short read leaves `out` partly
 *         filled; a caller that got false must not use it.
 */
template <typename Payload>
bool decode(const ReplayMessage& message, Message<Payload>& out) {
  CdrReader cdr(message.data, message.size);
  read_header(cdr, out.header);
  decode_payload(cdr, out.data);
  return cdr.ok();
}

/**
 * What a bulk topic reserves per message unless the caller says otherwise.
 *
 * Being wrong is not fatal either way: too small and the encode reallocates
 * once on the pushing thread, which McapRecorder::grew() counts, and too large
 * only costs reserved memory -- capped by Config::topic_memory_mb, which shrinks
 * the queue rather than letting the reservation grow. Pass a size to add_topic()
 * for a camera this does not fit.
 */
inline constexpr std::size_t kDefaultBulkBytes = 4u * 1024 * 1024;

using ImageMessage = Message<Image>;

/** @brief The ROS encoding name for a pixel format, e.g. "bayer_rggb8". */
std::string_view ros_encoding(PixelFormat format);

void encode_payload(CdrWriter& cdr, const Image& image, const NoContext&);

template <>
struct MessageTraits<ImageMessage> {
  // Nothing topic-constant left: every message carries its own geometry.
  using Context = NoContext;
  static constexpr std::string_view type = schema::kImageType;
  static constexpr std::string_view schema_text = schema::kImage;
  static std::size_t max_bytes(const Context&) { return kDefaultBulkBytes; }
};

/** The topic-constant CONTENT of a stereo_msgs/DisparityImage -- not a size. */
struct DisparityContext {
  uint32_t width = 0;
  uint32_t height = 0;
  float focal_length_px = 0.0f;  ///< Rectified fx, on the network's grid.
  float baseline_m = 0.0f;
  float min_disparity = 0.0f;
  float max_disparity = 0.0f;
};

/** @brief stereo_msgs/DisparityImage with a borrowed float plane -- a pinned pool slot. */
struct DisparityView {
  const void* data = nullptr;
  std::size_t bytes = 0;
};

using DisparityMessage = Message<DisparityView>;

/** Not the generic encode(): the nested sensor_msgs/Image carries a second header. */
void encode(CdrWriter& cdr, const Topic<DisparityMessage>& topic, const DisparityMessage& message);

template <>
struct MessageTraits<DisparityMessage> {
  using Context = DisparityContext;
  static constexpr std::string_view type = schema::kDisparityImageType;
  static constexpr std::string_view schema_text = schema::kDisparityImage;
  static std::size_t max_bytes(const Context&) { return kDefaultBulkBytes; }
};

void encode_payload(CdrWriter& cdr, const Imu& imu, const NoContext&);

template <>
struct MessageTraits<ImuMessage> {
  using Context = NoContext;
  static constexpr std::string_view type = schema::kImuType;
  static constexpr std::string_view schema_text = schema::kImu;

  /** 320 bytes of body, plus the header and a frame_id. */
  static std::size_t max_bytes(const Context&) { return 512; }
};

void encode_payload(CdrWriter& cdr, const MagneticField& field, const NoContext&);

template <>
struct MessageTraits<MagneticFieldMessage> {
  using Context = NoContext;
  static constexpr std::string_view type = schema::kMagneticFieldType;
  static constexpr std::string_view schema_text = schema::kMagneticField;
  static std::size_t max_bytes(const Context&) { return 256; }  ///< 96 of body, plus the header.
};

void encode_payload(CdrWriter& cdr, const FluidPressure& pressure, const NoContext&);

template <>
struct MessageTraits<FluidPressureMessage> {
  using Context = NoContext;
  static constexpr std::string_view type = schema::kFluidPressureType;
  static constexpr std::string_view schema_text = schema::kFluidPressure;
  static std::size_t max_bytes(const Context&) { return 128; }
};

}  // namespace perception::ros_msg
