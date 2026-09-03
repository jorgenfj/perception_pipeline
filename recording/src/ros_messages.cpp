#include "ros_messages.hpp"

namespace perception::ros_msg {
namespace {

void write_image_body(CdrWriter& cdr, uint32_t width, uint32_t height, std::string_view encoding,
                      uint32_t step, const void* data, std::size_t bytes) {
  cdr.u32(height);
  cdr.u32(width);
  cdr.str(encoding);
  cdr.u8(0);  // is_bigendian: aarch64 and x86 are both little
  cdr.u32(step);
  cdr.bytes(data, bytes);
}

}  // namespace

void write_header(CdrWriter& cdr, uint64_t stamp_ns, std::string_view frame_id) {
  cdr.i32(static_cast<int32_t>(stamp_ns / 1000000000ull));
  cdr.u32(static_cast<uint32_t>(stamp_ns % 1000000000ull));
  cdr.str(frame_id);
}

std::string_view ros_encoding(PixelFormat format) {
  switch (format) {
    case PixelFormat::Bayer8_RGGB:
      return "bayer_rggb8";
    case PixelFormat::Bayer8_GRBG:
      return "bayer_grbg8";
    case PixelFormat::Bayer8_GBRG:
      return "bayer_gbrg8";
    case PixelFormat::Bayer8_BGGR:
      return "bayer_bggr8";
    case PixelFormat::GRAY8:
      return "mono8";
    case PixelFormat::RGB8:
      return "rgb8";
    case PixelFormat::RGBA8:
      return "rgba8";
  }
  return "mono8";
}

void encode_payload(CdrWriter& cdr, const Image& image, const NoContext&) {
  write_image_body(cdr, image.desc.width, image.desc.height, ros_encoding(image.desc.format),
                   image.desc.stride_bytes, image.data, image.bytes);
}

void encode(CdrWriter& cdr, const Topic<DisparityMessage>& topic, const DisparityMessage& message) {
  const DisparityContext& c = topic.context;

  write_header(cdr, message.header);
  write_header(cdr, message.header);
  write_image_body(cdr, c.width, c.height, "32FC1", c.width * 4, message.data.data,
                   message.data.bytes);

  cdr.f32(c.focal_length_px);
  cdr.f32(c.baseline_m);
  cdr.u32(0);
  cdr.u32(0);
  cdr.u32(c.height);
  cdr.u32(c.width);
  cdr.b(true);
  cdr.f32(c.min_disparity);
  cdr.f32(c.max_disparity);
  cdr.f32(0.0f);
}

void encode_payload(CdrWriter& cdr, const Imu& imu, const NoContext&) {
  cdr.f64_array(imu.orientation.data(), 4);
  cdr.f64_array(imu.orientation_covariance.data(), 9);
  cdr.f64_array(imu.angular_velocity.data(), 3);
  cdr.f64_array(imu.angular_velocity_covariance.data(), 9);
  cdr.f64_array(imu.linear_acceleration.data(), 3);
  cdr.f64_array(imu.linear_acceleration_covariance.data(), 9);
}

void encode_payload(CdrWriter& cdr, const MagneticField& field, const NoContext&) {
  cdr.f64_array(field.magnetic_field.data(), 3);
  cdr.f64_array(field.magnetic_field_covariance.data(), 9);
}

void encode_payload(CdrWriter& cdr, const FluidPressure& pressure, const NoContext&) {
  cdr.f64(pressure.fluid_pressure);
  cdr.f64(pressure.variance);
}

void read_header(CdrReader& cdr, Header& out) {
  const int32_t sec = cdr.i32();
  const uint32_t nsec = cdr.u32();
  out.frame_id = cdr.str();
  if (sec < 0) {
    out.stamp_ns = 0;
    return;
  }
  out.stamp_ns = static_cast<uint64_t>(sec) * 1000000000ull + nsec;
}

void decode_payload(CdrReader& cdr, Imu& imu) {
  cdr.f64_array(imu.orientation.data(), 4);
  cdr.f64_array(imu.orientation_covariance.data(), 9);
  cdr.f64_array(imu.angular_velocity.data(), 3);
  cdr.f64_array(imu.angular_velocity_covariance.data(), 9);
  cdr.f64_array(imu.linear_acceleration.data(), 3);
  cdr.f64_array(imu.linear_acceleration_covariance.data(), 9);
}

void decode_payload(CdrReader& cdr, MagneticField& field) {
  cdr.f64_array(field.magnetic_field.data(), 3);
  cdr.f64_array(field.magnetic_field_covariance.data(), 9);
}

void decode_payload(CdrReader& cdr, FluidPressure& pressure) {
  pressure.fluid_pressure = cdr.f64();
  pressure.variance = cdr.f64();
}

}  // namespace perception::ros_msg
