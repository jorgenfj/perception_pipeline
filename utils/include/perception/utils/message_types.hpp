#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace perception {

enum class PixelFormat : uint8_t {
  Bayer8_RGGB,
  Bayer8_GRBG,
  Bayer8_GBRG,
  Bayer8_BGGR,
  GRAY8,
  RGB8,
  RGBA8,
};

constexpr std::size_t bytes_per_pixel(PixelFormat format) {
  switch (format) {
    case PixelFormat::RGB8:
      return 3;
    case PixelFormat::RGBA8:
      return 4;
    case PixelFormat::Bayer8_RGGB:
    case PixelFormat::Bayer8_GRBG:
    case PixelFormat::Bayer8_GBRG:
    case PixelFormat::Bayer8_BGGR:
    case PixelFormat::GRAY8:
      break;
  }
  return 1;
}

// Geometry of one image. `stride_bytes` is the row pitch and may exceed
// width * bytes_per_pixel
struct ImageDesc {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t stride_bytes = 0;
  PixelFormat format = PixelFormat::Bayer8_RGGB;

  constexpr std::size_t bytes() const {
    return static_cast<std::size_t>(height) * stride_bytes;
  }
};

constexpr ImageDesc packed_desc(uint32_t width, uint32_t height, PixelFormat format) {
  return ImageDesc{width, height, static_cast<uint32_t>(width * bytes_per_pixel(format)), format};
}

}  // namespace perception

namespace perception::ros_msg {

/**
 * @brief std_msgs/Header.
 */
struct Header {
  uint64_t stamp_ns{};
  std::string frame_id;
};

/** @brief A header and a payload. The payload is POD; this is not, and does not need to be. */
template <typename Payload>
struct Message {
  Header header;
  Payload data;
};

/**
 * @brief sensor_msgs/Image, less its header: geometry plus borrowed pixels.
 */
struct Image {
  ImageDesc desc;
  const void* data = nullptr;
  std::size_t bytes = 0;
};

/**
 * @brief sensor_msgs/Imu, less its header.
 *
 * A covariance whose first element is -1 means that quantity is not estimated,
 * per REP-145
 */
struct Imu {
  std::array<double, 4> orientation{0.0, 0.0, 0.0, 1.0};  ///< x, y, z, w -- ROS's order.
  std::array<double, 9> orientation_covariance{-1.0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::array<double, 3> angular_velocity{};  ///< rad/s
  std::array<double, 9> angular_velocity_covariance{};

  std::array<double, 3> linear_acceleration{};  ///< m/s^2
  std::array<double, 9> linear_acceleration_covariance{};
};

/** @brief sensor_msgs/MagneticField, less its header. */
struct MagneticField {
  std::array<double, 3> magnetic_field{};  ///< tesla
  std::array<double, 9> magnetic_field_covariance{};
};

/** @brief sensor_msgs/FluidPressure, less its header. */
struct FluidPressure {
  double fluid_pressure{};  ///< Absolute pascals -- not depth, not gauge.
  double variance{};        ///< 0 means unknown, per the message definition.
};

using ImuMessage = Message<Imu>;
using MagneticFieldMessage = Message<MagneticField>;
using FluidPressureMessage = Message<FluidPressure>;

}  // namespace perception::ros_msg
