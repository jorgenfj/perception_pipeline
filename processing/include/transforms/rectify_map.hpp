#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include <perception/geometry/camera_model.hpp>
#include <perception/geometry/stereo.hpp>

#include "transforms/ess_preprocess.hpp"

namespace perception {

struct EssRectifier {
  std::unique_ptr<EssPreprocessTransform> transform;

  // The rectification restated for the network's grid
  geometry::StereoRectification rectification;

  // The grid the calibration was solved at.
  geometry::ImageSize source_rectified;
};

// Build one camera's map against the rectification resized onto ESS's input grid,
// and upload it.
inline EssRectifier make_ess_rectifier(const geometry::StereoCalibration& calibration, int camera,
                                       EssNormalization normalization = {}) {
  if (camera != 0 && camera != 1) {
    throw std::runtime_error("make_ess_rectifier: camera must be 0 or 1");
  }
  const geometry::ImageSize target{kEssFullWidth, kEssFullHeight};

  EssRectifier rectifier;
  rectifier.rectification = geometry::resize_rectification(calibration.rectification, target,
                                                           geometry::ResizeFit::Crop);
  rectifier.source_rectified = calibration.rectification.size;

  const std::vector<float> map = geometry::build_rectify_map(
      calibration.cameras[camera], rectifier.rectification.cameras[camera], target);

  rectifier.transform = std::make_unique<EssPreprocessTransform>(
      map, calibration.size.width, calibration.size.height, target.width, target.height,
      normalization);

  return rectifier;
}

}  // namespace perception
