#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include <perception/utils/camera_model.hpp>
#include <perception/utils/stereo.hpp>

#include "transforms/ess_preprocess.hpp"

namespace perception {

struct EssRectifier {
  std::unique_ptr<EssPreprocessTransform> transform;

  // The rectification restated for the network's grid
  utils::StereoRectification rectification;

  // The grid the calibration was solved at.
  utils::ImageSize source_rectified;
};

// Build one camera's map against the rectification resized onto ESS's input grid,
// and upload it.
inline EssRectifier make_ess_rectifier(const utils::StereoCalibration& calibration, int camera,
                                       EssNormalization normalization = {}) {
  if (camera != 0 && camera != 1) {
    throw std::runtime_error("make_ess_rectifier: camera must be 0 or 1");
  }
  const utils::ImageSize target{kEssFullWidth, kEssFullHeight};

  EssRectifier rectifier;
  rectifier.rectification = utils::resize_rectification(calibration.rectification, target,
                                                           utils::ResizeFit::Crop);
  rectifier.source_rectified = calibration.rectification.size;

  const std::vector<float> map = utils::build_rectify_map(
      calibration.cameras[camera], rectifier.rectification.cameras[camera], target);

  rectifier.transform = std::make_unique<EssPreprocessTransform>(
      map, calibration.size.width, calibration.size.height, target.width, target.height,
      normalization);

  return rectifier;
}

}  // namespace perception
