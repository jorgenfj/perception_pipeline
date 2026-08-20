#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace perception {

// Reads `output` (max_detections x 6 float rows: x1, y1, x2, y2, score,
// class_id -- TensorRT's own output layout, in model input-pixel space)
// directly from device memory and, for every row scoring at least
// `conf_threshold`, rasterizes a box outline plus a filled label chip
// ("CLASS NAME 0.87", via a small built-in 5x7 bitmap font and a COCO-80
// name table) into `surface`, an RGBA8 CUDA surface sized img_w x img_h
// (e.g. the GL interop array GlViewer::present_gpu_boxes maps for its frame
// copy). Box and chip share one color per class_id, generated deterministically
// rather than looked up from a fixed palette. Coordinates are un-letterboxed
// with (scale, pad_x, pad_y) -- see compute_letterbox() in
// transforms/yolo_preprocess.hpp, the same math YoloEngine::decode()'s CPU
// path uses. Nothing here ever touches host memory.
void draw_detections(const float* output, uint32_t max_detections, float conf_threshold,
                     float scale, float pad_x, float pad_y, uint32_t img_w, uint32_t img_h,
                     cudaSurfaceObject_t surface, cudaStream_t stream);

}  // namespace perception
