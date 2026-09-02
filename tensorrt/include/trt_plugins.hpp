#pragma once

#include <string>

namespace perception {

// Loads a TensorRT plugin shared library so its operators exist before an
// engine that uses them is deserialized.
//
// Some plugins -- NVIDIA's ESS ones among them -- register their creators from
// static initializers and export no getCreators() entry point, so TensorRT
// cannot pull them out of the plan and the library has to already be resident
// when deserializeCudaEngine() runs. Without it the plan fails to load with
// "Cannot find plugin: <op>" followed by "Serialization assertion creator
// failed", which reads like a corrupt or mismatched engine and is not.
//
// Idempotent: loading the same path twice is a no-op, so every engine that
// wants a plugin can ask for it without the callers coordinating. Throws
// std::runtime_error with dlerror()'s text if the library cannot be loaded;
// an empty path is a no-op, meaning "this engine needs no plugins".
void load_trt_plugins(const std::string& library_path);

}  // namespace perception
