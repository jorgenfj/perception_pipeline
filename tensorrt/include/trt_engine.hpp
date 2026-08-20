#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace perception {

// Deserializes and runs a TensorRT engine file. 
class TrtEngine {
 public:
  struct TensorInfo {
    std::string name;
    bool is_input = false;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;

    std::size_t element_count() const;
  };

  // Throws if the file cannot be read or does not deserialize -- e.g. it was
  // built against a different TensorRT version or GPU/DLA architecture than
  // this device runs, which TensorRT itself refuses at load time.
  explicit TrtEngine(const std::string& engine_path, int device_id = 0);
  ~TrtEngine();

  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo& tensor(const std::string& name) const;

  // Binds a device pointer for one I/O tensor. Must be called for every
  // tensor in tensors() before enqueue(); the engine here is built with a
  // static input shape (trtexec's default), so there is no setInputShape step.
  void set_tensor_address(const std::string& name, void* device_ptr);

  // context->enqueueV3 on `stream`. Every tensor address must already be set.
  void enqueue(cudaStream_t stream);

 private:
  struct Deleter {
    template <typename T>
    void operator()(T* p) const {
      delete p;
    }
  };

  std::unique_ptr<nvinfer1::IRuntime, Deleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, Deleter> context_;
  std::vector<TensorInfo> tensors_;
  int device_id_;
};

}  // namespace perception
