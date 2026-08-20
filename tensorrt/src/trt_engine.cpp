#include "trt_engine.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "cuda_util.hpp"

namespace perception {
namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    // kVERBOSE and kINFO are noise once an engine is known to load cleanly --
    // this runs on every inference, not once at build time.
    if (severity == Severity::kVERBOSE || severity == Severity::kINFO) return;
    std::fprintf(stderr, "trt: %s\n", msg);
  }
};

Logger g_logger;

std::vector<char> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("TrtEngine: cannot open engine file '" + path + "'");
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("TrtEngine: engine file '" + path + "' is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(static_cast<std::size_t>(size));
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("TrtEngine: failed reading engine file '" + path + "'");
  }
  return buffer;
}

}  // namespace

std::size_t TrtEngine::TensorInfo::element_count() const {
  std::size_t count = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      throw std::runtime_error("TrtEngine: tensor '" + name +
                               "' has a dynamic or zero dimension -- this wrapper only "
                               "supports engines built with a fully static input shape");
    }
    count *= static_cast<std::size_t>(dims.d[i]);
  }
  return count;
}

TrtEngine::TrtEngine(const std::string& engine_path, int device_id) : device_id_(device_id) {
  cuda_error_check(cudaSetDevice(device_id_), "TrtEngine: cudaSetDevice");

  const std::vector<char> blob = read_file(engine_path);

  runtime_.reset(nvinfer1::createInferRuntime(g_logger));
  if (!runtime_) throw std::runtime_error("TrtEngine: createInferRuntime failed");

  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) {
    throw std::runtime_error("TrtEngine: deserializeCudaEngine failed for '" + engine_path +
                             "' -- built for a different TensorRT version, GPU, or DLA config "
                             "than this device?");
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("TrtEngine: createExecutionContext failed");

  const int32_t nb_tensors = engine_->getNbIOTensors();
  tensors_.reserve(static_cast<std::size_t>(nb_tensors));
  for (int32_t i = 0; i < nb_tensors; ++i) {
    TensorInfo info;
    info.name = engine_->getIOTensorName(i);
    info.is_input = engine_->getTensorIOMode(info.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    info.dims = engine_->getTensorShape(info.name.c_str());
    info.dtype = engine_->getTensorDataType(info.name.c_str());
    tensors_.push_back(std::move(info));
  }
}

TrtEngine::~TrtEngine() = default;

const TrtEngine::TensorInfo& TrtEngine::tensor(const std::string& name) const {
  for (const TensorInfo& info : tensors_) {
    if (info.name == name) return info;
  }
  throw std::runtime_error("TrtEngine: no such tensor '" + name + "'");
}

void TrtEngine::set_tensor_address(const std::string& name, void* device_ptr) {
  if (!context_->setTensorAddress(name.c_str(), device_ptr)) {
    throw std::runtime_error("TrtEngine: setTensorAddress failed for '" + name + "'");
  }
}

void TrtEngine::enqueue(cudaStream_t stream) {
  if (!context_->enqueueV3(stream)) {
    throw std::runtime_error("TrtEngine: enqueueV3 failed");
  }
}

}  // namespace perception
