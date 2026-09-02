#include "trt_plugins.hpp"

#include <dlfcn.h>

#include <cstdio>
#include <mutex>
#include <set>
#include <stdexcept>

namespace perception {
namespace {

std::mutex g_mutex;
std::set<std::string> g_loaded;

}  // namespace

void load_trt_plugins(const std::string& library_path) {
  if (library_path.empty()) return;

  const std::lock_guard<std::mutex> lock(g_mutex);
  if (g_loaded.count(library_path)) return;

  // RTLD_NOW so a plugin built against a CUDA runtime this machine does not
  // have fails here, by name, rather than at the first call into it -- the
  // lazy-binding version of that failure surfaces inside TensorRT as an
  // assertion with no mention of the library that caused it.
  //
  // RTLD_GLOBAL because the plugin's static initializers register creators into
  // libnvinfer's registry, and the symbols they resolve against have to be the
  // ones already loaded into this process rather than a second private copy.
  //
  // Never dlclose()d: the registry keeps raw pointers to creators that live in
  // this library, and unloading it leaves TensorRT holding dangling ones. It is
  // opened once per process and stays for the life of it.
  void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    const char* error = dlerror();
    throw std::runtime_error("load_trt_plugins: cannot load '" + library_path +
                             "': " + (error ? error : "unknown dlopen failure"));
  }

  g_loaded.insert(library_path);
  std::printf("trt: plugins %s\n", library_path.c_str());
}

}  // namespace perception
