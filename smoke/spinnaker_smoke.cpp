// Smoke test: proves the Spinnaker SDK headers, libraries and GenTL producer
// are wired up correctly. Enumerates cameras without initialising them, so it
// is safe to run against a live rig. Zero cameras is a pass -- it means the API
// works, just that none are attached or reachable on this network.

#include <Spinnaker.h>
#include <SpinGenApi/SpinnakerGenApi.h>

#include <cstdio>
#include <string>

namespace {

/// Reads a string node, tolerating models that do not expose it.
std::string readString(Spinnaker::GenApi::INodeMap& nodeMap, const char* name) {
  const Spinnaker::GenApi::CStringPtr node = nodeMap.GetNode(name);
  if (!IsReadable(node)) return "<unavailable>";
  return std::string(node->GetValue().c_str());
}

/// GigE cameras report their address as a packed 32-bit integer.
std::string readIPv4(Spinnaker::GenApi::INodeMap& nodeMap, const char* name) {
  const Spinnaker::GenApi::CIntegerPtr node = nodeMap.GetNode(name);
  if (!IsReadable(node)) return "<unavailable>";
  const auto v = static_cast<unsigned long long>(node->GetValue());
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%llu.%llu.%llu.%llu", (v >> 24) & 0xFF, (v >> 16) & 0xFF,
                (v >> 8) & 0xFF, v & 0xFF);
  return buf;
}

}  // namespace

int main() {
  Spinnaker::SystemPtr system;
  try {
    system = Spinnaker::System::GetInstance();
  } catch (const Spinnaker::Exception& e) {
    std::printf("FAILED to initialise Spinnaker: %s\n", e.what());
    return 1;
  }

  const Spinnaker::LibraryVersion version = system->GetLibraryVersion();
  std::printf("Spinnaker %u.%u.%u.%u\n", version.major, version.minor, version.type,
              version.build);

  int status = 0;
  try {
    Spinnaker::CameraList cameras = system->GetCameras();
    const unsigned count = cameras.GetSize();
    std::printf("cameras detected: %u\n", count);

    for (unsigned i = 0; i < count; ++i) {
      // The transport-layer nodemap is readable without calling Init(), so this
      // never disturbs a camera that another process is streaming from.
      Spinnaker::GenApi::INodeMap& tl = cameras.GetByIndex(i)->GetTLDeviceNodeMap();
      std::printf("  [%u] %s %s  serial=%s  ip=%s\n", i,
                  readString(tl, "DeviceVendorName").c_str(),
                  readString(tl, "DeviceModelName").c_str(),
                  readString(tl, "DeviceSerialNumber").c_str(),
                  readIPv4(tl, "GevDeviceIPAddress").c_str());
    }

    // Cameras must be released before the system instance, or Spinnaker warns
    // about outstanding references at shutdown.
    cameras.Clear();
  } catch (const Spinnaker::Exception& e) {
    std::printf("FAILED during enumeration: %s\n", e.what());
    status = 1;
  }

  system->ReleaseInstance();
  return status;
}
