#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tulpa::windows {

// Captures all application playback except hostProcessId and its descendants.
// Never opens a microphone. Callbacks run on the capture worker; copy PCM if it
// must outlive the callback. start/stop belong to the helper's control thread.
class SystemAudioCapture final {
 public:
  using PcmCallback = std::function<void(
      std::uint64_t firstSampleMonotonicNs, const std::vector<std::int16_t>&)>;
  using StatusCallback =
      std::function<void(const std::string& state, const std::string& reason)>;

  SystemAudioCapture(std::uint32_t hostProcessId, PcmCallback onPcm,
                     StatusCallback onStatus);
  ~SystemAudioCapture();
  SystemAudioCapture(const SystemAudioCapture&) = delete;
  SystemAudioCapture& operator=(const SystemAudioCapture&) = delete;
  void start();
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tulpa::windows
