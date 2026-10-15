#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace tulpa::windows {

struct SystemAudioWindow {
  std::string captureSessionId;
  std::uint64_t windowIndex = 0;
  double startedAt = 0;
  double endedAt = 0;
  std::string audioStatus;
  int audioSampleRate = 16'000;
  std::size_t audioCapturedSamples = 0;
  std::optional<std::string> audioGapReason;
  bool audioSilence = false;
  // Little-endian signed PCM16; absent for unavailable or complete digital silence.
  std::vector<std::uint8_t> pcmBytes;
};

// Same sequential windows as the macOS assembler: ten seconds, a one-second
// capture grace, cuts at explicit boundaries and every ten-minute wall bucket.
// The helper owns the timer and calls drainReady; callbacks can append safely.
class SystemAudioWindowBuffer final {
 public:
  SystemAudioWindowBuffer(std::string captureSessionId, double startedAt,
                         std::uint64_t startMonotonicNs,
                         std::string initialReason = "system_audio_starting");
  bool append(const std::vector<std::int16_t>& pcm);
  void markBackendRunning();
  void markBackendUnavailable(const std::string& reason);
  void markGap(const std::string& reason);
  void cut(double boundaryAt, double nowWallTime);
  std::vector<SystemAudioWindow> drainReady(std::uint64_t nowMonotonicNs);

 private:
  void markGapLocked(const std::string& reason);
  SystemAudioWindow finish(std::size_t samples);
  double nextStart() const;
  std::mutex mutex_;
  std::string sessionId_;
  double startedAt_;
  std::uint64_t startMonotonicNs_;
  std::uint64_t index_ = 0;
  std::uint64_t emittedSamples_ = 0;
  std::deque<std::int16_t> samples_;
  std::vector<double> boundaries_;
  std::optional<std::string> gapReason_;
  std::optional<std::string> unavailableReason_;
};

}  // namespace tulpa::windows
