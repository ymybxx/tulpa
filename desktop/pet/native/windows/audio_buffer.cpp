#include "audio_buffer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tulpa::windows {
namespace {
constexpr std::size_t kSampleRate = 16'000;
constexpr std::size_t kWindowSamples = kSampleRate * 10;
constexpr std::size_t kMaximumBufferedSamples = kWindowSamples * 2;
constexpr double kSampleTolerance = 0.5 / kSampleRate;
}

SystemAudioWindowBuffer::SystemAudioWindowBuffer(
    std::string captureSessionId, double startedAt,
    std::uint64_t startMonotonicNs, std::string initialReason)
    : sessionId_(std::move(captureSessionId)), startedAt_(startedAt),
      startMonotonicNs_(startMonotonicNs) {
  if (sessionId_.empty() || !std::isfinite(startedAt_))
    throw std::invalid_argument("invalid_system_audio_window_clock");
  if (!initialReason.empty()) unavailableReason_ = std::move(initialReason);
}

bool SystemAudioWindowBuffer::append(const std::vector<std::int16_t>& pcm) {
  if (pcm.empty() || pcm.size() > kSampleRate * 2) return false;
  std::lock_guard lock(mutex_);
  const auto accepted = std::min(pcm.size(), kMaximumBufferedSamples - samples_.size());
  samples_.insert(samples_.end(), pcm.begin(), pcm.begin() + accepted);
  if (accepted < pcm.size()) markGapLocked("capture_overflow");
  return accepted == pcm.size();
}

void SystemAudioWindowBuffer::markGapLocked(const std::string& reason) {
  if (!reason.empty() && !gapReason_) gapReason_ = reason;
}
void SystemAudioWindowBuffer::markGap(const std::string& reason) {
  std::lock_guard lock(mutex_);
  markGapLocked(reason);
}
void SystemAudioWindowBuffer::markBackendRunning() {
  std::lock_guard lock(mutex_);
  if (unavailableReason_) markGapLocked("capture_started_late");
  unavailableReason_.reset();
}
void SystemAudioWindowBuffer::markBackendUnavailable(const std::string& reason) {
  if (reason.empty()) return;
  std::lock_guard lock(mutex_);
  unavailableReason_ = reason;
  markGapLocked(reason);
}

double SystemAudioWindowBuffer::nextStart() const {
  return startedAt_ + static_cast<double>(emittedSamples_) / kSampleRate;
}

void SystemAudioWindowBuffer::cut(double boundaryAt, double nowWallTime) {
  if (!std::isfinite(boundaryAt) || !std::isfinite(nowWallTime)) return;
  std::lock_guard lock(mutex_);
  if (boundaryAt <= nextStart() || boundaryAt > nowWallTime + 1) return;
  if (std::find(boundaries_.begin(), boundaries_.end(), boundaryAt) != boundaries_.end())
    return;
  // Commands must not make this native buffer unbounded.
  if (boundaries_.size() >= 256) return;
  boundaries_.push_back(boundaryAt);
  std::sort(boundaries_.begin(), boundaries_.end());
}

std::vector<SystemAudioWindow> SystemAudioWindowBuffer::drainReady(
    std::uint64_t nowMonotonicNs) {
  std::lock_guard lock(mutex_);
  std::vector<SystemAudioWindow> ready;
  if (nowMonotonicNs < startMonotonicNs_) return ready;
  const double elapsed = static_cast<double>(nowMonotonicNs - startMonotonicNs_) / 1e9;
  // Bound each timer invocation after a long suspend. Subsequent invocations
  // drain the remaining honest unavailable windows without allocating them all.
  while (ready.size() < 64) {
    const double start = nextStart();
    std::erase_if(boundaries_, [start](double value) { return value <= start + kSampleTolerance; });
    const double bucketEnd = (std::floor((start + kSampleTolerance) / 600) + 1) * 600;
    double end = std::min(start + 10, bucketEnd);
    if (!boundaries_.empty()) end = std::min(end, boundaries_.front());
    if (elapsed < end - startedAt_ + 1) break;
    const auto count = static_cast<std::size_t>(std::llround((end - start) * kSampleRate));
    if (!count) break;
    ready.push_back(finish(std::min(count, kWindowSamples)));
  }
  return ready;
}

SystemAudioWindow SystemAudioWindowBuffer::finish(std::size_t count) {
  SystemAudioWindow result;
  result.captureSessionId = sessionId_;
  result.windowIndex = index_++;
  result.startedAt = nextStart();
  emittedSamples_ += count;
  result.endedAt = nextStart();
  result.audioCapturedSamples = std::min(count, samples_.size());
  auto gap = std::exchange(gapReason_, std::nullopt);
  if (!result.audioCapturedSamples) {
    result.audioStatus = "unavailable";
    result.audioGapReason = gap ? gap : unavailableReason_.value_or("no_samples");
    return result;
  }
  const bool complete = result.audioCapturedSamples == count && !gap && !unavailableReason_;
  result.audioStatus = complete ? "complete" : "partial";
  if (!complete)
    result.audioGapReason = gap ? gap : unavailableReason_.value_or("insufficient_samples");
  result.pcmBytes.resize(count * 2, 0);
  bool silent = true;
  for (std::size_t i = 0; i < result.audioCapturedSamples; ++i) {
    const auto sample = static_cast<std::uint16_t>(samples_.front());
    samples_.pop_front();
    result.pcmBytes[i * 2] = static_cast<std::uint8_t>(sample & 0xff);
    result.pcmBytes[i * 2 + 1] = static_cast<std::uint8_t>(sample >> 8);
    silent = silent && sample == 0;
  }
  result.audioSilence = complete && silent;
  if (result.audioSilence) result.pcmBytes.clear();
  return result;
}
}  // namespace tulpa::windows
