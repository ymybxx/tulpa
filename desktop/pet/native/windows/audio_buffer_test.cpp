#include "audio_buffer.h"

#include <cassert>
#include <cstdint>
#include <vector>

using tulpa::windows::SystemAudioWindowBuffer;

int main() {
  // Complete digital silence has coverage, but needs no base64 payload.
  SystemAudioWindowBuffer silent("silent", 100, 0, "");
  for (int i = 0; i < 10; ++i) assert(silent.append(std::vector<std::int16_t>(16'000)));
  assert(silent.drainReady(10'000'000'000).empty());
  auto silence = silent.drainReady(11'000'000'000);
  assert(silence.size() == 1 && silence[0].audioStatus == "complete");
  assert(silence[0].audioSilence && silence[0].pcmBytes.empty());
  assert(silence[0].audioCapturedSamples == 160'000);

  // A real missing capture is never compressed or reported as captured silence.
  SystemAudioWindowBuffer unavailable("off", 100, 0, "system_audio_disabled");
  auto absent = unavailable.drainReady(11'000'000'000);
  assert(absent[0].audioStatus == "unavailable" && !absent[0].audioSilence);
  assert(absent[0].audioGapReason == "system_audio_disabled");

  // Respect both wall-clock bucket and explicit cuts, without duplicating PCM.
  SystemAudioWindowBuffer split("cuts", 598, 0, "");
  std::vector<std::int16_t> oneSecond(16'000, 0x1234);
  for (int i = 0; i < 8; ++i) assert(split.append(oneSecond));
  split.cut(603, 603);
  auto first = split.drainReady(3'000'000'000);
  assert(first.size() == 1 && first[0].startedAt == 598 && first[0].endedAt == 600);
  assert(first[0].pcmBytes.size() == 64'000);
  assert(first[0].pcmBytes[0] == 0x34 && first[0].pcmBytes[1] == 0x12);
  auto second = split.drainReady(6'000'000'000);
  assert(second.size() == 1 && second[0].startedAt == 600 && second[0].endedAt == 603);
  assert(second[0].windowIndex == 1 && second[0].audioCapturedSamples == 48'000);

  // Gaps remain visible even when enough PCM is buffered; overflow is bounded.
  SystemAudioWindowBuffer overflow("bounded", 100, 0, "");
  for (int i = 0; i < 20; ++i) assert(overflow.append(oneSecond));
  assert(!overflow.append(oneSecond));
  auto windows = overflow.drainReady(31'000'000'000);
  assert(windows.size() == 3);
  assert(windows[0].audioStatus == "partial" && windows[0].audioGapReason == "capture_overflow");
  assert(windows[1].audioStatus == "complete");
  assert(windows[2].audioStatus == "unavailable");

  SystemAudioWindowBuffer recovery("recovery", 100, 0);
  recovery.markBackendRunning();
  assert(recovery.append(oneSecond));
  auto partial = recovery.drainReady(11'000'000'000);
  assert(partial[0].audioStatus == "partial" && partial[0].audioCapturedSamples == 16'000);
  assert(partial[0].audioGapReason == "capture_started_late");
  assert(partial[0].pcmBytes.size() == 320'000);
  assert(partial[0].pcmBytes.back() == 0);
}
