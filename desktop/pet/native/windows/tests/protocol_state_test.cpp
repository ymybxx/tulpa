#include "../protocol_state.h"
#include <cassert>
#include <limits>
using namespace tulpa::windows;
int main() {
  FrameSlot slot;
  Frame first{"first", 0, 10, 0, 1, 960, 540, {0xff, 0xd8, 0xff, 0xd9}};
  auto stored = slot.store(first, 10);
  assert(stored && stored->expires_at == 13 && stored->revision == 1);
  assert(!slot.claim("wrong", 11));
  assert(slot.claim("first", 11));
  assert(!slot.claim("first", 11));
  assert(slot.store(first, 11));
  assert(!slot.claim("first", 13));
  assert(!slot.store(first, 13));
  assert(!slot.store(first, std::numeric_limits<double>::quiet_NaN()));
  first.observed_at = 20;
  assert(slot.store(first, 20));
  slot.clear(); assert(!slot.claim("first", 21));
  first.jpeg.resize(200 * 1024 + 1); assert(!slot.store(first, 20));
  assert(private_window_title(L"Example - InPrivate"));
  assert(private_window_title(L"浏览器 - 无痕"));
  assert(!private_window_title(L"Example - Microsoft Edge"));
  assert(half_to_float(0x3c00) == 1); assert(half_to_float(0x4000) == 2);
  assert(half_to_float(0x7c00) == 0); assert(linear_to_srgb(1) == 255);
  assert(linear_to_srgb(0) == 0); assert(valid_interval(1'000)); assert(!valid_interval(0));
  assert(bounded_utf8("中文😀end", 5) == "中");
  assert(bounded_utf8("中文😀end", 8) == "中文");
  assert(bounded_utf8("中文😀end", 10) == "中文😀");
  assert(audio_gap_code("system_audio_activation_failed:0x80070005") == "system_audio_activation_failed");
  assert(audio_gap_code("capture_discontinuity") == "capture_discontinuity");
  assert(audio_gap_code("unexpected secret or error") == "system_audio_capture_failed");
}
