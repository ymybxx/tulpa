#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cwctype>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace tulpa::windows {
// Keep the same bounded, memory-only, single-use frame contract as macOS.
struct Frame {
  std::string id;
  std::uint64_t revision = 0;
  double observed_at = 0;
  double expires_at = 0;
  unsigned display = 1;
  unsigned width = 0, height = 0;
  std::vector<std::uint8_t> jpeg;
};
class FrameSlot {
 public:
  std::optional<Frame> store(Frame value, double now) {
    std::lock_guard lock(mutex_);
    if (value.id.empty() || value.jpeg.empty() || value.jpeg.size() > 200 * 1024 ||
        value.width == 0 || value.height == 0 || !std::isfinite(value.observed_at) ||
        !std::isfinite(now) || now >= value.observed_at + 3) return std::nullopt;
    value.revision = ++revision_;
    value.expires_at = value.observed_at + 3;
    frame_ = std::move(value);
    return frame_;
  }
  std::optional<Frame> claim(const std::string& id, double now) {
    std::lock_guard lock(mutex_);
    if (!frame_) return std::nullopt;
    if (!std::isfinite(now) || now >= frame_->expires_at) { frame_.reset(); return std::nullopt; }
    if (frame_->id != id) return std::nullopt;
    auto value = std::move(frame_); frame_.reset(); return value;
  }
  void clear() { std::lock_guard lock(mutex_); frame_.reset(); }
 private:
  std::mutex mutex_;
  std::uint64_t revision_ = 0;
  std::optional<Frame> frame_;
};

inline bool private_window_title(std::wstring value) {
  std::transform(value.begin(), value.end(), value.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
  for (const auto* marker : {L"inprivate", L"incognito", L"private browsing", L"private window", L"隐私浏览", L"隐身", L"无痕", L"无痕浏览", L"隱私瀏覽", L"無痕"})
    if (value.find(marker) != std::wstring::npos) return true;
  return false;
}
inline bool valid_interval(int milliseconds) { return milliseconds >= 250 && milliseconds <= 60'000; }
inline std::string audio_gap_code(const std::string& diagnostic) {
  const auto separator = diagnostic.find(':');
  const auto value = diagnostic.substr(0, separator);
  if (!value.empty() && value.size() <= 64 && std::all_of(value.begin(), value.end(), [](char ch) { return (ch >= 'a' && ch <= 'z') || ch == '_'; })) return value;
  return "system_audio_capture_failed";
}
inline std::string bounded_utf8(std::string value, size_t maximum) {
  if (value.size() <= maximum) return value;
  size_t end = maximum;
  while (end && (static_cast<unsigned char>(value[end]) & 0xc0) == 0x80) --end;
  value.resize(end); return value;
}

inline float half_to_float(std::uint16_t value) {
  const unsigned sign = value >> 15, exponent = (value >> 10) & 31, mantissa = value & 1023;
  const float result = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -24)
    : exponent == 31 ? 0.0f : std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, static_cast<int>(exponent) - 15);
  return sign ? -result : result;
}
inline std::uint8_t linear_to_srgb(float value) {
  value = std::clamp(value, 0.0f, 1.0f);
  const float encoded = value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
  return static_cast<std::uint8_t>(std::round(encoded * 255.0f));
}
}
