#pragma once
#include "common.h"
#include "protocol_state.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <thread>

namespace tulpa::windows {
struct PersistentCapture;
class ScreenCapture {
 public:
  ScreenCapture(DWORD host, Emit emit, Report report) : host_(host), emit_(std::move(emit)), report_(std::move(report)) {}
  ~ScreenCapture() { stop(); }
  static bool supported();
  void start(int interval, bool exclude);
  void stop();
  void update_interval(int interval);
  std::optional<Frame> claim(const std::string& id);
 private:
  Frame capture();
  bool host_excluded() const;
  DWORD host_;
  Emit emit_; Report report_;
  FrameSlot slot_;
  std::jthread worker_;
  std::mutex wait_mutex_;
  std::condition_variable wait_;
  std::atomic<int> interval_{1000};
  bool exclude_ = false;
  std::shared_ptr<PersistentCapture> session_;
};
json frame_metadata(const Frame& frame);
// Uses the production WIC encoder with synthetic pixels; never opens capture.
Frame synthetic_protocol_frame();
}
