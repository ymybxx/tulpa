#pragma once
#include "common.h"
#include <atomic>
#include <thread>

namespace tulpa::windows {
class ComputerHistory {
 public:
  ComputerHistory(DWORD host, Emit emit, Report report) : host_(host), emit_(std::move(emit)), report_(std::move(report)) {}
  ~ComputerHistory() { stop(); }
  void start();
  void stop();
 private:
  void run(std::stop_token stop);
  DWORD host_;
  Emit emit_; Report report_;
  std::jthread worker_;
  std::atomic<DWORD> thread_id_{0};
};
}
