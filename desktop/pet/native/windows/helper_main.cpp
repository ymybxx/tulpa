#include "audio.h"
#include "audio_buffer.h"
#include "computer_history.h"
#include "screen_capture.h"
#include <winrt/Windows.Devices.Enumeration.h>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <thread>

namespace tulpa::windows {
namespace {
std::uint64_t monotonic_ns() {
  LARGE_INTEGER count{}, frequency{}; QueryPerformanceCounter(&count); QueryPerformanceFrequency(&frequency);
  return static_cast<std::uint64_t>(count.QuadPart / frequency.QuadPart) * 1'000'000'000ULL
    + static_cast<std::uint64_t>(count.QuadPart % frequency.QuadPart) * 1'000'000'000ULL / static_cast<std::uint64_t>(frequency.QuadPart);
}
class Writer {
 public:
  void send(const std::string& event, const json& body, const std::string& request = "") {
    std::lock_guard lock(mutex_);
    json value{{"protocolVersion", 1}, {"sequence", ++sequence_}, {"sentAt", unix_time()}, {"event", event}, {"body", body}};
    if (!request.empty()) value["requestId"] = request;
    std::cout << value.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
  }
 private:
  std::mutex mutex_; std::uint64_t sequence_ = 0;
};
json disabled_modules() {
  return {{"workspace", false}, {"screen", false}, {"accessibility", false}, {"camera", false}, {"enhancedVision", false},
    {"computerHistory", false}, {"legacyScreenAnalysis", false}, {"excludeHostApplicationFromCapture", false}, {"systemAudio", false}};
}
json audio_window_payload(const SystemAudioWindow& window) {
  return {{"captureSessionId", window.captureSessionId}, {"windowIndex", window.windowIndex}, {"startedAt", window.startedAt}, {"endedAt", window.endedAt},
    {"audioStatus", window.audioStatus}, {"audioSampleRate", window.audioSampleRate}, {"audioCapturedSamples", window.audioCapturedSamples}, {"audioSilence", window.audioSilence},
    {"audioGapReason", window.audioGapReason ? json(*window.audioGapReason) : json(nullptr)}, {"audioPcmBase64", window.pcmBytes.empty() ? json(nullptr) : json(base64(window.pcmBytes))}};
}
void protocol_fixtures() {
  // Exercises production serializers and WIC with synthetic data only. No
  // desktop, accessibility provider, audio device, or profile is accessed.
  Writer writer;
  const auto frame = synthetic_protocol_frame();
  writer.send("environment_frame_available", frame_metadata(frame));
  auto payload = frame_metadata(frame); payload["mimeType"] = "image/jpeg"; payload["jpegBase64"] = base64(frame.jpeg);
  writer.send("environment_frame", payload, "fixture");
  std::wstring long_text;
  for (int i = 0; i < 5000; ++i) long_text += L"中\U0001F600";
  long_text += std::wstring{L'\0', L'\x7f', L'\x2028', L'\x2029', L'\xd800'};
  const json app{{"bundleIdentifier", "win32:fixture.exe"}, {"name", context_text(L" Fixture\n应用 ", 200)}};
  const json window{{"title", context_text(long_text, 512)}};
  const json target{{"role", "UIA:50004"}, {"title", wire_text(long_text, 512)}};
  const auto emit = [&](const std::string& kind, json fields = json::object()) { writer.send("interaction_event", history_event(kind, unix_time(), app, window, std::move(fields))); };
  emit("session.started", {{"ax", {{"mode", "fullTree"}, {"text", wire_text(long_text, 24000)}, {"truncated", true}}}});
  emit("window.changed");
  for (const auto* kind : {"mouse.click", "mouse.context_menu"}) emit(kind, {{"mouse", {{"button", "left"}, {"clickCount", 1}, {"location", {{"x", -100}, {"y", 75}}}, {"target", target}}}});
  emit("mouse.drag", {{"mouse", {{"button", "left"}, {"origin", {{"x", 0}, {"y", 0}}}, {"destination", {{"x", 100}, {"y", 75}}}, {"target", target}}}});
  emit("keyboard.text_input", {{"keyboard", {{"text", wire_text(long_text, 8192)}, {"target", target}}}});
  emit("keyboard.shortcut", {{"keyboard", {{"keyEquivalent", "C"}, {"modifiers", json::array({"control"})}, {"target", target}}}});
  emit("keyboard.submit", {{"keyboard", {{"keyEquivalent", "return"}, {"target", target}}}});
  emit("selection.changed", {{"selection", {{"text", wire_text(long_text, 8192)}, {"truncated", true}, {"target", target}}}});
  writer.send("interaction_events_suppressed", {{"count", 2}, {"timestamp", unix_time()}});
  SystemAudioWindowBuffer disabled(unique_id(), 1000, 0, "system_audio_disabled");
  for (const auto& value : disabled.drainReady(11'000'000'000ULL)) writer.send("system_audio_window", audio_window_payload(value));
  SystemAudioWindowBuffer silence(unique_id(), 1000, 0, "");
  for (int i = 0; i < 5; ++i) silence.append(std::vector<std::int16_t>(32000, 0));
  for (const auto& value : silence.drainReady(11'000'000'000ULL)) writer.send("system_audio_window", audio_window_payload(value));
}
class Runtime {
 public:
  explicit Runtime(DWORD host) : host_(host),
    screen_(host, [this](const auto& event, const auto& value) { writer_.send(event, value); }, [this](const auto& state, const auto& reason) { report("screen", state, reason); }),
    history_(host, [this](const auto& event, const auto& value) { writer_.send(event, value); }, [this](const auto& state, const auto& reason) { report("computerHistory", state, reason); }) {
    for (const auto* name : {"workspace", "screen", "accessibility", "camera", "computerHistory", "systemAudio"}) {
      const bool supported = std::string(name) == "screen" ? ScreenCapture::supported()
        : std::string(name) == "computerHistory" || (std::string(name) == "systemAudio" && host_ != 0);
      capabilities_[name] = capability(supported, false, "disabled", supported ? "" : "unsupported_on_windows");
    }
  }
  ~Runtime() { stop(); }
  void hello() { writer_.send("hello", {{"helperVersion", "0.1.0-windows"}, {"platform", "win32"}, {"minimumWindowsBuild", 22000}, {"capabilities", snapshot()}}); status(); }
  bool command(const json& command) {
    const auto id = command.value("requestId", std::string());
    const auto name = command.value("command", std::string());
    const auto error = [&](const char* code, const char* message) { writer_.send("error", {{"code", code}, {"message", message}, {"fatal", false}}, id); };
    if (!command.contains("protocolVersion") || command["protocolVersion"] != 1) { error("unsupported_protocol_version", "Expected protocolVersion 1"); return true; }
    if (name == "configure") {
      if (!command.contains("modules") || !command["modules"].is_object()) { error("missing_modules", "configure requires modules"); return true; }
      json next = disabled_modules();
      for (auto& [key, value] : next.items()) {
        if (command["modules"].contains(key)) {
          if (!command["modules"][key].is_boolean()) { error("invalid_modules", "Module values must be booleans"); return true; }
          value = command["modules"][key];
        }
      }
      if (command.contains("environmentFrameIntervalMs") && !command["environmentFrameIntervalMs"].is_number_integer()) { error("invalid_environment_frame_interval", "Expected an integer interval"); return true; }
      const int interval = command.value("environmentFrameIntervalMs", 1000);
      if (!valid_interval(interval)) { error("invalid_environment_frame_interval", "Interval must be between 250 and 60000"); return true; }
      configure(next, interval);
      writer_.send("ack", {{"command", name}}, id); status(id);
    } else if (name == "status") status(id);
    else if (name == "ping") writer_.send("ack", {{"command", name}}, id);
    else if (name == "claim_environment_frame") {
      const auto frame = screen_.claim(command.value("frameId", std::string()));
      if (!frame) writer_.send("environment_frame_unavailable", {{"code", "unavailable"}}, id);
      else { auto body = frame_metadata(*frame); body["mimeType"] = "image/jpeg"; body["jpegBase64"] = base64(frame->jpeg); writer_.send("environment_frame", body, id); }
    } else if (name == "claim_vision_candidate") writer_.send("vision_candidate_unavailable", {{"code", "unavailable"}}, id);
    else if (name == "cut_environment_window") {
      if (command.contains("boundaryAt") && command["boundaryAt"].is_number() && audio_buffer_) audio_buffer_->cut(command["boundaryAt"].get<double>(), unix_time());
      writer_.send("ack", {{"command", name}}, id);
    } else if (name == "shutdown") { writer_.send("ack", {{"command", name}}, id); stop(); status(id, "stopped"); return false; }
    else error("unknown_command", "Unsupported helper command");
    return true;
  }
  void invalid_command() { writer_.send("error", {{"code", "invalid_command"}, {"message", "Expected a bounded JSON command"}, {"fatal", false}}); }
 private:
  json snapshot() { std::lock_guard lock(state_mutex_); return capabilities_; }
  void status(const std::string& id = "", const char* phase = "ready") {
    json capabilities, modules;
    { std::lock_guard lock(state_mutex_); capabilities = capabilities_; modules = modules_; }
    writer_.send("status", {{"phase", phase}, {"modules", modules}, {"capabilities", capabilities}}, id);
  }
  void report(const std::string& module, const std::string& state, const std::string& reason) {
    bool changed = false;
    { std::lock_guard lock(state_mutex_); auto& value = capabilities_[module];
      if (!value.value("requested", false)) return;
      const auto actual = state == "stopped" ? "disabled" : state;
      changed = value.value("state", "") != actual || value.value("reason", "") != reason;
      value["state"] = actual;
      if (reason.empty()) value.erase("reason"); else value["reason"] = reason;
    }
    if (changed) { if (!reason.empty()) std::cerr << module << ": " << reason << '\n'; status(); }
  }
  void configure(const json& next, int interval) {
    { std::lock_guard lock(state_mutex_);
      if (next == modules_) { screen_.update_interval(interval); return; }
    }
    stop();
    { std::lock_guard lock(state_mutex_); modules_ = next;
      for (auto& [key, value] : capabilities_.items()) {
        value["requested"] = next.value(key, false);
        value["state"] = !next.value(key, false) ? "disabled" : "unavailable";
        if (next.value(key, false)) value["reason"] = value.value("supported", false) ? "starting" : "unsupported_on_windows";
      }
    }
    if (next.value("screen", false)) {
      if (next.value("legacyScreenAnalysis", false) || next.value("enhancedVision", false)) report("screen", "unavailable", "legacy_vision_unsupported_on_windows");
      else screen_.start(interval, next.value("excludeHostApplicationFromCapture", false));
    }
    if (next.value("computerHistory", false)) history_.start();
    if (next.value("screen", false) || next.value("systemAudio", false)) {
      const bool audio_enabled = next.value("systemAudio", false) && host_;
      audio_buffer_ = std::make_shared<SystemAudioWindowBuffer>(unique_id(), unix_time(), monotonic_ns(), audio_enabled ? "system_audio_starting" : "system_audio_disabled");
      auto buffer = audio_buffer_;
      if (audio_enabled) {
      audio_ = std::make_unique<SystemAudioCapture>(host_, [buffer](std::uint64_t, const std::vector<std::int16_t>& pcm) { buffer->append(pcm); },
        [this, buffer](const std::string& state, const std::string& reason) {
          if (state == "running") { buffer->markBackendRunning(); if (!reason.empty()) buffer->markGap(audio_gap_code(reason)); }
          else buffer->markBackendUnavailable(reason.empty() ? "system_audio_stopped" : audio_gap_code(reason));
          report("systemAudio", state, reason);
        });
      audio_->start();
      }
      audio_timer_ = std::jthread([this, buffer](std::stop_token stop) {
        while (!stop.stop_requested()) {
          for (const auto& window : buffer->drainReady(monotonic_ns())) {
            writer_.send("system_audio_window", audio_window_payload(window));
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      });
    }
  }
  void stop() {
    screen_.stop(); history_.stop();
    if (audio_timer_.joinable()) { audio_timer_.request_stop(); audio_timer_.join(); }
    if (audio_) { audio_->stop(); audio_.reset(); }
    audio_buffer_.reset();
    std::lock_guard lock(state_mutex_);
    for (auto& [key, value] : capabilities_.items()) { value["requested"] = false; value["state"] = "disabled"; }
    modules_ = disabled_modules();
  }
  DWORD host_;
  Writer writer_;
  std::mutex state_mutex_;
  json capabilities_ = json::object(), modules_ = disabled_modules();
  ScreenCapture screen_; ComputerHistory history_;
  std::shared_ptr<SystemAudioWindowBuffer> audio_buffer_;
  std::unique_ptr<SystemAudioCapture> audio_;
  std::jthread audio_timer_;
};

std::string microphone_permission() {
  try {
    using namespace winrt::Windows::Devices::Enumeration;
    const auto value = DeviceAccessInformation::CreateFromDeviceClass(DeviceClass::AudioCapture).CurrentStatus();
    if (value == DeviceAccessStatus::Allowed) return "granted";
    if (value == DeviceAccessStatus::DeniedBySystem || value == DeviceAccessStatus::DeniedByUser) return "denied";
  } catch (...) {}
  return "not-determined";
}
}
}
int wmain(int argc, wchar_t** argv) {
  using namespace tulpa::windows;
  try {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    DWORD host = parent_pid(); bool probe = false, warmup = false, audio = false, screen = true, fixtures = false;
    for (int i = 1; i < argc; ++i) {
      if (std::wstring(argv[i]) == L"--host-pid" && i + 1 < argc) host = static_cast<DWORD>(std::stoul(argv[++i]));
      else if (std::wstring(argv[i]) == L"--permission-probe") probe = true;
      else if (std::wstring(argv[i]) == L"--capture-warmup") warmup = true;
      else if (std::wstring(argv[i]) == L"--audio-only") { audio = true; screen = false; }
      else if (std::wstring(argv[i]) == L"--with-system-audio") audio = true;
      else if (std::wstring(argv[i]) == L"--protocol-fixtures") fixtures = true;
    }
    if (fixtures) { protocol_fixtures(); return 0; }
    if (probe) {
      std::cout << json{{"screenCapture", ScreenCapture::supported() && desktop_unlocked()}, {"microphone", microphone_permission()}}.dump() << std::endl;
      return 0;
    }
    if (warmup) {
      std::mutex mutex; std::condition_variable ready; bool screen_ok = false, audio_done = !audio; std::string audio_state = "disabled";
      ScreenCapture capture(host, [&](const auto&, const auto&) { std::lock_guard lock(mutex); screen_ok = true; ready.notify_all(); }, [](const auto&, const auto&) {});
      SystemAudioCapture loopback(host, [](auto, const auto&) {}, [&](const auto& state, const auto&) { std::lock_guard lock(mutex); audio_state = state; audio_done = state == "running" || state == "failed"; ready.notify_all(); });
      if (screen) capture.start(1000, false); if (audio) loopback.start();
      { std::unique_lock lock(mutex); ready.wait_for(lock, std::chrono::seconds(8), [&] { return (!screen || screen_ok) && audio_done; }); }
      std::string observed_audio_state; bool observed_screen_ok = false;
      { std::lock_guard lock(mutex); observed_audio_state = audio_state; observed_screen_ok = screen_ok; }
      capture.stop(); loopback.stop();
      std::cout << json{{"screen", observed_screen_ok}, {"systemAudio", observed_audio_state}}.dump() << std::endl;
      return 0;
    }
    Runtime runtime(host); runtime.hello();
    // Refuse oversized lines before unbounded allocation, and never echo input/secrets.
    std::string line; line.reserve(4096); bool oversized = false; char ch;
    while (std::cin.get(ch)) {
      if (ch != '\n') { if (line.size() < 1024 * 1024) line += ch; else oversized = true; continue; }
      if (oversized) runtime.invalid_command();
      else if (!line.empty()) { try { if (!runtime.command(json::parse(line))) break; } catch (...) { runtime.invalid_command(); } }
      line.clear(); oversized = false;
    }
    return 0;
  } catch (const winrt::hresult_error& error) { std::cerr << "windows_helper: " << error_code(error.code()) << '\n'; }
    catch (const std::exception& error) { std::cerr << "windows_helper: " << error.what() << '\n'; }
  return 1;
}
