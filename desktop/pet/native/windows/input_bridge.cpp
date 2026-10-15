#include <windows.h>
#include <shobjidl.h>
#include <node_api.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

namespace {
struct Bridge {
  napi_env env{};
  std::mutex mutex;
  std::condition_variable ready;
  napi_threadsafe_function movement{}, clicks{};
  std::thread thread;
  DWORD thread_id = 0;
  HHOOK hook{};
  bool started = false;
  std::map<DWORD, HANDLE> jobs;
  HANDLE host_job{};
};
Bridge* active = nullptr;
napi_value boolean(napi_env env, bool value) { napi_value result; napi_get_boolean(env, value, &result); return result; }
Bridge* state(napi_env env) { void* value{}; napi_get_instance_data(env, &value); return static_cast<Bridge*>(value); }
HWND window_arg(napi_env env, napi_callback_info info) {
  size_t argc = 1, length = 0; napi_value arg{}; void* bytes{};
  napi_get_cb_info(env, info, &argc, &arg, nullptr, nullptr);
  if (argc != 1 || napi_get_buffer_info(env, arg, &bytes, &length) != napi_ok || length != sizeof(HWND)) {
    napi_throw_type_error(env, nullptr, "Expected a native window handle"); return nullptr;
  }
  HWND window{}; std::memcpy(&window, bytes, sizeof(window));
  DWORD pid = 0; GetWindowThreadProcessId(window, &pid);
  return IsWindow(window) && pid == GetCurrentProcessId() ? window : nullptr;
}
void deliver(napi_env env, napi_value callback, void*, void*) {
  if (!env || !callback) return;
  napi_value receiver; napi_get_undefined(env, &receiver); napi_call_function(env, receiver, callback, 0, nullptr, nullptr);
}
LRESULT CALLBACK mouse_hook(int code, WPARAM message, LPARAM data) {
  if (code >= 0 && active) {
    std::lock_guard lock(active->mutex);
    auto callback = message == WM_MOUSEMOVE ? active->movement
      : (message == WM_LBUTTONDOWN || message == WM_RBUTTONDOWN || message == WM_MBUTTONDOWN || message == WM_XBUTTONDOWN) ? active->clicks : nullptr;
    if (callback) napi_call_threadsafe_function(callback, nullptr, napi_tsfn_nonblocking);
  }
  return CallNextHookEx(nullptr, code, message, data);
}
bool ensure_monitor(Bridge* bridge) {
  std::unique_lock lock(bridge->mutex);
  if (bridge->thread.joinable()) return bridge->hook != nullptr;
  bridge->started = false;
  bridge->thread = std::thread([bridge] {
    MSG message{}; PeekMessageW(&message, nullptr, 0, 0, PM_NOREMOVE);
    HMODULE module{};
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      reinterpret_cast<LPCWSTR>(&mouse_hook), &module);
    { std::lock_guard lock(bridge->mutex); bridge->thread_id = GetCurrentThreadId();
      bridge->hook = SetWindowsHookExW(WH_MOUSE_LL, mouse_hook, module, 0); bridge->started = true; }
    bridge->ready.notify_all();
    while (GetMessageW(&message, nullptr, 0, 0) > 0) { TranslateMessage(&message); DispatchMessageW(&message); }
    std::lock_guard lock(bridge->mutex);
    if (bridge->hook) UnhookWindowsHookEx(bridge->hook);
    bridge->hook = nullptr;
  });
  bridge->ready.wait(lock, [&] { return bridge->started; });
  return bridge->hook != nullptr;
}
napi_value start_monitor(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value callback{}; void* mode{};
  napi_get_cb_info(env, info, &argc, &callback, nullptr, &mode);
  napi_valuetype type{};
  if (argc != 1 || napi_typeof(env, callback, &type) != napi_ok || type != napi_function) { napi_throw_type_error(env, nullptr, "Expected a pointer callback"); return nullptr; }
  auto* bridge = state(env);
  if (!ensure_monitor(bridge)) return boolean(env, false);
  napi_value name; napi_create_string_utf8(env, "WindowsPetPointer", NAPI_AUTO_LENGTH, &name);
  napi_threadsafe_function next{};
  if (napi_create_threadsafe_function(env, callback, nullptr, name, 1, 1, nullptr, nullptr, nullptr, deliver, &next) != napi_ok) return boolean(env, false);
  napi_unref_threadsafe_function(env, next);
  std::lock_guard lock(bridge->mutex);
  auto& channel = mode ? bridge->clicks : bridge->movement;
  if (channel) napi_release_threadsafe_function(channel, napi_tsfn_abort);
  channel = next; return boolean(env, true);
}
napi_value stop_monitor(napi_env env, napi_callback_info info) {
  void* mode{}; napi_get_cb_info(env, info, nullptr, nullptr, nullptr, &mode);
  auto* bridge = state(env); std::lock_guard lock(bridge->mutex);
  auto& channel = mode ? bridge->clicks : bridge->movement;
  if (channel) napi_release_threadsafe_function(channel, napi_tsfn_abort);
  channel = nullptr; napi_value result; napi_get_undefined(env, &result); return result;
}
napi_value is_active(napi_env env, napi_callback_info) { DWORD pid{}; GetWindowThreadProcessId(GetForegroundWindow(), &pid); return boolean(env, pid == GetCurrentProcessId()); }
napi_value focus_window(napi_env env, napi_callback_info info) {
  const HWND window = window_arg(env, info);
  if (!window || !IsWindowVisible(window) || IsIconic(window)) return boolean(env, false);
  // Invoked only for an explicit composer/menu action, never pointer hover.
  // Respect Windows' foreground-lock policy: do not use AttachThreadInput tricks.
  const bool accepted = SetForegroundWindow(window) != FALSE;
  if (accepted) SetFocus(window);
  return boolean(env, accepted);
}
napi_value no_activation(napi_env env, napi_callback_info) { napi_value result; napi_get_null(env, &result); return result; }
napi_value no_restore(napi_env env, napi_callback_info) { return boolean(env, false); }
napi_value nonactivating_window(napi_env env, napi_callback_info info) {
  const HWND window = window_arg(env, info);
  if (!window) return boolean(env, false);
  SetLastError(ERROR_SUCCESS);
  const auto previous = SetWindowLongPtrW(window, GWL_EXSTYLE, GetWindowLongPtrW(window, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW);
  if (!previous && GetLastError() != ERROR_SUCCESS) return boolean(env, false);
  return boolean(env, SetWindowPos(window, nullptr, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED) != FALSE);
}
napi_value resign_window(napi_env env, napi_callback_info info) {
  const HWND window = window_arg(env, info);
  if (!window || GetFocus() != window) return boolean(env, false);
  SetFocus(nullptr); return boolean(env, true);
}
napi_value set_cursor(napi_env env, napi_callback_info info) {
  size_t argc = 1, count = 0; napi_value value{}; char text[24]{};
  napi_get_cb_info(env, info, &argc, &value, nullptr, nullptr);
  if (argc != 1 || napi_get_value_string_utf8(env, value, text, sizeof(text), &count) != napi_ok) { napi_throw_type_error(env, nullptr, "Expected a cursor kind"); return nullptr; }
  const wchar_t* kind = std::strcmp(text, "pointer") == 0 ? IDC_HAND : std::strcmp(text, "grabbing") == 0 ? IDC_SIZEALL : IDC_ARROW;
  SetCursor(LoadCursorW(nullptr, kind)); return boolean(env, true);
}
napi_value attach_job(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value value{}; std::uint32_t pid{};
  napi_get_cb_info(env, info, &argc, &value, nullptr, nullptr);
  if (argc != 1 || napi_get_value_uint32(env, value, &pid) != napi_ok || !pid || pid == GetCurrentProcessId()) return boolean(env, false);
  auto* bridge = state(env);
  if (bridge->jobs.contains(pid)) return boolean(env, true);
  HANDLE process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, FALSE, pid);
  if (!process) return boolean(env, false);
  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{}; limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  const bool ok = job && SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) && AssignProcessToJobObject(job, process);
  CloseHandle(process);
  if (ok) bridge->jobs.emplace(pid, job); else if (job) CloseHandle(job);
  return boolean(env, ok);
}
napi_value initialize_host_job(napi_env env, napi_callback_info) {
  auto* bridge = state(env);
  if (bridge->host_job) return boolean(env, true);
  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{}; limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  const bool ok = job && SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) && AssignProcessToJobObject(job, GetCurrentProcess());
  if (ok) bridge->host_job = job; else if (job) CloseHandle(job);
  return boolean(env, ok);
}
napi_value terminate_job(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value value{}; std::uint32_t pid{};
  napi_get_cb_info(env, info, &argc, &value, nullptr, nullptr);
  if (argc != 1 || napi_get_value_uint32(env, value, &pid) != napi_ok) return boolean(env, false);
  auto* bridge = state(env); const auto found = bridge->jobs.find(pid);
  if (found == bridge->jobs.end()) return boolean(env, false);
  const bool result = TerminateJobObject(found->second, 1) != FALSE;
  CloseHandle(found->second); bridge->jobs.erase(found); return boolean(env, result);
}
void cleanup(napi_env, void* data, void*) {
  auto* bridge = static_cast<Bridge*>(data);
  if (bridge->thread.joinable()) { PostThreadMessageW(bridge->thread_id, WM_QUIT, 0, 0); bridge->thread.join(); }
  active = nullptr;
  if (bridge->movement) napi_release_threadsafe_function(bridge->movement, napi_tsfn_abort);
  if (bridge->clicks) napi_release_threadsafe_function(bridge->clicks, napi_tsfn_abort);
  for (const auto& [pid, job] : bridge->jobs) CloseHandle(job);
  // The host is a member: closing host_job here could interrupt Electron's
  // shutdown writes. The OS closes it only when the host process exits.
  delete bridge;
}
}
NAPI_MODULE_INIT() {
  auto* bridge = new Bridge(); bridge->env = env; active = bridge;
  napi_set_instance_data(env, bridge, cleanup, nullptr);
  napi_property_descriptor methods[] = {
    {"isActive", nullptr, is_active, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"makeKeyWindow", nullptr, focus_window, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"acquire", nullptr, no_activation, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"restore", nullptr, no_restore, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"resignKeyWindow", nullptr, resign_window, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"setCursor", nullptr, set_cursor, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"setNonActivating", nullptr, nonactivating_window, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"startMouseMonitor", nullptr, start_monitor, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"stopMouseMonitor", nullptr, stop_monitor, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"startClickMonitor", nullptr, start_monitor, nullptr, nullptr, nullptr, napi_default, reinterpret_cast<void*>(1)},
    {"stopClickMonitor", nullptr, stop_monitor, nullptr, nullptr, nullptr, napi_default, reinterpret_cast<void*>(1)},
    {"attachProcessToJob", nullptr, attach_job, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"initializeHostJob", nullptr, initialize_host_job, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"terminateProcessJob", nullptr, terminate_job, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods); return exports;
}
