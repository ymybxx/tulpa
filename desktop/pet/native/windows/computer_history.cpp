#include "computer_history.h"
#include "protocol_state.h"
#include <uiautomation.h>
#include <deque>
#include <map>
#include <set>

namespace tulpa::windows {
namespace {
struct Input { UINT message; POINT point{}; DWORD key{}; double at = 0; bool control = false, alt = false, shift = false, windows = false; HWND window{}; };
struct Hooks { std::deque<Input> queue; unsigned suppressed = 0; bool changed = true; };
thread_local Hooks* current_hooks = nullptr;
void queue_input(Input value) {
  if (!current_hooks) return;
  value.window = GetForegroundWindow();
  if (current_hooks->queue.size() >= 128) { ++current_hooks->suppressed; return; }
  current_hooks->queue.push_back(value);
}
void CALLBACK foreground_hook(HWINEVENTHOOK, DWORD, HWND, LONG, LONG, DWORD, DWORD) { if (current_hooks) current_hooks->changed = true; }
LRESULT CALLBACK mouse_hook(int code, WPARAM message, LPARAM data) {
  if (code >= 0 && (message == WM_LBUTTONDOWN || message == WM_LBUTTONUP || message == WM_RBUTTONDOWN || message == WM_RBUTTONUP || message == WM_MBUTTONDOWN || message == WM_MBUTTONUP)) {
    const auto& event = *reinterpret_cast<MSLLHOOKSTRUCT*>(data);
    if (!(event.flags & LLMHF_INJECTED)) queue_input({static_cast<UINT>(message), event.pt, 0, unix_time()});
  }
  return CallNextHookEx(nullptr, code, message, data);
}
LRESULT CALLBACK key_hook(int code, WPARAM message, LPARAM data) {
  if (code >= 0 && (message == WM_KEYDOWN || message == WM_SYSKEYDOWN)) {
    const auto& event = *reinterpret_cast<KBDLLHOOKSTRUCT*>(data);
    if (!(event.flags & LLKHF_INJECTED)) queue_input({static_cast<UINT>(message), {}, event.vkCode, unix_time(),
      (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0, (event.flags & LLKHF_ALTDOWN) != 0,
      (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0, (GetAsyncKeyState(VK_LWIN) & 0x8000) != 0 || (GetAsyncKeyState(VK_RWIN) & 0x8000) != 0});
  }
  return CallNextHookEx(nullptr, code, message, data);
}
std::wstring bstr_text(BSTR value, size_t limit = 2048) {
  if (!value) return {};
  std::wstring result(value, std::min<size_t>(SysStringLen(value), limit)); SysFreeString(value);
  if (!result.empty() && result.back() >= 0xd800 && result.back() <= 0xdbff) result.pop_back();
  return result;
}
bool safe_element(IUIAutomationElement* element) {
  BOOL password = TRUE;
  return element && SUCCEEDED(element->get_CurrentIsPassword(&password)) && !password;
}
std::wstring element_name(IUIAutomationElement* element) { BSTR text{}; if (!element || FAILED(element->get_CurrentName(&text))) return {}; return bstr_text(text, 512); }
json target(IUIAutomationElement* element) {
  if (!safe_element(element)) return json::object();
  CONTROLTYPEID type = 0; element->get_CurrentControlType(&type);
  return {{"role", "UIA:" + std::to_string(type)}, {"title", wire_text(element_name(element), 512)}};
}
std::wstring focused_text(IUIAutomationElement* element) {
  if (!safe_element(element)) return {};
  winrt::com_ptr<IUIAutomationValuePattern> value;
  if (SUCCEEDED(element->GetCurrentPatternAs(UIA_ValuePatternId, __uuidof(IUIAutomationValuePattern), value.put_void()))) {
    BSTR text{}; if (SUCCEEDED(value->get_CurrentValue(&text))) return bstr_text(text, 8192);
  }
  // Raw key codes are not text, in particular for Chinese IME composition.
  // Only accessible committed values are observed; unsupported providers stay absent.
  return {};
}
std::wstring selection_text(IUIAutomationElement* element) {
  if (!safe_element(element)) return {};
  winrt::com_ptr<IUIAutomationTextPattern> pattern;
  if (FAILED(element->GetCurrentPatternAs(UIA_TextPatternId, __uuidof(IUIAutomationTextPattern), pattern.put_void()))) return {};
  winrt::com_ptr<IUIAutomationTextRangeArray> ranges;
  if (FAILED(pattern->GetSelection(ranges.put()))) return {};
  int count = 0; if (FAILED(ranges->get_Length(&count)) || count == 0) return {};
  winrt::com_ptr<IUIAutomationTextRange> range;
  if (FAILED(ranges->GetElement(0, range.put()))) return {};
  BSTR text{}; if (FAILED(range->GetText(4096, &text))) return {};
  return bstr_text(text, 4096);
}
std::wstring process_name(DWORD pid) {
  HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
  if (!process) return {};
  wchar_t path[32768]{}; DWORD length = 32768;
  const bool ok = QueryFullProcessImageNameW(process, 0, path, &length) != FALSE; CloseHandle(process);
  if (!ok) return {};
  std::wstring value(path, length); const auto slash = value.find_last_of(L"\\/"); return value.substr(slash == std::wstring::npos ? 0 : slash + 1);
}
json tree_snapshot(IUIAutomation* automation, IUIAutomationElement* root) {
  winrt::com_ptr<IUIAutomationTreeWalker> walker;
  if (FAILED(automation->get_ControlViewWalker(walker.put()))) return {{"mode", "fullTree"}, {"text", ""}, {"truncated", true}};
  std::vector<std::pair<winrt::com_ptr<IUIAutomationElement>, unsigned>> stack;
  winrt::com_ptr<IUIAutomationElement> retained; retained.copy_from(root); stack.emplace_back(retained, 0);
  std::string text; unsigned nodes = 0; bool truncated = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(75);
  while (!stack.empty()) {
    if (++nodes > 80 || text.size() > 24000 || std::chrono::steady_clock::now() > deadline) { truncated = true; break; }
    auto [element, depth] = std::move(stack.back()); stack.pop_back();
    if (!safe_element(element.get())) continue;
    const auto name = wire_text(element_name(element.get()), 512);
    if (!name.empty()) { text.append(depth * 2, ' '); text += name + "\n"; }
    if (depth >= 8) { truncated = true; continue; }
    winrt::com_ptr<IUIAutomationElement> child;
    if (FAILED(walker->GetFirstChildElement(element.get(), child.put()))) continue;
    unsigned siblings = 0;
    while (child && siblings++ < 30 && stack.size() < 80) {
      stack.emplace_back(child, depth + 1);
      winrt::com_ptr<IUIAutomationElement> next;
      if (FAILED(walker->GetNextSiblingElement(child.get(), next.put()))) break;
      child = std::move(next);
    }
  }
  return {{"mode", "fullTree"}, {"text", text}, {"truncated", truncated}};
}
}

void ComputerHistory::start() { stop(); worker_ = std::jthread([this](std::stop_token stop) { run(stop); }); }
void ComputerHistory::stop() {
  if (worker_.joinable()) { worker_.request_stop(); const DWORD thread = thread_id_.load(); if (thread) PostThreadMessageW(thread, WM_QUIT, 0, 0); worker_.join(); }
  thread_id_ = 0;
}
void ComputerHistory::run(std::stop_token stop) {
  winrt::init_apartment(winrt::apartment_type::multi_threaded);
  Hooks hooks; current_hooks = &hooks;
  HHOOK mouse{}, keyboard{}; HWINEVENTHOOK foreground{};
  try {
    winrt::com_ptr<IUIAutomation> automation;
    winrt::check_hresult(CoCreateInstance(CLSID_CUIAutomation, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(automation.put())));
    if (auto advanced = automation.try_as<IUIAutomation2>()) { advanced->put_ConnectionTimeout(100); advanced->put_TransactionTimeout(100); }
    MSG message{}; PeekMessageW(&message, nullptr, 0, 0, PM_NOREMOVE); thread_id_ = GetCurrentThreadId();
    mouse = SetWindowsHookExW(WH_MOUSE_LL, mouse_hook, GetModuleHandleW(nullptr), 0);
    keyboard = SetWindowsHookExW(WH_KEYBOARD_LL, key_hook, GetModuleHandleW(nullptr), 0);
    foreground = SetWinEventHook(EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, nullptr, foreground_hook, 0, 0, WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS);
    if (!mouse || !keyboard || !foreground) throw std::runtime_error("computer_history_hooks_unavailable");
    const UINT_PTR timer = SetTimer(nullptr, 0, 250, nullptr);
    if (!timer) throw std::runtime_error("computer_history_timer_unavailable");
    report_("running", "");
    HWND previous_window{}; std::wstring previous_title, previous_value, previous_selection;
    winrt::com_ptr<IUIAutomationElement> previous_focus;
    std::map<std::string, POINT> down;
    bool first = true;
    while (!stop.stop_requested() && GetMessageW(&message, nullptr, 0, 0) > 0) {
      if (message.message != WM_TIMER) { TranslateMessage(&message); DispatchMessageW(&message); continue; }
      HWND window = GetForegroundWindow(); DWORD pid = 0; GetWindowThreadProcessId(window, &pid);
      const auto title = window_title(window);
      if (!desktop_unlocked() || !window || pid == host_ || pid == GetCurrentProcessId() || private_window_title(title)) {
        previous_window = nullptr; previous_focus = nullptr; previous_value.clear(); previous_selection.clear(); hooks.queue.clear(); down.clear(); continue;
      }
      winrt::com_ptr<IUIAutomationElement> root, focused;
      if (FAILED(automation->ElementFromHandle(window, root.put())) || !safe_element(root.get())) { hooks.queue.clear(); continue; }
      automation->GetFocusedElement(focused.put());
      int focused_pid = 0; if (focused) focused->get_CurrentProcessId(&focused_pid);
      if (static_cast<DWORD>(focused_pid) != pid) focused = nullptr;
      const auto name = process_name(pid);
      const auto app_name = context_text(name, 200), window_title_text = context_text(title, 512);
      json app = json::object(), window_info = json::object();
      if (!app_name.empty()) { app["bundleIdentifier"] = "win32:" + context_text(name, 249); app["name"] = app_name; }
      if (!window_title_text.empty()) window_info["title"] = window_title_text;
      const auto send = [&](const char* kind, double timestamp, json payload = json::object()) {
        // An event delayed by UIA must never cross an application/privacy boundary.
        if (GetForegroundWindow() != window || !desktop_unlocked() || private_window_title(window_title(window))) return;
        emit_("interaction_event", history_event(kind, timestamp, app, window_info, std::move(payload)));
      };
      if (window != previous_window || title != previous_title || hooks.changed) {
        send(first ? "session.started" : "window.changed", unix_time(), {{"ax", tree_snapshot(automation.get(), root.get())}});
        first = false; previous_window = window; previous_title = title; hooks.changed = false;
      }
      BOOL same = FALSE; if (previous_focus && focused) automation->CompareElements(previous_focus.get(), focused.get(), &same);
      const auto value = focused_text(focused.get());
      const auto selection = selection_text(focused.get());
      if (same && value != previous_value && !value.empty() && safe_element(focused.get())) {
        size_t prefix = 0; while (prefix < value.size() && prefix < previous_value.size() && value[prefix] == previous_value[prefix]) ++prefix;
        size_t suffix = 0; while (suffix < value.size() - prefix && suffix < previous_value.size() - prefix && value[value.size() - suffix - 1] == previous_value[previous_value.size() - suffix - 1]) ++suffix;
        const auto inserted = value.substr(prefix, value.size() - prefix - suffix);
        if (!inserted.empty()) send("keyboard.text_input", unix_time(), {{"keyboard", {{"text", wire_text(inserted, 8192)}, {"target", target(focused.get())}}}});
      }
      if (selection != previous_selection && !selection.empty()) send("selection.changed", unix_time(), {{"selection", {{"text", wire_text(selection, 8192)}, {"truncated", wire_text(selection, 32768).size() > 8192}, {"target", target(focused.get())}}}});
      previous_focus = focused; previous_value = value; previous_selection = selection;
      unsigned processed = 0;
      while (!hooks.queue.empty() && processed++ < 32) {
        const auto input = hooks.queue.front(); hooks.queue.pop_front();
        if (input.window != window || unix_time() - input.at > 1) { ++hooks.suppressed; continue; }
        if (input.message == WM_KEYDOWN || input.message == WM_SYSKEYDOWN) {
          if (!safe_element(focused.get()) || input.key == VK_CONTROL || input.key == VK_SHIFT || input.key == VK_MENU || input.key == VK_LWIN || input.key == VK_RWIN) continue;
          if (!input.control && !input.alt && !input.windows && input.key != VK_RETURN) continue;
          json modifiers = json::array(); if (input.control) modifiers.push_back("control"); if (input.alt) modifiers.push_back("alt"); if (input.shift) modifiers.push_back("shift"); if (input.windows) modifiers.push_back("windows");
          wchar_t key[64]{}; GetKeyNameTextW(static_cast<LONG>(MapVirtualKeyW(input.key, MAPVK_VK_TO_VSC) << 16), key, 64);
          send(input.key == VK_RETURN ? "keyboard.submit" : "keyboard.shortcut", input.at,
            {{"keyboard", {{"keyEquivalent", input.key == VK_RETURN ? "return" : wire_text(key, 128)}, {"modifiers", modifiers}, {"target", target(focused.get())}}}});
        } else {
          const std::string button = (input.message == WM_LBUTTONDOWN || input.message == WM_LBUTTONUP) ? "left" : (input.message == WM_RBUTTONDOWN || input.message == WM_RBUTTONUP) ? "right" : "other";
          if (input.message == WM_LBUTTONDOWN || input.message == WM_RBUTTONDOWN || input.message == WM_MBUTTONDOWN) { down[button] = input.point; continue; }
          auto start = down.find(button); if (start == down.end()) continue;
          const POINT origin = start->second; down.erase(start);
          winrt::com_ptr<IUIAutomationElement> hit; automation->ElementFromPoint(input.point, hit.put());
          int hit_pid = 0; if (hit) hit->get_CurrentProcessId(&hit_pid);
          if (static_cast<DWORD>(hit_pid) != pid || !safe_element(hit.get())) continue;
          const auto point = [](POINT p) { return json{{"x", p.x}, {"y", p.y}}; };
          const bool dragged = std::hypot(static_cast<double>(origin.x - input.point.x), static_cast<double>(origin.y - input.point.y)) >= 4;
          json mouse_data{{"button", button}, {"target", target(hit.get())}};
          if (dragged) { mouse_data["origin"] = point(origin); mouse_data["destination"] = point(input.point); }
          else { mouse_data["location"] = point(input.point); mouse_data["clickCount"] = 1; }
          send(dragged ? "mouse.drag" : button == "right" ? "mouse.context_menu" : "mouse.click", input.at, {{"mouse", mouse_data}});
        }
      }
      if (hooks.suppressed) { emit_("interaction_events_suppressed", {{"count", hooks.suppressed}, {"timestamp", unix_time()}}); hooks.suppressed = 0; }
    }
    KillTimer(nullptr, timer);
  } catch (const winrt::hresult_error& error) { report_("failed", "computer_history_" + error_code(error.code())); }
    catch (const std::exception& error) { report_("failed", error.what()); }
  if (foreground) UnhookWinEvent(foreground);
  if (mouse) UnhookWindowsHookEx(mouse); if (keyboard) UnhookWindowsHookEx(keyboard);
  current_hooks = nullptr; thread_id_ = 0; winrt::uninit_apartment();
}
}
