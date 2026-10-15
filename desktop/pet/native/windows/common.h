#pragma once
#include <windows.h>
#include <objbase.h>
#include <wincrypt.h>
#include <tlhelp32.h>
#include <winrt/base.h>
#include "protocol_state.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <functional>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace tulpa::windows {
using json = nlohmann::json;
using Emit = std::function<void(const std::string&, const json&)>;
using Report = std::function<void(const std::string&, const std::string&)>;
inline double unix_time() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
inline std::string utf8(const std::wstring& value) { return winrt::to_string(value); }
inline std::string error_code(HRESULT code) { std::ostringstream out; out << "0x" << std::hex << static_cast<unsigned long>(code); return out.str(); }
inline std::string unique_id() {
  GUID value{}; winrt::check_hresult(CoCreateGuid(&value)); wchar_t text[40]{}; StringFromGUID2(value, text, 40);
  std::wstring id(text + 1, 36);
  for (auto& ch : id) ch = static_cast<wchar_t>(std::towupper(ch));
  return utf8(id);
}
inline std::string wire_text(const std::wstring& source, size_t maximum) {
  std::wstring clean; clean.reserve(source.size());
  for (size_t index = 0; index < source.size(); ++index) {
    const wchar_t ch = source[index];
    if (ch >= 0xd800 && ch <= 0xdbff) {
      if (index + 1 < source.size() && source[index + 1] >= 0xdc00 && source[index + 1] <= 0xdfff) { clean += ch; clean += source[++index]; }
      continue;
    }
    if (ch >= 0xdc00 && ch <= 0xdfff) continue;
    if (ch != 0 && ch != 0x7f && ch != 0x2028 && ch != 0x2029 && (ch >= 0x20 || ch == '\n' || ch == '\r' || ch == '\t')) clean += ch;
  }
  auto result = utf8(clean);
  return bounded_utf8(std::move(result), maximum);
}
inline std::string context_text(std::wstring source, size_t maximum) {
  for (auto& ch : source) if (ch < 0x20 || ch == 0x7f || ch == 0x2028 || ch == 0x2029) ch = L' ';
  while (!source.empty() && std::iswspace(source.front())) source.erase(source.begin());
  while (!source.empty() && std::iswspace(source.back())) source.pop_back();
  auto result = wire_text(source, maximum);
  while (!result.empty() && result.back() == ' ') result.pop_back();
  return result;
}
inline std::string base64(const std::vector<std::uint8_t>& bytes) {
  if (bytes.empty()) return "";
  DWORD size = 0;
  if (!CryptBinaryToStringA(bytes.data(), static_cast<DWORD>(bytes.size()), CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF, nullptr, &size)) throw std::runtime_error("base64_size_failed");
  std::string result(size, '\0');
  if (!CryptBinaryToStringA(bytes.data(), static_cast<DWORD>(bytes.size()), CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF, result.data(), &size)) throw std::runtime_error("base64_failed");
  while (!result.empty() && result.back() == '\0') result.pop_back();
  return result;
}
inline bool desktop_unlocked() {
  HDESK desktop = OpenInputDesktop(0, FALSE, DESKTOP_READOBJECTS);
  if (!desktop) return false;
  wchar_t name[128]{}; DWORD bytes = 0;
  const bool available = GetUserObjectInformationW(desktop, UOI_NAME, name, sizeof(name), &bytes) && _wcsicmp(name, L"Default") == 0;
  CloseDesktop(desktop); return available;
}
inline DWORD parent_pid() {
  DWORD result = 0;
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snapshot == INVALID_HANDLE_VALUE) return 0;
  PROCESSENTRY32 entry{sizeof(entry)};
  if (Process32First(snapshot, &entry)) do { if (entry.th32ProcessID == GetCurrentProcessId()) { result = entry.th32ParentProcessID; break; } } while (Process32Next(snapshot, &entry));
  CloseHandle(snapshot); return result;
}
inline std::wstring window_title(HWND window) { wchar_t text[1024]{}; GetWindowTextW(window, text, 1024); return text; }
inline json capability(bool supported, bool requested, const std::string& state, const std::string& reason = "", const std::string& permission = "not_required") {
  json value{{"supported", supported}, {"requested", requested}, {"state", state}, {"permission", permission}};
  if (!reason.empty()) value["reason"] = reason;
  return value;
}
inline json history_event(const std::string& kind, double timestamp, const json& app, const json& window, json payload = json::object()) {
  payload.update({{"id", unique_id()}, {"kind", kind}, {"timestamp", timestamp}, {"app", app}, {"window", window}});
  return payload;
}
}
