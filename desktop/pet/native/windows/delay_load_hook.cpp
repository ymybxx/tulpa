// Adapted from node-gyp src/win_delay_load_hook.cc (MIT; see node-gyp-LICENSE.txt).
// Delay-load Node-API from the executable hosting this addon, including Electron
// and ELECTRON_RUN_AS_NODE. Directly importing node.exe would fail in tulpa.exe.
#include <windows.h>
#include <delayimp.h>
#include <cstring>
static FARPROC WINAPI resolve_node(unsigned event, PDelayLoadInfo info) {
  if (event == dliNotePreLoadLibrary && (_stricmp(info->szDll, "node.exe") == 0 || _stricmp(info->szDll, "node.dll") == 0))
    return reinterpret_cast<FARPROC>(GetModuleHandleW(L"libnode.dll") ? GetModuleHandleW(L"libnode.dll") : GetModuleHandleW(nullptr));
  return nullptr;
}
decltype(__pfnDliNotifyHook2) __pfnDliNotifyHook2 = resolve_node;
