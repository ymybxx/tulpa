#import <AppKit/AppKit.h>
#include <node_api.h>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include "mouse-monitor.h"

namespace {
struct ActivationState {
  uint64_t spaceGeneration = 0;
  id __strong spaceObserver = nil;
};

struct PreviousApplication {
  NSRunningApplication* __strong application;
  uint64_t spaceGeneration;
};

NSWindow* WindowFromHandle(napi_env env, napi_callback_info info) {
  size_t count = 1;
  napi_value handle;
  void* bytes = nullptr;
  size_t length = 0;
  napi_get_cb_info(env, info, &count, &handle, nullptr, nullptr);
  if (count != 1 || napi_get_buffer_info(env, handle, &bytes, &length) != napi_ok ||
      length != sizeof(void*)) {
    napi_throw_type_error(env, nullptr, "Expected a native window handle");
    return nil;
  }
  // Electron's macOS handle is the content NSView, not the NSWindow.
  void* pointer = nullptr;
  std::memcpy(&pointer, bytes, sizeof(pointer));
  NSView* view = (__bridge NSView*)pointer;
  return view.window;
}

bool IsFrontmostApplication() {
  return NSWorkspace.sharedWorkspace.frontmostApplication.processIdentifier ==
    NSRunningApplication.currentApplication.processIdentifier;
}

// Activating an app takes the user to a Space holding one of its windows
// (System Settings' "switch to a Space with open windows", on by default).
// The pet's own panels are on every Space; a document window elsewhere, such
// as the main window, would pull the user there. The menu bar icon's window
// counts as off-Space in a full-screen Space, where the menu bar is hidden,
// but never takes the user anywhere; neither it nor a panel can become main.
bool HasWindowOffActiveSpace() {
  for (NSWindow* window in NSApp.windows) {
    if (window.isVisible && !window.isMiniaturized && !window.isOnActiveSpace &&
        window.canBecomeMainWindow) {
      return true;
    }
  }
  return false;
}

napi_value Boolean(napi_env env, bool value) {
  napi_value result;
  napi_get_boolean(env, value, &result);
  return result;
}

napi_value IsActive(napi_env env, napi_callback_info) {
  // A nonactivating panel can make NSApp.isActive report YES while another
  // process still owns system gesture delivery. Use the workspace authority.
  return Boolean(env, IsFrontmostApplication());
}

void ReleaseApplication(napi_env, void* data, void*) {
  delete static_cast<PreviousApplication*>(data);
}

napi_value MakeKeyWindow(napi_env env, napi_callback_info info) {
  @autoreleasepool {
    NSWindow* window = WindowFromHandle(env, info);
    if (!window) return Boolean(env, false);
    // Never bring an off-Space window here, or order an already visible panel
    // front. Qt's Cocoa focus path similarly uses makeKeyWindow directly.
    if (!window.isVisible || !window.isOnActiveSpace || !window.canBecomeKeyWindow) {
      return Boolean(env, false);
    }
    [window makeKeyWindow];
    return Boolean(env, window.isKeyWindow);
  }
}

napi_value Acquire(napi_env env, napi_callback_info info) {
  @autoreleasepool {
    void* data = nullptr;
    napi_get_cb_info(env, info, nullptr, nullptr, nullptr, &data);
    auto* state = static_cast<ActivationState*>(data);
    napi_value result;
    napi_get_null(env, &result);
    if (IsFrontmostApplication()) return result;
    // Live, the main window open on a desktop dragged the user out of a
    // full-screen Claude on hover. Pinches wait until it is here or closed.
    if (HasWindowOffActiveSpace()) return result;

    NSRunningApplication* previous = NSWorkspace.sharedWorkspace.frontmostApplication;
    if (!previous || previous.processIdentifier == NSRunningApplication.currentApplication.processIdentifier) {
      return result;
    }
    // Keep the application object, not a PID that could be reused after exit.
    auto* retained = new PreviousApplication{previous, state->spaceGeneration};
    if (napi_create_external(env, retained, ReleaseApplication, nullptr, &result) != napi_ok) {
      delete retained;
      napi_throw_error(env, nullptr, "Unable to retain the previous application");
      return nullptr;
    }
    [NSRunningApplication.currentApplication activateWithOptions:NSApplicationActivateIgnoringOtherApps];
    return result;
  }
}

napi_value Restore(napi_env env, napi_callback_info info) {
  @autoreleasepool {
    size_t count = 1;
    napi_value token;
    void* retained = nullptr;
    void* data = nullptr;
    napi_get_cb_info(env, info, &count, &token, nullptr, &data);
    if (count != 1 || napi_get_value_external(env, token, &retained) != napi_ok) {
      napi_throw_type_error(env, nullptr, "Expected an application activation token");
      return nullptr;
    }
    auto* state = static_cast<ActivationState*>(data);
    auto* captured = static_cast<PreviousApplication*>(retained);
    // A panel on every Space can stay key while the user changes desktops.
    // Window blur alone does not invalidate the old foreground application.
    // Reactivating it now could take the user back to its previous desktop.
    if (captured->spaceGeneration != state->spaceGeneration) return Boolean(env, false);
    NSRunningApplication* previous = captured->application;
    if (!IsFrontmostApplication() || previous.isTerminated) return Boolean(env, false);
    return Boolean(env, [previous activateWithOptions:NSApplicationActivateIgnoringOtherApps]);
  }
}

napi_value ResignKeyWindow(napi_env env, napi_callback_info info) {
  @autoreleasepool {
    // Only resign this panel,
    // never a management window the user may have opened in the meantime.
    NSWindow* window = WindowFromHandle(env, info);
    if (!window) return Boolean(env, false);
    if (!window.isKeyWindow) return Boolean(env, false);
    // BrowserWindow.blur() uses orderOut/orderBack on macOS. Resigning key
    // directly releases input without removing the panel from its Space.
    [window resignKeyWindow];
    return Boolean(env, true);
  }
}

// macOS ignores cursor changes from an app that is not active, and the pet
// takes the mouse without activating tulpa (another app's full-screen Space,
// the main window on another desktop). WindowServer honours them once the
// connection is marked as setting the cursor in the background; that property
// is private, so it is looked up at run time and skipped when missing.
void AllowCursorInBackground() {
  static bool done = false;
  if (done) return;
  done = true;
  using DefaultConnection = int (*)();
  using SetProperty = int (*)(int, int, CFStringRef, CFTypeRef);
  void* graphics = dlopen(
    "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", RTLD_LAZY);
  if (!graphics) return;
  auto connection = reinterpret_cast<DefaultConnection>(dlsym(graphics, "_CGSDefaultConnection"));
  auto setProperty = reinterpret_cast<SetProperty>(dlsym(graphics, "CGSSetConnectionProperty"));
  if (!connection || !setProperty) return;
  int cid = connection();
  setProperty(cid, cid, CFSTR("SetsCursorInBackground"), kCFBooleanTrue);
}

napi_value SetCursor(napi_env env, napi_callback_info info) {
  @autoreleasepool {
    size_t count = 1;
    napi_value value;
    napi_get_cb_info(env, info, &count, &value, nullptr, nullptr);
    char kind[16] = {0};
    size_t length = 0;
    if (count != 1 || napi_get_value_string_utf8(env, value, kind, sizeof(kind), &length) != napi_ok) {
      napi_throw_type_error(env, nullptr, "Expected a cursor kind");
      return nullptr;
    }
    // The page sets its cursor while the pet window still lets the mouse
    // through, and Chromium never sends it again once the window takes the
    // mouse: live, the pet glowed under a plain arrow until clicked. The
    // system cursor is set here directly.
    AllowCursorInBackground();
    NSCursor* cursor = std::strcmp(kind, "pointer") == 0 ? NSCursor.pointingHandCursor
      : std::strcmp(kind, "grabbing") == 0 ? NSCursor.closedHandCursor
      : NSCursor.arrowCursor;
    [cursor set];
    return Boolean(env, true);
  }
}

void CleanupActivation(void* data) {
  auto* state = static_cast<ActivationState*>(data);
  [NSWorkspace.sharedWorkspace.notificationCenter removeObserver:state->spaceObserver];
  delete state;
}
}

NAPI_MODULE_INIT() {
  InstallMouseMonitor(env, exports);
  auto* state = new ActivationState();
  state->spaceObserver = [NSWorkspace.sharedWorkspace.notificationCenter
      addObserverForName:NSWorkspaceActiveSpaceDidChangeNotification object:nil
      queue:NSOperationQueue.mainQueue usingBlock:^(NSNotification*) {
        state->spaceGeneration++;
      }];
  napi_add_env_cleanup_hook(env, CleanupActivation, state);
  napi_property_descriptor methods[] = {
    {"isActive", nullptr, IsActive, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"makeKeyWindow", nullptr, MakeKeyWindow, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"acquire", nullptr, Acquire, nullptr, nullptr, nullptr, napi_default, state},
    {"restore", nullptr, Restore, nullptr, nullptr, nullptr, napi_default, state},
    {"resignKeyWindow", nullptr, ResignKeyWindow, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"setCursor", nullptr, SetCursor, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
  return exports;
}
