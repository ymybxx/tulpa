#import <AppKit/AppKit.h>
#include "mouse-monitor.h"

namespace {
struct Monitor {
  napi_threadsafe_function callback = nullptr;
  id __strong globalObserver = nil;
  id __strong localObserver = nil;
  bool running = true;
  bool pending = false;
};

struct Bridge {
  Monitor* monitor = nullptr;
  NSEventMask mask = NSEventMaskMouseMoved;
  // Also watch this app's own windows, not only other apps'.
  bool local = true;
};

void Stop(Bridge* bridge) {
  Monitor* monitor = bridge->monitor;
  if (!monitor) return;
  bridge->monitor = nullptr;
  monitor->running = false;
  if (monitor->globalObserver) [NSEvent removeMonitor:monitor->globalObserver];
  if (monitor->localObserver) [NSEvent removeMonitor:monitor->localObserver];
  monitor->globalObserver = nil;
  monitor->localObserver = nil;
  // The TSFN owns the allocation until every queued callback is drained.
  napi_release_threadsafe_function(monitor->callback, napi_tsfn_abort);
}

void Deliver(napi_env env, napi_value callback, void* context, void*) {
  auto* monitor = static_cast<Monitor*>(context);
  monitor->pending = false;
  if (!env || !monitor->running) return;
  napi_value receiver;
  napi_get_undefined(env, &receiver);
  // Read the current cursor in JS, not the position of a potentially stale
  // queued event. Own-app DOM events may have moved it in the meantime.
  napi_call_function(env, receiver, callback, 0, nullptr, nullptr);
}

void FinalizeMonitor(napi_env, void* data, void*) {
  delete static_cast<Monitor*>(data);
}

void NotifyMovement(Monitor* monitor) {
  if (!monitor->running || monitor->pending) return;
  monitor->pending = true;
  if (napi_call_threadsafe_function(monitor->callback, nullptr, napi_tsfn_nonblocking) != napi_ok) {
    monitor->pending = false;
  }
}

napi_value Start(napi_env env, napi_callback_info info) {
  size_t count = 1;
  napi_value callback;
  void* data;
  napi_get_cb_info(env, info, &count, &callback, nullptr, &data);
  napi_valuetype type;
  if (count != 1 || napi_typeof(env, callback, &type) != napi_ok || type != napi_function) {
    napi_throw_type_error(env, nullptr, "Expected a mouse movement callback");
    return nullptr;
  }
  if (![NSThread isMainThread]) {
    napi_throw_error(env, nullptr, "The pet mouse monitor requires the main thread");
    return nullptr;
  }
  auto* bridge = static_cast<Bridge*>(data);
  Stop(bridge);
  auto* monitor = new Monitor();
  napi_value resourceName;
  napi_create_string_utf8(env, "PetMouseMonitor", NAPI_AUTO_LENGTH, &resourceName);
  if (napi_create_threadsafe_function(env, callback, nullptr, resourceName, 1, 1,
      monitor, FinalizeMonitor, monitor, Deliver, &monitor->callback) != napi_ok) {
    delete monitor;
    napi_throw_error(env, nullptr, "Unable to create the pet mouse monitor");
    return nullptr;
  }
  bridge->monitor = monitor;
  napi_unref_threadsafe_function(env, monitor->callback);
  // The HUD must leave click-through mode over its entry even when movement
  // lands on our own management window. A global monitor excludes this app.
  // Coalesce both sources, observe only movement, and leave local events intact.
  // AppKit and Electron Node run these callbacks on the main thread.
  monitor->globalObserver = [NSEvent addGlobalMonitorForEventsMatchingMask:bridge->mask
      handler:^(NSEvent*) { NotifyMovement(monitor); }];
  if (bridge->local) {
    monitor->localObserver = [NSEvent addLocalMonitorForEventsMatchingMask:bridge->mask
        handler:^NSEvent*(NSEvent* event) {
          NotifyMovement(monitor);
          return event;
        }];
  }
  const bool installed = monitor->globalObserver != nil && (!bridge->local || monitor->localObserver != nil);
  if (!installed) Stop(bridge);
  napi_value result;
  napi_get_boolean(env, installed, &result);
  return result;
}

napi_value StopMonitor(napi_env env, napi_callback_info info) {
  void* data;
  napi_get_cb_info(env, info, nullptr, nullptr, nullptr, &data);
  Stop(static_cast<Bridge*>(data));
  napi_value result;
  napi_get_undefined(env, &result);
  return result;
}

void Cleanup(void* data) {
  auto* bridge = static_cast<Bridge*>(data);
  Stop(bridge);
  delete bridge;
}
}

void InstallMouseMonitor(napi_env env, napi_value exports) {
  auto* bridge = new Bridge();
  napi_add_env_cleanup_hook(env, Cleanup, bridge);
  // Presses anywhere: other apps, the desktop and tulpa's own windows (the
  // main window, the pet). An open panel closes on a press outside it: it
  // cannot rely on losing focus, since tulpa is often not the active app
  // while the pointer is on the pet. The caller ignores presses inside it.
  auto* clicks = new Bridge();
  clicks->mask = NSEventMaskLeftMouseDown | NSEventMaskRightMouseDown | NSEventMaskOtherMouseDown;
  clicks->local = true;
  napi_add_env_cleanup_hook(env, Cleanup, clicks);
  napi_property_descriptor methods[] = {
    {"startMouseMonitor", nullptr, Start, nullptr, nullptr, nullptr, napi_default, bridge},
    {"stopMouseMonitor", nullptr, StopMonitor, nullptr, nullptr, nullptr, napi_default, bridge},
    {"startClickMonitor", nullptr, Start, nullptr, nullptr, nullptr, napi_default, clicks},
    {"stopClickMonitor", nullptr, StopMonitor, nullptr, nullptr, nullptr, napi_default, clicks},
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
}
