import type {
  DesktopMainView,
  DesktopMainWindowState,
  DesktopRuntimeStatus,
} from "../src/shared/contracts";

export function resolveMainWindowState(
  requestedView: DesktopMainView,
  configured: boolean,
  runtime: DesktopRuntimeStatus,
): DesktopMainWindowState {
  return {
    view: resolveView(requestedView, configured, runtime),
    chatUrl: resolveChatUrl(runtime),
  };
}

function resolveView(
  requestedView: DesktopMainView,
  configured: boolean,
  runtime: DesktopRuntimeStatus,
): DesktopMainView {
  // Unconfigured users land in onboarding; an explicit settings or models
  // request still opens those pages for people who prefer manual setup.
  if (!configured) return requestedView === "chat" ? "onboarding" : requestedView;
  if (requestedView === "onboarding") return "chat";
  return requestedView === "chat" && runtime.phase === "failed" ? "settings" : requestedView;
}

function resolveChatUrl(runtime: DesktopRuntimeStatus): string | null {
  // startNow authenticates the browser before starting Companion. Audio startup
  // or recovery must not block a working chat API.
  if (
    !["starting-companion", "ready"].includes(runtime.phase)
    || !runtime.apiUrl
  ) return null;
  if (!runtime.voiceEnabled) return runtime.apiUrl;
  const url = new URL(runtime.apiUrl);
  url.searchParams.set("nativeAgentPlayback", "1");
  return url.toString();
}
