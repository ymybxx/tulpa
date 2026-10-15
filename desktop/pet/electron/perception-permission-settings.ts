import type { DesktopPerceptionPermissionKind } from "../src/shared/contracts";

const PERMISSION_SETTINGS_URLS: Record<DesktopPerceptionPermissionKind, string> = {
  accessibility:
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
  screen:
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
};

export function perceptionPermissionSettingsUrl(kind: unknown, platform = process.platform): string | null {
  if (platform === "win32") return kind === "screen" || kind === "accessibility" ? "ms-settings:privacy" : null;
  return kind === "accessibility" || kind === "screen"
    ? PERMISSION_SETTINGS_URLS[kind]
    : null;
}
