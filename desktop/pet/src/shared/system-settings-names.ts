import type { SystemPermissionKind } from "./contracts";

/**
 * What macOS calls the places a permission switch lives, which changed across
 * versions: 录屏与系统录音 (with 仅系统录音 below it) only exists from macOS 15;
 * 13 and 14 say 屏幕录制; 12 still has 系统偏好设置 › 安全性与隐私.
 */
export interface SystemSettingsNames {
  /** The privacy page, e.g. 隐私与安全性. */
  privacy: string;
  /** The screen recording list. */
  screen: string;
  /** The audio-only list under it, when this macOS has one. */
  audioOnly: string | null;
}

/** `darwinMajor` is the kernel's major version: 24 is macOS 15, 22 is macOS 13. */
export function systemSettingsNames(darwinMajor: number, locale: "zh-CN" | "en" = "zh-CN", platform: string = "darwin"): SystemSettingsNames {
  if (platform === "win32") return locale === "en"
    ? { privacy: "Privacy & security", screen: "Screenshots and apps", audioOnly: null }
    : { privacy: "隐私和安全性", screen: "屏幕截图和应用", audioOnly: null };
  if (locale === "en") {
    if (darwinMajor >= 24) return { privacy: "Privacy & Security", screen: "Screen & System Audio Recording", audioOnly: "System Audio Recording Only" };
    if (darwinMajor >= 22) return { privacy: "Privacy & Security", screen: "Screen Recording", audioOnly: null };
    return { privacy: "Security & Privacy › Privacy", screen: "Screen Recording", audioOnly: null };
  }
  if (darwinMajor >= 24) return { privacy: "隐私与安全性", screen: "录屏与系统录音", audioOnly: "仅系统录音" };
  if (darwinMajor >= 22) return { privacy: "隐私与安全性", screen: "屏幕录制", audioOnly: null };
  return { privacy: "安全性与隐私 › 隐私", screen: "屏幕录制", audioOnly: null };
}

/** Where a permission's switch is on this Mac, e.g. 隐私与安全性 › 屏幕录制. */
export function permissionPlace(kind: SystemPermissionKind, names: SystemSettingsNames, locale: "zh-CN" | "en" = "zh-CN"): string {
  const pairs: Record<string, string> = { "隐私和安全性": "Privacy & security", "屏幕截图和应用": "Screenshots and apps", "隐私与安全性": "Privacy & Security", "录屏与系统录音": "Screen & System Audio Recording", "仅系统录音": "System Audio Recording Only", "屏幕录制": "Screen Recording", "安全性与隐私 › 隐私": "Security & Privacy › Privacy" };
  const local = (value: string) => locale === "en" ? pairs[value] ?? value : Object.entries(pairs).find(([, en]) => en === value)?.[0] ?? value;
  names = { privacy: local(names.privacy), screen: local(names.screen), audioOnly: names.audioOnly && local(names.audioOnly) };
  const list = kind === "microphone" ? (locale === "en" ? "Microphone" : "麦克风")
    : kind === "systemAudio" && names.audioOnly ? `${names.screen} › ${names.audioOnly}` : names.screen;
  return `${names.privacy} › ${list}`;
}
