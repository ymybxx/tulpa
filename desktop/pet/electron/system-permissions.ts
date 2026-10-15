import type { SystemPermissionKind, SystemPermissionState } from "../src/shared/contracts";

const SETTINGS_URLS: Record<SystemPermissionKind, string> = {
  microphone: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
  screen: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
  // Its own list sits on the same page, under "仅系统录音".
  systemAudio: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
};

export interface SystemPermissionDeps {
  platform: NodeJS.Platform;
  /** Electron's systemPreferences.getMediaAccessStatus. */
  mediaAccessStatus(kind: SystemPermissionKind): string;
  askForMicrophone(): Promise<boolean>;
  /** Touches the capture API so macOS registers the app in the Screen Recording list. */
  primeScreenCapture(): Promise<void>;
  openExternal(url: string): Promise<void>;
  /**
   * Re-reads a permission in a fresh helper process. macOS only updates a
   * running app's earlier answer after it relaunches, so the in-process status goes stale.
   */
  probe?(kind: SystemPermissionKind): Promise<boolean | null>;
  /** Recording system audio has no readable status; see SystemAudioPermission. */
  systemAudio?: {
    status(): SystemPermissionState;
    refresh(): Promise<SystemPermissionState>;
    ask(): Promise<SystemPermissionState>;
  };
}

/** Privacy lists show the app bundle's name, which is "Electron" for development runs. */
export function settingsAppName(executablePath: string): string {
  const bundle = executablePath.match(/([^/]+)\.app\/Contents\/MacOS\//u);
  return bundle?.[1] ?? "tulpa";
}

export function systemPermissionKind(value: unknown): SystemPermissionKind {
  if (value === "microphone" || value === "screen" || value === "systemAudio") return value;
  throw new Error("不支持的系统权限类型");
}

export function systemPermissionState(
  kind: SystemPermissionKind,
  deps: Pick<SystemPermissionDeps, "platform" | "mediaAccessStatus" | "systemAudio">,
): SystemPermissionState {
  if (kind === "systemAudio") return deps.systemAudio?.status() ?? "not-required";
  if (deps.platform !== "darwin") {
    if (kind === "screen") return "not-required";
    if (deps.platform !== "win32") return "not-required";
  }
  const raw = deps.mediaAccessStatus(kind);
  if (raw === "granted") return "granted";
  if (raw === "not-determined") return "not-determined";
  if (raw === "denied" || raw === "restricted") return "denied";
  return "not-determined";
}

/** Like systemPermissionState, but confirms a possibly stale answer out of process. */
export async function currentSystemPermission(
  kind: SystemPermissionKind,
  deps: Pick<SystemPermissionDeps, "platform" | "mediaAccessStatus" | "probe" | "systemAudio">,
): Promise<SystemPermissionState> {
  if (kind === "systemAudio") return deps.systemAudio ? deps.systemAudio.refresh() : "not-required";
  const state = systemPermissionState(kind, deps);
  if (state === "granted" || state === "not-required" || deps.platform !== "darwin" || !deps.probe) return state;
  return (await deps.probe(kind)) === true ? "granted" : state;
}

/**
 * Microphone can be asked in place; Screen Recording on macOS can only be
 * granted in System Settings, so that pane is opened after registering the app.
 */
export async function requestSystemPermission(
  kind: SystemPermissionKind,
  deps: SystemPermissionDeps,
): Promise<SystemPermissionState> {
  const current = await currentSystemPermission(kind, deps);
  if (current === "granted" || current === "not-required") return current;
  if (deps.platform === "win32") {
    if (kind === "systemAudio") return deps.systemAudio ? deps.systemAudio.ask() : "not-required";
    // Windows desktop microphone access is controlled in Settings; Electron's
    // askForMediaAccess is macOS-only. Unknown is never treated as a grant.
    await deps.openExternal(kind === "microphone" ? "ms-settings:privacy-microphone" : "ms-settings:privacy-graphicscaptureprogrammatic");
    return current;
  }
  if (kind === "systemAudio") {
    // Once refused, only System Settings can change it (macOS shows no question again).
    if (current === "denied") {
      await deps.openExternal(SETTINGS_URLS.systemAudio);
      return current;
    }
    // A short recording makes macOS ask. It cannot wait for the answer, so the
    // user answers the question first; asking again goes to System Settings.
    return deps.systemAudio ? deps.systemAudio.ask() : "not-required";
  }
  if (kind === "microphone" && current === "not-determined") {
    await deps.askForMicrophone();
    const answered = systemPermissionState(kind, deps);
    if (answered !== "not-determined") return answered;
  }
  if (deps.platform !== "darwin") return systemPermissionState(kind, deps);
  if (kind === "screen") await deps.primeScreenCapture().catch(() => undefined);
  const afterPrime = await currentSystemPermission(kind, deps);
  if (afterPrime === "granted") return afterPrime;
  await deps.openExternal(SETTINGS_URLS[kind]);
  return afterPrime;
}
