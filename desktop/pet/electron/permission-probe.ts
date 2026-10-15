import { execFile } from "node:child_process";

import type { SystemPermissionKind } from "../src/shared/contracts";

type Runner = (file: string, args: string[], timeoutMs: number) => Promise<string>;

const defaultRunner: Runner = (file, args, timeoutMs) => new Promise((resolve, reject) => {
  execFile(file, args, { timeout: timeoutMs, windowsHide: true }, (error, stdout) => error ? reject(error) : resolve(stdout));
});

export interface CaptureWarmupResult {
  screen: boolean;
  systemAudio: string;
}

/**
 * Makes the perception helper capture once, so macOS asks its follow-up
 * questions (direct screen access, system audio) now, one after another. The
 * helper waits while a question is on screen, hence the long timeout.
 */
export async function warmUpCapture(
  helperPath: string,
  what: { screen: boolean; audio: boolean },
  run: Runner = defaultRunner,
): Promise<CaptureWarmupResult | null> {
  const mode = !what.screen ? ["--audio-only"] : what.audio ? ["--with-system-audio"] : [];
  try {
    const output = await run(helperPath, ["--capture-warmup", ...mode], 180_000);
    const value = JSON.parse(output.trim().split("\n").pop() ?? "") as Partial<CaptureWarmupResult>;
    if (typeof value.screen !== "boolean" || typeof value.systemAudio !== "string") return null;
    return { screen: value.screen, systemAudio: value.systemAudio };
  } catch {
    return null;
  }
}

/**
 * Asks a freshly started perception helper whether a permission is granted.
 * macOS keeps a running app's earlier answer for Screen Recording and the
 * microphone until it relaunches; a new process reads the current one.
 * Returns null when the helper is unavailable or answers something unexpected.
 */
export async function probePermission(
  helperPath: string,
  kind: SystemPermissionKind,
  run: Runner = defaultRunner,
): Promise<boolean | null> {
  try {
    const output = await run(helperPath, ["--permission-probe"], 5_000);
    const value = JSON.parse(output.trim().split("\n").pop() ?? "") as { screenCapture?: unknown; microphone?: unknown };
    if (kind === "screen") return typeof value.screenCapture === "boolean" ? value.screenCapture : null;
    return typeof value.microphone === "string" ? value.microphone === "granted" : null;
  } catch {
    return null;
  }
}
