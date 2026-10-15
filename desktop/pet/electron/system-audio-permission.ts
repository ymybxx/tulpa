import type { SystemPermissionState } from "../src/shared/contracts";
import type { PerceptionCapabilityReport } from "./perception-manager";

/** What the perception helper reports when macOS refuses to let it record system audio. */
export const SYSTEM_AUDIO_PERMISSION_REQUIRED = "core_audio_process_tap_permission_required";

/** How often a refused permission is re-checked while someone may be in System Settings. */
const RECHECK_MS = 4_000;

export interface SystemAudioPermissionDeps {
  supported(): boolean;
  /** Screen Recording also lets the helper record system audio (its fallback). */
  screenGranted(): boolean;
  /** The running perception helper's latest system-audio report, if it runs one. */
  capability(): PerceptionCapabilityReport | undefined;
  /** Records for a moment: true when it worked, false when refused, null when it could not tell. */
  probe(): Promise<boolean | null>;
  now?(): number;
}

/**
 * macOS keeps no readable answer for "record system audio": a refusal looks
 * the same as never having been asked, and asking again shows nothing. So the
 * answer is taken from what captures show — the running helper's report, or a
 * short capture made here. The helper falls back to ScreenCaptureKit audio,
 * which Screen Recording alone allows, so without other evidence the answer
 * follows Screen Recording.
 */
export class SystemAudioPermission {
  private answer: SystemPermissionState | null = null;
  private probedAt = Number.NEGATIVE_INFINITY;
  private probing: Promise<SystemPermissionState> | null = null;

  constructor(private readonly deps: SystemAudioPermissionDeps) {}

  status(): SystemPermissionState {
    if (!this.deps.supported()) return "not-required";
    const capability = this.deps.capability();
    if (capability?.state === "running") return "granted";
    if (capability?.reason === SYSTEM_AUDIO_PERMISSION_REQUIRED) return "denied";
    if (this.answer) return this.answer;
    return this.deps.screenGranted() ? "granted" : "not-determined";
  }

  /** Captures briefly: the first time macOS asks the user, afterwards it only answers. */
  ask(): Promise<SystemPermissionState> {
    if (!this.deps.supported()) return Promise.resolve("not-required");
    this.probing ??= this.deps.probe()
      .then((worked) => {
        if (worked !== null) this.answer = worked ? "granted" : "denied";
        return this.status();
      })
      .finally(() => {
        this.probedAt = this.now();
        this.probing = null;
      });
    return this.probing;
  }

  /**
   * The current answer; a refusal nobody else is watching is re-checked every
   * few seconds, so switching it on in System Settings shows up here.
   */
  async refresh(): Promise<SystemPermissionState> {
    const state = this.status();
    const watched = this.deps.capability() !== undefined;
    if (state === "denied" && !watched && this.now() - this.probedAt >= RECHECK_MS) return this.ask();
    return state;
  }

  private now(): number {
    return this.deps.now?.() ?? Date.now();
  }
}
