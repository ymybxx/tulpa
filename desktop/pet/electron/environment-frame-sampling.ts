import type { PerceptionPolicy } from "./perception-policy";
import type { CompanionSpeechActivityEvent } from "./runtime-manager";

export const SCREEN_FRAME_INTERVAL_MS = 1_000;
export const AUDIOVISUAL_FRAME_INTERVAL_MS = 1_000;
export const SPEECH_FRAME_INTERVAL_MS = 1_000;
export const SPEECH_FRAME_BURST_TIMEOUT_MS = 30_000;

interface AdaptiveEnvironmentFrameSamplingOptions {
  apply: (intervalMs: number, forceConfigure: boolean) => void;
  schedule?: (callback: () => void, delayMs: number) => NodeJS.Timeout;
  cancelSchedule?: (timer: NodeJS.Timeout) => void;
  speechTimeoutMs?: number;
}

/** Maps perception mode and live microphone speech into one native cadence. */
export class AdaptiveEnvironmentFrameSampling {
  private screenEnabled = false;
  private readonly apply: (intervalMs: number, forceConfigure: boolean) => void;
  private readonly schedule: (
    callback: () => void,
    delayMs: number,
  ) => NodeJS.Timeout;
  private readonly cancelSchedule: (timer: NodeJS.Timeout) => void;
  private readonly speechTimeoutMs: number;
  private baselineMs = SCREEN_FRAME_INTERVAL_MS;
  private speechActive = false;
  private speechTimer: NodeJS.Timeout | null = null;

  constructor(options: AdaptiveEnvironmentFrameSamplingOptions) {
    this.apply = options.apply;
    this.schedule = options.schedule ?? setTimeout;
    this.cancelSchedule = options.cancelSchedule ?? clearTimeout;
    this.speechTimeoutMs = options.speechTimeoutMs ?? SPEECH_FRAME_BURST_TIMEOUT_MS;
    if (!Number.isSafeInteger(this.speechTimeoutMs) || this.speechTimeoutMs < 1) {
      throw new Error("speech frame burst timeout is invalid");
    }
  }

  setPolicy(policy: PerceptionPolicy): void {
    this.screenEnabled = policy.screenEnabled;
    if (!this.screenEnabled) {
      this.clearSpeechTimer();
      this.speechActive = false;
      return;
    }
    const baseline = policy.audioEnabled
      ? AUDIOVISUAL_FRAME_INTERVAL_MS
      : SCREEN_FRAME_INTERVAL_MS;
    if (baseline === this.baselineMs) return;
    this.baselineMs = baseline;
    if (!this.speechActive) this.apply(this.baselineMs, false);
  }

  handleSpeechActivity(event: CompanionSpeechActivityEvent): void {
    if (!this.screenEnabled) return;
    this.clearSpeechTimer();
    if (event.active) {
      this.speechActive = true;
      // Force the configure even for screen-only's existing 1s cadence so the
      // helper captures the utterance's zero-second frame immediately.
      this.apply(SPEECH_FRAME_INTERVAL_MS, true);
      this.speechTimer = this.schedule(() => {
        this.speechTimer = null;
        this.speechActive = false;
        this.apply(this.baselineMs, false);
      }, this.speechTimeoutMs);
      return;
    }
    if (!this.speechActive) return;
    this.speechActive = false;
    this.apply(this.baselineMs, false);
  }

  dispose(): void {
    this.clearSpeechTimer();
    this.speechActive = false;
  }

  private clearSpeechTimer(): void {
    if (!this.speechTimer) return;
    this.cancelSchedule(this.speechTimer);
    this.speechTimer = null;
  }
}
