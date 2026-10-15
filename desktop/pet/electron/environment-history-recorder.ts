import {
  parseEnvironmentFramePayload,
  parseSystemAudioWindow,
  type EnvironmentFramePayload,
  type SystemAudioWindow,
} from "./perception-manager";
import { perceptionHistoryEnabled, type PerceptionPolicy } from "./perception-policy";

export interface EnvironmentHistoryTarget {
  apiUrl: string;
  apiKey: string;
  fishId: string;
}

export interface EnvironmentHistoryWindow extends SystemAudioWindow {
  frames: EnvironmentFramePayload[];
}

export type EnvironmentHistoryRecorderErrorCode =
  | "invalid_frame"
  | "invalid_audio"
  | "queue_overflow"
  | "upload_failed";

export interface EnvironmentHistoryRecorderOptions {
  fetcher?: (input: string | URL | Request, init?: RequestInit) => Promise<Response>;
  now?: () => number;
  requestTimeoutMs?: number;
  retryDelaysMs?: readonly number[];
  onErrorCode?: (code: EnvironmentHistoryRecorderErrorCode) => void;
}

interface PendingWindow {
  body: string;
  attempt: number;
  readyAt: number;
}

const FRAME_RETENTION_MS = 15_000;
const MAX_RECENT_FRAMES = 16;
const MAX_SEEN_WINDOWS = 64;
const MAX_PENDING_WINDOWS = 6;
const MAX_WINDOW_FRAMES = 10;

/**
 * Native owns the only ten-second media clock and sends one frozen audio
 * window. Electron retains only enough timestamped JPEGs to select ten views
 * with that window before handing the immutable request to the upload worker.
 */
export class EnvironmentHistoryRecorder {
  private readonly fetcher: NonNullable<EnvironmentHistoryRecorderOptions["fetcher"]>;
  private readonly now: () => number;
  private readonly requestTimeoutMs: number;
  private readonly retryDelaysMs: readonly number[];
  private enabled = false;
  private screenEnabled = true;
  private audioEnabled = true;
  private stopped = false;
  private target: EnvironmentHistoryTarget | null = null;
  private targetPending = false;
  private generation = 0;
  private recentFrames: EnvironmentFramePayload[] = [];
  private readonly seenWindowKeys = new Set<string>();
  private pending: PendingWindow[] = [];
  private retryTimer: NodeJS.Timeout | null = null;
  private active: { work: PendingWindow; controller: AbortController } | null = null;
  private worker: Promise<void> | null = null;

  constructor(private readonly options: EnvironmentHistoryRecorderOptions = {}) {
    this.fetcher = options.fetcher ?? fetch;
    this.now = options.now ?? Date.now;
    this.requestTimeoutMs = options.requestTimeoutMs ?? 2_000;
    this.retryDelaysMs = options.retryDelaysMs ?? [1_000, 5_000];
    if (!Number.isFinite(this.requestTimeoutMs) || this.requestTimeoutMs < 1) {
      throw new Error("requestTimeoutMs must be positive");
    }
    if (this.retryDelaysMs.some((delay) => !Number.isFinite(delay) || delay < 0)) {
      throw new Error("retryDelaysMs must contain nonnegative delays");
    }
  }

  setEnabled(enabled: boolean): void {
    if (this.stopped || enabled === this.enabled) return;
    this.enabled = enabled;
    this.resetTransientState();
  }

  setPolicy(policy: PerceptionPolicy): void {
    if (this.stopped) return;
    const enabled = perceptionHistoryEnabled(policy);
    if (enabled === this.enabled && policy.screenEnabled === this.screenEnabled
      && policy.audioEnabled === this.audioEnabled) return;
    this.enabled = enabled;
    this.screenEnabled = policy.screenEnabled;
    this.audioEnabled = policy.audioEnabled;
    this.resetTransientState();
  }

  /** A capture-filter change invalidates frames already queued under the old filter. */
  clearCapturedWindow(): void {
    if (!this.stopped) this.resetTransientState();
  }

  setTarget(target: EnvironmentHistoryTarget | null): void {
    if (this.stopped) return;
    if (target?.apiUrl === this.target?.apiUrl
      && target?.apiKey === this.target?.apiKey
      && target?.fishId === this.target?.fishId) {
      if (this.targetPending) {
        this.targetPending = false;
        this.kickWorker();
      }
      return;
    }
    this.target = target ? { ...target } : null;
    this.targetPending = false;
    this.resetTransientState();
  }

  suspendTarget(): void {
    if (this.stopped || this.targetPending) return;
    this.targetPending = true;
    this.generation += 1;
    this.active?.controller.abort();
    this.active = null;
    this.worker = null;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
  }

  handleFrame(payload: EnvironmentFramePayload): void {
    if (!this.canCapture() || !this.screenEnabled) return;
    try {
      const frame = parseEnvironmentFramePayload(payload);
      if (this.recentFrames.some((candidate) => candidate.frameId === frame.frameId)) return;
      this.recentFrames.push(frame);
      this.recentFrames.sort((left, right) => left.observedAt - right.observedAt);
      this.pruneFrames(this.now() / 1_000 - FRAME_RETENTION_MS / 1_000);
      if (this.recentFrames.length > MAX_RECENT_FRAMES) {
        this.recentFrames.splice(0, this.recentFrames.length - MAX_RECENT_FRAMES);
      }
    } catch {
      this.options.onErrorCode?.("invalid_frame");
    }
  }

  handleAudioWindow(payload: SystemAudioWindow): void {
    if (!this.canCapture()) return;
    let window: SystemAudioWindow;
    try {
      window = parseSystemAudioWindow(payload);
    } catch {
      this.options.onErrorCode?.("invalid_audio");
      return;
    }

    if (!this.audioEnabled) {
      window = {
        ...window, audioStatus: "unavailable", audioPcmBase64: null,
        audioCapturedSamples: 0, audioSilence: false, audioGapReason: "system_audio_disabled",
      };
    }

    const key = `${window.captureSessionId}:${window.windowIndex}`;
    if (this.seenWindowKeys.has(key)) return;
    this.recordWindowIdentity(key);

    const frames = this.screenEnabled ? this.framesForWindow(window.startedAt, window.endedAt) : [];
    this.pruneFrames(window.endedAt);
    if (frames.length === 0 && window.audioStatus === "unavailable") return;

    const frozen: EnvironmentHistoryWindow = { ...window, frames };
    if (this.pending.length >= MAX_PENDING_WINDOWS) {
      const discard = this.pending.findIndex((work) => work !== this.active?.work);
      if (discard >= 0) this.pending.splice(discard, 1);
      this.options.onErrorCode?.("queue_overflow");
    }
    const images = new Set<string>();
    const compactFrames = frozen.frames.map((frame) => {
      if (!frame.contentHash) return frame;
      if (!images.has(frame.contentHash)) {
        images.add(frame.contentHash);
        return frame;
      }
      const { jpegBase64: _jpeg, ...reference } = frame;
      return reference;
    });
    this.pending.push({ body: JSON.stringify({ ...frozen, frames: compactFrames }), attempt: 0, readyAt: this.now() });
    this.kickWorker();
  }

  stop(): void {
    this.stopped = true;
    this.enabled = false;
    this.target = null;
    this.resetTransientState();
  }

  private canCapture(): boolean {
    return this.enabled && !this.stopped && this.target !== null;
  }

  private resetTransientState(): void {
    this.generation += 1;
    this.active?.controller.abort();
    this.active = null;
    this.worker = null;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.recentFrames = [];
    this.seenWindowKeys.clear();
    this.pending = [];
  }

  private recordWindowIdentity(key: string): void {
    this.seenWindowKeys.add(key);
    while (this.seenWindowKeys.size > MAX_SEEN_WINDOWS) {
      const oldest = this.seenWindowKeys.values().next().value as string | undefined;
      if (oldest === undefined) break;
      this.seenWindowKeys.delete(oldest);
    }
  }

  private framesForWindow(startedAt: number, endedAt: number): EnvironmentFramePayload[] {
    const candidates = this.recentFrames.filter(
      (frame) => frame.observedAt >= startedAt && frame.observedAt < endedAt,
    );
    if (candidates.length <= MAX_WINDOW_FRAMES) return candidates.map((frame) => ({ ...frame }));
    // Include both ends of the actual observed interval. Do not fill missing
    // captures with duplicate frames or pick the following window's first frame.
    return Array.from({ length: MAX_WINDOW_FRAMES }, (_, index) => ({
      ...candidates[Math.round(index * (candidates.length - 1) / (MAX_WINDOW_FRAMES - 1))]!,
    }));
  }

  private pruneFrames(before: number): void {
    this.recentFrames = this.recentFrames.filter((frame) => frame.observedAt >= before);
  }

  private kickWorker(): void {
    if (this.worker || this.targetPending || !this.enabled || !this.target || !this.pending.length) return;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    const generation = this.generation;
    const running = this.runWorker(generation, this.target);
    this.worker = running;
    void running.finally(() => {
      if (this.worker !== running) return;
      this.worker = null;
      if (generation !== this.generation || !this.pending.length) return;
      const readyAt = Math.min(...this.pending.map((work) => work.readyAt));
      this.retryTimer = setTimeout(() => {
        this.retryTimer = null;
        this.kickWorker();
      }, Math.max(1, readyAt - this.now()));
      this.retryTimer.unref?.();
    });
  }

  private async runWorker(generation: number, target: EnvironmentHistoryTarget): Promise<void> {
    const failedThisPass = new Set<PendingWindow>();
    while (generation === this.generation) {
      const work = this.pending.find((item) => item.readyAt <= this.now() && !failedThisPass.has(item));
      if (!work) return;
      let retryable = false;
      let status: number | null = null;
      try {
        status = await this.upload(work, target);
        retryable = status === 429 || status === 503;
      } catch {
        retryable = true;
      }
      if (generation !== this.generation) return;
      if (status === 202) {
        this.pending = this.pending.filter((item) => item !== work);
        continue;
      }
      this.options.onErrorCode?.("upload_failed");
      const delay = this.retryDelaysMs[work.attempt];
      if (retryable && delay !== undefined) {
        work.attempt += 1;
        work.readyAt = this.now() + delay;
        failedThisPass.add(work);
      } else this.pending = this.pending.filter((item) => item !== work);
    }
  }

  private async upload(work: PendingWindow, target: EnvironmentHistoryTarget): Promise<number> {
    const controller = new AbortController();
    this.active = { work, controller };
    let rejectAbort: () => void = () => {};
    const aborted = new Promise<never>((_resolve, reject) => {
      rejectAbort = () => reject(new Error("History upload aborted"));
      controller.signal.addEventListener("abort", rejectAbort, { once: true });
    });
    const timer = setTimeout(() => controller.abort(), this.requestTimeoutMs);
    timer.unref?.();
    try {
      const response = await Promise.race([
        this.fetcher(
          `${target.apiUrl.replace(/\/$/u, "")}/internal/fishes/${encodeURIComponent(target.fishId)}`
            + "/environment-perception/history/windows",
          {
            method: "POST",
            headers: { Authorization: `Bearer ${target.apiKey}`, "Content-Type": "application/json" },
            body: work.body,
            signal: controller.signal,
          },
        ),
        aborted,
      ]);
      void response.body?.cancel().catch(() => {});
      return response.status;
    } finally {
      clearTimeout(timer);
      controller.signal.removeEventListener("abort", rejectAbort);
      if (this.active?.controller === controller) this.active = null;
    }
  }
}
