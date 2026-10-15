import { spawn, type SpawnOptionsWithoutStdio } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import type { Readable, Writable } from "node:stream";
import { StringDecoder } from "node:string_decoder";
import { attachProcessJob } from "./platform/process-tree";

export const PERCEPTION_PROTOCOL_VERSION = 1 as const;

export interface PerceptionModuleSelection {
  workspace: boolean;
  screen: boolean;
  accessibility: boolean;
  camera: boolean;
  /** Additive protocol field; omitted legacy callers normalize to false. */
  enhancedVision?: boolean;
  /** Additive protocol field; omitted legacy callers normalize to false. */
  computerHistory?: boolean;
  /** Additive field; omitted legacy callers retain the original analysis path. */
  legacyScreenAnalysis?: boolean;
  /** Exclude all Fish Memory windows from ScreenCaptureKit frames. */
  excludeHostApplicationFromCapture?: boolean;
  /** System playback audio only; independent from the screen input. */
  systemAudio?: boolean;
}

export const DISABLED_PERCEPTION_MODULES: Readonly<PerceptionModuleSelection> = Object.freeze({
  workspace: false,
  screen: false,
  accessibility: false,
  camera: false,
  enhancedVision: false,
  computerHistory: false,
  legacyScreenAnalysis: false,
  excludeHostApplicationFromCapture: false,
});

export interface ComputerHistoryInteractionEvent {
  id: string;
  timestamp: number;
  kind: ComputerHistoryInteractionKind;
  app: { bundleIdentifier?: string; name?: string };
  window: { title?: string; url?: string };
  ax?: ComputerHistoryAxPayload;
  mouse?: ComputerHistoryMousePayload;
  keyboard?: ComputerHistoryKeyboardPayload;
  selection?: ComputerHistorySelectionPayload;
}

export type ComputerHistoryInteractionKind =
  | "session.started"
  | "window.changed"
  | "mouse.click"
  | "mouse.drag"
  | "mouse.context_menu"
  | "selection.changed"
  | "keyboard.shortcut"
  | "keyboard.text_input"
  | "keyboard.submit";

export interface ComputerHistoryTarget {
  role: string;
  title?: string;
  description?: string;
}

export interface ComputerHistoryAxPayload {
  mode: "fullTree" | "diffFromPrevious";
  text: string;
  truncated: boolean;
}

export interface ComputerHistoryMousePayload {
  button: "left" | "right" | "other";
  clickCount?: number;
  location?: { x: number; y: number };
  origin?: { x: number; y: number };
  destination?: { x: number; y: number };
  target?: ComputerHistoryTarget;
}

export interface ComputerHistoryKeyboardPayload {
  keyEquivalent?: string;
  text?: string;
  modifiers?: string[];
  target?: ComputerHistoryTarget;
}

export interface ComputerHistorySelectionPayload {
  text?: string;
  truncated: boolean;
  selectedRange?: { location: number; length: number };
  target?: ComputerHistoryTarget;
}

export interface ComputerHistorySuppressedEvents {
  count: number;
  timestamp: number;
}

export interface VisionCandidateMetadata {
  candidateId: string;
  revision: number;
  observedAt: number;
  expiresAt: number;
  displayID: number;
  appBundleId: string;
  appName?: string;
  windowKey: string;
}

export interface VisionCandidatePayload extends VisionCandidateMetadata {
  mimeType: "image/jpeg";
  jpegBase64: string;
}

export interface EnvironmentFrameMetadata {
  frameId: string;
  revision: number;
  observedAt: number;
  expiresAt: number;
  displayID: number;
  pixelWidth: number;
  pixelHeight: number;
  byteLength: number;
  contentHash?: string;
}

export interface EnvironmentFramePayload extends EnvironmentFrameMetadata {
  mimeType: "image/jpeg";
  jpegBase64: string;
}

export interface SystemAudioWindow {
  captureSessionId: string;
  windowIndex: number;
  startedAt: number;
  endedAt: number;
  audioStatus: "complete" | "partial" | "unavailable";
  audioPcmBase64: string | null;
  audioSampleRate: 16000;
  audioCapturedSamples: number;
  audioGapReason: string | null;
  audioSilence?: boolean;
}

export type VisionCandidateClaimErrorCode =
  | "aborted"
  | "busy"
  | "helper_error"
  | "helper_restarted"
  | "helper_stopped"
  | "helper_unavailable"
  | "invalid_candidate_id"
  | "invalid_payload"
  | "invalid_timeout"
  | "stale_candidate"
  | "timeout"
  | "transport_error"
  | "unavailable";

/** Fixed-message error: candidate metadata and image bytes never reach logs. */
export class VisionCandidateClaimError extends Error {
  constructor(readonly code: VisionCandidateClaimErrorCode) {
    super(`Vision candidate claim failed (${code})`);
    this.name = "VisionCandidateClaimError";
  }
}

/** Fixed-message error: screen pixels never reach logs or generic events. */
export class EnvironmentFrameClaimError extends Error {
  constructor(readonly code: VisionCandidateClaimErrorCode) {
    super(`Environment frame claim failed (${code})`);
    this.name = "EnvironmentFrameClaimError";
  }
}

export type PerceptionManagerPhase =
  | "stopped"
  | "starting"
  | "ready"
  | "restarting"
  | "degraded";

export interface PerceptionManagerStatus {
  phase: PerceptionManagerPhase;
  detail: string;
  crashCount: number;
  helperPid: number | null;
  modules: PerceptionModuleSelection;
  capabilities: PerceptionCapabilities | null;
}

export type PerceptionCapabilityState = "disabled" | "running" | "unavailable" | "failed";
export type PerceptionCapabilityPermission =
  | "not_required"
  | "granted"
  | "denied"
  | "not_determined"
  | "restricted"
  | "unknown";

export interface PerceptionCapabilityReport {
  supported: boolean;
  requested: boolean;
  state: PerceptionCapabilityState;
  permission: PerceptionCapabilityPermission;
  reason?: string;
  computerHistoryInput?: ComputerHistoryInputReport;
}

export type ComputerHistoryEventTapState =
  | "disabled"
  | "permission_required"
  | "running"
  | "suspended"
  | "failed";

export interface ComputerHistoryInputChannelReport {
  state: PerceptionCapabilityState;
  receivedCount: number;
  lastEventAt?: number;
}

export interface ComputerHistoryInputReport {
  accessibilityPermission: PerceptionCapabilityPermission;
  listenEventPermission: PerceptionCapabilityPermission;
  eventTapState: ComputerHistoryEventTapState;
  mouse: ComputerHistoryInputChannelReport;
  keyboard: ComputerHistoryInputChannelReport;
  tapDisabledCount: number;
}

export interface PerceptionCapabilities {
  workspace: PerceptionCapabilityReport;
  screen: PerceptionCapabilityReport;
  accessibility: PerceptionCapabilityReport;
  camera: PerceptionCapabilityReport;
  computerHistory: PerceptionCapabilityReport;
  systemAudio?: PerceptionCapabilityReport;
}

export interface PerceptionHelperEnvelope {
  protocolVersion: typeof PERCEPTION_PROTOCOL_VERSION;
  sequence: number;
  sentAt: number;
  event: string;
  requestId?: string;
  body: unknown;
}

export interface PerceptionHelperProcess extends EventEmitter {
  readonly pid?: number;
  readonly stdin: Writable | null;
  readonly stdout: Readable | null;
  readonly stderr: Readable | null;
  kill(signal?: NodeJS.Signals): boolean;
}

export type PerceptionProcessSpawner = (
  command: string,
  args: readonly string[],
  options: SpawnOptionsWithoutStdio,
) => PerceptionHelperProcess;

export interface PerceptionManagerOptions {
  helperPath: string;
  helperArguments?: readonly string[];
  cwd?: string;
  spawnProcess?: PerceptionProcessSpawner;
  now?: () => number;
  schedule?: (callback: () => void, delayMs: number) => NodeJS.Timeout;
  cancelSchedule?: (timer: NodeJS.Timeout) => void;
  crashWindowMs?: number;
  maxCrashes?: number;
  restartDelaysMs?: readonly number[];
  handshakeTimeoutMs?: number;
  configureTimeoutMs?: number;
  pingIntervalMs?: number;
  pingTimeoutMs?: number;
  shutdownTimeoutMs?: number;
}

const DEFAULT_RESTART_DELAYS_MS = [500, 2_000] as const;
const DEFAULT_PROTOCOL_TIMEOUT_MS = 5_000;
const DEFAULT_PING_INTERVAL_MS = 15_000;
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 3_000;
const MAX_PROTOCOL_BUFFER_BYTES = 1024 * 1024;
const DEFAULT_VISION_CLAIM_TIMEOUT_MS = 3_000;
export const MAX_VISION_CANDIDATE_JPEG_BYTES = 200 * 1024;
export const DEFAULT_ENVIRONMENT_FRAME_INTERVAL_MS = 1_000;
export const MIN_ENVIRONMENT_FRAME_INTERVAL_MS = 250;
export const MAX_ENVIRONMENT_FRAME_INTERVAL_MS = 60_000;

interface PendingVisionCandidateClaim {
  requestId: string;
  candidateId: string;
  metadata: VisionCandidateMetadata;
  process: PerceptionHelperProcess;
  timer: NodeJS.Timeout;
  signal?: AbortSignal;
  abortHandler?: () => void;
  resolve: (payload: VisionCandidatePayload) => void;
  reject: (error: VisionCandidateClaimError) => void;
}

interface PendingEnvironmentFrameClaim {
  requestId: string;
  frameId: string;
  metadata: EnvironmentFrameMetadata;
  process: PerceptionHelperProcess;
  timer: NodeJS.Timeout;
  signal?: AbortSignal;
  abortHandler?: () => void;
  resolve: (payload: EnvironmentFramePayload) => void;
  reject: (error: EnvironmentFrameClaimError) => void;
}

/**
 * Owns only the native perception helper. It deliberately has no dependency on
 * the Python runtime or the voice lifecycle, so perception can degrade without
 * interrupting chat, memory, or audio.
 */
export class PerceptionManager extends EventEmitter {
  private readonly spawnProcess: PerceptionProcessSpawner;
  private readonly now: () => number;
  private readonly schedule: (callback: () => void, delayMs: number) => NodeJS.Timeout;
  private readonly cancelSchedule: (timer: NodeJS.Timeout) => void;
  private readonly crashWindowMs: number;
  private readonly maxCrashes: number;
  private readonly restartDelaysMs: readonly number[];
  private readonly handshakeTimeoutMs: number;
  private readonly configureTimeoutMs: number;
  private readonly pingIntervalMs: number;
  private readonly pingTimeoutMs: number;
  private readonly shutdownTimeoutMs: number;
  private readonly handledProcesses = new WeakSet<PerceptionHelperProcess>();
  private process: PerceptionHelperProcess | null = null;
  private shutdownPromise: Promise<void> | null = null;
  private decoder: JsonLineDecoder | null = null;
  private restartTimer: NodeJS.Timeout | null = null;
  private handshakeTimer: NodeJS.Timeout | null = null;
  private configureTimer: NodeJS.Timeout | null = null;
  private pingTimer: NodeJS.Timeout | null = null;
  private pingTimeoutTimer: NodeJS.Timeout | null = null;
  private pendingVisionClaim: PendingVisionCandidateClaim | null = null;
  private pendingEnvironmentFrameClaim: PendingEnvironmentFrameClaim | null = null;
  private pendingConfigureRequestId: string | null = null;
  private pendingPingRequestId: string | null = null;
  private latestVisionCandidate: VisionCandidateMetadata | null = null;
  private latestEnvironmentFrame: EnvironmentFrameMetadata | null = null;
  private environmentFrameIntervalMs = DEFAULT_ENVIRONMENT_FRAME_INTERVAL_MS;
  private requested = false;
  private crashTimes: number[] = [];
  private modules: PerceptionModuleSelection = { ...DISABLED_PERCEPTION_MODULES };
  private status: PerceptionManagerStatus = {
    phase: "stopped",
    detail: "感知服务未启动",
    crashCount: 0,
    helperPid: null,
    modules: { ...DISABLED_PERCEPTION_MODULES },
    capabilities: null,
  };

  constructor(private readonly options: PerceptionManagerOptions) {
    super();
    this.spawnProcess = options.spawnProcess ?? defaultSpawnProcess;
    this.now = options.now ?? Date.now;
    this.schedule = options.schedule ?? setTimeout;
    this.cancelSchedule = options.cancelSchedule ?? clearTimeout;
    this.crashWindowMs = options.crashWindowMs ?? 120_000;
    this.maxCrashes = options.maxCrashes ?? 3;
    this.restartDelaysMs = options.restartDelaysMs ?? DEFAULT_RESTART_DELAYS_MS;
    this.handshakeTimeoutMs = options.handshakeTimeoutMs ?? DEFAULT_PROTOCOL_TIMEOUT_MS;
    this.configureTimeoutMs = options.configureTimeoutMs ?? DEFAULT_PROTOCOL_TIMEOUT_MS;
    this.pingIntervalMs = options.pingIntervalMs ?? DEFAULT_PING_INTERVAL_MS;
    this.pingTimeoutMs = options.pingTimeoutMs ?? DEFAULT_PROTOCOL_TIMEOUT_MS;
    this.shutdownTimeoutMs = options.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS;
    if (this.maxCrashes < 1) throw new Error("maxCrashes must be at least 1");
    if (this.restartDelaysMs.length === 0) {
      throw new Error("restartDelaysMs must contain at least one delay");
    }
    if (
      this.handshakeTimeoutMs < 1
      || this.configureTimeoutMs < 1
      || this.pingIntervalMs < 1
      || this.pingTimeoutMs < 1
      || this.shutdownTimeoutMs < 1
    ) {
      throw new Error("perception protocol timeouts must be positive");
    }
  }

  currentStatus(): PerceptionManagerStatus {
    return cloneStatus(this.status);
  }

  start(modules: PerceptionModuleSelection = this.modules): void {
    const nextModules = cloneModules(modules);
    if (!sameModules(this.modules, nextModules)) {
      this.rejectPendingVisionClaim("stale_candidate");
      this.latestVisionCandidate = null;
      this.rejectPendingEnvironmentFrameClaim("stale_candidate");
      this.latestEnvironmentFrame = null;
    }
    this.modules = nextModules;
    this.updateStatus({ modules: this.modules });
    if (this.requested && (this.process || this.restartTimer)) {
      this.sendConfigureIfReady();
      return;
    }
    this.requested = true;
    if (this.status.phase === "degraded") this.crashTimes = [];
    if (this.shutdownPromise) {
      this.updateStatus({ phase: "starting", detail: "正在等待旧感知会话安全退出…" });
      return;
    }
    this.startProcess();
  }

  configure(modules: PerceptionModuleSelection): void {
    const nextModules = cloneModules(modules);
    if (!sameModules(this.modules, nextModules)) {
      this.rejectPendingVisionClaim("stale_candidate");
      this.latestVisionCandidate = null;
      this.rejectPendingEnvironmentFrameClaim("stale_candidate");
      this.latestEnvironmentFrame = null;
    }
    this.modules = nextModules;
    this.updateStatus({ modules: this.modules });
    this.sendConfigureIfReady();
  }

  setEnvironmentFrameInterval(
    intervalMs: number,
    forceConfigure = false,
  ): void {
    if (
      !Number.isSafeInteger(intervalMs)
      || intervalMs < MIN_ENVIRONMENT_FRAME_INTERVAL_MS
      || intervalMs > MAX_ENVIRONMENT_FRAME_INTERVAL_MS
    ) {
      throw new Error("environment frame interval is invalid");
    }
    if (intervalMs === this.environmentFrameIntervalMs && !forceConfigure) return;
    this.environmentFrameIntervalMs = intervalMs;
    this.sendConfigureIfReady();
  }

  claimVisionCandidate(
    candidateId: string,
    timeoutMs = DEFAULT_VISION_CLAIM_TIMEOUT_MS,
    signal?: AbortSignal,
  ): Promise<VisionCandidatePayload> {
    if (this.pendingVisionClaim) {
      return Promise.reject(new VisionCandidateClaimError("busy"));
    }
    if (!isCandidateId(candidateId)) {
      return Promise.reject(new VisionCandidateClaimError("invalid_candidate_id"));
    }
    if (!Number.isFinite(timeoutMs) || timeoutMs < 1 || timeoutMs > 30_000) {
      return Promise.reject(new VisionCandidateClaimError("invalid_timeout"));
    }
    if (signal?.aborted) {
      return Promise.reject(new VisionCandidateClaimError("aborted"));
    }
    const child = this.process;
    const metadata = this.latestVisionCandidate;
    if (
      !child
      || !child.stdin?.writable
      || !["ready", "degraded"].includes(this.status.phase)
      || this.modules.enhancedVision !== true
    ) {
      return Promise.reject(new VisionCandidateClaimError("helper_unavailable"));
    }
    if (
      !metadata
      || metadata.candidateId !== candidateId
      || isExpiredVisionCandidate(metadata, this.now())
    ) {
      return Promise.reject(new VisionCandidateClaimError("stale_candidate"));
    }

    return new Promise<VisionCandidatePayload>((resolve, reject) => {
      const requestId = randomUUID();
      const timer = this.schedule(() => {
        if (this.pendingVisionClaim?.requestId !== requestId) return;
        this.rejectPendingVisionClaim("timeout");
      }, timeoutMs);
      const abortHandler = signal
        ? () => {
          if (this.pendingVisionClaim?.requestId === requestId) {
            this.rejectPendingVisionClaim("aborted");
          }
        }
        : undefined;
      this.pendingVisionClaim = {
        requestId,
        candidateId,
        metadata,
        process: child,
        timer,
        ...(signal ? { signal } : {}),
        ...(abortHandler ? { abortHandler } : {}),
        resolve,
        reject,
      };
      if (signal && abortHandler) {
        signal.addEventListener("abort", abortHandler, { once: true });
        if (signal.aborted) {
          abortHandler();
          return;
        }
      }
      const sent = this.sendCommandTo(
        child,
        "claim_vision_candidate",
        { candidateId },
        requestId,
      );
      if (!sent && this.pendingVisionClaim?.requestId === requestId) {
        this.rejectPendingVisionClaim("transport_error");
      }
    });
  }

  claimEnvironmentFrame(
    frameId: string,
    timeoutMs = DEFAULT_VISION_CLAIM_TIMEOUT_MS,
    signal?: AbortSignal,
  ): Promise<EnvironmentFramePayload> {
    if (this.pendingEnvironmentFrameClaim) {
      return Promise.reject(new EnvironmentFrameClaimError("busy"));
    }
    if (!isCandidateId(frameId)) {
      return Promise.reject(new EnvironmentFrameClaimError("invalid_candidate_id"));
    }
    if (!Number.isFinite(timeoutMs) || timeoutMs < 1 || timeoutMs > 30_000) {
      return Promise.reject(new EnvironmentFrameClaimError("invalid_timeout"));
    }
    if (signal?.aborted) {
      return Promise.reject(new EnvironmentFrameClaimError("aborted"));
    }
    const child = this.process;
    const metadata = this.latestEnvironmentFrame;
    if (
      !child
      || !child.stdin?.writable
      || !["ready", "degraded"].includes(this.status.phase)
      || this.modules.screen !== true
    ) {
      return Promise.reject(new EnvironmentFrameClaimError("helper_unavailable"));
    }
    if (
      !metadata
      || metadata.frameId !== frameId
      || isExpiredEnvironmentFrame(metadata, this.now())
    ) {
      return Promise.reject(new EnvironmentFrameClaimError("stale_candidate"));
    }

    return new Promise<EnvironmentFramePayload>((resolve, reject) => {
      const requestId = randomUUID();
      const timer = this.schedule(() => {
        if (this.pendingEnvironmentFrameClaim?.requestId !== requestId) return;
        this.rejectPendingEnvironmentFrameClaim("timeout");
      }, timeoutMs);
      const abortHandler = signal
        ? () => {
          if (this.pendingEnvironmentFrameClaim?.requestId === requestId) {
            this.rejectPendingEnvironmentFrameClaim("aborted");
          }
        }
        : undefined;
      this.pendingEnvironmentFrameClaim = {
        requestId,
        frameId,
        metadata,
        process: child,
        timer,
        ...(signal ? { signal } : {}),
        ...(abortHandler ? { abortHandler } : {}),
        resolve,
        reject,
      };
      if (signal && abortHandler) {
        signal.addEventListener("abort", abortHandler, { once: true });
        if (signal.aborted) {
          abortHandler();
          return;
        }
      }
      const sent = this.sendCommandTo(
        child,
        "claim_environment_frame",
        { frameId },
        requestId,
      );
      if (!sent && this.pendingEnvironmentFrameClaim?.requestId === requestId) {
        this.rejectPendingEnvironmentFrameClaim("transport_error");
      }
    });
  }

  stop(detail = "感知服务已停止"): Promise<void> {
    this.requested = false;
    if (this.restartTimer) {
      this.cancelSchedule(this.restartTimer);
      this.restartTimer = null;
    }
    this.rejectPendingVisionClaim("helper_stopped");
    this.rejectPendingEnvironmentFrameClaim("helper_stopped");
    this.clearProtocolTimers();
    this.latestVisionCandidate = null;
    this.latestEnvironmentFrame = null;
    this.crashTimes = [];
    this.modules = { ...DISABLED_PERCEPTION_MODULES };
    const child = this.process;
    this.process = null;
    this.decoder = null;
    this.updateStatus({
      phase: "stopped",
      detail,
      helperPid: null,
      crashCount: 0,
      modules: this.modules,
      capabilities: null,
    });
    if (!child) return this.shutdownPromise ?? Promise.resolve();

    this.handledProcesses.add(child);
    const shutdown = this.gracefullyStopProcess(child);
    this.shutdownPromise = shutdown;
    void shutdown.finally(() => {
      if (this.shutdownPromise !== shutdown) return;
      this.shutdownPromise = null;
      if (this.requested && !this.process) this.startProcess();
    });
    return shutdown;
  }

  private startProcess(): void {
    if (!this.requested || this.process || this.shutdownPromise) return;
    this.rejectPendingVisionClaim("helper_restarted");
    this.rejectPendingEnvironmentFrameClaim("helper_restarted");
    this.latestVisionCandidate = null;
    this.latestEnvironmentFrame = null;
    this.updateStatus({
      phase: "starting",
      detail: "正在启动本机感知服务…",
      helperPid: null,
    });
    let child: PerceptionHelperProcess;
    try {
      child = this.spawnProcess(
        this.options.helperPath,
        this.options.helperArguments ?? [],
        {
          cwd: this.options.cwd,
          detached: false,
          windowsHide: true,
          stdio: ["pipe", "pipe", "pipe"],
          env: minimalHelperEnvironment(),
        },
      );
    } catch (error) {
      this.handleSpawnFailure(error instanceof Error ? error.message : String(error));
      return;
    }
    this.process = child;
    this.decoder = new JsonLineDecoder(MAX_PROTOCOL_BUFFER_BYTES);
    this.updateStatus({ helperPid: child.pid ?? null });
    this.handshakeTimer = this.schedule(() => {
      this.handshakeTimer = null;
      this.terminateUnresponsiveProcess(child, "感知服务启动握手超时");
    }, this.handshakeTimeoutMs);
    child.stdout?.on("data", (chunk: Buffer | string) => this.handleStdout(child, chunk));
    child.stderr?.on("data", (chunk: Buffer | string) => {
      this.emit("log", { role: "perception", line: chunk.toString().trimEnd() });
    });
    child.once("error", (error: Error) => {
      this.handleUnexpectedTermination(child, error.message);
    });
    child.stdin?.on("error", (error: Error) => {
      this.rejectPendingVisionClaim("transport_error", child);
      this.rejectPendingEnvironmentFrameClaim("transport_error", child);
      this.terminateUnresponsiveProcess(child, `感知服务输入管道异常: ${error.message}`);
    });
    child.once("exit", (code: number | null, signal: NodeJS.Signals | null) => {
      const reason = code === null ? signal ?? "unknown" : `code ${code}`;
      this.handleUnexpectedTermination(child, reason);
    });
  }

  private gracefullyStopProcess(child: PerceptionHelperProcess): Promise<void> {
    return new Promise((resolve) => {
      let settled = false;
      let timer: NodeJS.Timeout | null = null;
      const finish = () => {
        if (settled) return;
        settled = true;
        if (timer) this.cancelSchedule(timer);
        child.off("exit", finish);
        resolve();
      };
      child.once("exit", finish);
      const requestId = this.sendCommandTo(child, "shutdown");
      if (!requestId) {
        child.kill("SIGTERM");
        finish();
        return;
      }
      // EOF is a second graceful-shutdown signal. The helper drains the JSON
      // command first, awaits every ScreenCaptureKit stopCapture(), then exits.
      child.stdin?.end();
      timer = this.schedule(() => {
        // A wedged native helper must not block app quit forever. It has had a
        // full shutdown window to release ScreenCaptureKit before this fallback.
        child.kill("SIGKILL");
        finish();
      }, this.shutdownTimeoutMs);
      timer.unref?.();
    });
  }

  private handleStdout(child: PerceptionHelperProcess, chunk: Buffer | string): void {
    if (this.process !== child || !this.decoder) return;
    let lines: string[];
    try {
      lines = this.decoder.push(chunk);
    } catch (error) {
      this.emit("protocol-error", error instanceof Error ? error : new Error(String(error)));
      this.terminateUnresponsiveProcess(child, "感知服务协议输出超过限制");
      return;
    }
    for (const line of lines) {
      try {
        const envelope = parsePerceptionEnvelope(line);
        this.handleEnvelope(envelope);
      } catch {
        this.emit("protocol-error", new Error("Invalid perception helper protocol event"));
      }
    }
  }

  private handleEnvelope(envelope: PerceptionHelperEnvelope): void {
    // Raw media is private transport, never a generic event or log. Legacy
    // chunk/gap events are intentionally discarded during the native rollout.
    if (envelope.event === "system_audio_chunk" || envelope.event === "system_audio_gap") return;
    if (envelope.event === "system_audio_window") {
      this.handleSystemAudioWindow(envelope);
      return;
    }
    if (envelope.event === "interaction_events_suppressed") {
      try {
        this.emit(
          "interaction-events-suppressed",
          parseComputerHistorySuppressedEvents(envelope.body),
        );
      } catch {
        this.emit("protocol-error", new Error("Invalid suppressed interaction count"));
      }
      return;
    }
    if (envelope.event === "interaction_event") {
      try {
        this.emit(
          "interaction-event",
          parseComputerHistoryInteractionEvent(envelope.body),
        );
      } catch {
        this.emit("protocol-error", new Error("Invalid computer history event"));
        const timestamp = safeComputerHistoryTimestamp(envelope.body);
        if (timestamp !== null) {
          this.emit("interaction-events-suppressed", { count: 1, timestamp });
        }
      }
      return;
    }
    if (envelope.event === "vision_candidate_available") {
      this.handleVisionCandidateAvailable(envelope);
      return;
    }
    if (envelope.event === "environment_frame_available") {
      this.handleEnvironmentFrameAvailable(envelope);
      return;
    }
    if (
      envelope.event === "environment_frame"
      || envelope.event === "environment_frame_unavailable"
    ) {
      this.handleEnvironmentFrameResponse(envelope);
      return;
    }
    if (
      envelope.event === "vision_candidate"
      || envelope.event === "vision_candidate_unavailable"
    ) {
      this.handleVisionCandidateResponse(envelope);
      return;
    }
    if (
      envelope.event === "error"
      && envelope.requestId
      && envelope.requestId === this.pendingVisionClaim?.requestId
    ) {
      this.rejectPendingVisionClaim("helper_error");
      return;
    }
    if (
      envelope.event === "error"
      && envelope.requestId
      && envelope.requestId === this.pendingEnvironmentFrameClaim?.requestId
    ) {
      this.rejectPendingEnvironmentFrameClaim("helper_error");
      return;
    }
    this.emit("event", envelope);
    if (envelope.event === "hello") {
      this.clearHandshakeTimer();
      const capabilities = capabilitiesFromBody(envelope.body);
      this.updateStatus({
        phase: "ready",
        detail: "本机感知服务已就绪",
        crashCount: this.crashesInWindow(),
        capabilities: capabilities ?? this.status.capabilities,
      });
      this.sendConfigureIfReady();
      return;
    }
    if (envelope.event === "status") {
      const matchesPendingConfigure = Boolean(
        envelope.requestId
        && envelope.requestId === this.pendingConfigureRequestId
      );
      if (matchesPendingConfigure) {
        this.clearConfigureTimer();
      }
      const capabilities = capabilitiesFromBody(envelope.body);
      if (capabilities) {
        this.updateStatus({
          phase: requestedCapabilityUnavailable(capabilities, this.modules)
            ? "degraded"
            : "ready",
          detail: requestedCapabilityUnavailable(capabilities, this.modules)
            ? "部分感知能力当前不可用"
            : "本机感知服务已就绪",
          capabilities,
        });
      }
      if (matchesPendingConfigure) this.startLiveness();
      return;
    }
    if (
      envelope.event === "ack"
      && envelope.requestId
      && envelope.requestId === this.pendingPingRequestId
    ) {
      this.clearPingTimeoutTimer();
      this.scheduleNextPing();
      return;
    }
    if (envelope.event === "observation") this.emit("observation", envelope.body);
    if (envelope.event === "error") {
      if (
        envelope.requestId
        && envelope.requestId === this.pendingConfigureRequestId
      ) {
        this.clearConfigureTimer();
        this.updateStatus({
          phase: "degraded",
          detail: "感知服务拒绝了当前能力配置",
        });
      }
      this.emit("helper-error", envelope.body);
    }
  }

  private handleSystemAudioWindow(envelope: PerceptionHelperEnvelope): void {
    if (!this.requested || (!this.modules.screen && this.modules.systemAudio !== true)
      || !this.process || !["ready", "degraded"].includes(this.status.phase)) return;
    try {
      if (envelope.requestId !== undefined) throw new Error("Unexpected audio response");
      const window = parseSystemAudioWindow(envelope.body);
      // A disabled audio input only supplies a clock for the screen history.
      if (this.modules.systemAudio !== true
        && (window.audioStatus !== "unavailable" || window.audioGapReason !== "system_audio_disabled")) return;
      this.emit("system-audio-window", window);
    } catch {
      this.emit("protocol-error", new Error("Invalid system audio payload"));
    }
  }

  private handleVisionCandidateAvailable(envelope: PerceptionHelperEnvelope): void {
    if (envelope.requestId !== undefined) {
      this.emit("protocol-error", new Error("Unexpected vision candidate metadata"));
      return;
    }
    if (
      this.modules.enhancedVision !== true
      || !this.process
      || !["ready", "degraded"].includes(this.status.phase)
    ) return;
    let metadata: VisionCandidateMetadata;
    try {
      metadata = parseVisionCandidateMetadata(envelope.body);
    } catch {
      this.emit("protocol-error", new Error("Invalid vision candidate metadata"));
      return;
    }
    if (isExpiredVisionCandidate(metadata, this.now())) return;
    const previous = this.latestVisionCandidate;
    if (previous) {
      if (!isNewerVisionCandidate(metadata, previous)) return;
      if (
        this.pendingVisionClaim
        && this.pendingVisionClaim.candidateId !== metadata.candidateId
      ) {
        this.rejectPendingVisionClaim("stale_candidate");
      }
    }
    this.latestVisionCandidate = metadata;
    this.emit("vision-candidate-available", cloneVisionCandidateMetadata(metadata));
  }

  private handleVisionCandidateResponse(envelope: PerceptionHelperEnvelope): void {
    const pending = this.pendingVisionClaim;
    if (!pending || envelope.requestId !== pending.requestId) return;
    if (envelope.event === "vision_candidate_unavailable") {
      if (!isUnavailableVisionCandidateBody(envelope.body)) {
        this.emit("protocol-error", new Error("Invalid vision candidate unavailable response"));
        this.rejectPendingVisionClaim("invalid_payload");
        return;
      }
      if (this.latestVisionCandidate?.candidateId === pending.candidateId) {
        this.latestVisionCandidate = null;
      }
      this.rejectPendingVisionClaim("unavailable");
      return;
    }

    let payload: VisionCandidatePayload;
    try {
      payload = parseVisionCandidatePayload(envelope.body);
    } catch {
      this.emit("protocol-error", new Error("Invalid vision candidate payload"));
      this.rejectPendingVisionClaim("invalid_payload");
      return;
    }
    if (payload.candidateId !== pending.candidateId) {
      this.rejectPendingVisionClaim("stale_candidate");
      return;
    }
    if (
      !sameVisionCandidateMetadata(payload, pending.metadata)
      || isExpiredVisionCandidate(payload, this.now())
      || pending.process !== this.process
    ) {
      this.rejectPendingVisionClaim("stale_candidate");
      return;
    }
    this.latestVisionCandidate = null;
    this.resolvePendingVisionClaim(payload);
  }

  private handleEnvironmentFrameAvailable(envelope: PerceptionHelperEnvelope): void {
    if (envelope.requestId !== undefined) {
      this.emit("protocol-error", new Error("Unexpected environment frame metadata"));
      return;
    }
    if (
      this.modules.screen !== true
      || !this.process
      || !["ready", "degraded"].includes(this.status.phase)
    ) return;
    let metadata: EnvironmentFrameMetadata;
    try {
      metadata = parseEnvironmentFrameMetadata(envelope.body);
    } catch {
      this.emit("protocol-error", new Error("Invalid environment frame metadata"));
      return;
    }
    if (isExpiredEnvironmentFrame(metadata, this.now())) return;
    const previous = this.latestEnvironmentFrame;
    if (previous) {
      if (!isNewerEnvironmentFrame(metadata, previous)) return;
      if (
        this.pendingEnvironmentFrameClaim
        && this.pendingEnvironmentFrameClaim.frameId !== metadata.frameId
      ) {
        this.rejectPendingEnvironmentFrameClaim("stale_candidate");
      }
    }
    this.latestEnvironmentFrame = metadata;
    this.emit("environment-frame-available", { ...metadata });
  }

  private handleEnvironmentFrameResponse(envelope: PerceptionHelperEnvelope): void {
    const pending = this.pendingEnvironmentFrameClaim;
    if (!pending || envelope.requestId !== pending.requestId) return;
    if (envelope.event === "environment_frame_unavailable") {
      if (!isUnavailableVisionCandidateBody(envelope.body)) {
        this.emit("protocol-error", new Error("Invalid environment frame unavailable response"));
        this.rejectPendingEnvironmentFrameClaim("invalid_payload");
        return;
      }
      if (this.latestEnvironmentFrame?.frameId === pending.frameId) {
        this.latestEnvironmentFrame = null;
      }
      this.rejectPendingEnvironmentFrameClaim("unavailable");
      return;
    }
    let payload: EnvironmentFramePayload;
    try {
      payload = parseEnvironmentFramePayload(envelope.body);
    } catch {
      this.emit("protocol-error", new Error("Invalid environment frame payload"));
      this.rejectPendingEnvironmentFrameClaim("invalid_payload");
      return;
    }
    if (payload.frameId !== pending.frameId) {
      this.rejectPendingEnvironmentFrameClaim("stale_candidate");
      return;
    }
    if (
      !sameEnvironmentFrameMetadata(payload, pending.metadata)
      || isExpiredEnvironmentFrame(payload, this.now())
      || pending.process !== this.process
    ) {
      this.rejectPendingEnvironmentFrameClaim("stale_candidate");
      return;
    }
    this.latestEnvironmentFrame = null;
    this.resolvePendingEnvironmentFrameClaim(payload);
  }

  private sendConfigureIfReady(): void {
    if (!this.process || this.status.phase === "starting") return;
    this.clearLivenessTimers();
    this.clearConfigureTimer();
    const child = this.process;
    const requestId = this.sendCommandTo(child, "configure", {
      modules: this.modules,
      environmentFrameIntervalMs: this.environmentFrameIntervalMs,
    });
    if (!requestId) {
      this.terminateUnresponsiveProcess(child, "感知服务输入管道不可用");
      return;
    }
    this.pendingConfigureRequestId = requestId;
    this.configureTimer = this.schedule(() => {
      this.configureTimer = null;
      this.pendingConfigureRequestId = null;
      this.terminateUnresponsiveProcess(child, "感知服务配置响应超时");
    }, this.configureTimeoutMs);
  }

  private startLiveness(): void {
    this.clearLivenessTimers();
    this.scheduleNextPing();
  }

  private scheduleNextPing(): void {
    if (!this.requested || !this.process || this.configureTimer) return;
    if (this.pingTimer || this.pingTimeoutTimer || this.pendingPingRequestId) return;
    const child = this.process;
    this.pingTimer = this.schedule(() => {
      this.pingTimer = null;
      if (!this.requested || this.process !== child) return;
      const requestId = this.sendCommandTo(child, "ping");
      if (!requestId) {
        this.terminateUnresponsiveProcess(child, "感知服务心跳管道不可用");
        return;
      }
      this.pendingPingRequestId = requestId;
      this.pingTimeoutTimer = this.schedule(() => {
        this.pingTimeoutTimer = null;
        this.pendingPingRequestId = null;
        this.terminateUnresponsiveProcess(child, "感知服务心跳响应超时");
      }, this.pingTimeoutMs);
    }, this.pingIntervalMs);
  }

  cutEnvironmentWindow(boundaryAt: number): void {
    if (this.process && Number.isFinite(boundaryAt)) {
      this.sendCommandTo(this.process, "cut_environment_window", { boundaryAt });
    }
  }

  private sendCommandTo(
    child: PerceptionHelperProcess,
    command: string,
    fields: {
      modules?: PerceptionModuleSelection;
      environmentFrameIntervalMs?: number;
      candidateId?: string;
      frameId?: string;
      boundaryAt?: number;
    } = {},
    requestId = randomUUID(),
  ): string | null {
    if (!child.stdin?.writable) return null;
    const payload = {
      protocolVersion: PERCEPTION_PROTOCOL_VERSION,
      requestId,
      command,
      ...fields,
    };
    try {
      child.stdin.write(`${JSON.stringify(payload)}\n`);
      return requestId;
    } catch (error) {
      this.rejectPendingVisionClaim("transport_error", child);
      this.rejectPendingEnvironmentFrameClaim("transport_error", child);
      if (!this.handledProcesses.has(child)) {
        this.terminateUnresponsiveProcess(
          child,
          error instanceof Error ? error.message : String(error),
        );
      }
      return null;
    }
  }

  private handleSpawnFailure(reason: string): void {
    this.recordCrashAndMaybeRestart(reason);
  }

  private handleUnexpectedTermination(
    child: PerceptionHelperProcess,
    reason: string,
  ): void {
    if (this.handledProcesses.has(child)) return;
    this.handledProcesses.add(child);
    if (this.process === child) {
      this.rejectPendingVisionClaim("helper_restarted", child);
      this.rejectPendingEnvironmentFrameClaim("helper_restarted", child);
      this.latestVisionCandidate = null;
      this.latestEnvironmentFrame = null;
      this.clearProtocolTimers();
      this.process = null;
      this.decoder = null;
    }
    if (!this.requested) return;
    this.recordCrashAndMaybeRestart(reason);
  }

  private terminateUnresponsiveProcess(
    child: PerceptionHelperProcess,
    reason: string,
  ): void {
    if (this.handledProcesses.has(child) || this.process !== child) return;
    this.handledProcesses.add(child);
    this.rejectPendingVisionClaim("helper_restarted", child);
    this.rejectPendingEnvironmentFrameClaim("helper_restarted", child);
    this.latestVisionCandidate = null;
    this.latestEnvironmentFrame = null;
    this.clearProtocolTimers();
    this.process = null;
    this.decoder = null;
    child.kill("SIGTERM");
    if (this.requested) this.recordCrashAndMaybeRestart(reason);
  }

  private clearProtocolTimers(): void {
    this.clearHandshakeTimer();
    this.clearConfigureTimer();
    this.clearLivenessTimers();
  }

  private clearHandshakeTimer(): void {
    if (!this.handshakeTimer) return;
    this.cancelSchedule(this.handshakeTimer);
    this.handshakeTimer = null;
  }

  private clearConfigureTimer(): void {
    if (this.configureTimer) this.cancelSchedule(this.configureTimer);
    this.configureTimer = null;
    this.pendingConfigureRequestId = null;
  }

  private clearLivenessTimers(): void {
    if (this.pingTimer) this.cancelSchedule(this.pingTimer);
    this.pingTimer = null;
    this.clearPingTimeoutTimer();
  }

  private clearPingTimeoutTimer(): void {
    if (this.pingTimeoutTimer) this.cancelSchedule(this.pingTimeoutTimer);
    this.pingTimeoutTimer = null;
    this.pendingPingRequestId = null;
  }

  private resolvePendingVisionClaim(payload: VisionCandidatePayload): void {
    const pending = this.pendingVisionClaim;
    if (!pending) return;
    this.pendingVisionClaim = null;
    this.cancelSchedule(pending.timer);
    if (pending.signal && pending.abortHandler) {
      pending.signal.removeEventListener("abort", pending.abortHandler);
    }
    pending.resolve(cloneVisionCandidatePayload(payload));
  }

  private rejectPendingVisionClaim(
    code: VisionCandidateClaimErrorCode,
    process?: PerceptionHelperProcess,
  ): void {
    const pending = this.pendingVisionClaim;
    if (!pending || (process && pending.process !== process)) return;
    this.pendingVisionClaim = null;
    this.cancelSchedule(pending.timer);
    if (pending.signal && pending.abortHandler) {
      pending.signal.removeEventListener("abort", pending.abortHandler);
    }
    if (this.latestVisionCandidate?.candidateId === pending.candidateId) {
      this.latestVisionCandidate = null;
    }
    pending.reject(new VisionCandidateClaimError(code));
  }

  private resolvePendingEnvironmentFrameClaim(payload: EnvironmentFramePayload): void {
    const pending = this.pendingEnvironmentFrameClaim;
    if (!pending) return;
    this.pendingEnvironmentFrameClaim = null;
    this.cancelSchedule(pending.timer);
    if (pending.signal && pending.abortHandler) {
      pending.signal.removeEventListener("abort", pending.abortHandler);
    }
    pending.resolve({ ...payload });
  }

  private rejectPendingEnvironmentFrameClaim(
    code: VisionCandidateClaimErrorCode,
    process?: PerceptionHelperProcess,
  ): void {
    const pending = this.pendingEnvironmentFrameClaim;
    if (!pending || (process && pending.process !== process)) return;
    this.pendingEnvironmentFrameClaim = null;
    this.cancelSchedule(pending.timer);
    if (pending.signal && pending.abortHandler) {
      pending.signal.removeEventListener("abort", pending.abortHandler);
    }
    if (this.latestEnvironmentFrame?.frameId === pending.frameId) {
      this.latestEnvironmentFrame = null;
    }
    pending.reject(new EnvironmentFrameClaimError(code));
  }

  private recordCrashAndMaybeRestart(reason: string): void {
    if (!this.requested) return;
    const now = this.now();
    this.crashTimes = this.crashTimes.filter((timestamp) => now - timestamp < this.crashWindowMs);
    this.crashTimes.push(now);
    const crashCount = this.crashTimes.length;
    if (crashCount >= this.maxCrashes) {
      this.requested = false;
      this.updateStatus({
        phase: "degraded",
        detail: `感知服务连续异常退出 (${reason})，已停用感知；聊天、记忆和语音不受影响`,
        crashCount,
        helperPid: null,
      });
      return;
    }
    const delayIndex = Math.min(crashCount - 1, this.restartDelaysMs.length - 1);
    const delayMs = this.restartDelaysMs[delayIndex]!;
    this.updateStatus({
      phase: "restarting",
      detail: `感知服务异常退出，${delayMs}ms 后恢复 (${crashCount}/${this.maxCrashes})`,
      crashCount,
      helperPid: null,
    });
    this.restartTimer = this.schedule(() => {
      this.restartTimer = null;
      this.startProcess();
    }, delayMs);
  }

  private crashesInWindow(): number {
    const now = this.now();
    this.crashTimes = this.crashTimes.filter((timestamp) => now - timestamp < this.crashWindowMs);
    return this.crashTimes.length;
  }

  private updateStatus(patch: Partial<PerceptionManagerStatus>): void {
    this.status = {
      ...this.status,
      ...patch,
      modules: cloneModules(patch.modules ?? this.status.modules),
      capabilities: patch.capabilities === undefined
        ? this.status.capabilities
        : patch.capabilities,
    };
    this.emit("status", this.currentStatus());
  }
}

export class JsonLineDecoder {
  private buffer = "";
  private readonly decoder = new StringDecoder("utf8");

  constructor(private readonly maxBufferBytes = MAX_PROTOCOL_BUFFER_BYTES) {}

  push(chunk: Buffer | string): string[] {
    this.buffer += typeof chunk === "string" ? chunk : this.decoder.write(chunk);
    if (Buffer.byteLength(this.buffer, "utf8") > this.maxBufferBytes) {
      this.buffer = "";
      throw new Error("Perception helper protocol buffer exceeded its size limit");
    }
    const parts = this.buffer.split("\n");
    this.buffer = parts.pop() ?? "";
    return parts
      .map((line) => line.endsWith("\r") ? line.slice(0, -1) : line)
      .filter((line) => line.trim().length > 0);
  }
}

export function parsePerceptionEnvelope(line: string): PerceptionHelperEnvelope {
  const value: unknown = JSON.parse(line);
  if (!isRecord(value)) throw new Error("Perception helper event must be an object");
  if (value.protocolVersion !== PERCEPTION_PROTOCOL_VERSION) {
    throw new Error(`Unsupported perception protocol version: ${String(value.protocolVersion)}`);
  }
  if (!Number.isSafeInteger(value.sequence) || Number(value.sequence) < 1) {
    throw new Error("Perception helper event sequence must be a positive integer");
  }
  if (typeof value.sentAt !== "number" || !Number.isFinite(value.sentAt)) {
    throw new Error("Perception helper event sentAt must be finite");
  }
  if (typeof value.event !== "string" || value.event.length === 0) {
    throw new Error("Perception helper event name is required");
  }
  if (value.requestId !== undefined && typeof value.requestId !== "string") {
    throw new Error("Perception helper requestId must be a string");
  }
  return {
    protocolVersion: PERCEPTION_PROTOCOL_VERSION,
    sequence: Number(value.sequence),
    sentAt: value.sentAt,
    event: value.event,
    ...(typeof value.requestId === "string" ? { requestId: value.requestId } : {}),
    body: value.body,
  };
}

const UUID_PATTERN = /^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$/u;
const HISTORY_UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu;
const HISTORY_KINDS = new Set<ComputerHistoryInteractionKind>([
  "session.started",
  "window.changed",
  "mouse.click",
  "mouse.drag",
  "mouse.context_menu",
  "selection.changed",
  "keyboard.shortcut",
  "keyboard.text_input",
  "keyboard.submit",
]);
const MAX_HISTORY_EVENT_BYTES = 192 * 1024;
const MAX_HISTORY_TIMESTAMP_SECONDS = 253_402_300_799.999;
const APP_BUNDLE_ID_PATTERN = /^[a-z0-9][a-z0-9.-]{0,254}$/u;
const CANONICAL_BASE64_PATTERN = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
const METADATA_REQUIRED_KEYS = [
  "candidateId",
  "revision",
  "observedAt",
  "expiresAt",
  "displayID",
  "appBundleId",
  "windowKey",
] as const;
const VISION_CANDIDATE_TTL_SECONDS = 5;
const ENVIRONMENT_FRAME_TTL_SECONDS = 3;
const ENVIRONMENT_FRAME_MAX_LONG_EDGE = 960;
const TIMESTAMP_EPSILON_SECONDS = 0.001;
const MAX_CLOCK_SKEW_MS = 5_000;

export function parseComputerHistoryInteractionEvent(
  value: unknown,
): ComputerHistoryInteractionEvent {
  if (!isRecord(value)) throw new Error("Invalid computer history event");
  const allowedKeys = new Set([
    "id",
    "timestamp",
    "kind",
    "app",
    "window",
    "ax",
    "mouse",
    "keyboard",
    "selection",
  ]);
  if (Object.keys(value).some((key) => !allowedKeys.has(key))) {
    throw new Error("Invalid computer history event");
  }
  if (
    typeof value.id !== "string"
    || !HISTORY_UUID_PATTERN.test(value.id)
    || typeof value.timestamp !== "number"
    || !Number.isFinite(value.timestamp)
    || value.timestamp <= 0
    || value.timestamp > MAX_HISTORY_TIMESTAMP_SECONDS
    || typeof value.kind !== "string"
    || !HISTORY_KINDS.has(value.kind as ComputerHistoryInteractionKind)
  ) {
    throw new Error("Invalid computer history event");
  }
  const app = historyContext(value.app, {
    bundleIdentifier: 255,
    name: 200,
  });
  const window = historyContext(value.window, {
    title: 512,
    url: 2_048,
  });
  const event: ComputerHistoryInteractionEvent = {
    id: value.id,
    timestamp: value.timestamp,
    kind: value.kind as ComputerHistoryInteractionKind,
    app,
    window,
  };
  if (value.ax !== undefined) event.ax = historyAx(value.ax);
  if (value.mouse !== undefined) event.mouse = historyMouse(value.mouse);
  if (value.keyboard !== undefined) event.keyboard = historyKeyboard(value.keyboard);
  if (value.selection !== undefined) event.selection = historySelection(value.selection);
  validateHistoryPayloadForKind(event);
  if (Buffer.byteLength(JSON.stringify(event), "utf8") > MAX_HISTORY_EVENT_BYTES) {
    throw new Error("Invalid computer history event");
  }
  return event;
}

export function parseComputerHistorySuppressedEvents(
  value: unknown,
): ComputerHistorySuppressedEvents {
  requireExactKeys(value, ["count", "timestamp"]);
  if (
    !Number.isSafeInteger(value.count)
    || Number(value.count) < 1
    || Number(value.count) > 1_000_000
  ) throw new Error("Invalid suppressed interaction count");
  const timestamp = safeComputerHistoryTimestamp(value);
  if (timestamp === null) throw new Error("Invalid suppressed interaction count");
  return { count: Number(value.count), timestamp };
}

function safeComputerHistoryTimestamp(value: unknown): number | null {
  if (!isRecord(value)) return null;
  return typeof value.timestamp === "number"
    && Number.isFinite(value.timestamp)
    && value.timestamp > 0
    && value.timestamp <= MAX_HISTORY_TIMESTAMP_SECONDS
    ? value.timestamp
    : null;
}

function historyContext(
  value: unknown,
  fields: Readonly<Record<string, number>>,
): Record<string, string> {
  if (!isRecord(value)) throw new Error("Invalid computer history event");
  if (Object.keys(value).some((key) => !(key in fields))) {
    throw new Error("Invalid computer history event");
  }
  const result: Record<string, string> = {};
  for (const [key, maximum] of Object.entries(fields)) {
    const candidate = value[key];
    if (candidate === undefined) continue;
    if (
      typeof candidate !== "string"
      || candidate !== candidate.trim()
      || !candidate
      || [...candidate].length > maximum
      || /[\u0000-\u001f\u007f\u2028\u2029]/u.test(candidate)
    ) {
      throw new Error("Invalid computer history event");
    }
    result[key] = candidate;
  }
  return result;
}

function historyAx(value: unknown): ComputerHistoryAxPayload {
  requireExactKeys(value, ["mode", "text", "truncated"]);
  if (
    value.mode !== "fullTree"
    && value.mode !== "diffFromPrevious"
  ) throw new Error("Invalid computer history event");
  if (
    typeof value.text !== "string"
    || Buffer.byteLength(value.text, "utf8") > 72 * 1024
    || typeof value.truncated !== "boolean"
  ) throw new Error("Invalid computer history event");
  return { mode: value.mode, text: value.text, truncated: value.truncated };
}

function historyMouse(value: unknown): ComputerHistoryMousePayload {
  requireAllowedKeys(value, [
    "button",
    "clickCount",
    "location",
    "origin",
    "destination",
    "target",
  ]);
  if (value.button !== "left" && value.button !== "right" && value.button !== "other") {
    throw new Error("Invalid computer history event");
  }
  const result: ComputerHistoryMousePayload = { button: value.button };
  if (value.clickCount !== undefined) {
    if (!Number.isSafeInteger(value.clickCount) || Number(value.clickCount) < 0) {
      throw new Error("Invalid computer history event");
    }
    result.clickCount = Number(value.clickCount);
  }
  if (value.location !== undefined) result.location = historyCoords(value.location);
  if (value.origin !== undefined) result.origin = historyCoords(value.origin);
  if (value.destination !== undefined) {
    result.destination = historyCoords(value.destination);
  }
  if (value.target !== undefined) result.target = historyTarget(value.target);
  requirePayloadSize(result);
  return result;
}

function historyKeyboard(value: unknown): ComputerHistoryKeyboardPayload {
  requireAllowedKeys(value, ["keyEquivalent", "text", "modifiers", "target"]);
  const result: ComputerHistoryKeyboardPayload = {};
  if (value.keyEquivalent !== undefined) {
    result.keyEquivalent = requiredHistoryText(value.keyEquivalent, 128, true);
  }
  if (value.text !== undefined) {
    result.text = requiredHistoryText(value.text, 8_192, true);
  }
  if (value.modifiers !== undefined) {
    if (
      !Array.isArray(value.modifiers)
      || value.modifiers.length > 16
      || value.modifiers.some((item) => !validHistoryText(item, 32, false))
    ) throw new Error("Invalid computer history event");
    result.modifiers = [...value.modifiers] as string[];
  }
  if (value.target !== undefined) result.target = historyTarget(value.target);
  requirePayloadSize(result);
  return result;
}

function historySelection(value: unknown): ComputerHistorySelectionPayload {
  requireAllowedKeys(value, ["text", "truncated", "selectedRange", "target"]);
  if (typeof value.truncated !== "boolean") {
    throw new Error("Invalid computer history event");
  }
  const result: ComputerHistorySelectionPayload = { truncated: value.truncated };
  if (value.text !== undefined) result.text = requiredHistoryText(value.text, 8_192, true);
  if (value.selectedRange !== undefined) {
    requireExactKeys(value.selectedRange, ["location", "length"]);
    if (
      !Number.isSafeInteger(value.selectedRange.location)
      || Number(value.selectedRange.location) < 0
      || !Number.isSafeInteger(value.selectedRange.length)
      || Number(value.selectedRange.length) < 0
    ) throw new Error("Invalid computer history event");
    result.selectedRange = {
      location: Number(value.selectedRange.location),
      length: Number(value.selectedRange.length),
    };
  }
  if (value.target !== undefined) result.target = historyTarget(value.target);
  requirePayloadSize(result);
  return result;
}

function historyTarget(value: unknown): ComputerHistoryTarget {
  requireAllowedKeys(value, ["role", "title", "description"]);
  const result: ComputerHistoryTarget = {
    role: requiredHistoryText(value.role, 128, false),
  };
  if (value.title !== undefined) result.title = requiredHistoryText(value.title, 512, true);
  if (value.description !== undefined) {
    result.description = requiredHistoryText(value.description, 1_024, true);
  }
  return result;
}

function historyCoords(value: unknown): { x: number; y: number } {
  requireExactKeys(value, ["x", "y"]);
  if (
    typeof value.x !== "number"
    || !Number.isFinite(value.x)
    || typeof value.y !== "number"
    || !Number.isFinite(value.y)
  ) throw new Error("Invalid computer history event");
  return { x: value.x, y: value.y };
}

function validateHistoryPayloadForKind(event: ComputerHistoryInteractionEvent): void {
  const has = {
    ax: event.ax !== undefined,
    mouse: event.mouse !== undefined,
    keyboard: event.keyboard !== undefined,
    selection: event.selection !== undefined,
  };
  if (event.kind === "session.started" || event.kind === "window.changed") {
    if (has.mouse || has.keyboard || has.selection) throw new Error("Invalid computer history event");
  } else if (event.kind.startsWith("mouse.")) {
    if (!has.mouse || has.keyboard || has.selection) throw new Error("Invalid computer history event");
    if (event.kind === "mouse.drag") {
      if (!event.mouse?.origin || !event.mouse.destination || event.mouse.location) {
        throw new Error("Invalid computer history event");
      }
    } else if (event.mouse?.origin || event.mouse?.destination) {
      throw new Error("Invalid computer history event");
    }
  } else if (event.kind === "keyboard.shortcut") {
    if (!has.keyboard || has.mouse || has.selection) throw new Error("Invalid computer history event");
  } else if (event.kind === "keyboard.text_input") {
    if (!has.keyboard || has.ax || has.mouse || has.selection) {
      throw new Error("Invalid computer history event");
    }
  } else if (event.kind === "keyboard.submit") {
    if (
      !has.keyboard
      || has.mouse
      || has.selection
      || (event.keyboard?.keyEquivalent !== "return"
        && event.keyboard?.keyEquivalent !== "enter")
      || event.keyboard.text !== undefined
    ) {
      throw new Error("Invalid computer history event");
    }
  } else if (event.kind === "selection.changed") {
    if (!has.selection || has.ax || has.mouse || has.keyboard) {
      throw new Error("Invalid computer history event");
    }
  }
}

function requireAllowedKeys(
  value: unknown,
  allowed: readonly string[],
): asserts value is Record<string, unknown> {
  if (!isRecord(value) || Object.keys(value).some((key) => !allowed.includes(key))) {
    throw new Error("Invalid computer history event");
  }
}

function requireExactKeys(
  value: unknown,
  keys: readonly string[],
): asserts value is Record<string, unknown> {
  requireAllowedKeys(value, keys);
  if (keys.some((key) => !Object.prototype.hasOwnProperty.call(value, key))) {
    throw new Error("Invalid computer history event");
  }
}

function requiredHistoryText(value: unknown, maximum: number, allowEmpty: boolean): string {
  if (!validHistoryText(value, maximum, allowEmpty)) {
    throw new Error("Invalid computer history event");
  }
  return value;
}

function validHistoryText(
  value: unknown,
  maximum: number,
  allowEmpty: boolean,
): value is string {
  return typeof value === "string"
    && (allowEmpty || value.length > 0)
    && Buffer.byteLength(value, "utf8") <= maximum
    && !/[\u0000\u007f\u2028\u2029]/u.test(value);
}

function requirePayloadSize(value: object): void {
  if (Buffer.byteLength(JSON.stringify(value), "utf8") > 32 * 1024) {
    throw new Error("Invalid computer history event");
  }
}

export function parseVisionCandidateMetadata(value: unknown): VisionCandidateMetadata {
  return parseVisionCandidateMetadataAllowing(value, []);
}

export function parseVisionCandidatePayload(value: unknown): VisionCandidatePayload {
  const metadata = parseVisionCandidateMetadataAllowing(value, [
    "mimeType",
    "jpegBase64",
  ]);
  if (!isRecord(value) || value.mimeType !== "image/jpeg") {
    throw new Error("Invalid vision candidate payload");
  }
  const jpegBase64 = validatedVisionCandidateJpeg(value.jpegBase64);
  return { ...metadata, mimeType: "image/jpeg", jpegBase64 };
}

export function parseEnvironmentFrameMetadata(value: unknown): EnvironmentFrameMetadata {
  return parseEnvironmentFrameMetadataAllowing(value, []);
}

export function parseEnvironmentFramePayload(value: unknown): EnvironmentFramePayload {
  const metadata = parseEnvironmentFrameMetadataAllowing(value, [
    "mimeType",
    "jpegBase64",
  ]);
  if (!isRecord(value) || value.mimeType !== "image/jpeg") {
    throw new Error("Invalid environment frame payload");
  }
  const jpegBase64 = validatedVisionCandidateJpeg(value.jpegBase64);
  if (Buffer.from(jpegBase64, "base64").byteLength !== metadata.byteLength) {
    throw new Error("Invalid environment frame payload");
  }
  if (metadata.contentHash && createHash("sha256").update(Buffer.from(jpegBase64, "base64")).digest("hex") !== metadata.contentHash) {
    throw new Error("Environment frame content hash mismatch");
  }
  return { ...metadata, mimeType: "image/jpeg", jpegBase64 };
}

function parseEnvironmentFrameMetadataAllowing(
  value: unknown,
  extraKeys: readonly string[],
): EnvironmentFrameMetadata {
  if (!isRecord(value)) throw new Error("Invalid environment frame metadata");
  const required = [
    "frameId",
    "revision",
    "observedAt",
    "expiresAt",
    "displayID",
    "pixelWidth",
    "pixelHeight",
    "byteLength",
  ] as const;
  const allowed = new Set<string>([...required, ...extraKeys, "contentHash"]);
  if (
    required.some((key) => !Object.prototype.hasOwnProperty.call(value, key))
    || Object.keys(value).some((key) => !allowed.has(key))
    || !isCandidateId(value.frameId)
    || !isPositiveSafeInteger(value.revision)
    || !isFinitePositiveNumber(value.observedAt)
    || !isFinitePositiveNumber(value.expiresAt)
    || value.expiresAt <= value.observedAt
    || Math.abs(
      value.expiresAt - value.observedAt - ENVIRONMENT_FRAME_TTL_SECONDS,
    ) > TIMESTAMP_EPSILON_SECONDS
    || !Number.isSafeInteger(value.displayID)
    || Number(value.displayID) < 0
    || Number(value.displayID) > 0xffff_ffff
    || !isPositiveSafeInteger(value.pixelWidth)
    || !isPositiveSafeInteger(value.pixelHeight)
    || Math.max(Number(value.pixelWidth), Number(value.pixelHeight))
      > ENVIRONMENT_FRAME_MAX_LONG_EDGE
    || !isPositiveSafeInteger(value.byteLength)
    || Number(value.byteLength) > MAX_VISION_CANDIDATE_JPEG_BYTES
    || (value.contentHash !== undefined && (typeof value.contentHash !== "string"
      || !/^[0-9a-f]{64}$/u.test(value.contentHash)))
  ) {
    throw new Error("Invalid environment frame metadata");
  }
  return {
    frameId: value.frameId,
    revision: Number(value.revision),
    observedAt: value.observedAt,
    expiresAt: value.expiresAt,
    displayID: Number(value.displayID),
    pixelWidth: Number(value.pixelWidth),
    pixelHeight: Number(value.pixelHeight),
    byteLength: Number(value.byteLength),
    ...(typeof value.contentHash === "string" ? { contentHash: value.contentHash } : {}),
  };
}

function parseVisionCandidateMetadataAllowing(
  value: unknown,
  extraKeys: readonly string[],
): VisionCandidateMetadata {
  if (!isRecord(value)) throw new Error("Invalid vision candidate metadata");
  const allowedKeys = new Set<string>([
    ...METADATA_REQUIRED_KEYS,
    "appName",
    ...extraKeys,
  ]);
  const keys = Object.keys(value);
  if (
    METADATA_REQUIRED_KEYS.some((key) => !Object.prototype.hasOwnProperty.call(value, key))
    || keys.some((key) => !allowedKeys.has(key))
  ) {
    throw new Error("Invalid vision candidate metadata");
  }
  if (!isCandidateId(value.candidateId) || !isCandidateId(value.windowKey)) {
    throw new Error("Invalid vision candidate metadata");
  }
  if (!isPositiveSafeInteger(value.revision)) {
    throw new Error("Invalid vision candidate metadata");
  }
  if (
    !isFinitePositiveNumber(value.observedAt)
    || !isFinitePositiveNumber(value.expiresAt)
    || value.expiresAt <= value.observedAt
    || Math.abs(
      value.expiresAt - value.observedAt - VISION_CANDIDATE_TTL_SECONDS,
    ) > TIMESTAMP_EPSILON_SECONDS
  ) {
    throw new Error("Invalid vision candidate metadata");
  }
  if (
    !Number.isSafeInteger(value.displayID)
    || Number(value.displayID) < 0
    || Number(value.displayID) > 0xffff_ffff
  ) {
    throw new Error("Invalid vision candidate metadata");
  }
  if (
    typeof value.appBundleId !== "string"
    || !APP_BUNDLE_ID_PATTERN.test(value.appBundleId)
  ) {
    throw new Error("Invalid vision candidate metadata");
  }
  const appName = optionalSingleLine(value.appName, 200);
  return {
    candidateId: value.candidateId,
    revision: Number(value.revision),
    observedAt: value.observedAt,
    expiresAt: value.expiresAt,
    displayID: Number(value.displayID),
    appBundleId: value.appBundleId,
    ...(appName ? { appName } : {}),
    windowKey: value.windowKey,
  };
}

function validatedVisionCandidateJpeg(value: unknown): string {
  if (
    typeof value !== "string"
    || !value
    || !CANONICAL_BASE64_PATTERN.test(value)
  ) {
    throw new Error("Invalid vision candidate payload");
  }
  const bytes = Buffer.from(value, "base64");
  if (
    bytes.toString("base64") !== value
    || bytes.byteLength > MAX_VISION_CANDIDATE_JPEG_BYTES
    || bytes.byteLength < 5
    || bytes[0] !== 0xff
    || bytes[1] !== 0xd8
    || bytes[2] !== 0xff
    || bytes.at(-2) !== 0xff
    || bytes.at(-1) !== 0xd9
  ) {
    throw new Error("Invalid vision candidate payload");
  }
  return value;
}

function isUnavailableVisionCandidateBody(value: unknown): boolean {
  return isRecord(value)
    && Object.keys(value).length === 1
    && value.code === "unavailable";
}

function isCandidateId(value: unknown): value is string {
  return typeof value === "string" && UUID_PATTERN.test(value);
}

function isPositiveSafeInteger(value: unknown): boolean {
  return Number.isSafeInteger(value) && Number(value) > 0;
}

function isFinitePositiveNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function optionalSingleLine(value: unknown, maximum: number): string | null {
  if (value === undefined) return null;
  if (
    typeof value !== "string"
    || !value
    || value !== value.trim()
    || [...value].length > maximum
    || /[\u0000-\u001f\u007f\u2028\u2029]/u.test(value)
  ) {
    throw new Error("Invalid vision candidate metadata");
  }
  return value;
}

function isExpiredVisionCandidate(
  metadata: VisionCandidateMetadata,
  nowMs: number,
): boolean {
  return metadata.expiresAt * 1_000 <= nowMs
    || metadata.observedAt * 1_000 > nowMs + MAX_CLOCK_SKEW_MS;
}

function isNewerVisionCandidate(
  candidate: VisionCandidateMetadata,
  previous: VisionCandidateMetadata,
): boolean {
  return candidate.revision > previous.revision
    && candidate.observedAt >= previous.observedAt;
}

function sameVisionCandidateMetadata(
  candidate: VisionCandidateMetadata,
  expected: VisionCandidateMetadata,
): boolean {
  return candidate.candidateId === expected.candidateId
    && candidate.revision === expected.revision
    && candidate.observedAt === expected.observedAt
    && candidate.expiresAt === expected.expiresAt
    && candidate.displayID === expected.displayID
    && candidate.appBundleId === expected.appBundleId
    && candidate.appName === expected.appName
    && candidate.windowKey === expected.windowKey;
}

function cloneVisionCandidateMetadata(
  metadata: VisionCandidateMetadata,
): VisionCandidateMetadata {
  return { ...metadata };
}

function cloneVisionCandidatePayload(
  payload: VisionCandidatePayload,
): VisionCandidatePayload {
  return { ...payload };
}

function isExpiredEnvironmentFrame(
  metadata: EnvironmentFrameMetadata,
  nowMs: number,
): boolean {
  return metadata.expiresAt * 1_000 <= nowMs
    || metadata.observedAt * 1_000 > nowMs + MAX_CLOCK_SKEW_MS;
}

function isNewerEnvironmentFrame(
  frame: EnvironmentFrameMetadata,
  previous: EnvironmentFrameMetadata,
): boolean {
  return frame.revision > previous.revision
    && frame.observedAt >= previous.observedAt;
}

function sameEnvironmentFrameMetadata(
  frame: EnvironmentFrameMetadata,
  expected: EnvironmentFrameMetadata,
): boolean {
  return frame.frameId === expected.frameId
    && frame.revision === expected.revision
    && frame.observedAt === expected.observedAt
    && frame.expiresAt === expected.expiresAt
    && frame.displayID === expected.displayID
    && frame.pixelWidth === expected.pixelWidth
    && frame.pixelHeight === expected.pixelHeight
    && frame.byteLength === expected.byteLength
    && frame.contentHash === expected.contentHash;
}

function defaultSpawnProcess(
  command: string,
  args: readonly string[],
  options: SpawnOptionsWithoutStdio,
): PerceptionHelperProcess {
  const child = spawn(command, [...args], options);
  attachProcessJob(child);
  return child as PerceptionHelperProcess;
}

function minimalHelperEnvironment(): NodeJS.ProcessEnv {
  const result: NodeJS.ProcessEnv = {};
  const keys = new Set(["HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA", "APPDATA"]);
  for (const [key, value] of Object.entries(process.env)) {
    if (keys.has(process.platform === "win32" ? key.toUpperCase() : key)) result[key] = value;
  }
  return result;
}

function capabilitiesFromBody(body: unknown): PerceptionCapabilities | null {
  if (!isRecord(body)) return null;
  const candidate = isRecord(body.capabilities) ? body.capabilities : null;
  if (!candidate) return null;
  const workspace = capabilityFromUnknown(candidate.workspace);
  const screen = capabilityFromUnknown(candidate.screen);
  const accessibility = capabilityFromUnknown(candidate.accessibility);
  const camera = capabilityFromUnknown(candidate.camera);
  const computerHistory = candidate.computerHistory === undefined
    ? legacyComputerHistoryCapability()
    : capabilityFromUnknown(candidate.computerHistory);
  const systemAudio = candidate.systemAudio === undefined
    ? undefined : capabilityFromUnknown(candidate.systemAudio);
  if (candidate.systemAudio !== undefined && !systemAudio) return null;
  return workspace && screen && accessibility && camera && computerHistory
    ? { workspace, screen, accessibility, camera, computerHistory,
        ...(systemAudio ? { systemAudio } : {}) }
    : null;
}

function legacyComputerHistoryCapability(): PerceptionCapabilityReport {
  return {
    supported: false,
    requested: false,
    state: "disabled",
    permission: "not_determined",
  };
}

function capabilityFromUnknown(value: unknown): PerceptionCapabilityReport | null {
  if (!isRecord(value)) return null;
  const states: PerceptionCapabilityState[] = ["disabled", "running", "unavailable", "failed"];
  const permissions: PerceptionCapabilityPermission[] = [
    "not_required",
    "granted",
    "denied",
    "not_determined",
    "restricted",
    "unknown",
  ];
  if (
    typeof value.supported !== "boolean" ||
    typeof value.requested !== "boolean" ||
    !states.includes(value.state as PerceptionCapabilityState) ||
    !permissions.includes(value.permission as PerceptionCapabilityPermission) ||
    (value.reason !== undefined && typeof value.reason !== "string")
  ) return null;
  const computerHistoryInput = value.computerHistoryInput === undefined
    ? undefined
    : computerHistoryInputFromUnknown(value.computerHistoryInput);
  if (value.computerHistoryInput !== undefined && !computerHistoryInput) return null;
  return {
    supported: value.supported,
    requested: value.requested,
    state: value.state as PerceptionCapabilityState,
    permission: value.permission as PerceptionCapabilityPermission,
    ...(typeof value.reason === "string" ? { reason: value.reason } : {}),
    ...(computerHistoryInput ? { computerHistoryInput } : {}),
  };
}

function computerHistoryInputFromUnknown(value: unknown): ComputerHistoryInputReport | null {
  if (!isRecord(value)) return null;
  const permissions: PerceptionCapabilityPermission[] = [
    "not_required",
    "granted",
    "denied",
    "not_determined",
    "restricted",
    "unknown",
  ];
  const eventTapStates: ComputerHistoryEventTapState[] = [
    "disabled",
    "permission_required",
    "running",
    "suspended",
    "failed",
  ];
  const mouse = computerHistoryInputChannelFromUnknown(value.mouse);
  const keyboard = computerHistoryInputChannelFromUnknown(value.keyboard);
  if (
    !permissions.includes(value.accessibilityPermission as PerceptionCapabilityPermission)
    || !permissions.includes(value.listenEventPermission as PerceptionCapabilityPermission)
    || !eventTapStates.includes(value.eventTapState as ComputerHistoryEventTapState)
    || !mouse
    || !keyboard
    || !Number.isSafeInteger(value.tapDisabledCount)
    || Number(value.tapDisabledCount) < 0
  ) return null;
  return {
    accessibilityPermission: value.accessibilityPermission as PerceptionCapabilityPermission,
    listenEventPermission: value.listenEventPermission as PerceptionCapabilityPermission,
    eventTapState: value.eventTapState as ComputerHistoryEventTapState,
    mouse,
    keyboard,
    tapDisabledCount: Number(value.tapDisabledCount),
  };
}

function computerHistoryInputChannelFromUnknown(
  value: unknown,
): ComputerHistoryInputChannelReport | null {
  if (!isRecord(value)) return null;
  const states: PerceptionCapabilityState[] = ["disabled", "running", "unavailable", "failed"];
  if (
    !states.includes(value.state as PerceptionCapabilityState)
    || !Number.isSafeInteger(value.receivedCount)
    || Number(value.receivedCount) < 0
    || (value.lastEventAt !== undefined && (
      typeof value.lastEventAt !== "number"
      || !Number.isFinite(value.lastEventAt)
      || value.lastEventAt <= 0
      || value.lastEventAt > MAX_HISTORY_TIMESTAMP_SECONDS
    ))
  ) return null;
  return {
    state: value.state as PerceptionCapabilityState,
    receivedCount: Number(value.receivedCount),
    ...(typeof value.lastEventAt === "number" ? { lastEventAt: value.lastEventAt } : {}),
  };
}

function requestedCapabilityUnavailable(
  capabilities: PerceptionCapabilities,
  modules: PerceptionModuleSelection,
): boolean {
  return (
    (modules.workspace && capabilities.workspace.state !== "running") ||
    (modules.screen && capabilities.screen.state !== "running") ||
    (modules.accessibility && capabilities.accessibility.state !== "running") ||
    (modules.camera && capabilities.camera.state !== "running") ||
    (modules.systemAudio === true && capabilities.systemAudio?.state !== "running") ||
    (modules.computerHistory === true && capabilities.computerHistory.state !== "running")
  );
}

function cloneModules(modules: PerceptionModuleSelection): PerceptionModuleSelection {
  return {
    workspace: modules.workspace,
    screen: modules.screen,
    accessibility: modules.accessibility,
    camera: modules.camera,
    enhancedVision: modules.enhancedVision === true,
    computerHistory: modules.computerHistory === true,
    legacyScreenAnalysis: modules.legacyScreenAnalysis !== false,
    excludeHostApplicationFromCapture: modules.excludeHostApplicationFromCapture === true,
    ...(modules.systemAudio === undefined ? {} : { systemAudio: modules.systemAudio === true }),
  };
}

function sameModules(
  left: PerceptionModuleSelection,
  right: PerceptionModuleSelection,
): boolean {
  return left.workspace === right.workspace
    && left.screen === right.screen
    && left.accessibility === right.accessibility
    && left.camera === right.camera
    && left.enhancedVision === right.enhancedVision
    && left.computerHistory === right.computerHistory
    && left.legacyScreenAnalysis === right.legacyScreenAnalysis
    && left.excludeHostApplicationFromCapture === right.excludeHostApplicationFromCapture
    && (left.systemAudio === true) === (right.systemAudio === true);
}

function cloneStatus(status: PerceptionManagerStatus): PerceptionManagerStatus {
  return {
    ...status,
    modules: cloneModules(status.modules),
    capabilities: status.capabilities
      ? {
          workspace: { ...status.capabilities.workspace },
          screen: { ...status.capabilities.screen },
          accessibility: { ...status.capabilities.accessibility },
          camera: { ...status.capabilities.camera },
          computerHistory: cloneCapability(status.capabilities.computerHistory),
          ...(status.capabilities.systemAudio
            ? { systemAudio: cloneCapability(status.capabilities.systemAudio) } : {}),
        }
      : null,
  };
}

function cloneCapability(value: PerceptionCapabilityReport): PerceptionCapabilityReport {
  const input = value.computerHistoryInput;
  return {
    ...value,
    ...(input
      ? {
          computerHistoryInput: {
            ...input,
            mouse: { ...input.mouse },
            keyboard: { ...input.keyboard },
          },
        }
      : {}),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

// Early cuts and older helpers may emit shorter windows.
const SYSTEM_AUDIO_WINDOW_SECONDS = 10;
const SYSTEM_AUDIO_WINDOW_BYTES = SYSTEM_AUDIO_WINDOW_SECONDS * 16_000 * 2;

export function parseSystemAudioWindow(value: unknown): SystemAudioWindow {
  const allowedKeys = [
    "captureSessionId", "windowIndex", "startedAt", "endedAt", "audioStatus",
    "audioPcmBase64", "audioSampleRate", "audioCapturedSamples", "audioGapReason",
    "audioSilence",
  ] as const;
  requireAllowedKeys(value, allowedKeys);
  const requiredKeys = allowedKeys.filter(
    (key) => key !== "audioPcmBase64" && key !== "audioGapReason" && key !== "audioSilence",
  );
  if (requiredKeys.some((key) => !Object.prototype.hasOwnProperty.call(value, key))) {
    throw new Error("Invalid system audio payload");
  }
  const audioPcmBase64 = value.audioPcmBase64 ?? null;
  const audioGapReason = value.audioGapReason ?? null;
  const audioSilence = value.audioSilence === undefined ? false : value.audioSilence;
  if (!isRecord(value)
    || typeof value.captureSessionId !== "string"
    || !HISTORY_UUID_PATTERN.test(value.captureSessionId)
    || !Number.isSafeInteger(value.windowIndex) || Number(value.windowIndex) < 0
    || typeof value.startedAt !== "number" || !Number.isFinite(value.startedAt)
    || value.startedAt <= 0 || value.startedAt > MAX_HISTORY_TIMESTAMP_SECONDS
    || typeof value.endedAt !== "number" || !Number.isFinite(value.endedAt)
    || value.endedAt <= 0 || value.endedAt > MAX_HISTORY_TIMESTAMP_SECONDS
    || value.endedAt <= value.startedAt
    || value.endedAt - value.startedAt > SYSTEM_AUDIO_WINDOW_SECONDS + TIMESTAMP_EPSILON_SECONDS
    || !["complete", "partial", "unavailable"].includes(String(value.audioStatus))
    || value.audioSampleRate !== 16000
    || typeof audioSilence !== "boolean"
    || !Number.isSafeInteger(value.audioCapturedSamples)
    || Number(value.audioCapturedSamples) < 0
    || Number(value.audioCapturedSamples) > 160_000
    || (audioGapReason !== null
      && (typeof audioGapReason !== "string"
        || !/^[a-z_]{1,64}$/u.test(audioGapReason)))) {
    throw new Error("Invalid system audio payload");
  }

  const expectedSamples = Math.round((Number(value.endedAt) - Number(value.startedAt)) * 16_000);
  const expectedBytes = expectedSamples * 2;
  const status = value.audioStatus as SystemAudioWindow["audioStatus"];
  let pcm: string | null = null;
  if (audioSilence) {
    if (status !== "complete" || value.audioCapturedSamples !== expectedSamples
      || audioPcmBase64 !== null || audioGapReason !== null) {
      throw new Error("Invalid system audio silence payload");
    }
  } else if (status === "unavailable") {
    if (audioPcmBase64 !== null || value.audioCapturedSamples !== 0) {
      throw new Error("Invalid system audio payload");
    }
  } else {
    if (typeof audioPcmBase64 !== "string"
      || audioPcmBase64.length > Math.ceil(SYSTEM_AUDIO_WINDOW_BYTES / 3) * 4
      || !CANONICAL_BASE64_PATTERN.test(audioPcmBase64)) {
      throw new Error("Invalid system audio payload");
    }
    const bytes = Buffer.from(audioPcmBase64, "base64");
    if (bytes.length !== expectedBytes
      || bytes.toString("base64") !== audioPcmBase64
      || (status === "complete" && Number(value.audioCapturedSamples) !== expectedSamples)) {
      throw new Error("Invalid system audio payload");
    }
    pcm = audioPcmBase64;
  }

  return {
    captureSessionId: value.captureSessionId,
    windowIndex: Number(value.windowIndex),
    startedAt: value.startedAt,
    endedAt: value.endedAt,
    audioStatus: status,
    audioPcmBase64: pcm,
    audioSampleRate: 16000,
    audioCapturedSamples: Number(value.audioCapturedSamples),
    audioGapReason: audioGapReason as string | null,
    ...(audioSilence ? { audioSilence: true } : {}),
  };
}
