import { createHash, randomUUID } from "node:crypto";
import {
  appendFile,
  mkdir,
  open,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";

import {
  parseComputerHistoryInteractionEvent,
  type ComputerHistoryInteractionEvent,
} from "./perception-manager";
import { ComputerHistorySelfObservationGuard } from "./computer-history-self-observation";

const SCHEMA_VERSION = 1 as const;
const SEGMENT_DURATION_MS = 10 * 60 * 1_000;
const DEFAULT_GRACE_MS = 5_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 180_000;
const DEFAULT_RETRY_DELAYS_MS = [5_000, 30_000, 2 * 60_000, 10 * 60_000] as const;
const DEFAULT_REFRESH_COOLDOWN_MS = 60_000;
export const MAX_COMPUTER_HISTORY_SEGMENT_BYTES = 512 * 1024;
const MAX_RESPONSE_BYTES = 64 * 1024;
const MAX_PENDING_AGE_MS = 7 * 24 * 60 * 60 * 1_000;
const SELF_OBSERVATION_RECOVERY_MAX_AGE_MS = 2 * SEGMENT_DURATION_MS;
const SELF_OBSERVATION_RECOVERY_EVENT_LIMIT = 256;
const MAX_HISTORY_TIMESTAMP_MS = 253_402_300_799_999;
const SEGMENT_NAME_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$/u;
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu;
const DIGEST_PATTERN = /^[0-9a-f]{64}$/u;
const UTC_ISO_PATTERN = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?Z$/u;
const STORED_EVENT_KEYS = new Set([
  "schemaVersion",
  "ordinal",
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

type Fetcher = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

export interface ComputerHistoryConnection {
  apiUrl: string;
  apiKey: string;
}

export interface ComputerHistoryStoredEvent {
  schemaVersion: typeof SCHEMA_VERSION;
  ordinal: number;
  id: string;
  timestamp: string;
  kind: ComputerHistoryInteractionEvent["kind"];
  app: ComputerHistoryInteractionEvent["app"];
  window: ComputerHistoryInteractionEvent["window"];
  ax?: ComputerHistoryInteractionEvent["ax"];
  mouse?: ComputerHistoryInteractionEvent["mouse"];
  keyboard?: ComputerHistoryInteractionEvent["keyboard"];
  selection?: ComputerHistoryInteractionEvent["selection"];
}

export interface ComputerHistorySegmentMetadata {
  schemaVersion: typeof SCHEMA_VERSION;
  segmentId: string;
  startedAt: string;
  endedAt: string;
  createdAt: string;
  fishId: string | null;
  eventCount: number;
  meaningfulEventCount: number;
  suppressedEventCount: number;
  summaryStatus: "pending";
  checkpoint?: ComputerHistoryCheckpoint | null;
  refreshRequest?: ComputerHistoryRefreshRequest | null;
}

export interface ComputerHistoryCheckpoint {
  schemaVersion: typeof SCHEMA_VERSION;
  segmentId: string;
  startedAt: string;
  endedAt: string;
  revision: number;
  throughEventCount: number;
  throughByteOffset: number;
  throughSuppressedEventCount: number;
  deltaChainDigest: string;
  lastSuccessfulSummaryAt: string | null;
  status: "active" | "finalized";
  summary: string | null;
  applications: string[];
  finalizedAt: string | null;
  observationId: string | null;
  finalEventDigest: string | null;
}

export interface ComputerHistoryRefreshRequest {
  requestId: string;
  requestedAt: string;
  /** Server revision visible when this Ask froze its high-water mark. */
  baseRevision: number;
  throughEventCount: number;
  throughByteOffset: number;
  throughSuppressedEventCount: number;
  finalize: boolean;
}

export interface ComputerHistoryCheckpointRequest {
  schemaVersion: typeof SCHEMA_VERSION;
  requestId: string;
  startedAt: string;
  endedAt: string;
  baseRevision: number;
  fromEventCount: number;
  throughEventCount: number;
  fromByteOffset: number;
  throughByteOffset: number;
  throughSuppressedEventCount: number;
  deltaDigest: string;
  deltaEventsJsonl: string;
  finalize: boolean;
  finalEventDigest?: string;
}

export interface ComputerHistorySummarizeRequest {
  schemaVersion: typeof SCHEMA_VERSION;
  segmentId: string;
  startedAt: string;
  endedAt: string;
  eventCount: number;
  suppressedEventCount: number;
  eventDigest: string;
  eventsJsonl: string;
}

export type ComputerHistoryRecorderErrorCode =
  | "local_api_failed"
  | "segment_invalid"
  | "storage_failed"
  | "summary_failed";

export interface ComputerHistoryRecorderOptions {
  rootPath: string;
  connection: () => Promise<ComputerHistoryConnection | null>;
  deliveryMode?: "summary" | "context";
  fetcher?: Fetcher;
  now?: () => number;
  schedule?: (callback: () => void, delayMs: number) => NodeJS.Timeout;
  cancelSchedule?: (timer: NodeJS.Timeout) => void;
  graceMs?: number;
  requestTimeoutMs?: number;
  retryDelaysMs?: readonly number[];
  refreshCooldownMs?: number;
  idFactory?: () => string;
  onErrorCode?: (code: ComputerHistoryRecorderErrorCode) => void;
}

interface SegmentState {
  name: string;
  directory: string;
  metadata: ComputerHistorySegmentMetadata;
  byteLength: number;
  eventIds: Set<string>;
  legacyFullSummary: boolean;
}

interface IncrementalWork {
  kind: "incremental";
  name: string;
  directory: string;
  metadata: ComputerHistorySegmentMetadata;
  request: ComputerHistoryRefreshRequest;
}

interface LegacyWork {
  kind: "legacy";
  name: string;
  directory: string;
  metadata: ComputerHistorySegmentMetadata;
}

type RecorderWork = IncrementalWork | LegacyWork;

/** Durable, ordered handoff from native interaction events to local summarization. */
export class ComputerHistoryRecorder {
  private readonly fetcher: Fetcher;
  private readonly now: () => number;
  private readonly schedule: (callback: () => void, delayMs: number) => NodeJS.Timeout;
  private readonly cancelSchedule: (timer: NodeJS.Timeout) => void;
  private readonly graceMs: number;
  private readonly requestTimeoutMs: number;
  private readonly retryDelaysMs: readonly number[];
  private readonly refreshCooldownMs: number;
  private readonly idFactory: () => string;
  private enabled = false;
  private stopped = false;
  private operation: Promise<void> = Promise.resolve();
  private wakeTimer: NodeJS.Timeout | null = null;
  private readonly segmentRetries = new Map<string, { attempt: number; afterMs: number }>();
  private activeRequest: AbortController | null = null;
  private worker: Promise<void> | null = null;
  private workerWakeRequested = false;
  private readonly segmentCache = new Map<string, SegmentState>();
  private readonly selfObservationGuard: ComputerHistorySelfObservationGuard;
  private selfObservationRecoveryPending = true;

  constructor(private readonly options: ComputerHistoryRecorderOptions) {
    this.fetcher = options.fetcher ?? fetch;
    this.now = options.now ?? Date.now;
    this.schedule = options.schedule ?? setTimeout;
    this.cancelSchedule = options.cancelSchedule ?? clearTimeout;
    this.graceMs = options.graceMs ?? DEFAULT_GRACE_MS;
    this.requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
    this.retryDelaysMs = options.retryDelaysMs ?? DEFAULT_RETRY_DELAYS_MS;
    this.refreshCooldownMs = options.refreshCooldownMs ?? DEFAULT_REFRESH_COOLDOWN_MS;
    this.idFactory = options.idFactory ?? randomUUID;
    this.selfObservationGuard = new ComputerHistorySelfObservationGuard(options.rootPath);
    if (
      this.graceMs < 0
      || this.requestTimeoutMs < 1
      || this.refreshCooldownMs < 0
      || !Number.isFinite(this.refreshCooldownMs)
      || this.retryDelaysMs.length === 0
      || this.retryDelaysMs.some((delay) => delay < 0 || !Number.isFinite(delay))
    ) throw new Error("Invalid computer history recorder timing configuration");
  }

  setEnabled(enabled: boolean): Promise<void> {
    this.stopped = false;
    this.enabled = enabled;
    this.clearWakeTimer();
    if (!enabled) {
      this.activeRequest?.abort();
      this.selfObservationGuard.reset();
      this.selfObservationRecoveryPending = true;
    }
    const queued = this.enqueue(async () => {
      if (!this.enabled && !await pathExists(this.options.rootPath)) return;
      await mkdir(this.options.rootPath, { recursive: true, mode: 0o700 });
      await this.recoverDeletingDirectories();
      if (this.enabled) {
        await this.restoreSelfObservationContext();
      }
      else await this.maintainDisabledRetention();
    }).catch(() => {
      this.options.onErrorCode?.("storage_failed");
    });
    void queued.then(() => this.kickWorker());
    return queued;
  }

  record(event: ComputerHistoryInteractionEvent): Promise<void> {
    if (!this.enabled) return Promise.resolve();
    const frozen = structuredClone(event);
    const queued = this.enqueue(async () => {
      if (!this.enabled) return;
      try {
        await mkdir(this.options.rootPath, { recursive: true, mode: 0o700 });
        const startMs = segmentStart(event.timestamp * 1_000);
        const endMs = startMs + SEGMENT_DURATION_MS;
        const name = segmentName(startMs);
        if (this.now() >= endMs + this.graceMs) return;
        const segment = await this.openSegment(name, startMs, endMs);
        await this.appendEvent(segment, this.selfObservationGuard.sanitize(frozen));
      } catch {
        this.options.onErrorCode?.("storage_failed");
      }
    });
    void queued.then(() => this.kickWorker());
    return queued;
  }

  recordSuppressed(count: number, timestamp: number): Promise<void> {
    if (!this.enabled) return Promise.resolve();
    if (
      !Number.isSafeInteger(count)
      || count < 1
      || count > 1_000_000
      || !Number.isFinite(timestamp)
      || timestamp <= 0
    ) return Promise.resolve();
    const queued = this.enqueue(async () => {
      if (!this.enabled) return;
      try {
        await mkdir(this.options.rootPath, { recursive: true, mode: 0o700 });
        const startMs = segmentStart(timestamp * 1_000);
        const endMs = startMs + SEGMENT_DURATION_MS;
        const name = segmentName(startMs);
        if (this.now() >= endMs + this.graceMs) return;
        const segment = await this.openSegment(name, startMs, endMs);
        const nextSuppressed = Math.min(
          Number.MAX_SAFE_INTEGER,
          segment.metadata.suppressedEventCount + count,
        );
        segment.metadata = {
          ...segment.metadata,
          suppressedEventCount: nextSuppressed,
        };
        await writeMetadataAtomically(
          path.join(segment.directory, "metadata.json"),
          segment.metadata,
        );
      } catch {
        this.options.onErrorCode?.("storage_failed");
      }
    });
    void queued.then(() => this.kickWorker());
    return queued;
  }

  /**
   * Freeze the latest active-segment high-water mark after an Ask. This method
   * performs no network I/O and never reads the JSONL body on the writer queue.
   */
  requestRefresh(fishId?: string): Promise<void> {
    if (this.options.deliveryMode === "context") {
      this.kickWorker();
      return Promise.resolve();
    }
    if (!this.enabled) return Promise.resolve();
    if (fishId !== undefined && !UUID_PATTERN.test(fishId)) return Promise.resolve();
    const queued = this.enqueue(async () => {
      if (!this.enabled) return;
      const startMs = segmentStart(this.now());
      const name = segmentName(startMs);
      const directory = path.join(this.options.rootPath, name);
      if (!await pathExists(directory)) return;
      const segment = await this.openSegment(name, startMs, startMs + SEGMENT_DURATION_MS);
      if (segment.legacyFullSummary || segment.metadata.meaningfulEventCount === 0) return;
      if (fishId !== undefined && segment.metadata.fishId !== null
        && segment.metadata.fishId !== fishId) return;
      const checkpoint = segment.metadata.checkpoint ?? null;
      if (
        checkpoint?.lastSuccessfulSummaryAt
        && this.now() - Date.parse(checkpoint.lastSuccessfulSummaryAt) < this.refreshCooldownMs
      ) return;
      if (segment.metadata.eventCount <= (checkpoint?.throughEventCount ?? 0)) return;
      const current = segment.metadata.refreshRequest ?? null;
      if (
        current
        && current.throughEventCount >= segment.metadata.eventCount
        && current.throughByteOffset >= segment.byteLength
        && current.throughSuppressedEventCount >= segment.metadata.suppressedEventCount
      ) return;
      segment.metadata = {
        ...segment.metadata,
        fishId: fishId ?? segment.metadata.fishId,
        refreshRequest: this.newRefreshRequest(segment, false),
      };
      await this.writeSegmentMetadata(segment);
    }).catch(() => {
      this.options.onErrorCode?.("storage_failed");
    });
    void queued.then(() => {
      this.clearWakeTimer();
      this.kickWorker();
    });
    return queued;
  }

  async stop(): Promise<void> {
    this.stopped = true;
    this.enabled = false;
    this.clearWakeTimer();
    this.activeRequest?.abort();
    await this.operation;
    await this.worker;
    await this.operation;
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const next = this.operation.then(operation, operation);
    this.operation = next.then(() => undefined, () => undefined);
    return next;
  }

  private async openSegment(
    name: string,
    startMs: number,
    endMs: number,
  ): Promise<SegmentState> {
    const cached = this.segmentCache.get(name);
    if (cached) return cached;
    const directory = path.join(this.options.rootPath, name);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    const metadataPath = path.join(directory, "metadata.json");
    let metadata: ComputerHistorySegmentMetadata;
    let legacyFullSummary = false;
    try {
      const parsed = parseMetadata(JSON.parse(await readFile(metadataPath, "utf8")) as unknown);
      metadata = parsed.metadata;
      legacyFullSummary = parsed.legacyFullSummary;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      metadata = {
        schemaVersion: SCHEMA_VERSION,
        segmentId: this.idFactory(),
        startedAt: new Date(startMs).toISOString(),
        endedAt: new Date(endMs).toISOString(),
        createdAt: new Date(this.now()).toISOString(),
        fishId: null,
        eventCount: 0,
        meaningfulEventCount: 0,
        suppressedEventCount: 0,
        summaryStatus: "pending",
        checkpoint: null,
        refreshRequest: null,
      };
      await writeMetadataAtomically(metadataPath, metadata);
    }
    if (
      Date.parse(metadata.startedAt) !== startMs
      || Date.parse(metadata.endedAt) !== endMs
    ) throw new Error("Computer history segment metadata does not match its directory");
    const eventsPath = path.join(directory, "events.jsonl");
    const eventsJsonl = await readFile(eventsPath, "utf8").catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return "";
      throw error;
    });
    const counts = validateEventsJsonl(eventsJsonl);
    if (
      metadata.eventCount !== counts.eventCount
      || metadata.meaningfulEventCount !== counts.meaningfulEventCount
    ) {
      metadata = {
        ...metadata,
        eventCount: counts.eventCount,
        meaningfulEventCount: counts.meaningfulEventCount,
      };
      await writeMetadataAtomically(metadataPath, metadata);
    }
    const state = {
      name,
      directory,
      metadata,
      byteLength: Buffer.byteLength(eventsJsonl, "utf8"),
      eventIds: counts.eventIds,
      legacyFullSummary,
    };
    validateMetadataWatermarks(state);
    this.segmentCache.set(name, state);
    return state;
  }

  private async appendEvent(
    segment: SegmentState,
    event: ComputerHistoryInteractionEvent,
  ): Promise<void> {
    if (segment.eventIds.has(event.id)) return;
    const stored: ComputerHistoryStoredEvent = {
      schemaVersion: SCHEMA_VERSION,
      ordinal: segment.metadata.eventCount + 1,
      id: event.id,
      timestamp: new Date(event.timestamp * 1_000).toISOString(),
      kind: event.kind,
      app: { ...event.app },
      window: { ...event.window },
      ...(event.ax ? { ax: structuredClone(event.ax) } : {}),
      ...(event.mouse ? { mouse: structuredClone(event.mouse) } : {}),
      ...(event.keyboard ? { keyboard: structuredClone(event.keyboard) } : {}),
      ...(event.selection ? { selection: structuredClone(event.selection) } : {}),
    };
    const line = `${JSON.stringify(stored)}\n`;
    const lineBytes = Buffer.byteLength(line, "utf8");
    if (segment.byteLength + lineBytes > MAX_COMPUTER_HISTORY_SEGMENT_BYTES) {
      segment.metadata = {
        ...segment.metadata,
        suppressedEventCount: Math.min(
          Number.MAX_SAFE_INTEGER,
          segment.metadata.suppressedEventCount + 1,
        ),
      };
      await writeMetadataAtomically(
        path.join(segment.directory, "metadata.json"),
        segment.metadata,
      );
      return;
    }
    await appendFile(path.join(segment.directory, "events.jsonl"), line, {
      encoding: "utf8",
      mode: 0o600,
    });
    segment.byteLength += lineBytes;
    segment.eventIds.add(event.id);
    segment.metadata = {
      ...segment.metadata,
      eventCount: segment.metadata.eventCount + 1,
      meaningfulEventCount: segment.metadata.meaningfulEventCount
        + (event.kind === "session.started" ? 0 : 1),
    };
    await writeMetadataAtomically(
      path.join(segment.directory, "metadata.json"),
      segment.metadata,
    );
  }

  private async writeSegmentMetadata(segment: SegmentState): Promise<void> {
    await writeMetadataAtomically(path.join(segment.directory, "metadata.json"), segment.metadata);
  }

  private newRefreshRequest(
    segment: SegmentState,
    finalize: boolean,
  ): ComputerHistoryRefreshRequest {
    return {
      requestId: randomUUID(),
      requestedAt: new Date(this.now()).toISOString(),
      baseRevision: segment.metadata.checkpoint?.revision ?? 0,
      throughEventCount: segment.metadata.eventCount,
      throughByteOffset: segment.byteLength,
      throughSuppressedEventCount: segment.metadata.suppressedEventCount,
      finalize,
    };
  }

  /** Persist a server high-water mark without discarding a newer Ask request. */
  private async commitCheckpoint(
    name: string,
    checkpoint: ComputerHistoryCheckpoint,
    completedRequestId?: string,
  ): Promise<void> {
    const startMs = Date.parse(checkpoint.startedAt);
    const segment = await this.openSegment(name, startMs, Date.parse(checkpoint.endedAt));
    if (segment.metadata.segmentId !== checkpoint.segmentId) {
      throw new Error("Computer history checkpoint segment does not match local metadata");
    }
    const current = segment.metadata.refreshRequest;
    segment.metadata = {
      ...segment.metadata,
      checkpoint,
      refreshRequest: current?.requestId === completedRequestId ? null : current,
    };
    await this.writeSegmentMetadata(segment);
  }

  private async clearRefreshRequest(name: string, requestId: string): Promise<void> {
    const segment = this.segmentCache.get(name);
    if (!segment || segment.metadata.refreshRequest?.requestId !== requestId) return;
    segment.metadata = { ...segment.metadata, refreshRequest: null };
    await this.writeSegmentMetadata(segment);
  }

  private async rotateRefreshRequestId(name: string, requestId: string): Promise<void> {
    const segment = this.segmentCache.get(name);
    if (!segment || segment.metadata.refreshRequest?.requestId !== requestId) return;
    segment.metadata = {
      ...segment.metadata,
      refreshRequest: this.newRefreshRequest(segment, segment.metadata.refreshRequest.finalize),
    };
    await this.writeSegmentMetadata(segment);
  }

  private async deleteSegmentIfCurrent(
    name: string,
    checkpoint?: ComputerHistoryCheckpoint,
  ): Promise<void> {
    const segment = this.segmentCache.get(name);
    if (!segment) return;
    if (checkpoint) {
      const current = segment.metadata.checkpoint;
      if (
        !current
        || current.status !== "finalized"
        || current.segmentId !== checkpoint.segmentId
        || current.revision !== checkpoint.revision
        || current.finalEventDigest !== checkpoint.finalEventDigest
        || current.throughEventCount !== segment.metadata.eventCount
        || current.throughByteOffset !== segment.byteLength
        || current.throughSuppressedEventCount !== segment.metadata.suppressedEventCount
      ) return;
    }
    await this.deleteConfirmedSegment(segment);
  }

  private kickWorker(): void {
    if (!this.enabled || this.stopped) return;
    if (this.worker) {
      this.workerWakeRequested = true;
      return;
    }
    this.workerWakeRequested = false;
    const running = this.runWorker();
    this.worker = running;
    void running.finally(() => {
      if (this.worker === running) this.worker = null;
      if (this.enabled && !this.stopped && this.workerWakeRequested) this.kickWorker();
    });
  }

  private async runWorker(): Promise<void> {
    this.clearWakeTimer();
    if (this.options.deliveryMode === "context") {
      await this.forwardContextEvents();
      return;
    }
    const failedThisPass = new Set<string>();
    while (this.enabled && !this.stopped) {
      const work = await this.enqueue(() => this.prepareNextWork(failedThisPass));
      if (!work) return;
      try {
        if (work.kind === "legacy") await this.submitLegacyWork(work);
        else await this.submitIncrementalWork(work);
        this.segmentRetries.delete(work.name);
      } catch {
        if (!this.enabled || this.stopped) return;
        this.options.onErrorCode?.("summary_failed");
        const attempt = this.segmentRetries.get(work.name)?.attempt ?? 0;
        const delay = this.retryDelaysMs[Math.min(attempt, this.retryDelaysMs.length - 1)]!;
        this.segmentRetries.set(work.name, { attempt: attempt + 1, afterMs: this.now() + delay });
        // A failed segment must not block newer activity. Even zero-delay
        // retries yield to the scheduler after trying the other segments.
        failedThisPass.add(work.name);
      }
    }
  }

  private async forwardContextEvents(): Promise<void> {
    try {
      const connection = await this.options.connection();
      if (!connection) { this.scheduleWake(1_000); return; }
      const segments = await this.enqueue(() => this.pendingSegments());
      for (const segment of segments) {
        if (!this.enabled || this.stopped) return;
        const offsetPath = path.join(segment.directory, "context-offset.json");
        const checkpoint = await readFile(offsetPath, "utf8")
          .then((s) => JSON.parse(s) as { offset: number; fishId: string })
          .catch(() => null);
        const eventsPath = path.join(segment.directory, "events.jsonl");
        const length = (await stat(eventsPath)).size;
        let offset = checkpoint?.offset ?? 0;
        if (!Number.isSafeInteger(offset) || offset < 0 || offset > length) offset = 0;
        if (offset === length) {
          if (checkpoint && this.now() >= Date.parse(segment.metadata.endedAt) + this.graceMs) {
            await this.enqueue(async () => {
              // Recheck on the writer queue: a late event may have arrived
              // between the first stat and this cleanup operation.
              if ((await stat(eventsPath)).size === offset) {
                await this.deleteConfirmedSegment(segment);
              }
            });
          }
          continue;
        }
        const body = await readFile(eventsPath);
        let fishId = checkpoint?.fishId ?? segment.metadata.fishId;
        if (!fishId) fishId = await this.resolveAssistantFish(connection);
        if (!fishId || !UUID_PATTERN.test(fishId)) { this.scheduleWake(1_000); return; }
        while (offset < body.length && this.enabled && !this.stopped) {
          const events: unknown[] = [];
          let next = offset;
          while (events.length < 32) {
            const end = body.indexOf(10, next);
            if (end === -1) break;
            const line = body.subarray(next, end).toString("utf8");
            next = end + 1;
            if (line.trim()) events.push(JSON.parse(line));
          }
          if (!events.length) break;
          const response = await this.fetchLocal(
            `${connection.apiUrl}/internal/fishes/${encodeURIComponent(fishId)}/history/events`,
            {
              method: "POST",
              headers: {
                ...authorizationHeaders(connection.apiKey),
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ events }),
            },
          );
          if (!response.ok) throw new Error("Unable to deliver context events");
          const temporary = `${offsetPath}.tmp`;
          await writeFile(temporary, JSON.stringify({ fishId, offset: next }), { mode: 0o600 });
          await rename(temporary, offsetPath);
          offset = next;
        }
      }
    } catch {
      if (this.enabled && !this.stopped) {
        this.options.onErrorCode?.("summary_failed");
        this.scheduleWake(3_000);
      }
    }
  }

  private async prepareNextWork(failedThisPass: ReadonlySet<string>): Promise<RecorderWork | null> {
    if (!this.enabled || this.stopped) return null;
    const segments = await this.pendingSegments();
    let nextEligibleAt: number | null = null;
    for (const segment of segments) {
      const endedAt = Date.parse(segment.metadata.endedAt);
      if (this.now() >= endedAt + MAX_PENDING_AGE_MS) {
        await this.deleteConfirmedSegment(segment);
        continue;
      }
      const eligibleAt = endedAt + this.graceMs;
      const closed = this.now() >= eligibleAt;
      if (closed && segment.metadata.meaningfulEventCount === 0) {
        await this.deleteConfirmedSegment(segment);
        continue;
      }
      const retry = this.segmentRetries.get(segment.name);
      if (retry && (this.now() < retry.afterMs || failedThisPass.has(segment.name))) {
        nextEligibleAt = nextEligibleAt === null
          ? retry.afterMs
          : Math.min(nextEligibleAt, retry.afterMs);
        continue;
      }
      if (closed && segment.legacyFullSummary) {
        return {
          kind: "legacy",
          name: segment.name,
          directory: segment.directory,
          metadata: structuredClone(segment.metadata),
        };
      }
      if (closed) {
        const request = segment.metadata.refreshRequest;
        if (
          !request
          || !request.finalize
          || request.throughEventCount !== segment.metadata.eventCount
          || request.throughByteOffset !== segment.byteLength
          || request.throughSuppressedEventCount !== segment.metadata.suppressedEventCount
        ) {
          segment.metadata = {
            ...segment.metadata,
            refreshRequest: this.newRefreshRequest(segment, true),
          };
          await this.writeSegmentMetadata(segment);
        }
      }
      if (segment.metadata.refreshRequest) {
        return {
          kind: "incremental",
          name: segment.name,
          directory: segment.directory,
          metadata: structuredClone(segment.metadata),
          request: structuredClone(segment.metadata.refreshRequest),
        };
      }
      if (!closed) {
        nextEligibleAt = nextEligibleAt === null
          ? eligibleAt
          : Math.min(nextEligibleAt, eligibleAt);
      }
    }
    if (nextEligibleAt !== null) this.scheduleWake(nextEligibleAt - this.now());
    return null;
  }

  private async pendingSegments(): Promise<SegmentState[]> {
    await this.recoverDeletingDirectories();
    const entries = await readdir(this.options.rootPath, { withFileTypes: true });
    const states: SegmentState[] = [];
    const names = entries
      .filter((entry) => entry.isDirectory() && SEGMENT_NAME_PATTERN.test(entry.name))
      .map((entry) => entry.name)
      .sort();
    for (const name of names) {
      try {
        const startMs = Date.parse(name.replace(
          /T(\d{2})-(\d{2})-(\d{2})Z$/u,
          "T$1:$2:$3Z",
        ));
        if (this.now() >= startMs + SEGMENT_DURATION_MS + MAX_PENDING_AGE_MS) {
          await this.deleteExpiredDirectory(name);
          continue;
        }
        states.push(await this.openSegment(
          name,
          startMs,
          startMs + SEGMENT_DURATION_MS,
        ));
      } catch {
        this.options.onErrorCode?.("segment_invalid");
        this.scheduleWake(this.retryDelaysMs.at(-1)!);
        return [];
      }
    }
    return states;
  }

  private async submitIncrementalWork(work: IncrementalWork): Promise<void> {
    const connection = await this.options.connection();
    if (!connection) throw new Error("Local API unavailable");
    const fishId = work.metadata.fishId ?? await this.resolveAssistantFish(connection);
    if (work.metadata.fishId === null) {
      await this.enqueue(() => this.pinSegmentFish(work.name, fishId));
    }
    const base = `${connection.apiUrl}/internal/fishes/${encodeURIComponent(fishId)}`
      + "/computer-history/segments";
    const checkpointUrl = `${base}/${encodeURIComponent(work.metadata.segmentId)}/checkpoint`;
    const existing = await this.fetchLocal(checkpointUrl, {
      headers: authorizationHeaders(connection.apiKey),
    });
    let checkpoint: ComputerHistoryCheckpoint | null = null;
    if (existing.status === 200) {
      checkpoint = parseCheckpoint(await boundedResponseJson(existing), work.metadata);
      await this.commitCheckpoint(work.name, checkpoint);
      if (checkpoint.status === "finalized") {
        await this.confirmFinalizedCheckpoint(base, connection, work, checkpoint);
        return;
      }
    } else if (existing.status === 404) {
      if (await this.recoverFinalObservation(base, connection, work)) return;
    } else {
      throw new Error("Unable to reconcile computer history checkpoint");
    }

    if (
      !work.request.finalize
      && checkpoint
      && checkpoint.lastSuccessfulSummaryAt
      && this.now() - Date.parse(checkpoint.lastSuccessfulSummaryAt) < this.refreshCooldownMs
    ) {
      await this.clearRefreshRequest(work.name, work.request.requestId);
      return;
    }
    if (checkpoint && checkpointCoversRequest(checkpoint, work.request) && !work.request.finalize) {
      await this.clearRefreshRequest(work.name, work.request.requestId);
      return;
    }

    const fromEventCount = checkpoint?.throughEventCount ?? 0;
    const fromByteOffset = checkpoint?.throughByteOffset ?? 0;
    if (fromEventCount > work.request.throughEventCount) {
      throw new Error("Computer history checkpoint is ahead of the frozen request");
    }
    if (
      fromByteOffset > work.request.throughByteOffset
      || (checkpoint
        && checkpoint.throughSuppressedEventCount > work.request.throughSuppressedEventCount)
    ) throw new Error("Computer history checkpoint is ahead of the frozen request");
    if (fromEventCount === work.request.throughEventCount && !work.request.finalize) {
      await this.clearRefreshRequest(work.name, work.request.requestId);
      return;
    }
    const deltaEventsJsonl = await readJsonlRange(
      path.join(work.directory, "events.jsonl"),
      fromByteOffset,
      work.request.throughByteOffset,
    );
    validateEventsJsonlRange(
      deltaEventsJsonl,
      fromEventCount,
      work.request.throughEventCount,
    );
    const request: ComputerHistoryCheckpointRequest = {
      schemaVersion: SCHEMA_VERSION,
      requestId: work.request.requestId,
      startedAt: work.metadata.startedAt,
      endedAt: work.metadata.endedAt,
      baseRevision: checkpoint?.revision ?? 0,
      fromEventCount,
      throughEventCount: work.request.throughEventCount,
      fromByteOffset,
      throughByteOffset: work.request.throughByteOffset,
      throughSuppressedEventCount: work.request.throughSuppressedEventCount,
      deltaDigest: createHash("sha256").update(deltaEventsJsonl, "utf8").digest("hex"),
      deltaEventsJsonl,
      finalize: work.request.finalize,
    };
    if (request.finalize) {
      const fullEventsJsonl = await readJsonlRange(
        path.join(work.directory, "events.jsonl"),
        0,
        work.request.throughByteOffset,
      );
      request.finalEventDigest = createHash("sha256")
        .update(fullEventsJsonl, "utf8")
        .digest("hex");
    }
    const response = await this.fetchLocal(checkpointUrl, {
      method: "POST",
      headers: {
        ...authorizationHeaders(connection.apiKey),
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
    if (response.status === 409) {
      await this.rotateRefreshRequestId(work.name, work.request.requestId);
      return;
    }
    if (!response.ok) throw new Error("Unable to refresh computer history checkpoint");
    const updated = parseCheckpoint(await boundedResponseJson(response), work.metadata);
    requireMatchingCheckpoint(updated, request);
    await this.commitCheckpoint(work.name, updated, work.request.requestId);
    if (request.finalize) {
      await this.confirmFinalizedCheckpoint(base, connection, work, updated);
    }
  }

  private async confirmFinalizedCheckpoint(
    base: string,
    connection: ComputerHistoryConnection,
    work: IncrementalWork,
    checkpoint: ComputerHistoryCheckpoint,
  ): Promise<void> {
    if (
      checkpoint.status !== "finalized"
      || !checkpoint.observationId
      || !checkpoint.finalEventDigest
      || checkpoint.throughEventCount !== work.metadata.eventCount
      || checkpoint.throughByteOffset !== work.request.throughByteOffset
      || checkpoint.throughSuppressedEventCount !== work.metadata.suppressedEventCount
    ) throw new Error("Final computer history checkpoint does not match the segment");
    const fullEventsJsonl = await readJsonlRange(
      path.join(work.directory, "events.jsonl"),
      0,
      work.request.throughByteOffset,
    );
    const digest = createHash("sha256").update(fullEventsJsonl, "utf8").digest("hex");
    if (checkpoint.finalEventDigest !== digest) {
      throw new Error("Final computer history checkpoint digest does not match");
    }
    const observation = await this.fetchLocal(
      `${base}/${encodeURIComponent(work.metadata.segmentId)}`,
      { headers: authorizationHeaders(connection.apiKey) },
    );
    if (observation.status !== 200) throw new Error("Unable to confirm final history Observation");
    await requireMatchingResponse(observation, {
      schemaVersion: SCHEMA_VERSION,
      segmentId: work.metadata.segmentId,
      startedAt: work.metadata.startedAt,
      endedAt: work.metadata.endedAt,
      eventCount: work.metadata.eventCount,
      suppressedEventCount: work.metadata.suppressedEventCount,
      eventDigest: digest,
      eventsJsonl: "",
    }, checkpoint.observationId);
    const acknowledgement = new URL(
      `${base}/${encodeURIComponent(work.metadata.segmentId)}/checkpoint`,
    );
    acknowledgement.searchParams.set("observationId", checkpoint.observationId);
    acknowledgement.searchParams.set("finalEventDigest", digest);
    const acknowledged = await this.fetchLocal(acknowledgement.toString(), {
      method: "DELETE",
      headers: authorizationHeaders(connection.apiKey),
    });
    if (acknowledged.status !== 204 && acknowledged.status !== 404) {
      throw new Error("Unable to acknowledge final history checkpoint");
    }
    await this.enqueue(() => this.deleteSegmentIfCurrent(work.name, checkpoint));
  }

  private async recoverFinalObservation(
    base: string,
    connection: ComputerHistoryConnection,
    work: IncrementalWork,
  ): Promise<boolean> {
    const observation = await this.fetchLocal(
      `${base}/${encodeURIComponent(work.metadata.segmentId)}`,
      { headers: authorizationHeaders(connection.apiKey) },
    );
    if (observation.status === 404) return false;
    if (observation.status !== 200) {
      throw new Error("Unable to reconcile final history Observation");
    }
    const request = await this.enqueue(async () => {
      const segment = await this.openSegment(
        work.name,
        Date.parse(work.metadata.startedAt),
        Date.parse(work.metadata.endedAt),
      );
      return summarizeRequest(segment);
    });
    await requireMatchingResponse(observation, request);
    await this.enqueue(() => this.deleteSegmentIfCurrent(work.name));
    return true;
  }

  private async submitLegacyWork(work: LegacyWork): Promise<void> {
    const segment = await this.enqueue(() => this.openSegment(
      work.name,
      Date.parse(work.metadata.startedAt),
      Date.parse(work.metadata.endedAt),
    ));
    const request = await summarizeRequest(segment);
    const connection = await this.options.connection();
    if (!connection) throw new Error("Local API unavailable");
    const fishId = await this.resolveAssistantFish(connection);
    const base = `${connection.apiUrl}/internal/fishes/${encodeURIComponent(fishId)}`
      + "/computer-history/segments";
    const existing = await this.fetchLocal(
      `${base}/${encodeURIComponent(request.segmentId)}`,
      { headers: authorizationHeaders(connection.apiKey) },
    );
    if (existing.status === 200) {
      await requireMatchingResponse(existing, request);
      await this.enqueue(() => this.deleteSegmentIfCurrent(work.name));
      return;
    }
    if (existing.status !== 404) throw new Error("Unable to reconcile segment");
    const response = await this.fetchLocal(`${base}/summarize`, {
      method: "POST",
      headers: {
        ...authorizationHeaders(connection.apiKey),
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error("Unable to summarize segment");
    await requireMatchingResponse(response, request);
    await this.enqueue(() => this.deleteSegmentIfCurrent(work.name));
  }

  private async resolveAssistantFish(connection: ComputerHistoryConnection): Promise<string> {
    const response = await this.fetchLocal(`${connection.apiUrl}/v1/fishes/assistant`, {
      headers: authorizationHeaders(connection.apiKey),
    });
    if (!response.ok) throw new Error("Unable to resolve assistant Fish");
    const body = await boundedResponseJson(response);
    if (!isRecord(body) || typeof body.fishId !== "string" || !UUID_PATTERN.test(body.fishId)) {
      throw new Error("Unable to resolve assistant Fish");
    }
    return body.fishId;
  }

  private async pinSegmentFish(name: string, fishId: string): Promise<void> {
    const segment = this.segmentCache.get(name);
    if (!segment) throw new Error("Computer history segment is not open");
    if (segment.metadata.fishId !== null && segment.metadata.fishId !== fishId) {
      throw new Error("Computer history segment is already pinned to another Fish");
    }
    if (segment.metadata.fishId === fishId) return;
    segment.metadata = { ...segment.metadata, fishId };
    await this.writeSegmentMetadata(segment);
  }

  private async fetchLocal(input: string, init: RequestInit): Promise<Response> {
    const controller = new AbortController();
    this.activeRequest = controller;
    const timer = setTimeout(() => controller.abort(), this.requestTimeoutMs);
    timer.unref?.();
    try {
      return await this.fetcher(input, { ...init, signal: controller.signal });
    } finally {
      clearTimeout(timer);
      if (this.activeRequest === controller) this.activeRequest = null;
    }
  }

  private async deleteConfirmedSegment(segment: SegmentState): Promise<void> {
    const deleting = `${segment.directory}.deleting-${segment.metadata.segmentId}`;
    await rename(segment.directory, deleting);
    this.segmentCache.delete(segment.name);
    this.segmentRetries.delete(segment.name);
    await rm(deleting, { recursive: true, force: true });
  }

  private async deleteExpiredDirectory(name: string): Promise<void> {
    const directory = path.join(this.options.rootPath, name);
    const deleting = `${directory}.deleting-${randomUUID()}`;
    await rename(directory, deleting);
    this.segmentCache.delete(name);
    this.segmentRetries.delete(name);
    await rm(deleting, { recursive: true, force: true });
  }

  private async maintainDisabledRetention(): Promise<void> {
    if (this.enabled || this.stopped) return;
    const entries = await readdir(this.options.rootPath, { withFileTypes: true });
    let nextExpiry: number | null = null;
    for (const entry of entries) {
      if (!entry.isDirectory() || !SEGMENT_NAME_PATTERN.test(entry.name)) continue;
      const startMs = Date.parse(entry.name.replace(
        /T(\d{2})-(\d{2})-(\d{2})Z$/u,
        "T$1:$2:$3Z",
      ));
      if (!Number.isFinite(startMs)) continue;
      const expiresAt = startMs + SEGMENT_DURATION_MS + MAX_PENDING_AGE_MS;
      if (this.now() >= expiresAt) await this.deleteExpiredDirectory(entry.name);
      else nextExpiry = nextExpiry === null ? expiresAt : Math.min(nextExpiry, expiresAt);
    }
    if (nextExpiry !== null) this.scheduleWake(nextExpiry - this.now(), true);
  }

  private async recoverDeletingDirectories(): Promise<void> {
    const entries = await readdir(this.options.rootPath, { withFileTypes: true }).catch(
      (error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") return [];
        throw error;
      },
    );
    for (const entry of entries) {
      if (entry.isDirectory() && entry.name.includes(".deleting-")) {
        await rm(path.join(this.options.rootPath, entry.name), {
          recursive: true,
          force: true,
        });
      }
    }
  }

  private async restoreSelfObservationContext(): Promise<void> {
    if (!this.selfObservationRecoveryPending) return;
    const entries = await readdir(this.options.rootPath, { withFileTypes: true });
    const names = entries
      .filter((entry) => entry.isDirectory() && SEGMENT_NAME_PATTERN.test(entry.name))
      .map((entry) => entry.name)
      .sort()
      .reverse();
    const recent: ComputerHistoryInteractionEvent[] = [];
    const oldestAllowed = this.now() - SELF_OBSERVATION_RECOVERY_MAX_AGE_MS;
    scanRecentEvents:
    for (const name of names) {
      const eventsJsonl = await readFile(
        path.join(this.options.rootPath, name, "events.jsonl"),
        "utf8",
      ).catch((error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") return "";
        throw error;
      });
      const lines = eventsJsonl.trimEnd().split("\n").filter(Boolean).reverse();
      for (const line of lines) {
        let interaction: ComputerHistoryInteractionEvent | null = null;
        try {
          interaction = storedInteractionEvent(JSON.parse(line) as unknown);
        } catch {
          // openSegment performs authoritative validation. Recovery is best
          // effort and must not block the core recorder on a partial file.
        }
        if (!interaction) continue;
        if (interaction.timestamp * 1_000 < oldestAllowed) {
          break scanRecentEvents;
        }
        recent.push(interaction);
        if (recent.length >= SELF_OBSERVATION_RECOVERY_EVENT_LIMIT) {
          break scanRecentEvents;
        }
      }
    }
    this.selfObservationGuard.restoreFromRecentEvents(recent);
    this.selfObservationRecoveryPending = false;
  }

  private scheduleWake(delayMs: number, whileDisabled = false): void {
    if (this.stopped || (!this.enabled && !whileDisabled) || this.wakeTimer) return;
    this.wakeTimer = this.schedule(() => {
      this.wakeTimer = null;
      if (this.enabled) this.kickWorker();
      else {
        void this.enqueue(() => this.maintainDisabledRetention()).catch(() => {
          this.options.onErrorCode?.("storage_failed");
        });
      }
    }, Math.max(0, Math.min(delayMs, 2_147_000_000)));
    this.wakeTimer.unref?.();
  }

  private clearWakeTimer(): void {
    if (this.wakeTimer) this.cancelSchedule(this.wakeTimer);
    this.wakeTimer = null;
  }
}

function segmentStart(timestampMs: number): number {
  if (
    !Number.isFinite(timestampMs)
    || timestampMs <= 0
    || timestampMs > MAX_HISTORY_TIMESTAMP_MS
  ) {
    throw new Error("Invalid computer history event timestamp");
  }
  return Math.floor(timestampMs / SEGMENT_DURATION_MS) * SEGMENT_DURATION_MS;
}

function segmentName(startMs: number): string {
  return new Date(startMs).toISOString().replace(/:/gu, "-").replace(".000", "");
}

async function summarizeRequest(segment: SegmentState): Promise<ComputerHistorySummarizeRequest> {
  const eventsJsonl = await readFile(path.join(segment.directory, "events.jsonl"), "utf8");
  const counts = validateEventsJsonl(eventsJsonl);
  if (counts.eventCount !== segment.metadata.eventCount) {
    throw new Error("Computer history event count changed unexpectedly");
  }
  return {
    schemaVersion: SCHEMA_VERSION,
    segmentId: segment.metadata.segmentId,
    startedAt: segment.metadata.startedAt,
    endedAt: segment.metadata.endedAt,
    eventCount: counts.eventCount,
    suppressedEventCount: segment.metadata.suppressedEventCount,
    eventDigest: createHash("sha256").update(eventsJsonl, "utf8").digest("hex"),
    eventsJsonl,
  };
}

function validateEventsJsonl(
  value: string,
): { eventCount: number; meaningfulEventCount: number; eventIds: Set<string> } {
  if (!value) {
    return { eventCount: 0, meaningfulEventCount: 0, eventIds: new Set() };
  }
  if (!value.endsWith("\n")) throw new Error("Computer history JSONL is incomplete");
  const lines = value.slice(0, -1).split("\n");
  const eventIds = new Set<string>();
  let meaningfulEventCount = 0;
  for (let index = 0; index < lines.length; index += 1) {
    const item = JSON.parse(lines[index]!) as unknown;
    if (
      !isRecord(item)
      || item.schemaVersion !== SCHEMA_VERSION
      || item.ordinal !== index + 1
      || typeof item.id !== "string"
      || !UUID_PATTERN.test(item.id)
      || typeof item.timestamp !== "string"
      || !Number.isFinite(Date.parse(item.timestamp))
      || new Date(item.timestamp).toISOString() !== item.timestamp
      || eventIds.has(item.id)
      || Object.keys(item).some((key) => !STORED_EVENT_KEYS.has(key))
    ) throw new Error("Computer history JSONL is invalid");
    const {
      schemaVersion: _schemaVersion,
      ordinal: _ordinal,
      timestamp,
      ...interaction
    } = item;
    parseComputerHistoryInteractionEvent({
      ...interaction,
      timestamp: Date.parse(timestamp as string) / 1_000,
    });
    eventIds.add(item.id);
    if (item.kind !== "session.started") meaningfulEventCount += 1;
  }
  return {
    eventCount: lines.length,
    meaningfulEventCount,
    eventIds,
  };
}

function storedInteractionEvent(value: unknown): ComputerHistoryInteractionEvent | null {
  if (
    !isRecord(value)
    || value.schemaVersion !== SCHEMA_VERSION
    || !Number.isSafeInteger(value.ordinal)
    || Number(value.ordinal) < 1
    || !validIsoDate(value.timestamp)
  ) return null;
  const {
    schemaVersion: _schemaVersion,
    ordinal: _ordinal,
    timestamp,
    ...interaction
  } = value;
  const timestampMs = Date.parse(timestamp);
  if (!Number.isFinite(timestampMs)) return null;
  return parseComputerHistoryInteractionEvent({
    ...interaction,
    timestamp: timestampMs / 1_000,
  });
}

function parseMetadata(value: unknown): {
  metadata: ComputerHistorySegmentMetadata;
  legacyFullSummary: boolean;
} {
  if (!isRecord(value)) throw new Error("Computer history metadata is invalid");
  const baseKeys = [
    "schemaVersion",
    "segmentId",
    "startedAt",
    "endedAt",
    "createdAt",
    "eventCount",
    "meaningfulEventCount",
    "suppressedEventCount",
    "summaryStatus",
  ];
  const isLegacy = Object.keys(value).length === baseKeys.length;
  const previousIncrementalKeys = [...baseKeys, "checkpoint", "refreshRequest"];
  const keys = [...baseKeys, "fishId", "checkpoint", "refreshRequest"];
  const isPreviousIncremental = (
    Object.keys(value).length === previousIncrementalKeys.length
    && previousIncrementalKeys.every((key) => (
      Object.prototype.hasOwnProperty.call(value, key)
    ))
  );
  if (
    (!isLegacy && !isPreviousIncremental && (
      Object.keys(value).length !== keys.length
      || keys.some((key) => !Object.prototype.hasOwnProperty.call(value, key))
    ))
    || (isLegacy && baseKeys.some((key) => !Object.prototype.hasOwnProperty.call(value, key)))
    || value.schemaVersion !== SCHEMA_VERSION
    || typeof value.segmentId !== "string"
    || !UUID_PATTERN.test(value.segmentId)
    || !validIsoDate(value.startedAt)
    || !validIsoDate(value.endedAt)
    || !validIsoDate(value.createdAt)
    || (!isLegacy && !isPreviousIncremental && value.fishId !== null && (
      typeof value.fishId !== "string" || !UUID_PATTERN.test(value.fishId)
    ))
    || !Number.isSafeInteger(value.eventCount)
    || Number(value.eventCount) < 0
    || !Number.isSafeInteger(value.meaningfulEventCount)
    || Number(value.meaningfulEventCount) < 0
    || Number(value.meaningfulEventCount) > Number(value.eventCount)
    || !Number.isSafeInteger(value.suppressedEventCount)
    || Number(value.suppressedEventCount) < 0
    || value.summaryStatus !== "pending"
  ) throw new Error("Computer history metadata is invalid");
  if (isLegacy) {
    return {
      metadata: {
        ...value,
        fishId: null,
        checkpoint: null,
        refreshRequest: null,
      } as ComputerHistorySegmentMetadata,
      legacyFullSummary: true,
    };
  }
  const checkpoint = value.checkpoint === null ? null : parseCheckpoint(
    value.checkpoint,
    value as unknown as Pick<ComputerHistorySegmentMetadata, "segmentId" | "startedAt" | "endedAt">,
  );
  const refreshRequest = value.refreshRequest === null
    ? null
    : parseRefreshRequest(value.refreshRequest);
  return {
    metadata: {
      ...value,
      fishId: isPreviousIncremental ? null : value.fishId,
      checkpoint,
      refreshRequest,
    } as ComputerHistorySegmentMetadata,
    legacyFullSummary: false,
  };
}

function parseRefreshRequest(value: unknown): ComputerHistoryRefreshRequest {
  if (
    !isRecord(value)
    || Object.keys(value).length !== 7
    || typeof value.requestId !== "string"
    || !UUID_PATTERN.test(value.requestId)
    || !validIsoDate(value.requestedAt)
    || !validNonnegativeInteger(value.baseRevision)
    || !validNonnegativeInteger(value.throughEventCount)
    || !validNonnegativeInteger(value.throughByteOffset)
    || !validNonnegativeInteger(value.throughSuppressedEventCount)
    || typeof value.finalize !== "boolean"
  ) throw new Error("Computer history refresh request is invalid");
  return value as unknown as ComputerHistoryRefreshRequest;
}

function parseCheckpoint(
  value: unknown,
  segment: Pick<ComputerHistorySegmentMetadata, "segmentId" | "startedAt" | "endedAt">,
): ComputerHistoryCheckpoint {
  const keys = [
    "schemaVersion",
    "segmentId",
    "startedAt",
    "endedAt",
    "revision",
    "throughEventCount",
    "throughByteOffset",
    "throughSuppressedEventCount",
    "deltaChainDigest",
    "lastSuccessfulSummaryAt",
    "status",
    "summary",
    "applications",
    "finalizedAt",
    "observationId",
    "finalEventDigest",
  ];
  if (
    !isRecord(value)
    || Object.keys(value).length !== keys.length
    || keys.some((key) => !Object.prototype.hasOwnProperty.call(value, key))
    || value.schemaVersion !== SCHEMA_VERSION
    || value.segmentId !== segment.segmentId
    || !sameInstant(value.startedAt, segment.startedAt)
    || !sameInstant(value.endedAt, segment.endedAt)
    || !validPositiveInteger(value.revision)
    || !validNonnegativeInteger(value.throughEventCount)
    || !validNonnegativeInteger(value.throughByteOffset)
    || !validNonnegativeInteger(value.throughSuppressedEventCount)
    || typeof value.deltaChainDigest !== "string"
    || !DIGEST_PATTERN.test(value.deltaChainDigest)
    || (value.lastSuccessfulSummaryAt !== null && !validIsoDate(value.lastSuccessfulSummaryAt))
    || (value.summary !== null && typeof value.summary !== "string")
    || !Array.isArray(value.applications)
    || value.applications.length > 64
    || value.applications.some((app) => typeof app !== "string")
    || (value.status !== "active" && value.status !== "finalized")
    || (value.finalizedAt !== null && !validIsoDate(value.finalizedAt))
    || (value.observationId !== null && (
      typeof value.observationId !== "string" || !UUID_PATTERN.test(value.observationId)
    ))
    || (value.finalEventDigest !== null && (
      typeof value.finalEventDigest !== "string" || !DIGEST_PATTERN.test(value.finalEventDigest)
    ))
  ) throw new Error("Computer history checkpoint is invalid");
  const finalFieldsPresent = value.finalizedAt !== null
    && value.observationId !== null
    && value.finalEventDigest !== null;
  const finalFieldsAbsent = value.finalizedAt === null
    && value.observationId === null
    && value.finalEventDigest === null;
  if (
    (value.status === "finalized" && !finalFieldsPresent)
    || (value.status === "active" && !finalFieldsAbsent)
  ) {
    throw new Error("Computer history checkpoint final state is invalid");
  }
  return value as unknown as ComputerHistoryCheckpoint;
}

function validNonnegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

function validPositiveInteger(value: unknown): value is number {
  return validNonnegativeInteger(value) && Number(value) > 0;
}

function validateMetadataWatermarks(segment: SegmentState): void {
  const { metadata, byteLength } = segment;
  const checkpoint = metadata.checkpoint;
  const request = metadata.refreshRequest;
  if (
    checkpoint
    && (
      checkpoint.throughEventCount > metadata.eventCount
      || checkpoint.throughByteOffset > byteLength
      || checkpoint.throughSuppressedEventCount > metadata.suppressedEventCount
    )
  ) throw new Error("Computer history checkpoint high-water mark is invalid");
  if (
    request
    && (
      request.throughEventCount > metadata.eventCount
      || request.throughByteOffset > byteLength
      || request.throughSuppressedEventCount > metadata.suppressedEventCount
      || (checkpoint && request.throughEventCount < checkpoint.throughEventCount)
      || (checkpoint && request.throughByteOffset < checkpoint.throughByteOffset)
      || (checkpoint && request.throughSuppressedEventCount < checkpoint.throughSuppressedEventCount)
    )
  ) throw new Error("Computer history refresh high-water mark is invalid");
}

function checkpointCoversRequest(
  checkpoint: ComputerHistoryCheckpoint,
  request: ComputerHistoryRefreshRequest,
): boolean {
  return checkpoint.throughEventCount >= request.throughEventCount
    && checkpoint.throughByteOffset >= request.throughByteOffset
    && checkpoint.throughSuppressedEventCount >= request.throughSuppressedEventCount;
}

function requireMatchingCheckpoint(
  checkpoint: ComputerHistoryCheckpoint,
  request: ComputerHistoryCheckpointRequest,
): void {
  if (
    checkpoint.throughEventCount !== request.throughEventCount
    || checkpoint.throughByteOffset !== request.throughByteOffset
    || checkpoint.throughSuppressedEventCount !== request.throughSuppressedEventCount
    || (request.finalize && (
      checkpoint.status !== "finalized"
      || checkpoint.finalEventDigest !== request.finalEventDigest
    ))
  ) throw new Error("Computer history checkpoint response does not match request");
}

async function readJsonlRange(
  source: string,
  fromByteOffset: number,
  throughByteOffset: number,
): Promise<string> {
  if (
    !validNonnegativeInteger(fromByteOffset)
    || !validNonnegativeInteger(throughByteOffset)
    || throughByteOffset < fromByteOffset
  ) throw new Error("Computer history JSONL range is invalid");
  const length = throughByteOffset - fromByteOffset;
  if (length === 0) return "";
  const handle = await open(source, "r");
  try {
    const buffer = Buffer.allocUnsafe(length);
    const { bytesRead } = await handle.read(buffer, 0, length, fromByteOffset);
    if (bytesRead !== length) throw new Error("Computer history JSONL range is incomplete");
    return buffer.toString("utf8");
  } finally {
    await handle.close();
  }
}

function validateEventsJsonlRange(
  value: string,
  fromEventCount: number,
  throughEventCount: number,
): void {
  if (throughEventCount < fromEventCount) throw new Error("Computer history event range is invalid");
  if (throughEventCount === fromEventCount) {
    if (value) throw new Error("Computer history empty event range has content");
    return;
  }
  if (!value.endsWith("\n")) throw new Error("Computer history JSONL range is incomplete");
  const lines = value.slice(0, -1).split("\n");
  if (lines.length !== throughEventCount - fromEventCount) {
    throw new Error("Computer history event range count is invalid");
  }
  for (let index = 0; index < lines.length; index += 1) {
    const item = JSON.parse(lines[index]!) as unknown;
    if (!isRecord(item) || item.ordinal !== fromEventCount + index + 1) {
      throw new Error("Computer history event range ordinal is invalid");
    }
    if (!storedInteractionEvent(item)) throw new Error("Computer history event range is invalid");
  }
}

async function writeMetadataAtomically(
  destination: string,
  metadata: ComputerHistorySegmentMetadata,
): Promise<void> {
  const temporary = `${destination}.${randomUUID()}.tmp`;
  await writeFile(temporary, `${JSON.stringify(metadata)}\n`, { encoding: "utf8", mode: 0o600 });
  await rename(temporary, destination);
}

async function requireMatchingResponse(
  response: Response,
  request: ComputerHistorySummarizeRequest,
  expectedObservationId?: string,
): Promise<void> {
  const body = await boundedResponseJson(response);
  if (
    !isRecord(body)
    || !isRecord(body.payload)
    || typeof body.payload.segmentId !== "string"
    || !UUID_PATTERN.test(body.payload.segmentId)
    || typeof body.payload.eventDigest !== "string"
    || !DIGEST_PATTERN.test(body.payload.eventDigest)
    || body.payload.segmentId !== request.segmentId
    || body.payload.eventDigest !== request.eventDigest
    || body.payload.eventCount !== request.eventCount
    || body.payload.suppressedEventCount !== request.suppressedEventCount
    || !sameInstant(body.startedAt, request.startedAt)
    || !sameInstant(body.endedAt, request.endedAt)
    || (expectedObservationId !== undefined && body.id !== expectedObservationId)
  ) throw new Error("Computer history summary confirmation does not match");
}

function sameInstant(value: unknown, expected: string): boolean {
  return typeof value === "string"
    && Number.isFinite(Date.parse(value))
    && Date.parse(value) === Date.parse(expected);
}

async function boundedResponseJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (Buffer.byteLength(text, "utf8") > MAX_RESPONSE_BYTES) {
    throw new Error("Computer history response is too large");
  }
  return JSON.parse(text) as unknown;
}

function authorizationHeaders(apiKey: string): Record<string, string> {
  return { Authorization: `Bearer ${apiKey}` };
}

function validIsoDate(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const match = UTC_ISO_PATTERN.exec(value);
  if (!match) return false;
  const milliseconds = (match[2] ?? "").padEnd(3, "0").slice(0, 3);
  const normalized = `${match[1]}.${milliseconds}Z`;
  return Number.isFinite(Date.parse(value))
    && new Date(value).toISOString() === normalized;
}

async function pathExists(value: string): Promise<boolean> {
  try {
    await stat(value);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
