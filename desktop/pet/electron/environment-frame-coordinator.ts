import type {
  EnvironmentFrameMetadata,
  EnvironmentFramePayload,
} from "./perception-manager";
import {
  parseEnvironmentFrameMetadata,
  parseEnvironmentFramePayload,
} from "./perception-manager";
import {
  DISABLED_PERCEPTION_POLICY,
  perceptionFramesEnabled,
  perceptionHistoryEnabled,
  type PerceptionPolicy,
} from "./perception-policy";

type Fetcher = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

export interface EnvironmentFrameConnection {
  apiUrl: string;
  apiKey: string;
}

export type EnvironmentFrameCoordinatorErrorCode =
  | "claim_failed"
  | "local_api_failed"
  | "state_failed"
  | "upload_failed";

export interface EnvironmentFrameCoordinatorOptions {
  claim: (
    frameId: string,
    signal: AbortSignal,
  ) => Promise<EnvironmentFramePayload>;
  connection: () => Promise<EnvironmentFrameConnection | null>;
  fetcher?: Fetcher;
  now?: () => number;
  requestTimeoutMs?: number;
  stateRetryDelaysMs?: readonly number[];
  onErrorCode?: (code: EnvironmentFrameCoordinatorErrorCode) => void;
  onTarget?: (target: { apiUrl: string; apiKey: string; fishId: string } | null) => void;
  onTargetPending?: () => void;
  onFrame?: (frame: EnvironmentFramePayload) => void;
}

interface ActiveUpload {
  id: number;
  metadata: EnvironmentFrameMetadata;
  controller: AbortController;
}

interface EnvironmentFrameTarget {
  connection: EnvironmentFrameConnection;
  fishId: string;
}

interface AssistantBinding {
  connection: EnvironmentFrameConnection;
  fishId: string | null;
}

const DEFAULT_REQUEST_TIMEOUT_MS = 2_000;
const DEFAULT_STATE_RETRY_DELAYS_MS = [1_000, 5_000, 30_000] as const;
const MAX_CLOCK_SKEW_MS = 5_000;
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu;

/**
 * Claims native frames exactly once and moves them only to the authenticated
 * local in-memory cache. It never persists, logs, or calls a model provider.
 */
export class EnvironmentFrameCoordinator {
  private readonly fetcher: Fetcher;
  private readonly now: () => number;
  private readonly requestTimeoutMs: number;
  private readonly stateRetryDelaysMs: readonly number[];
  private enabled = false;
  private policy: PerceptionPolicy = { ...DISABLED_PERCEPTION_POLICY };
  private stateSynchronized = false;
  private stateSynchronizationPending = false;
  private target: EnvironmentFrameTarget | null = null;
  private policyTarget: EnvironmentFrameTarget | null = null;
  private clearTarget: EnvironmentFrameTarget | null = null;
  private bindingChangedAtMs = 0;
  private active: ActiveUpload | null = null;
  private nextRunId = 0;
  private stateGeneration = 0;
  private stateOperation: Promise<void> = Promise.resolve();
  private stateRetryTimer: NodeJS.Timeout | null = null;
  private stateRetryAttempt = 0;
  private clearRetryTimer: NodeJS.Timeout | null = null;
  private clearRetryAttempt = 0;
  private cachedPayload: EnvironmentFramePayload | null = null;

  constructor(private readonly options: EnvironmentFrameCoordinatorOptions) {
    this.fetcher = options.fetcher ?? fetch;
    this.now = options.now ?? Date.now;
    this.requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
    this.stateRetryDelaysMs = options.stateRetryDelaysMs
      ?? DEFAULT_STATE_RETRY_DELAYS_MS;
    if (!Number.isFinite(this.requestTimeoutMs) || this.requestTimeoutMs < 1) {
      throw new Error("requestTimeoutMs must be positive");
    }
    if (
      this.stateRetryDelaysMs.length === 0
      || this.stateRetryDelaysMs.some((delay) => !Number.isFinite(delay) || delay < 0)
    ) throw new Error("stateRetryDelaysMs must contain nonnegative delays");
  }

  setEnabled(enabled: boolean): Promise<void> {
    return this.setPolicy({
      operationEnabled: enabled,
      screenEnabled: enabled,
      audioEnabled: false,
    });
  }

  get currentFishId(): string | null {
    return this.target?.fishId ?? this.policyTarget?.fishId ?? null;
  }

  setPolicy(policy: PerceptionPolicy): Promise<void> {
    if (
      samePolicy(policy, this.policy)
      && (this.stateSynchronized || this.stateSynchronizationPending)
    ) return this.stateOperation;
    this.clearStateRetry();
    this.clearClearRetry();
    this.stateRetryAttempt = 0;
    return this.queueStateSynchronization({ ...policy });
  }

  /** Re-read the persisted binding after a change or event-channel reconnect. */
  refreshBinding(): Promise<void> {
    this.clearStateRetry();
    this.clearClearRetry();
    this.stateRetryAttempt = 0;
    this.clearRetryAttempt = 0;
    this.bindingChangedAtMs = this.now();
    return this.queueStateSynchronization({ ...this.policy });
  }

  private queueStateSynchronization(policy: PerceptionPolicy): Promise<void> {
    this.cancel();
    this.cachedPayload = null;
    this.policy = policy;
    this.enabled = perceptionFramesEnabled(policy);
    this.stateSynchronized = false;
    this.stateSynchronizationPending = true;
    this.target = null;
    if (perceptionHistoryEnabled(policy)) this.options.onTargetPending?.();
    else this.options.onTarget?.(null);
    const generation = ++this.stateGeneration;
    const synchronize = async (): Promise<void> => {
      try {
        if (generation !== this.stateGeneration) return;
        const resolved = await this.resolveConnectionAndFish();
        if (!resolved) throw new CoordinatorFailure("state_failed");
        if (generation !== this.stateGeneration) return;
        await this.revokePreviousTarget(resolved);
        if (generation !== this.stateGeneration) return;
        if (resolved.fishId === null) {
          this.stateSynchronized = true;
          this.stateSynchronizationPending = false;
          this.options.onTarget?.(null);
          return;
        }
        const target = { connection: resolved.connection, fishId: resolved.fishId };
        // Track even an uncertain PUT: it may reach the API before a timeout or
        // binding change. The next queued transition must revoke that target.
        this.policyTarget = target;
        const response = await this.fetchLocal(
          stateUrl(resolved.connection, resolved.fishId),
          {
            method: "PUT",
            headers: jsonHeaders(resolved.connection.apiKey),
            body: JSON.stringify(policy),
          },
        );
        if (!response.ok) throw new CoordinatorFailure("state_failed");
        if (generation !== this.stateGeneration) return;
        this.stateSynchronized = true;
        this.stateSynchronizationPending = false;
        this.stateRetryAttempt = 0;
        if (this.enabled) {
          this.target = target;
          this.clearTarget = target;
        } else {
          this.clearTarget = null;
        }
        if (perceptionHistoryEnabled(policy)) {
          this.options.onTarget?.({ ...resolved.connection, fishId: resolved.fishId });
        }
      } catch (error) {
        if (generation === this.stateGeneration) {
          this.report(error, "state_failed");
          this.scheduleStateRetry(policy, generation);
        }
      }
    };
    const queued = this.stateOperation.then(synchronize, synchronize);
    this.stateOperation = queued.then(() => undefined, () => undefined);
    return queued;
  }

  handleFrame(value: unknown): Promise<void> {
    this.cancel();
    if (!this.enabled || !this.stateSynchronized || !this.target) return Promise.resolve();
    let metadata: EnvironmentFrameMetadata;
    try {
      metadata = parseEnvironmentFrameMetadata(value);
    } catch {
      return Promise.resolve();
    }
    if (!isFresh(metadata, this.now())
      || metadata.observedAt * 1_000 < this.bindingChangedAtMs) return Promise.resolve();
    const run: ActiveUpload = {
      id: ++this.nextRunId,
      metadata,
      controller: new AbortController(),
    };
    this.active = run;
    return this.process(run)
      .catch((error: unknown) => this.report(error, "upload_failed", run))
      .finally(() => {
        if (this.active?.id === run.id) this.active = null;
      });
  }

  async clearFrame(): Promise<void> {
    this.clearClearRetry();
    this.clearRetryAttempt = 0;
    await this.clearFrameInternal();
  }

  private async clearFrameInternal(): Promise<void> {
    this.cancel();
    this.cachedPayload = null;
    try {
      const resolved = this.target ?? this.clearTarget;
      if (!resolved) return;
      const response = await this.fetchLocal(
        frameUrl(resolved.connection, resolved.fishId),
        {
          method: "DELETE",
          headers: authorizationHeaders(resolved.connection.apiKey),
        },
      );
      if (!response.ok) throw new CoordinatorFailure("local_api_failed");
      this.clearRetryAttempt = 0;
    } catch (error) {
      this.report(error, "local_api_failed");
      this.scheduleClearRetry();
    }
  }

  cancel(): void {
    const active = this.active;
    this.active = null;
    active?.controller.abort("environment_frame_superseded");
  }

  private scheduleStateRetry(policy: PerceptionPolicy, generation: number): void {
    if (this.stateRetryTimer || generation !== this.stateGeneration) return;
    const delay = this.stateRetryDelaysMs[
      Math.min(this.stateRetryAttempt, this.stateRetryDelaysMs.length - 1)
    ]!;
    this.stateRetryAttempt += 1;
    this.stateRetryTimer = setTimeout(() => {
      this.stateRetryTimer = null;
      if (generation !== this.stateGeneration || policy !== this.policy) return;
      void this.queueStateSynchronization(policy);
    }, delay);
    this.stateRetryTimer.unref?.();
  }

  private clearStateRetry(): void {
    if (this.stateRetryTimer) clearTimeout(this.stateRetryTimer);
    this.stateRetryTimer = null;
  }

  private scheduleClearRetry(): void {
    if (this.clearRetryTimer || !this.clearTarget) return;
    const delay = this.stateRetryDelaysMs[
      Math.min(this.clearRetryAttempt, this.stateRetryDelaysMs.length - 1)
    ]!;
    this.clearRetryAttempt += 1;
    this.clearRetryTimer = setTimeout(() => {
      this.clearRetryTimer = null;
      void this.clearFrameInternal();
    }, delay);
    this.clearRetryTimer.unref?.();
  }

  private clearClearRetry(): void {
    if (this.clearRetryTimer) clearTimeout(this.clearRetryTimer);
    this.clearRetryTimer = null;
  }

  private async process(run: ActiveUpload): Promise<void> {
    let payload: EnvironmentFramePayload;
    const cached = this.cachedPayload;
    const reuse = !!run.metadata.contentHash && cached?.contentHash === run.metadata.contentHash
      && cached.displayID === run.metadata.displayID
      && cached.pixelWidth === run.metadata.pixelWidth && cached.pixelHeight === run.metadata.pixelHeight
      && cached.byteLength === run.metadata.byteLength;
    try {
      payload = reuse && cached ? { ...cached, ...run.metadata } : parseEnvironmentFramePayload(
        await this.options.claim(run.metadata.frameId, run.controller.signal),
      );
    } catch (error) {
      if (run.controller.signal.aborted) throw new CoordinatorCancelled();
      throw new CoordinatorFailure("claim_failed", { cause: error });
    }
    this.requireCurrent(run);
    if (!sameMetadata(payload, run.metadata) || !isFresh(payload, this.now())) {
      return;
    }
    const resolved = this.target;
    if (!resolved) return;
    const { jpegBase64: _jpeg, ...reference } = payload;
    let response = await this.fetchLocal(
      frameUrl(resolved.connection, resolved.fishId),
      {
        method: reuse ? "PATCH" : "PUT",
        headers: jsonHeaders(resolved.connection.apiKey),
        body: JSON.stringify(reuse ? reference : payload),
      },
      run,
    );
    this.requireCurrent(run);
    if (reuse && response.status === 409) {
      response = await this.fetchLocal(frameUrl(resolved.connection, resolved.fishId), {
        method: "PUT", headers: jsonHeaders(resolved.connection.apiKey), body: JSON.stringify(payload),
      }, run);
      this.requireCurrent(run);
    }
    if (!response.ok) throw new CoordinatorFailure("upload_failed");
    this.cachedPayload = payload;
    if (perceptionHistoryEnabled(this.policy)) this.options.onFrame?.(payload);
    this.clearClearRetry();
  }

  private async revokePreviousTarget(binding: AssistantBinding): Promise<void> {
    const previous = this.policyTarget;
    if (!previous || (previous.connection.apiUrl === binding.connection.apiUrl
      && previous.fishId === binding.fishId)) return;
    // A restarted local API owns a fresh cache. Never wait on its dead predecessor.
    if (previous.connection.apiUrl === binding.connection.apiUrl) {
      const response = await this.fetchLocal(
        stateUrl(binding.connection, previous.fishId),
        {
          method: "PUT",
          headers: jsonHeaders(binding.connection.apiKey),
          body: JSON.stringify(DISABLED_PERCEPTION_POLICY),
        },
      );
      // Deleting the old Fish already removes its ability to receive frames.
      if (!response.ok && response.status !== 404) throw new CoordinatorFailure("state_failed");
    }
    this.policyTarget = null;
    this.clearTarget = null;
  }

  private async resolveConnectionAndFish(): Promise<AssistantBinding | null> {
    const connection = await this.options.connection();
    if (!connection) return null;
    const response = await this.fetchLocal(
      `${connection.apiUrl}/v1/fishes/assistant`,
      { headers: authorizationHeaders(connection.apiKey) },
    );
    if (!response.ok) throw new CoordinatorFailure("local_api_failed");
    let body: unknown;
    try {
      body = await response.json() as unknown;
    } catch {
      throw new CoordinatorFailure("local_api_failed");
    }
    if (!isRecord(body) || (body.fishId !== null
      && (typeof body.fishId !== "string" || !UUID_PATTERN.test(body.fishId)))) {
      throw new CoordinatorFailure("local_api_failed");
    }
    return { connection, fishId: body.fishId };
  }

  private async fetchLocal(
    input: string,
    init: RequestInit,
    run?: ActiveUpload,
  ): Promise<Response> {
    if (run) this.requireCurrent(run);
    const controller = new AbortController();
    const cancel = () => controller.abort();
    run?.controller.signal.addEventListener("abort", cancel, { once: true });
    const timer = setTimeout(cancel, this.requestTimeoutMs);
    timer.unref?.();
    try {
      return await this.fetcher(input, { ...init, signal: controller.signal });
    } catch {
      if (run?.controller.signal.aborted) throw new CoordinatorCancelled();
      throw new CoordinatorFailure("local_api_failed");
    } finally {
      clearTimeout(timer);
      run?.controller.signal.removeEventListener("abort", cancel);
    }
  }

  private requireCurrent(run: ActiveUpload): void {
    if (
      !this.enabled
      || !this.stateSynchronized
      || this.active?.id !== run.id
      || run.controller.signal.aborted
    ) {
      throw new CoordinatorCancelled();
    }
    if (!isFresh(run.metadata, this.now())) throw new CoordinatorCancelled();
  }

  private report(
    error: unknown,
    fallback: EnvironmentFrameCoordinatorErrorCode,
    run?: ActiveUpload,
  ): void {
    if (error instanceof CoordinatorCancelled || run?.controller.signal.aborted) return;
    if (error instanceof CoordinatorFailure) {
      this.options.onErrorCode?.(error.code);
      return;
    }
    this.options.onErrorCode?.(fallback);
  }
}

class CoordinatorFailure extends Error {
  constructor(
    readonly code: EnvironmentFrameCoordinatorErrorCode,
    options?: ErrorOptions,
  ) {
    super(code, options);
  }
}

class CoordinatorCancelled extends Error {}

function isFresh(metadata: EnvironmentFrameMetadata, nowMs: number): boolean {
  return metadata.expiresAt * 1_000 > nowMs
    && metadata.observedAt * 1_000 <= nowMs + MAX_CLOCK_SKEW_MS;
}

function sameMetadata(
  payload: EnvironmentFramePayload,
  metadata: EnvironmentFrameMetadata,
): boolean {
  return payload.frameId === metadata.frameId
    && payload.revision === metadata.revision
    && payload.observedAt === metadata.observedAt
    && payload.expiresAt === metadata.expiresAt
    && payload.displayID === metadata.displayID
    && payload.pixelWidth === metadata.pixelWidth
    && payload.pixelHeight === metadata.pixelHeight
    && payload.byteLength === metadata.byteLength
    && payload.contentHash === metadata.contentHash;
}

function samePolicy(left: PerceptionPolicy, right: PerceptionPolicy): boolean {
  return left.operationEnabled === right.operationEnabled
    && left.screenEnabled === right.screenEnabled
    && left.audioEnabled === right.audioEnabled
    && (left.audioSupported ?? true) === (right.audioSupported ?? true);
}

function stateUrl(connection: EnvironmentFrameConnection, fishId: string): string {
  return `${connection.apiUrl}/internal/fishes/${encodeURIComponent(fishId)}`
    + "/environment-perception/state";
}

function frameUrl(connection: EnvironmentFrameConnection, fishId: string): string {
  return `${connection.apiUrl}/internal/fishes/${encodeURIComponent(fishId)}`
    + "/environment-perception/frame";
}

function authorizationHeaders(apiKey: string): Record<string, string> {
  return { Authorization: `Bearer ${apiKey}` };
}

function jsonHeaders(apiKey: string): Record<string, string> {
  return {
    ...authorizationHeaders(apiKey),
    "Content-Type": "application/json",
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
