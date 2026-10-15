import { createHash } from "node:crypto";

import type {
  VisionCandidateMetadata,
  VisionCandidatePayload,
} from "./perception-manager";
import {
  isSafeVisionBaseUrl,
  SparseVisionClient,
  SparseVisionClientError,
} from "./sparse-vision-client";
import { redactSensitiveText } from "./sensitive-text";

type Fetcher = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

export interface SparseVisionRuntimeConfiguration {
  enabled: boolean;
  baseUrl: string;
  model: string | null;
  apiKey: string;
}

export interface SparseVisionLocalConnection {
  apiUrl: string;
  apiKey: string;
}

export type SparseVisionCoordinatorErrorCode =
  | "claim_failed"
  | "configuration_invalid"
  | "local_api_failed"
  | "persist_failed"
  | "provider_failed";

export interface SparseVisionCoordinatorOptions {
  claim: (
    candidateId: string,
    signal: AbortSignal,
  ) => Promise<VisionCandidatePayload>;
  configuration: () => Promise<SparseVisionRuntimeConfiguration>;
  connection: () => Promise<SparseVisionLocalConnection | null>;
  fetcher?: Fetcher;
  now?: () => number;
  localRequestTimeoutMs?: number;
  onErrorCode?: (code: SparseVisionCoordinatorErrorCode) => void;
}

interface ActiveRun {
  id: number;
  metadata: VisionCandidateMetadata;
  controller: AbortController;
}

const CANDIDATE_TTL_MS = 5_000;
const RESULT_TTL_MS = 12_000;
const DEFAULT_LOCAL_REQUEST_TIMEOUT_MS = 2_000;
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu;
const APP_BUNDLE_ID_PATTERN = /^[a-z0-9][a-z0-9.-]{0,254}$/u;

/**
 * One-candidate coordinator. It never owns capture bytes beyond the duration
 * of one explicitly enabled provider request, and it only persists a bounded
 * text result after local policy and freshness checks pass.
 */
export class SparseVisionCoordinator {
  private readonly fetcher: Fetcher;
  private readonly now: () => number;
  private readonly localRequestTimeoutMs: number;
  private active: ActiveRun | null = null;
  private nextRunId = 0;

  constructor(private readonly options: SparseVisionCoordinatorOptions) {
    this.fetcher = options.fetcher ?? fetch;
    this.now = options.now ?? Date.now;
    this.localRequestTimeoutMs = options.localRequestTimeoutMs
      ?? DEFAULT_LOCAL_REQUEST_TIMEOUT_MS;
    if (!Number.isFinite(this.localRequestTimeoutMs) || this.localRequestTimeoutMs < 1) {
      throw new Error("localRequestTimeoutMs must be positive");
    }
  }

  handleCandidate(value: unknown): Promise<void> {
    this.cancel();
    const metadata = parseCandidateMetadata(value);
    if (!metadata || !isFreshCandidate(metadata, this.now())) return Promise.resolve();
    const run: ActiveRun = {
      id: ++this.nextRunId,
      metadata,
      controller: new AbortController(),
    };
    this.active = run;
    return this.process(run)
      .catch((error: unknown) => this.reportFailure(error, run))
      .finally(() => {
        if (this.active?.id === run.id) this.active = null;
      });
  }

  /** Workspace or Accessibility changes revoke an in-flight image result. */
  handleContextObservation(value: unknown): void {
    if (!isRecord(value)) return;
    const source = normalizedText(value.source);
    if (source === "accessibility" || source === "workspace") {
      this.cancel();
      return;
    }
    if (
      source === null
      && normalizedText(value.kind)
      && Object.prototype.hasOwnProperty.call(value, "frontmostApplication")
    ) {
      this.cancel();
    }
  }

  cancel(): void {
    const active = this.active;
    this.active = null;
    active?.controller.abort("vision_context_changed");
  }

  private async process(run: ActiveRun): Promise<void> {
    const configuration = await this.options.configuration();
    this.requireCurrent(run);
    if (!isEnabledConfiguration(configuration)) return;
    const connection = await this.options.connection();
    this.requireCurrent(run);
    if (!connection) return;
    const fishId = await this.resolveAssistantFish(connection, run);
    if (!fishId) return;
    if (!await this.preflight(connection, fishId, run)) return;
    this.requireFreshCandidate(run);

    const payload = await this.options.claim(
      run.metadata.candidateId,
      run.controller.signal,
    );
    this.requireCurrent(run);
    if (!sameCandidate(payload, run.metadata)) return;
    const fishAfterClaim = await this.resolveAssistantFish(connection, run);
    if (!fishAfterClaim || fishAfterClaim !== fishId) return;
    // Re-check immediately before the only network request containing pixels.
    // The first check prevents needless claims; this one closes a policy-change
    // race while the candidate was being claimed from the local helper.
    if (!await this.preflight(connection, fishId, run)) return;
    this.requireCurrent(run);
    this.requireFreshCandidate(run);

    const client = new SparseVisionClient({
      enabled: true,
      baseUrl: configuration.baseUrl,
      model: configuration.model,
      apiKey: configuration.apiKey,
      fetcher: this.fetcher,
    });
    const result = await client.analyze({
      imageDataUrl: `data:image/jpeg;base64,${payload.jpegBase64}`,
      signal: run.controller.signal,
    });
    this.requireCurrent(run);
    if (!result || this.now() > run.metadata.observedAt * 1_000 + RESULT_TTL_MS) return;
    const fishBeforePersist = await this.resolveAssistantFish(connection, run);
    if (!fishBeforePersist || fishBeforePersist !== fishId) return;
    this.requireCurrent(run);
    if (this.now() > run.metadata.observedAt * 1_000 + RESULT_TTL_MS) return;

    const summary = redactSensitiveText(result.summary, 400);
    if (!summary) return;
    const semanticDigest = digest(`${run.metadata.appBundleId}\n${summary}`);
    const candidateKey = digest(run.metadata.candidateId);
    const response = await this.fetchLocal(
      `${connection.apiUrl}/internal/fishes/${encodeURIComponent(fishId)}/observations/states`,
      {
        method: "POST",
        headers: localHeaders(connection.apiKey),
        body: JSON.stringify({
          source: "vision",
          observationType: "screen_context",
          scopeKey: "frontmost-window",
          semanticKey: `${run.metadata.appBundleId}:${semanticDigest}`,
          summary,
          payload: {
            revision: run.metadata.revision,
            displayID: run.metadata.displayID,
            salience: result.salience,
            candidateKey,
          },
          appBundleId: run.metadata.appBundleId,
          ...(run.metadata.appName ? { appName: run.metadata.appName } : {}),
          observedAt: new Date(run.metadata.observedAt * 1_000).toISOString(),
        }),
      },
      run,
    );
    if (!response.ok && response.status !== 204) {
      throw new CoordinatorFailure("persist_failed");
    }
  }

  private async resolveAssistantFish(
    connection: SparseVisionLocalConnection,
    run: ActiveRun,
  ): Promise<string | null> {
    const response = await this.fetchLocal(
      `${connection.apiUrl}/v1/fishes/assistant`,
      { headers: { Authorization: `Bearer ${connection.apiKey}` } },
      run,
    );
    if (!response.ok) throw new CoordinatorFailure("local_api_failed");
    const body = await response.json() as unknown;
    if (!isRecord(body) || !isUuid(body.fishId)) {
      throw new CoordinatorFailure("local_api_failed");
    }
    return body.fishId;
  }

  private async preflight(
    connection: SparseVisionLocalConnection,
    fishId: string,
    run: ActiveRun,
  ): Promise<boolean> {
    const url = new URL(
      `${connection.apiUrl}/internal/fishes/${encodeURIComponent(fishId)}`
      + "/observation-app-policies/preflight",
    );
    url.searchParams.set("appBundleId", run.metadata.appBundleId);
    const response = await this.fetchLocal(
      url,
      { headers: { Authorization: `Bearer ${connection.apiKey}` } },
      run,
    );
    if (!response.ok) return false;
    let body: unknown;
    try {
      body = await response.json() as unknown;
    } catch {
      return false;
    }
    return isRecord(body)
      && Object.keys(body).length === 1
      && body.allowed === true;
  }

  private async fetchLocal(
    input: string | URL,
    init: RequestInit,
    run: ActiveRun,
  ): Promise<Response> {
    this.requireCurrent(run);
    const controller = new AbortController();
    const cancel = () => controller.abort();
    run.controller.signal.addEventListener("abort", cancel, { once: true });
    const timer = setTimeout(cancel, this.localRequestTimeoutMs);
    timer.unref?.();
    try {
      return await this.fetcher(input, { ...init, signal: controller.signal });
    } catch (error) {
      if (run.controller.signal.aborted) throw new CoordinatorCancelled();
      throw new CoordinatorFailure("local_api_failed");
    } finally {
      clearTimeout(timer);
      run.controller.signal.removeEventListener("abort", cancel);
    }
  }

  private requireFreshCandidate(run: ActiveRun): void {
    this.requireCurrent(run);
    if (!isFreshCandidate(run.metadata, this.now())) throw new CoordinatorCancelled();
  }

  private requireCurrent(run: ActiveRun): void {
    if (this.active?.id !== run.id || run.controller.signal.aborted) {
      throw new CoordinatorCancelled();
    }
  }

  private reportFailure(error: unknown, run: ActiveRun): void {
    if (error instanceof CoordinatorCancelled || run.controller.signal.aborted) return;
    if (error instanceof CoordinatorFailure) {
      this.options.onErrorCode?.(error.code);
      return;
    }
    if (error instanceof SparseVisionClientError) {
      if (error.code !== "aborted") this.options.onErrorCode?.("provider_failed");
      return;
    }
    this.options.onErrorCode?.("claim_failed");
  }
}

class CoordinatorCancelled extends Error {}

class CoordinatorFailure extends Error {
  constructor(readonly code: SparseVisionCoordinatorErrorCode) {
    super(code);
  }
}

function parseCandidateMetadata(value: unknown): VisionCandidateMetadata | null {
  if (!isRecord(value)) return null;
  const allowedKeys = new Set([
    "candidateId",
    "revision",
    "observedAt",
    "expiresAt",
    "displayID",
    "appBundleId",
    "appName",
    "windowKey",
  ]);
  if (Object.keys(value).some((key) => !allowedKeys.has(key))) return null;
  if (
    !isUuid(value.candidateId)
    || !isUuid(value.windowKey)
    || !Number.isSafeInteger(value.revision)
    || Number(value.revision) < 1
    || !finitePositive(value.observedAt)
    || !finitePositive(value.expiresAt)
    || value.expiresAt <= value.observedAt
    || value.expiresAt * 1_000 - value.observedAt * 1_000 > CANDIDATE_TTL_MS + 1
    || !Number.isSafeInteger(value.displayID)
    || Number(value.displayID) < 0
    || typeof value.appBundleId !== "string"
    || !APP_BUNDLE_ID_PATTERN.test(value.appBundleId)
  ) return null;
  const appName = optionalSingleLine(value.appName, 200);
  if (appName === false) return null;
  return {
    candidateId: value.candidateId,
    revision: Number(value.revision),
    observedAt: value.observedAt,
    expiresAt: value.expiresAt,
    displayID: Number(value.displayID),
    appBundleId: value.appBundleId,
    ...(typeof appName === "string" ? { appName } : {}),
    windowKey: value.windowKey,
  };
}

function isFreshCandidate(metadata: VisionCandidateMetadata, nowMs: number): boolean {
  return Number.isFinite(nowMs)
    && metadata.observedAt * 1_000 <= nowMs + 1_000
    && metadata.expiresAt * 1_000 > nowMs
    && metadata.expiresAt * 1_000 - metadata.observedAt * 1_000 <= CANDIDATE_TTL_MS + 1;
}

function sameCandidate(
  payload: VisionCandidatePayload,
  metadata: VisionCandidateMetadata,
): boolean {
  return payload.candidateId === metadata.candidateId
    && payload.revision === metadata.revision
    && payload.observedAt === metadata.observedAt
    && payload.expiresAt === metadata.expiresAt
    && payload.displayID === metadata.displayID
    && payload.appBundleId === metadata.appBundleId
    && payload.appName === metadata.appName
    && payload.windowKey === metadata.windowKey
    && payload.mimeType === "image/jpeg";
}

function isEnabledConfiguration(
  value: SparseVisionRuntimeConfiguration,
): value is SparseVisionRuntimeConfiguration & { model: string } {
  if (!value.enabled || typeof value.model !== "string" || !value.model.trim()) return false;
  if (typeof value.apiKey !== "string" || !value.apiKey.trim()) return false;
  return isSafeVisionBaseUrl(value.baseUrl);
}

function localHeaders(apiKey: string): Record<string, string> {
  return {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
  };
}

function digest(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex").slice(0, 24);
}

function singleLine(value: unknown, maximum: number): string | null {
  if (typeof value !== "string") return null;
  const normalized = value.split(/\s+/u).join(" ").trim();
  return normalized ? [...normalized].slice(0, maximum).join("") : null;
}

function optionalSingleLine(value: unknown, maximum: number): string | null | false {
  if (value === undefined) return null;
  if (typeof value !== "string" || value !== value.trim()) return false;
  const normalized = singleLine(value, maximum);
  return normalized && [...value].length <= maximum ? normalized : false;
}

function finitePositive(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function isUuid(value: unknown): value is string {
  return typeof value === "string" && UUID_PATTERN.test(value);
}

function normalizedText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
