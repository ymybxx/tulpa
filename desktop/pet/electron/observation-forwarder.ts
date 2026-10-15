import { createHash } from "node:crypto";

import { redactSensitiveText } from "./sensitive-text";

export interface ObservationRuntimeConnection {
  apiUrl: string;
  apiKey: string;
}

export interface ObservationProactivityConfiguration {
  enabled: boolean;
  profile: "quiet" | "balanced" | "lively";
}

type Fetcher = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

interface ForwardableObservation {
  source: string;
  laneKey: string;
  semanticKey: string;
  revision: number;
  observedAtMs: number;
  body: Record<string, unknown>;
}

export interface ObservationForwarderOptions {
  connection: () => Promise<ObservationRuntimeConnection | null>;
  fetcher?: Fetcher;
  now?: () => number;
  requestTimeoutMs?: number;
  onError?: (error: Error) => void;
}

const OBSERVATION_HEARTBEAT_MS: Readonly<Record<string, number>> = Object.freeze({
  accessibility: 2_000,
  camera: 5_000,
  screen: 30_000,
  workspace: 30_000,
});
const DEFAULT_REQUEST_TIMEOUT_MS = 3_000;

/** Persists only latest structured observations; native image buffers never enter this class. */
export class ObservationForwarder {
  private readonly fetcher: Fetcher;
  private readonly now: () => number;
  private readonly requestTimeoutMs: number;
  private readonly pending = new Map<string, ForwardableObservation>();
  private readonly active = new Map<string, ForwardableObservation>();
  private readonly lastForwarded = new Map<
    string,
    { semanticKey: string; forwardedAt: number }
  >();
  private drainPromise: Promise<void> | null = null;
  private cachedAssistant: { apiUrl: string; fishId: string } | null = null;
  private desiredProactivity: ObservationProactivityConfiguration | null = null;
  private syncedProactivity: string | null = null;
  private proactivitySyncPromise: Promise<void> | null = null;

  constructor(private readonly options: ObservationForwarderOptions) {
    this.fetcher = options.fetcher ?? fetch;
    this.now = options.now ?? Date.now;
    this.requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
    if (this.requestTimeoutMs < 1) throw new Error("requestTimeoutMs must be positive");
  }

  publish(value: unknown): Promise<void> {
    const observation = parseObservation(value);
    if (!observation) return Promise.resolve();
    const previous = this.pending.get(observation.laneKey)
      ?? this.active.get(observation.laneKey);
    if (!previous || isSameOrNewer(observation, previous)) {
      this.pending.set(observation.laneKey, observation);
    }
    if (!this.drainPromise) {
      this.drainPromise = this.drain().finally(() => {
        this.drainPromise = null;
      });
    }
    return this.drainPromise;
  }

  clearConnectionCache(): void {
    this.cachedAssistant = null;
    this.lastForwarded.clear();
    this.syncedProactivity = null;
  }

  configureProactivity(configuration: ObservationProactivityConfiguration): Promise<void> {
    this.desiredProactivity = { ...configuration };
    if (this.syncedProactivity !== proactivityFingerprint(configuration)) {
      this.syncedProactivity = null;
    }
    return this.syncProactivity();
  }

  retryProactivityConfiguration(): Promise<void> {
    return this.syncProactivity();
  }

  private async drain(): Promise<void> {
    while (this.pending.size > 0) {
      const observations = [...this.pending.values()];
      this.pending.clear();
      for (const observation of observations) {
        const newerPending = this.pending.get(observation.laneKey);
        if (newerPending && isSameOrNewer(newerPending, observation)) continue;
        this.active.set(observation.laneKey, observation);
        try {
          if (!this.shouldForward(observation)) continue;
          if (await this.forward(observation)) {
            this.lastForwarded.set(observation.laneKey, {
              semanticKey: observation.semanticKey,
              forwardedAt: this.now(),
            });
          }
        } catch (error) {
          this.options.onError?.(error instanceof Error ? error : new Error(String(error)));
        } finally {
          if (this.active.get(observation.laneKey) === observation) {
            this.active.delete(observation.laneKey);
          }
        }
      }
    }
  }

  private shouldForward(observation: ForwardableObservation): boolean {
    const previous = this.lastForwarded.get(observation.laneKey);
    if (!previous || previous.semanticKey !== observation.semanticKey) return true;
    const elapsed = this.now() - previous.forwardedAt;
    return elapsed < 0
      || elapsed >= (OBSERVATION_HEARTBEAT_MS[observation.source] ?? 30_000);
  }

  private async forward(observation: ForwardableObservation): Promise<boolean> {
    const connection = await this.options.connection();
    if (!connection) return false;
    const fishId = await this.resolveAssistantFish(connection);
    if (!fishId) return false;
    await this.syncProactivity();
    const response = await this.fetchWithTimeout(
      `${connection.apiUrl}/internal/fishes/${encodeURIComponent(fishId)}/observations/states`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${connection.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(observation.body),
      },
    );
    if (!response.ok && response.status !== 204) {
      throw new Error(`Observation API rejected ${observation.source} state (${response.status})`);
    }
    return true;
  }

  private async resolveAssistantFish(
    connection: ObservationRuntimeConnection,
  ): Promise<string | null> {
    if (this.cachedAssistant?.apiUrl === connection.apiUrl) {
      return this.cachedAssistant.fishId;
    }
    const response = await this.fetchWithTimeout(`${connection.apiUrl}/v1/fishes/assistant`, {
      headers: { Authorization: `Bearer ${connection.apiKey}` },
    });
    if (!response.ok) throw new Error(`Unable to resolve assistant Fish (${response.status})`);
    const body = await response.json() as { fishId?: unknown };
    if (typeof body.fishId !== "string" || !body.fishId) return null;
    this.cachedAssistant = { apiUrl: connection.apiUrl, fishId: body.fishId };
    return body.fishId;
  }

  private syncProactivity(): Promise<void> {
    if (!this.proactivitySyncPromise) {
      this.proactivitySyncPromise = this.drainProactivityConfiguration()
        .catch((error) => {
          this.options.onError?.(
            error instanceof Error ? error : new Error(String(error)),
          );
        })
        .finally(() => {
          this.proactivitySyncPromise = null;
        });
    }
    return this.proactivitySyncPromise;
  }

  private async drainProactivityConfiguration(): Promise<void> {
    while (this.desiredProactivity) {
      if (
        this.syncedProactivity === proactivityFingerprint(this.desiredProactivity)
      ) return;
      const desired = { ...this.desiredProactivity };
      const connection = await this.options.connection();
      if (!connection) return;
      const fishId = await this.resolveAssistantFish(connection);
      if (!fishId) return;
      await this.postProactivityConfiguration(connection, fishId, desired);
      if (
        this.desiredProactivity
        && proactivityFingerprint(desired) === proactivityFingerprint(
          this.desiredProactivity,
        )
      ) {
        this.syncedProactivity = proactivityFingerprint(desired);
      }
    }
  }

  private async postProactivityConfiguration(
    connection: ObservationRuntimeConnection,
    fishId: string,
    configuration: ObservationProactivityConfiguration,
  ): Promise<void> {
    const response = await this.fetchWithTimeout(
      `${connection.apiUrl}/internal/fishes/${encodeURIComponent(fishId)}`
        + "/observations/proactivity",
      {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${connection.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(configuration),
      },
    );
    if (!response.ok && response.status !== 204) {
      throw new Error(`Observation proactivity configuration failed (${response.status})`);
    }
  }

  private async fetchWithTimeout(
    input: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.requestTimeoutMs);
    timer.unref?.();
    try {
      return await this.fetcher(input, { ...init, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }
}

function parseObservation(value: unknown): ForwardableObservation | null {
  if (!isRecord(value)) return null;
  const source = normalizedText(value.source);
  const evaluation = isRecord(value.evaluation) ? value.evaluation : null;
  const revision = positiveInteger(value.revision)
    ?? (source === "screen" ? positiveInteger(evaluation?.revision) : null);
  const observedAtSeconds = finiteNumber(value.observedAt);
  const observedAt = observedAtSeconds === null ? null : isoTimestamp(observedAtSeconds);
  if (revision === null || observedAt === null) return null;
  if (source === "accessibility") return accessibilityObservation(value, revision, observedAt);
  if (source === "camera") return cameraObservation(value, revision, observedAt);
  if (source === "screen") {
    return normalizedText(value.kind) === "screen_text"
      ? screenTextObservation(value, revision, observedAt)
      : screenObservation(value, revision, observedAt);
  }
  if (source === "workspace") return workspaceObservation(value, revision, observedAt);
  if (
    source === null
    && normalizedText(value.kind)
    && Object.prototype.hasOwnProperty.call(value, "frontmostApplication")
  ) {
    return workspaceObservation(value, revision, observedAt);
  }
  return null;
}

function screenTextObservation(
  value: Record<string, unknown>,
  revision: number,
  observedAt: string,
): ForwardableObservation | null {
  if (containsRawMedia(value)) return null;
  const displayID = nonNegativeInteger(value.displayID);
  const appBundleId = normalizedBundleIdentifier(value.appBundleId);
  const appName = boundedSingleLineText(value.appName, 200);
  const nativeWindowKey = normalizedUuid(value.windowKey);
  const rawText = boundedSingleLineText(value.text, 2_000);
  const text = rawText ? redactSensitiveText(rawText, 1_800) : null;
  if (displayID === null || !appBundleId || !nativeWindowKey || !text) return null;

  const windowKey = stableDigest(`${appBundleId}\n${nativeWindowKey}`);
  const contentKey = stableDigest(`${appBundleId}\n${text}`);
  const scopeKey = "frontmost-window";
  return {
    source: "screen",
    laneKey: `screen:text:${scopeKey}`,
    semanticKey: `${appBundleId}:${contentKey}`,
    revision,
    observedAtMs: Date.parse(observedAt),
    body: {
      source: "screen",
      observationType: "screen_context",
      scopeKey,
      semanticKey: `${appBundleId}:${contentKey}`,
      summary: `当前窗口文字：${text}`,
      payload: {
        revision,
        displayID,
        windowKey,
      },
      appBundleId,
      ...(appName ? { appName } : {}),
      observedAt,
    },
  };
}

function workspaceObservation(
  value: Record<string, unknown>,
  revision: number,
  observedAt: string,
): ForwardableObservation | null {
  const kind = normalizedText(value.kind);
  if (!kind) return null;
  const application = isRecord(value.frontmostApplication)
    ? value.frontmostApplication
    : null;
  const processIdentifier = application
    ? nonNegativeInteger(application.processIdentifier)
    : null;
  if (application && processIdentifier === null) return null;
  const bundleIdentifier = boundedText(application?.bundleIdentifier, 255)?.toLowerCase() ?? null;
  const appName = boundedText(application?.name, 200);
  const semanticKey = bundleIdentifier
    ?? (processIdentifier !== null ? `pid-${processIdentifier}` : "no-frontmost-app");
  const scopeKey = "active-app";
  return {
    source: "workspace",
    laneKey: `workspace:${scopeKey}`,
    semanticKey,
    revision,
    observedAtMs: Date.parse(observedAt),
    body: {
      source: "workspace",
      observationType: "activity",
      scopeKey,
      semanticKey,
      summary: appName ? `正在使用 ${appName}` : "当前没有前台应用",
      payload: {
        workspaceEvent: kind,
        revision,
        ...(processIdentifier !== null ? { processIdentifier } : {}),
      },
      ...(bundleIdentifier ? { appBundleId: bundleIdentifier } : {}),
      ...(appName ? { appName } : {}),
      observedAt,
    },
  };
}

function accessibilityObservation(
  value: Record<string, unknown>,
  revision: number,
  observedAt: string,
): ForwardableObservation | null {
  const kind = normalizedText(value.kind);
  const processIdentifier = nonNegativeInteger(value.processIdentifier);
  if (!kind || processIdentifier === null || typeof value.secureField !== "boolean") return null;
  const bundleIdentifier = boundedText(value.bundleIdentifier, 255)?.toLowerCase() ?? null;
  const appName = boundedText(value.applicationName, 200);
  const windowTitle = redactedBoundedText(value.windowTitle, 512);
  const focusedRole = boundedText(value.focusedRole, 128);
  const secureField = value.secureField;
  const focusedTitle = secureField ? null : redactedBoundedText(value.focusedTitle, 512);
  const focusedValue = secureField ? null : redactedBoundedText(value.focusedValue, 512);
  const semanticKey = boundedText(
    [bundleIdentifier ?? `pid-${processIdentifier}`, windowTitle, focusedRole]
      .filter(Boolean)
      .join(":"),
    512,
  ) ?? `pid-${processIdentifier}`;
  const visibleContext = secureField ? "安全输入界面" : focusedValue ?? focusedTitle;
  const scopeKey = "focused-ui";
  return {
    source: "accessibility",
    laneKey: `accessibility:${scopeKey}`,
    semanticKey,
    revision,
    observedAtMs: Date.parse(observedAt),
    body: {
      source: "accessibility",
      observationType: "application_context",
      scopeKey,
      semanticKey,
      summary: boundedText(
        [appName, windowTitle, visibleContext]
          .filter(Boolean)
          .join(" · ") || "当前应用界面发生变化",
        2_000,
      ),
      payload: {
        accessibilityEvent: kind,
        revision,
        processIdentifier,
        secureField,
        ...(focusedRole ? { focusedRole } : {}),
        ...(focusedTitle ? { focusedTitle } : {}),
        ...(focusedValue ? { focusedValue } : {}),
      },
      ...(bundleIdentifier ? { appBundleId: bundleIdentifier } : {}),
      ...(appName ? { appName } : {}),
      observedAt,
    },
  };
}

function cameraObservation(
  value: Record<string, unknown>,
  revision: number,
  observedAt: string,
): ForwardableObservation | null {
  if (typeof value.facePresent !== "boolean") return null;
  const faceCount = nonNegativeInteger(value.faceCount);
  if (faceCount === null) return null;
  const face = isRecord(value.primaryFace) ? value.primaryFace : null;
  const attention = value.facePresent ? unitNumber(face?.attentionProxy) : null;
  const fatigue = value.facePresent ? unitNumber(face?.fatigueProxy) : null;
  const smile = value.facePresent ? unitNumber(face?.smileProxy) : null;
  const attentionLevel = bucket(attention);
  const fatigueLevel = bucket(fatigue);
  const smileLevel = bucket(smile);
  const semanticKey = value.facePresent
    ? `present:${attentionLevel}:${fatigueLevel}:${smileLevel}`
    : "absent";
  const labels = value.facePresent
    ? [
        "用户在摄像头前",
        attention !== null && attention < 0.35 ? "注意力偏离" : null,
        fatigue !== null && fatigue > 0.7 ? "出现疲倦迹象" : null,
        smile !== null && smile > 0.65 ? "出现微笑迹象" : null,
      ]
    : ["用户未出现在摄像头画面中"];
  const scopeKey = "camera-presence";
  return {
    source: "camera",
    laneKey: `camera:${scopeKey}`,
    semanticKey,
    revision,
    observedAtMs: Date.parse(observedAt),
    body: {
      source: "camera",
      observationType: "user_state",
      scopeKey,
      semanticKey,
      summary: labels.filter(Boolean).join(" · "),
      payload: {
        revision,
        facePresent: value.facePresent,
        faceCount,
        ...(value.facePresent ? { attentionLevel, fatigueLevel, smileLevel } : {}),
      },
      observedAt,
    },
  };
}

function screenObservation(
  value: Record<string, unknown>,
  revision: number,
  observedAt: string,
): ForwardableObservation | null {
  const evaluation = isRecord(value.evaluation) ? value.evaluation : null;
  const state = normalizedText(evaluation?.state);
  const displayID = nonNegativeInteger(value.displayID);
  if (!state || !["stable", "changing", "continuous_motion"].includes(state) || displayID === null) {
    return null;
  }
  const summary = {
    stable: "当前屏幕画面已稳定",
    changing: "当前屏幕画面正在变化",
    continuous_motion: "当前屏幕处于连续运动状态",
  }[state] ?? `当前屏幕状态：${state}`;
  const scopeKey = `display-${displayID}`;
  const semanticKey = `${scopeKey}:${state}`;
  return {
    source: "screen",
    laneKey: `screen:${scopeKey}`,
    semanticKey,
    revision,
    observedAtMs: Date.parse(observedAt),
    body: {
      source: "screen",
      observationType: "screen_activity",
      scopeKey,
      semanticKey,
      summary,
      payload: {
        revision,
        displayID,
        state,
        measurement: "sampled_pixels",
        pixelChangeRatio: unitNumber(evaluation?.dirtyAreaRatio) ?? 0,
        compositorDirtyRectCount:
          nonNegativeInteger(evaluation?.dirtyRectCount) ?? 0,
        significantChange: evaluation?.significantChange === true,
      },
      observedAt,
    },
  };
}

function bucket(value: number | null): string {
  if (value === null) return "unknown";
  if (value < 0.35) return "low";
  if (value > 0.7) return "high";
  return "medium";
}

function proactivityFingerprint(
  value: ObservationProactivityConfiguration,
): string {
  return `${value.enabled ? "1" : "0"}:${value.profile}`;
}

function isSameOrNewer(
  candidate: ForwardableObservation,
  reference: ForwardableObservation,
): boolean {
  return candidate.observedAtMs > reference.observedAtMs
    || (
      candidate.observedAtMs === reference.observedAtMs
      && candidate.revision >= reference.revision
    );
}

function isoTimestamp(seconds: number): string | null {
  const date = new Date(seconds * 1_000);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

function boundedText(value: unknown, maximum: number): string | null {
  const text = normalizedText(value);
  return text ? text.slice(0, maximum) : null;
}

function boundedSingleLineText(value: unknown, maximum: number): string | null {
  const text = normalizedText(value);
  if (!text) return null;
  const singleLine = text.split(/\s+/u).join(" ");
  return singleLine ? singleLine.slice(0, maximum) : null;
}

function redactedBoundedText(value: unknown, maximum: number): string | null {
  const text = boundedSingleLineText(value, maximum);
  if (!text) return null;
  const redacted = redactSensitiveText(text, maximum);
  return redacted || null;
}

function normalizedBundleIdentifier(value: unknown): string | null {
  const normalized = normalizedText(value)?.toLowerCase() ?? null;
  return normalized && /^[a-z0-9][a-z0-9.-]{0,254}$/u.test(normalized)
    ? normalized
    : null;
}

function normalizedUuid(value: unknown): string | null {
  return typeof value === "string"
    && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu.test(value)
    ? value.toUpperCase()
    : null;
}

function stableDigest(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex").slice(0, 24);
}

function containsRawMedia(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(containsRawMedia);
  if (!isRecord(value)) {
    return typeof value === "string" && value.trimStart().toLowerCase().startsWith("data:image/");
  }
  return Object.entries(value).some(([key, child]) => {
    const normalizedKey = key
      .replace(/(?<=[a-z0-9])(?=[A-Z])/gu, "_")
      .toLowerCase()
      .replace(/[^a-z0-9]+/gu, "_")
      .replace(/^_+|_+$/gu, "");
    if ([
      "base64",
      "frame",
      "frame_bytes",
      "image",
      "image_data",
      "image_url",
      "jpeg",
      "ocr_blocks",
      "ocr_tokens",
      "pixel_data",
      "png",
      "raw_ocr",
      "raw_pixels",
      "screen_shot",
      "screenshot",
    ].includes(normalizedKey)) return true;
    return containsRawMedia(child);
  });
}

function normalizedText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function positiveInteger(value: unknown): number | null {
  return Number.isSafeInteger(value) && Number(value) > 0 ? Number(value) : null;
}

function nonNegativeInteger(value: unknown): number | null {
  return Number.isSafeInteger(value) && Number(value) >= 0 ? Number(value) : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function unitNumber(value: unknown): number | null {
  const number = finiteNumber(value);
  return number !== null && number >= 0 && number <= 1 ? number : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
