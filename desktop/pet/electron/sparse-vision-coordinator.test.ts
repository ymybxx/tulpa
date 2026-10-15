import { describe, expect, it, vi } from "vitest";

import type {
  VisionCandidateMetadata,
  VisionCandidatePayload,
} from "./perception-manager";
import {
  SparseVisionCoordinator,
  type SparseVisionCoordinatorErrorCode,
  type SparseVisionRuntimeConfiguration,
} from "./sparse-vision-coordinator";

const NOW = 1_788_200_000_000;
const FISH_ID = "00000000-0000-0000-0000-000000000001";
const OTHER_FISH_ID = "00000000-0000-0000-0000-000000000002";
const CANDIDATE_ID = "00000000-0000-0000-0000-000000000010";
const WINDOW_KEY = "00000000-0000-0000-0000-000000000020";
const API_URL = "http://127.0.0.1:8123";

interface CapturedRequest {
  url: string;
  init?: RequestInit;
}

describe("SparseVisionCoordinator", () => {
  it("persists only a structured summary after local preflight and exact claim", async () => {
    const requests: CapturedRequest[] = [];
    const claim = vi.fn(async () => payload());
    const coordinator = createCoordinator({
      claim,
      requests,
      providerResult: { summary: "正在编辑代码", salience: "low" },
    });

    await coordinator.handleCandidate(metadata());

    expect(claim).toHaveBeenCalledOnce();
    expect(requests.map((request) => request.url)).toEqual([
      `${API_URL}/v1/fishes/assistant`,
      `${API_URL}/internal/fishes/${FISH_ID}/observation-app-policies/preflight?appBundleId=com.example.editor`,
      `${API_URL}/v1/fishes/assistant`,
      `${API_URL}/internal/fishes/${FISH_ID}/observation-app-policies/preflight?appBundleId=com.example.editor`,
      "https://provider.example/v1/chat/completions",
      `${API_URL}/v1/fishes/assistant`,
      `${API_URL}/internal/fishes/${FISH_ID}/observations/states`,
    ]);
    const persisted = bodyOf(requests[6]!);
    expect(persisted).toMatchObject({
      source: "vision",
      observationType: "screen_context",
      scopeKey: "frontmost-window",
      summary: "正在编辑代码",
      appBundleId: "com.example.editor",
      appName: "Editor",
      payload: {
        revision: 7,
        displayID: 2,
        salience: "low",
      },
    });
    expect(persisted.semanticKey).toMatch(/^com\.example\.editor:[0-9a-f]{24}$/u);
    expect((persisted.payload as Record<string, unknown>).candidateKey)
      .toMatch(/^[0-9a-f]{24}$/u);
    const serialized = JSON.stringify(persisted);
    expect(serialized).not.toContain(CANDIDATE_ID);
    expect(serialized).not.toContain(WINDOW_KEY);
    expect(serialized).not.toContain(jpegBase64());
    expect(serialized).not.toContain("Document title");
  });

  it.each([
    [{ enabled: false }, "disabled"],
    [{ model: null }, "missing model"],
    [{ apiKey: "" }, "missing key"],
    [{ baseUrl: "http://192.168.1.20/v1" }, "insecure remote URL"],
  ] as const)("does not claim or call a provider when configuration is %s", async (patch) => {
    const requests: CapturedRequest[] = [];
    const claim = vi.fn(async () => payload());
    const coordinator = createCoordinator({
      claim,
      requests,
      configuration: { ...configuration(), ...patch },
    });

    await coordinator.handleCandidate(metadata());

    expect(claim).not.toHaveBeenCalled();
    expect(requests).toHaveLength(0);
  });

  it("fails closed when the app recording policy denies the candidate", async () => {
    const requests: CapturedRequest[] = [];
    const claim = vi.fn(async () => payload());
    const coordinator = createCoordinator({ requests, claim, preflightAllowed: false });

    await coordinator.handleCandidate(metadata());

    expect(claim).not.toHaveBeenCalled();
    expect(requests.some((request) => request.url.includes("chat/completions"))).toBe(false);
    expect(requests.some((request) => request.url.includes("observations/states"))).toBe(false);
  });

  it("rechecks app policy after claim and before the image-bearing request", async () => {
    const requests: CapturedRequest[] = [];
    const claim = vi.fn(async () => payload());
    let preflightCalls = 0;
    const coordinator = createCoordinator({
      requests,
      claim,
      fetchOverride: async (input, _init, fallback) => {
        if (!input.toString().includes("observation-app-policies/preflight")) {
          return fallback();
        }
        preflightCalls += 1;
        return Response.json({ allowed: preflightCalls === 1 });
      },
    });

    await coordinator.handleCandidate(metadata());

    expect(claim).toHaveBeenCalledOnce();
    expect(preflightCalls).toBe(2);
    expect(requests.some((request) => request.url.includes("chat/completions"))).toBe(false);
    expect(requests.some((request) => request.url.includes("observations/states"))).toBe(false);
  });

  it("does not upload when the candidate expires during the second preflight", async () => {
    const requests: CapturedRequest[] = [];
    let now = NOW;
    let preflightCalls = 0;
    const coordinator = createCoordinator({
      requests,
      now: () => now,
      fetchOverride: async (input, _init, fallback) => {
        if (!input.toString().includes("observation-app-policies/preflight")) {
          return fallback();
        }
        preflightCalls += 1;
        if (preflightCalls === 2) now = NOW + 4_001;
        return Response.json({ allowed: true });
      },
    });

    await coordinator.handleCandidate(metadata());

    expect(preflightCalls).toBe(2);
    expect(requests.some((request) => request.url.includes("chat/completions"))).toBe(false);
    expect(requests.some((request) => request.url.includes("observations/states"))).toBe(false);
  });

  it("drops the image before provider upload if the active Fish changes during claim", async () => {
    const requests: CapturedRequest[] = [];
    const claim = vi.fn(async () => payload());
    const coordinator = createCoordinator({
      requests,
      claim,
      assistantFishIds: [FISH_ID, OTHER_FISH_ID],
    });

    await coordinator.handleCandidate(metadata());

    expect(claim).toHaveBeenCalledOnce();
    expect(requests.some((request) => request.url.includes("chat/completions"))).toBe(false);
    expect(requests.some((request) => request.url.includes("observations/states"))).toBe(false);
  });

  it("drops a valid provider result if the active Fish changes before persistence", async () => {
    const requests: CapturedRequest[] = [];
    const coordinator = createCoordinator({
      requests,
      assistantFishIds: [FISH_ID, FISH_ID, OTHER_FISH_ID],
    });

    await coordinator.handleCandidate(metadata());

    expect(requests.some((request) => request.url.includes("chat/completions"))).toBe(true);
    expect(requests.some((request) => request.url.includes("observations/states"))).toBe(false);
  });

  it("rejects expired, camera-shaped and unknown-field candidates before any I/O", async () => {
    const requests: CapturedRequest[] = [];
    const claim = vi.fn(async () => payload());
    const coordinator = createCoordinator({ requests, claim });

    await coordinator.handleCandidate(metadata({ expiresAt: (NOW - 1) / 1_000 }));
    await coordinator.handleCandidate({ ...metadata(), source: "camera" });
    await coordinator.handleCandidate({ ...metadata(), image: "raw-canary" });

    expect(claim).not.toHaveBeenCalled();
    expect(requests).toHaveLength(0);
  });

  it("drops a claimed payload whose metadata does not exactly match", async () => {
    const requests: CapturedRequest[] = [];
    const coordinator = createCoordinator({
      requests,
      claim: vi.fn(async () => payload({ displayID: 99 })),
    });

    await coordinator.handleCandidate(metadata());

    expect(requests.some((request) => request.url.includes("chat/completions"))).toBe(false);
    expect(requests.some((request) => request.url.includes("observations/states"))).toBe(false);
  });

  it("newer candidates cancel an older provider request and only the latest may persist", async () => {
    const requests: CapturedRequest[] = [];
    let firstProviderSignal: AbortSignal | undefined;
    let releaseFirstStarted!: () => void;
    const firstStarted = new Promise<void>((resolve) => (releaseFirstStarted = resolve));
    let providerCalls = 0;
    const coordinator = createCoordinator({
      requests,
      fetchOverride: async (input, init, fallback) => {
        const url = input.toString();
        if (!url.includes("chat/completions")) return fallback();
        providerCalls += 1;
        if (providerCalls > 1) return completion("latest", "none");
        firstProviderSignal = init?.signal ?? undefined;
        releaseFirstStarted();
        return await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          }, { once: true });
        });
      },
      claim: vi.fn(async (id: string) => id === CANDIDATE_ID
        ? payload()
        : payload({
          candidateId: "00000000-0000-0000-0000-000000000011",
          revision: 8,
          observedAt: (NOW - 500) / 1_000,
          expiresAt: (NOW + 4_500) / 1_000,
          windowKey: "00000000-0000-0000-0000-000000000021",
        })),
    });

    const first = coordinator.handleCandidate(metadata());
    await firstStarted;
    const secondMetadata = metadata({
      candidateId: "00000000-0000-0000-0000-000000000011",
      revision: 8,
      observedAt: (NOW - 500) / 1_000,
      expiresAt: (NOW + 4_500) / 1_000,
      windowKey: "00000000-0000-0000-0000-000000000021",
    });
    const second = coordinator.handleCandidate(secondMetadata);
    await Promise.all([first, second]);

    expect(firstProviderSignal?.aborted).toBe(true);
    const stateBodies = requests
      .filter((request) => request.url.includes("observations/states"))
      .map(bodyOf);
    expect(stateBodies).toHaveLength(1);
    expect(stateBodies[0]?.summary).toBe("latest");
    expect((stateBodies[0]?.payload as Record<string, unknown>).revision).toBe(8);
  });

  it("context changes abort provider work without persisting an old result", async () => {
    const requests: CapturedRequest[] = [];
    let providerSignal: AbortSignal | undefined;
    let signalStarted!: () => void;
    const started = new Promise<void>((resolve) => (signalStarted = resolve));
    const coordinator = createCoordinator({
      requests,
      fetchOverride: async (input, init, fallback) => {
        if (!input.toString().includes("chat/completions")) return fallback();
        providerSignal = init?.signal ?? undefined;
        signalStarted();
        return await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          }, { once: true });
        });
      },
    });

    const pending = coordinator.handleCandidate(metadata());
    await started;
    coordinator.handleContextObservation({ source: "workspace", kind: "foreground_app_changed" });
    await pending;

    expect(providerSignal?.aborted).toBe(true);
    expect(requests.some((request) => request.url.includes("observations/states"))).toBe(false);
  });

  it("drops a provider result that arrives after the twelve-second result TTL", async () => {
    const requests: CapturedRequest[] = [];
    let now = NOW;
    const coordinator = createCoordinator({
      requests,
      now: () => now,
      fetchOverride: async (input, _init, fallback) => {
        if (!input.toString().includes("chat/completions")) return fallback();
        now = NOW + 12_001;
        return completion("old result", "high");
      },
    });

    await coordinator.handleCandidate(metadata());

    expect(requests.some((request) => request.url.includes("observations/states"))).toBe(false);
  });

  it("turns provider failures into fixed codes without exposing raw errors", async () => {
    const requests: CapturedRequest[] = [];
    const codes: SparseVisionCoordinatorErrorCode[] = [];
    const coordinator = createCoordinator({
      requests,
      onErrorCode: (code) => codes.push(code),
      fetchOverride: async (input, _init, fallback) => {
        if (!input.toString().includes("chat/completions")) return fallback();
        throw new Error(`${CANDIDATE_ID}:${jpegBase64()}:secret`);
      },
    });

    await coordinator.handleCandidate(metadata());

    expect(codes).toEqual(["provider_failed"]);
    expect(JSON.stringify(codes)).not.toContain("secret");
    expect(requests.some((request) => request.url.includes("observations/states"))).toBe(false);
  });

  it("redacts credentials echoed by the provider before persistence", async () => {
    const requests: CapturedRequest[] = [];
    const secret = "ghp_1234567890abcdefghijklmnopqrst";
    const coordinator = createCoordinator({
      requests,
      providerResult: {
        summary: `正在编辑配置 token=${secret}`,
        salience: "none",
      },
    });

    await coordinator.handleCandidate(metadata());

    const persisted = bodyOf(
      requests.find((request) => request.url.includes("observations/states"))!,
    );
    expect(persisted.summary).toContain("token=[REDACTED]");
    expect(JSON.stringify(persisted)).not.toContain(secret);
  });
});

function createCoordinator(options: {
  requests: CapturedRequest[];
  claim?: (candidateId: string) => Promise<VisionCandidatePayload>;
  configuration?: SparseVisionRuntimeConfiguration;
  preflightAllowed?: boolean;
  providerResult?: { summary: string; salience: "none" | "low" | "high" };
  now?: () => number;
  onErrorCode?: (code: SparseVisionCoordinatorErrorCode) => void;
  fetchOverride?: (
    input: string | URL | Request,
    init: RequestInit | undefined,
    fallback: () => Promise<Response>,
  ) => Promise<Response>;
  assistantFishIds?: string[];
}): SparseVisionCoordinator {
  let assistantCalls = 0;
  const fallback = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = input.toString();
    options.requests.push({ url, init });
    if (url.endsWith("/v1/fishes/assistant")) {
      const fishIds = options.assistantFishIds ?? [FISH_ID];
      const fishId = fishIds[Math.min(assistantCalls, fishIds.length - 1)]!;
      assistantCalls += 1;
      return Response.json({ fishId });
    }
    if (url.includes("observation-app-policies/preflight")) {
      return Response.json({ allowed: options.preflightAllowed ?? true });
    }
    if (url.includes("chat/completions")) {
      const result = options.providerResult ?? { summary: "screen summary", salience: "none" };
      return completion(result.summary, result.salience);
    }
    if (url.includes("observations/states")) return new Response(null, { status: 204 });
    throw new Error("unexpected request");
  };
  return new SparseVisionCoordinator({
    claim: options.claim ?? (async () => payload()),
    configuration: async () => options.configuration ?? configuration(),
    connection: async () => ({ apiUrl: API_URL, apiKey: "local-secret" }),
    now: options.now ?? (() => NOW),
    onErrorCode: options.onErrorCode,
    fetcher: async (input, init) => {
      const runFallback = () => fallback(input, init);
      if (options.fetchOverride) {
        const url = input.toString();
        options.requests.push({ url, init });
        const response = await options.fetchOverride(input, init, async () => {
          options.requests.pop();
          return runFallback();
        });
        return response;
      }
      return runFallback();
    },
  });
}

function configuration(): SparseVisionRuntimeConfiguration {
  return {
    enabled: true,
    baseUrl: "https://provider.example/v1",
    model: "vision-model",
    apiKey: "provider-secret",
  };
}

function metadata(
  overrides: Partial<VisionCandidateMetadata> = {},
): VisionCandidateMetadata {
  return {
    candidateId: CANDIDATE_ID,
    revision: 7,
    observedAt: (NOW - 1_000) / 1_000,
    expiresAt: (NOW + 4_000) / 1_000,
    displayID: 2,
    appBundleId: "com.example.editor",
    appName: "Editor",
    windowKey: WINDOW_KEY,
    ...overrides,
  };
}

function payload(
  overrides: Partial<VisionCandidatePayload> = {},
): VisionCandidatePayload {
  return {
    ...metadata(),
    mimeType: "image/jpeg",
    jpegBase64: jpegBase64(),
    ...overrides,
  };
}

function completion(summary: string, salience: "none" | "low" | "high"): Response {
  return Response.json({
    choices: [{ message: { content: JSON.stringify({ summary, salience }) } }],
  });
}

function jpegBase64(): string {
  return Buffer.from([0xff, 0xd8, 0xff, 0x01, 0xff, 0xd9]).toString("base64");
}

function bodyOf(request: CapturedRequest): Record<string, unknown> {
  return JSON.parse(String(request.init?.body)) as Record<string, unknown>;
}
