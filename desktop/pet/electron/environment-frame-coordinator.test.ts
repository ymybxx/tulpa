import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";

import {
  EnvironmentFrameCoordinator,
  type EnvironmentFrameCoordinatorErrorCode,
  type EnvironmentFrameCoordinatorOptions,
} from "./environment-frame-coordinator";
import type {
  EnvironmentFrameMetadata,
  EnvironmentFramePayload,
} from "./perception-manager";

const NOW = 1_788_200_000_000;
const API_URL = "http://127.0.0.1:8123";
const FISH_ID = "00000000-0000-0000-0000-000000000001";
const OTHER_FISH_ID = "00000000-0000-0000-0000-000000000002";
const FRAME_ID = "00000000-0000-0000-0000-000000000010";
const JPEG_BASE64 = "/9j//9k=";

interface CapturedRequest {
  url: string;
  init?: RequestInit;
}

describe("EnvironmentFrameCoordinator", () => {
  it("moves unchanged perception switches to the new assistant and rejects old frames", async () => {
    const requests: CapturedRequest[] = [];
    let fishId: string | null = FISH_ID;
    let now = NOW;
    const onTarget = vi.fn();
    const onTargetPending = vi.fn();
    const claim = vi.fn(async () => payload({ observedAt: now / 1_000, expiresAt: now / 1_000 + 3 }));
    const coordinator = new EnvironmentFrameCoordinator({
      claim, onTarget, onTargetPending, now: () => now,
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      fetcher: async (input, init) => {
        const url = input.toString();
        requests.push({ url, init });
        if (url.endsWith("/assistant")) return Response.json({ fishId });
        return new Response(null, { status: 204 });
      },
    });
    const policy = { operationEnabled: true, screenEnabled: true, audioEnabled: true };
    await coordinator.setPolicy(policy);
    await coordinator.handleFrame(metadata());
    fishId = OTHER_FISH_ID;
    now += 1_000;
    const refresh = coordinator.refreshBinding();
    await coordinator.handleFrame(metadata());
    await refresh;
    await coordinator.handleFrame(metadata());
    expect(claim).toHaveBeenCalledOnce();
    await coordinator.handleFrame(metadata({ observedAt: now / 1_000, expiresAt: now / 1_000 + 3 }));

    const states = requests.filter((r) => r.url.endsWith("/state"));
    expect(states.map((r) => [r.url.split("/")[5], JSON.parse(r.init!.body as string)])).toEqual([
      [FISH_ID, policy],
      [FISH_ID, { operationEnabled: false, screenEnabled: false, audioEnabled: false }],
      [OTHER_FISH_ID, policy],
    ]);
    expect(requests.filter((r) => r.url.endsWith("/frame")).map((r) => r.url.split("/")[5]))
      .toEqual([FISH_ID, OTHER_FISH_ID]);
    expect(onTargetPending).toHaveBeenCalledTimes(2);
    expect(onTarget).toHaveBeenLastCalledWith({ apiUrl: API_URL, apiKey: "secret", fishId: OTHER_FISH_ID });

    fishId = null;
    await coordinator.refreshBinding();
    await coordinator.handleFrame(metadata({ observedAt: now / 1_000, expiresAt: now / 1_000 + 3 }));
    expect(claim).toHaveBeenCalledTimes(2);
    expect(onTarget).toHaveBeenLastCalledWith(null);
    expect(JSON.parse(requests.at(-1)!.init!.body as string))
      .toEqual({ operationEnabled: false, screenEnabled: false, audioEnabled: false });
  });

  it("cancels an old assistant's pending claim when rebinding", async () => {
    const oldClaim = deferred<EnvironmentFramePayload>();
    const started = deferred<void>();
    let fishId = FISH_ID;
    let signal: AbortSignal | undefined;
    const uploads: string[] = [];
    const onFrame = vi.fn();
    const coordinator = new EnvironmentFrameCoordinator({
      claim: async (_id, inputSignal) => { signal = inputSignal; started.resolve(); return oldClaim.promise; },
      onFrame, now: () => NOW,
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      fetcher: async (input) => {
        const url = input.toString();
        if (url.endsWith("/assistant")) return Response.json({ fishId });
        if (url.endsWith("/frame")) uploads.push(url);
        return new Response(null, { status: 204 });
      },
    });
    await coordinator.setPolicy({ operationEnabled: true, screenEnabled: true, audioEnabled: true });
    const pending = coordinator.handleFrame(metadata());
    await started.promise;
    fishId = OTHER_FISH_ID;
    await coordinator.refreshBinding();
    expect(signal?.aborted).toBe(true);
    oldClaim.resolve(payload());
    await pending;
    expect(uploads).toEqual([]);
    expect(onFrame).not.toHaveBeenCalled();
  });

  it("revokes a superseded state PUT before enabling the latest assistant", async () => {
    const started = deferred<void>();
    const finishOld = deferred<void>();
    let fishId = FISH_ID;
    const states: [string, boolean][] = [];
    const onTarget = vi.fn();
    const coordinator = new EnvironmentFrameCoordinator({
      claim: async () => payload(), onTarget, now: () => NOW,
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      fetcher: async (input, init) => {
        const url = input.toString();
        if (url.endsWith("/assistant")) return Response.json({ fishId });
        const enabled = JSON.parse(init!.body as string).screenEnabled as boolean;
        states.push([url.split("/")[5]!, enabled]);
        if (states.length === 1) { started.resolve(); await finishOld.promise; }
        return new Response(null, { status: 204 });
      },
    });
    const first = coordinator.setPolicy({ operationEnabled: true, screenEnabled: true, audioEnabled: true });
    await started.promise;
    fishId = OTHER_FISH_ID;
    const second = coordinator.refreshBinding();
    finishOld.resolve();
    await Promise.all([first, second]);
    expect(states).toEqual([[FISH_ID, true], [FISH_ID, false], [OTHER_FISH_ID, true]]);
    expect(onTarget).toHaveBeenCalledOnce();
    expect(onTarget).toHaveBeenLastCalledWith({ apiUrl: API_URL, apiKey: "secret", fishId: OTHER_FISH_ID });
  });

  it("refreshes operation-only and disabled policies without enabling screen capture", async () => {
    let fishId = FISH_ID;
    const states: [string, object][] = [];
    const claim = vi.fn(async () => payload());
    const coordinator = new EnvironmentFrameCoordinator({
      claim, now: () => NOW,
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      fetcher: async (input, init) => {
        const url = input.toString();
        if (url.endsWith("/assistant")) return Response.json({ fishId });
        states.push([url.split("/")[5]!, JSON.parse(init!.body as string)]);
        return new Response(null, { status: 204 });
      },
    });
    await coordinator.setPolicy({ operationEnabled: true, screenEnabled: false, audioEnabled: false });
    fishId = OTHER_FISH_ID;
    await coordinator.refreshBinding();
    await coordinator.handleFrame(metadata());
    expect(states[1]).toEqual([FISH_ID, { operationEnabled: false, screenEnabled: false, audioEnabled: false }]);
    expect(states[2]).toEqual([OTHER_FISH_ID, { operationEnabled: true, screenEnabled: false, audioEnabled: false }]);
    await coordinator.setEnabled(false);
    await coordinator.refreshBinding();
    await coordinator.handleFrame(metadata());
    expect(claim).not.toHaveBeenCalled();
    expect(states.at(-1)![1]).toEqual({ operationEnabled: false, screenEnabled: false, audioEnabled: false });
  });

  it("resynchronizes after a reconnect without querying the binding on every frame", async () => {
    const requests: CapturedRequest[] = [];
    const coordinator = createCoordinator({ requests });
    await coordinator.setEnabled(true);
    await coordinator.refreshBinding();
    await coordinator.handleFrame(metadata());
    await coordinator.handleFrame(metadata());
    expect(requests.filter((r) => r.url.endsWith("/assistant"))).toHaveLength(2);
    expect(requests.filter((r) => r.url.endsWith("/frame"))).toHaveLength(2);
  });

  it("refreshes identical pixels with fresh timestamps and retries a missing reference", async () => {
    const contentHash = createHash("sha256").update(Buffer.from(JPEG_BASE64, "base64")).digest("hex");
    const requests: CapturedRequest[] = [];
    const claim = vi.fn(async () => payload({ contentHash }));
    const onFrame = vi.fn();
    let missing = false;
    const coordinator = new EnvironmentFrameCoordinator({
      claim, onFrame, now: () => NOW,
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      fetcher: async (input, init) => {
        requests.push({ url: input.toString(), init });
        if (input.toString().endsWith("/assistant")) return Response.json({ fishId: FISH_ID });
        return new Response(null, { status: missing && init?.method === "PATCH" ? 409 : 204 });
      },
    });
    await coordinator.setPolicy({ operationEnabled: false, screenEnabled: true, audioEnabled: true });
    await coordinator.handleFrame(metadata({ contentHash }));
    for (let index = 1; index <= 2; index += 1) {
      missing = index === 2;
      await coordinator.handleFrame(metadata({ contentHash, revision: index + 1,
        frameId: `00000000-0000-0000-0000-00000000001${index}`,
        observedAt: NOW / 1000 + index, expiresAt: NOW / 1000 + index + 3,
      }));
    }
    expect(claim).toHaveBeenCalledTimes(1);
    const uploads = requests.filter((request) => request.url.endsWith("/frame"));
    expect(uploads.map((r) => r.init?.method)).toEqual(["PUT", "PATCH", "PATCH", "PUT"]);
    expect(JSON.parse(uploads[1]!.init!.body as string).jpegBase64).toBeUndefined();
    expect(onFrame).toHaveBeenCalledTimes(3);
    expect(onFrame.mock.calls[2]![0].observedAt).toBe(NOW / 1000 + 2);
    await coordinator.clearFrame();
    await coordinator.handleFrame(metadata({ contentHash }));
    expect(claim).toHaveBeenCalledTimes(2);
  });
  it.each(Array.from({ length: 8 }, (_value, mask) => mask))(
    "synchronizes all choices and gates frame handoff for combination %i", async (mask) => {
      const requests: CapturedRequest[] = [];
      const claim = vi.fn(async () => payload());
      const onFrame = vi.fn();
      const onTarget = vi.fn();
      const onTargetPending = vi.fn();
      const coordinator = createCoordinator({ requests, claim, onFrame, onTarget, onTargetPending });
      const policy = {
        operationEnabled: Boolean(mask & 1),
        screenEnabled: Boolean(mask & 2),
        audioEnabled: Boolean(mask & 4),
      };
      const synchronized = coordinator.setPolicy(policy);
      expect(onTargetPending).toHaveBeenCalledTimes(mask & 6 ? 1 : 0);
      if (mask & 6) expect(onTarget).not.toHaveBeenCalled();
      await synchronized;
      await coordinator.handleFrame(metadata());
      expect(JSON.parse(requests[1]!.init!.body as string)).toEqual(policy);
      expect(claim).toHaveBeenCalledTimes(mask & 2 ? 1 : 0);
      expect(requests.filter((request) => request.url.endsWith("/frame")))
        .toHaveLength(mask & 2 ? 1 : 0);
      expect(onFrame).toHaveBeenCalledTimes(mask & 2 ? 1 : 0);
      expect(onTarget).toHaveBeenLastCalledWith(mask & 6
        ? { apiUrl: API_URL, apiKey: "secret", fishId: FISH_ID } : null);
    },
  );

  it("keeps visual summaries active when only audio is disabled", async () => {
    const requests: CapturedRequest[] = [];
    const onFrame = vi.fn();
    const onTarget = vi.fn();
    const coordinator = createCoordinator({ requests, onFrame, onTarget });
    await coordinator.setPolicy({ operationEnabled: false, screenEnabled: true, audioEnabled: true });
    await coordinator.handleFrame(metadata());
    await coordinator.setPolicy({ operationEnabled: false, screenEnabled: true, audioEnabled: false });
    expect(onTarget).toHaveBeenLastCalledWith({ apiUrl: API_URL, apiKey: "secret", fishId: FISH_ID });
    await coordinator.handleFrame(metadata());
    expect(requests.filter((request) => request.url.endsWith("/frame"))).toHaveLength(2);
    expect(onFrame).toHaveBeenCalledTimes(2);
  });

  it("treats the same in-flight and synchronized audiovisual policy as idempotent", async () => {
    const requests: CapturedRequest[] = [];
    const onTarget = vi.fn();
    const onTargetPending = vi.fn();
    const assistant = deferred<Response>();
    const assistantStarted = deferred<void>();
    let assistantRequests = 0;
    const policy = {
      operationEnabled: true,
      screenEnabled: true,
      audioEnabled: true,
    };
    const coordinator = new EnvironmentFrameCoordinator({
      claim: async () => payload(),
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      now: () => NOW,
      onTarget,
      onTargetPending,
      fetcher: async (input, init) => {
        const url = input.toString();
        requests.push({ url, init });
        if (url.endsWith("/v1/fishes/assistant")) {
          assistantRequests += 1;
          assistantStarted.resolve(undefined);
          return assistant.promise;
        }
        return new Response(null, { status: 204 });
      },
    });

    const first = coordinator.setPolicy(policy);
    const duplicateWhilePending = coordinator.setPolicy({ ...policy });
    await assistantStarted.promise;
    expect(assistantRequests).toBe(1);
    expect(onTargetPending).toHaveBeenCalledOnce();

    assistant.resolve(Response.json({ fishId: FISH_ID }));
    await Promise.all([first, duplicateWhilePending]);
    await coordinator.setPolicy({ ...policy });

    expect(requests.filter((request) => request.url.endsWith("/v1/fishes/assistant")))
      .toHaveLength(1);
    expect(requests.filter((request) => request.url.endsWith("/state"))).toHaveLength(1);
    expect(onTargetPending).toHaveBeenCalledOnce();
    expect(onTarget).toHaveBeenCalledOnce();
  });

  it("does not replace a scheduled retry when the same policy is repeated", async () => {
    vi.useFakeTimers();
    try {
      let statePuts = 0;
      const onTargetPending = vi.fn();
      const policy = {
        operationEnabled: false,
        screenEnabled: true,
        audioEnabled: true,
      };
      const coordinator = new EnvironmentFrameCoordinator({
        claim: async () => payload(),
        connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
        now: () => NOW,
        stateRetryDelaysMs: [1_000],
        onTargetPending,
        fetcher: async (input) => {
          const url = input.toString();
          if (url.endsWith("/v1/fishes/assistant")) {
            return Response.json({ fishId: FISH_ID });
          }
          if (url.endsWith("/state")) {
            statePuts += 1;
            return new Response(null, { status: statePuts === 1 ? 503 : 204 });
          }
          return new Response(null, { status: 204 });
        },
      });

      await coordinator.setPolicy(policy);
      await coordinator.setPolicy({ ...policy });
      expect(statePuts).toBe(1);
      expect(onTargetPending).toHaveBeenCalledOnce();

      await vi.advanceTimersByTimeAsync(1_000);
      expect(statePuts).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("claims once and uploads only to the authenticated local memory slot", async () => {
    const requests: CapturedRequest[] = [];
    const claim = vi.fn(async () => payload());
    const coordinator = createCoordinator({ requests, claim });

    await coordinator.setEnabled(true);
    await coordinator.handleFrame(metadata());

    expect(claim).toHaveBeenCalledOnce();
    expect(requests.map((item) => [item.url, item.init?.method ?? "GET"])).toEqual([
      [`${API_URL}/v1/fishes/assistant`, "GET"],
      [`${API_URL}/internal/fishes/${FISH_ID}/environment-perception/state`, "PUT"],
      [`${API_URL}/internal/fishes/${FISH_ID}/environment-perception/frame`, "PUT"],
    ]);
    expect(JSON.parse(requests[1]!.init!.body as string)).toEqual({ operationEnabled: true, screenEnabled: true, audioEnabled: false });
    expect(JSON.parse(requests[2]!.init!.body as string)).toEqual(payload());
    expect(requests[2]!.init!.headers).toMatchObject({
      Authorization: "Bearer secret",
      "Content-Type": "application/json",
    });
  });

  it("reuses the state-synchronized Fish target across consecutive frames", async () => {
    const requests: CapturedRequest[] = [];
    const second = metadata({
      frameId: "00000000-0000-0000-0000-000000000011",
      revision: 2,
      observedAt: NOW / 1_000 + 0.5,
      expiresAt: NOW / 1_000 + 3.5,
    });
    const claim = vi.fn(async (frameId: string) => frameId === FRAME_ID
      ? payload()
      : payload(second));
    const coordinator = createCoordinator({ requests, claim });

    await coordinator.setEnabled(true);
    await coordinator.handleFrame(metadata());
    await coordinator.handleFrame(second);

    expect(requests.filter((item) => item.url.endsWith("/v1/fishes/assistant")))
      .toHaveLength(1);
    expect(requests.filter((item) => item.url.endsWith("/frame"))).toHaveLength(2);
  });

  it("recovers on the next frame after one transient upload failure", async () => {
    const errors: EnvironmentFrameCoordinatorErrorCode[] = [];
    let framePuts = 0;
    let frameDeletes = 0;
    const second = metadata({
      frameId: "00000000-0000-0000-0000-000000000011",
      revision: 2,
      observedAt: NOW / 1_000 + 0.5,
      expiresAt: NOW / 1_000 + 3.5,
    });
    const claim = vi.fn(async (frameId: string) => frameId === FRAME_ID
      ? payload()
      : payload(second));
    const coordinator = new EnvironmentFrameCoordinator({
      claim,
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      now: () => NOW,
      onErrorCode: (code) => errors.push(code),
      fetcher: async (input, init) => {
        const url = input.toString();
        if (url.endsWith("/v1/fishes/assistant")) {
          return Response.json({ fishId: FISH_ID });
        }
        if (url.endsWith("/frame")) {
          if (init?.method === "DELETE") {
            frameDeletes += 1;
            return new Response(null, { status: 204 });
          }
          framePuts += 1;
          return new Response(null, { status: framePuts === 1 ? 503 : 204 });
        }
        return new Response(null, { status: 204 });
      },
    });

    await coordinator.setEnabled(true);
    await coordinator.handleFrame(metadata());
    await coordinator.clearFrame();
    await coordinator.handleFrame(second);

    expect(claim).toHaveBeenCalledTimes(2);
    expect(framePuts).toBe(2);
    expect(frameDeletes).toBe(1);
    expect(errors).toEqual(["upload_failed"]);
  });

  it("retries a failed state synchronization without blocking helper startup", async () => {
    let statePuts = 0;
    const claim = vi.fn(async () => payload());
    const coordinator = new EnvironmentFrameCoordinator({
      claim,
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      now: () => NOW,
      stateRetryDelaysMs: [0],
      fetcher: async (input) => {
        const url = input.toString();
        if (url.endsWith("/v1/fishes/assistant")) {
          return Response.json({ fishId: FISH_ID });
        }
        if (url.endsWith("/state")) {
          statePuts += 1;
          return new Response(null, { status: statePuts === 1 ? 503 : 204 });
        }
        return new Response(null, { status: 204 });
      },
    });

    await coordinator.setEnabled(true);
    await vi.waitFor(() => expect(statePuts).toBe(2));
    await coordinator.handleFrame(metadata());

    expect(claim).toHaveBeenCalledOnce();
  });

  it("retries an idempotent clear after a transient local API failure", async () => {
    let deletes = 0;
    const coordinator = new EnvironmentFrameCoordinator({
      claim: async () => payload(),
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      now: () => NOW,
      stateRetryDelaysMs: [0],
      fetcher: async (input, init) => {
        const url = input.toString();
        if (url.endsWith("/v1/fishes/assistant")) {
          return Response.json({ fishId: FISH_ID });
        }
        if (url.endsWith("/frame") && init?.method === "DELETE") {
          deletes += 1;
          return new Response(null, { status: deletes === 1 ? 503 : 204 });
        }
        return new Response(null, { status: 204 });
      },
    });

    await coordinator.setEnabled(true);
    await coordinator.handleFrame(metadata());
    await coordinator.clearFrame();
    await vi.waitFor(() => expect(deletes).toBe(2));
  });

  it("serializes overlapping state changes so the latest disable wins", async () => {
    const states: boolean[] = [];
    let releaseEnable!: () => void;
    let markEnableStarted!: () => void;
    const enableStarted = new Promise<void>((resolve) => (markEnableStarted = resolve));
    const enableGate = new Promise<void>((resolve) => (releaseEnable = resolve));
    const claim = vi.fn(async () => payload());
    const coordinator = new EnvironmentFrameCoordinator({
      claim,
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      now: () => NOW,
      fetcher: async (input, init) => {
        const url = input.toString();
        if (url.endsWith("/v1/fishes/assistant")) {
          return Response.json({ fishId: FISH_ID });
        }
        const enabled = (JSON.parse(init!.body as string) as { screenEnabled: boolean }).screenEnabled;
        states.push(enabled);
        if (enabled) {
          markEnableStarted();
          await enableGate;
        }
        return new Response(null, { status: 204 });
      },
    });

    const enabling = coordinator.setEnabled(true);
    await enableStarted;
    const disabling = coordinator.setEnabled(false);
    await Promise.resolve();
    expect(states).toEqual([true]);
    await coordinator.handleFrame(metadata());
    expect(claim).not.toHaveBeenCalled();
    releaseEnable();
    await Promise.all([enabling, disabling]);

    expect(states).toEqual([true, false]);
    await coordinator.handleFrame(metadata());
    expect(claim).not.toHaveBeenCalled();
  });

  it("never claims while disabled and clears remote state on disable", async () => {
    const requests: CapturedRequest[] = [];
    const claim = vi.fn(async () => payload());
    const coordinator = createCoordinator({ requests, claim });

    await coordinator.handleFrame(metadata());
    await coordinator.setEnabled(false);

    expect(claim).not.toHaveBeenCalled();
    expect(JSON.parse(requests[1]!.init!.body as string)).toEqual({ operationEnabled: false, screenEnabled: false, audioEnabled: false });
  });

  it("rejects expired and malformed metadata before claim or local I/O", async () => {
    const requests: CapturedRequest[] = [];
    const claim = vi.fn(async () => payload());
    const coordinator = createCoordinator({ requests, claim });
    await coordinator.setEnabled(true);
    requests.length = 0;

    await coordinator.handleFrame(metadata({ expiresAt: (NOW - 1) / 1_000 }));
    await coordinator.handleFrame({ ...metadata(), jpegBase64: JPEG_BASE64 });

    expect(claim).not.toHaveBeenCalled();
    expect(requests).toHaveLength(0);
  });

  it("cancels a superseded claim and uploads only the latest frame", async () => {
    const requests: CapturedRequest[] = [];
    let firstSignal: AbortSignal | undefined;
    let releaseFirst!: () => void;
    const firstStarted = new Promise<void>((resolve) => (releaseFirst = resolve));
    const claim = vi.fn(async (frameId: string, signal: AbortSignal) => {
      if (frameId !== FRAME_ID) return payload({
        frameId,
        revision: 2,
        observedAt: NOW / 1_000 + 0.5,
        expiresAt: NOW / 1_000 + 3.5,
      });
      firstSignal = signal;
      releaseFirst();
      await new Promise<never>((_resolve, reject) => {
        signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
      });
      throw new Error("unreachable");
    });
    const coordinator = createCoordinator({ requests, claim });
    await coordinator.setEnabled(true);
    requests.length = 0;

    const first = coordinator.handleFrame(metadata());
    await firstStarted;
    const secondId = "00000000-0000-0000-0000-000000000011";
    const second = coordinator.handleFrame(metadata({
      frameId: secondId,
      revision: 2,
      observedAt: NOW / 1_000 + 0.5,
      expiresAt: NOW / 1_000 + 3.5,
    }));
    await Promise.all([first, second]);

    expect(firstSignal?.aborted).toBe(true);
    expect(requests.filter((item) => item.url.endsWith("/frame"))).toHaveLength(1);
  });

  it("reports fixed error codes without exposing image bytes", async () => {
    const errors: EnvironmentFrameCoordinatorErrorCode[] = [];
    const coordinator = createCoordinator({
      requests: [],
      claim: vi.fn(async () => { throw new Error(`secret ${JPEG_BASE64}`); }),
      onErrorCode: (code) => errors.push(code),
    });
    await coordinator.setEnabled(true);

    await coordinator.handleFrame(metadata());

    expect(errors).toEqual(["claim_failed"]);
    expect(JSON.stringify(errors)).not.toContain(JPEG_BASE64);
  });
});

function createCoordinator(options: {
  requests: CapturedRequest[];
  claim?: (frameId: string, signal: AbortSignal) => Promise<EnvironmentFramePayload>;
  onErrorCode?: (code: EnvironmentFrameCoordinatorErrorCode) => void;
  onFrame?: EnvironmentFrameCoordinatorOptions["onFrame"];
  onTarget?: EnvironmentFrameCoordinatorOptions["onTarget"];
  onTargetPending?: EnvironmentFrameCoordinatorOptions["onTargetPending"];
}): EnvironmentFrameCoordinator {
  return new EnvironmentFrameCoordinator({
    claim: options.claim ?? (async () => payload()),
    connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
    now: () => NOW,
    onErrorCode: options.onErrorCode,
    onFrame: options.onFrame,
    onTarget: options.onTarget,
    onTargetPending: options.onTargetPending,
    fetcher: async (input, init) => {
      const url = input.toString();
      options.requests.push({ url, init });
      if (url.endsWith("/v1/fishes/assistant")) {
        return Response.json({ fishId: FISH_ID });
      }
      return new Response(null, { status: 204 });
    },
  });
}

function metadata(
  patch: Partial<EnvironmentFrameMetadata> = {},
): EnvironmentFrameMetadata {
  return {
    frameId: FRAME_ID,
    revision: 1,
    observedAt: NOW / 1_000,
    expiresAt: NOW / 1_000 + 3,
    displayID: 1,
    pixelWidth: 960,
    pixelHeight: 540,
    byteLength: 5,
    ...patch,
  };
}

function payload(
  patch: Partial<EnvironmentFramePayload> = {},
): EnvironmentFramePayload {
  return {
    ...metadata(patch),
    mimeType: "image/jpeg",
    jpegBase64: JPEG_BASE64,
    ...patch,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
