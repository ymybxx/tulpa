import { describe, expect, it } from "vitest";

import { ObservationForwarder } from "./observation-forwarder";

const API_URL = "http://127.0.0.1:8123";
const FISH_ID = "00000000-0000-0000-0000-000000000001";

interface CapturedRequest {
  url: string;
  init?: RequestInit;
}

function bodyOf(request: CapturedRequest): Record<string, unknown> {
  return JSON.parse(String(request.init?.body)) as Record<string, unknown>;
}

function stateRequests(requests: CapturedRequest[]): CapturedRequest[] {
  return requests.filter((request) => request.url.includes("/observations/states"));
}

function workspaceObservation(
  revision: number,
  bundleIdentifier = "com.microsoft.VSCode",
  name = "Visual Studio Code",
): Record<string, unknown> {
  return {
    revision,
    kind: "foreground_app_changed",
    observedAt: 1_788_200_000 + revision,
    frontmostApplication: {
      bundleIdentifier,
      name,
      processIdentifier: 42,
    },
  };
}

function accessibilityObservation(
  revision: number,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    source: "accessibility",
    revision,
    kind: "focused_value_changed",
    observedAt: 1_788_200_000 + revision,
    processIdentifier: 42,
    bundleIdentifier: "com.apple.Safari",
    applicationName: "Safari",
    windowTitle: "Login",
    focusedRole: "AXTextField",
    focusedTitle: "Account",
    focusedValue: "first value",
    secureField: false,
    ...overrides,
  };
}

function cameraObservation(
  revision: number,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    source: "camera",
    kind: "camera_face_observation",
    revision,
    observedAt: 1_788_200_000 + revision,
    facePresent: true,
    faceCount: 1,
    primaryFace: {
      confidence: 0.98,
      attentionProxy: 0.2,
      fatigueProxy: 0.8,
      smileProxy: 0.9,
    },
    ...overrides,
  };
}

function screenTextObservation(
  revision: number,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    source: "screen",
    kind: "screen_text",
    revision,
    observedAt: 1_788_200_000 + revision,
    displayID: 7,
    text: "Project plan\nProject plan   next step",
    appBundleId: "com.apple.TextEdit",
    appName: "TextEdit",
    windowKey: "00000000-0000-0000-0000-000000000040",
    ...overrides,
  };
}

function capturingForwarder(options: {
  requests: CapturedRequest[];
  now?: () => number;
  connection?: () => Promise<{ apiUrl: string; apiKey: string } | null>;
  requestTimeoutMs?: number;
  onError?: (error: Error) => void;
  stateResponse?: () => Promise<Response> | Response;
}): ObservationForwarder {
  return new ObservationForwarder({
    connection: options.connection
      ?? (async () => ({ apiUrl: API_URL, apiKey: "secret" })),
    now: options.now,
    requestTimeoutMs: options.requestTimeoutMs,
    onError: options.onError,
    fetcher: async (input, init) => {
      const url = input.toString();
      options.requests.push({ url, init });
      if (url.endsWith("/v1/fishes/assistant")) {
        return Response.json({ fishId: FISH_ID });
      }
      return options.stateResponse?.() ?? new Response(null, { status: 204 });
    },
  });
}

describe("ObservationForwarder", () => {
  it("synchronizes proactivity separately and reuses it for later observations", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });

    await forwarder.configureProactivity({ enabled: true, profile: "lively" });

    expect(requests).toHaveLength(2);
    expect(requests[1]!.url).toContain("/observations/proactivity");
    expect(bodyOf(requests[1]!)).toEqual({ enabled: true, profile: "lively" });

    await forwarder.publish(workspaceObservation(1));
    expect(
      requests.filter((request) => request.url.includes("/observations/proactivity")),
    ).toHaveLength(1);
    expect(stateRequests(requests)).toHaveLength(1);
  });

  it("maps a legacy workspace observation into isolated state storage", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });

    await forwarder.publish(workspaceObservation(4));

    expect(requests).toHaveLength(2);
    const request = requests[1]!;
    expect(request.url).toContain(`/internal/fishes/${FISH_ID}/observations/states`);
    expect(request.init?.headers).toMatchObject({ Authorization: "Bearer secret" });
    expect(bodyOf(request)).toMatchObject({
      source: "workspace",
      observationType: "activity",
      scopeKey: "active-app",
      semanticKey: "com.microsoft.vscode",
      summary: "正在使用 Visual Studio Code",
      payload: { workspaceEvent: "foreground_app_changed", revision: 4 },
    });
  });

  it("never forwards secure Accessibility values or titles", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });

    await forwarder.publish(accessibilityObservation(1, {
      secureField: true,
      focusedTitle: "password-title-secret",
      focusedValue: "hunter2",
    }));

    const body = bodyOf(stateRequests(requests)[0]!);
    expect(body).toMatchObject({
      source: "accessibility",
      observationType: "application_context",
      scopeKey: "focused-ui",
      semanticKey: "com.apple.safari:Login:AXTextField",
      summary: "Safari · Login · 安全输入界面",
      payload: {
        accessibilityEvent: "focused_value_changed",
        revision: 1,
        processIdentifier: 42,
        secureField: true,
        focusedRole: "AXTextField",
      },
    });
    expect(JSON.stringify(body)).not.toContain("hunter2");
    expect(JSON.stringify(body)).not.toContain("password-title-secret");
  });

  it("redacts credentials from non-secure Accessibility text and payload", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });
    const secret = "sk-1234567890abcdefghijklmnop";

    await forwarder.publish(accessibilityObservation(2, {
      windowTitle: `Config token=${secret}`,
      focusedValue: `Authorization: Bearer ${secret}`,
    }));

    const serialized = JSON.stringify(bodyOf(stateRequests(requests)[0]!));
    expect(serialized).toContain("[REDACTED]");
    expect(serialized).not.toContain(secret);
  });

  it("stores only coarse camera state and whitelisted payload fields", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });

    await forwarder.publish(cameraObservation(1, {
      primaryFace: {
        attentionProxy: 0.2,
        fatigueProxy: 0.8,
        smileProxy: 0.9,
        image: "must-not-leave-the-helper",
        rawPixels: [1, 2, 3],
      },
    }));

    const body = bodyOf(stateRequests(requests)[0]!);
    expect(body).toMatchObject({
      source: "camera",
      observationType: "user_state",
      scopeKey: "camera-presence",
      semanticKey: "present:low:high:high",
      summary: "用户在摄像头前 · 注意力偏离 · 出现疲倦迹象 · 出现微笑迹象",
      payload: {
        revision: 1,
        facePresent: true,
        faceCount: 1,
        attentionLevel: "low",
        fatigueLevel: "high",
        smileLevel: "high",
      },
    });
    expect(JSON.stringify(body)).not.toContain("primaryFace");
    expect(JSON.stringify(body)).not.toContain("must-not-leave-the-helper");
  });

  it("maps screen activity without pixels and accepts the transitional nested revision", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });

    await forwarder.publish({
      source: "screen",
      kind: "screen_change",
      observedAt: 1_788_200_000,
      displayID: 7,
      frameWidth: 3024,
      frameHeight: 1964,
      evaluation: {
        revision: 9,
        state: "continuous_motion",
        dirtyAreaRatio: 0.42,
        significantChange: true,
        rawPixels: "must-not-be-forwarded",
      },
    });

    expect(bodyOf(stateRequests(requests)[0]!)).toEqual({
      source: "screen",
      observationType: "screen_activity",
      scopeKey: "display-7",
      semanticKey: "display-7:continuous_motion",
      summary: "当前屏幕处于连续运动状态",
      payload: {
        revision: 9,
        displayID: 7,
        state: "continuous_motion",
        measurement: "sampled_pixels",
        pixelChangeRatio: 0.42,
        compositorDirtyRectCount: 0,
        significantChange: true,
      },
      observedAt: new Date(1_788_200_000_000).toISOString(),
    });
  });

  it("maps privacy-gated final screen text without retaining window text twice", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });

    await forwarder.publish(screenTextObservation(10));

    const body = bodyOf(stateRequests(requests)[0]!);
    expect(body).toMatchObject({
      source: "screen",
      observationType: "screen_context",
      scopeKey: "frontmost-window",
      summary: "当前窗口文字：Project plan Project plan next step",
      appBundleId: "com.apple.textedit",
      appName: "TextEdit",
      payload: {
        revision: 10,
        displayID: 7,
      },
    });
    expect(body.semanticKey).toMatch(/^com\.apple\.textedit:[0-9a-f]{24}$/u);
    expect((body.payload as Record<string, unknown>).windowKey).toMatch(/^[0-9a-f]{24}$/u);
    expect(JSON.stringify(body.payload)).not.toContain("Project plan");
    expect(JSON.stringify(body)).not.toContain("Project.txt");
  });

  it("fails closed for malformed or raw-media screen text events", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });

    await forwarder.publish(screenTextObservation(1, { appBundleId: null }));
    await forwarder.publish(screenTextObservation(2, { windowKey: "" }));
    await forwarder.publish(screenTextObservation(3, {
      rawPixels: "screen-canary",
    }));
    await forwarder.publish(screenTextObservation(4, {
      image: "data:image/jpeg;base64,screen-canary",
    }));

    expect(requests).toHaveLength(0);
  });

  it("redacts credentials from final OCR text before durable Observation storage", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });
    const secret = "sk-1234567890abcdefghijklmnop";

    await forwarder.publish(screenTextObservation(5, {
      text: `Editing .env OPENAI_API_KEY=${secret}`,
    }));

    const serialized = JSON.stringify(bodyOf(stateRequests(requests)[0]!));
    expect(serialized).toContain("OPENAI_API_KEY=[REDACTED]");
    expect(serialized).not.toContain(secret);
  });

  it("extends identical final screen text by heartbeat instead of every OCR event", async () => {
    const requests: CapturedRequest[] = [];
    let now = 0;
    const forwarder = capturingForwarder({ requests, now: () => now });

    await forwarder.publish(screenTextObservation(1));
    now = 29_999;
    await forwarder.publish(screenTextObservation(2));
    expect(stateRequests(requests)).toHaveLength(1);
    now = 30_000;
    await forwarder.publish(screenTextObservation(3));
    expect(stateRequests(requests)).toHaveLength(2);
  });

  it("keeps revisions independent across sources", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });

    await forwarder.publish({ ...workspaceObservation(100), source: "workspace" });
    await forwarder.publish(cameraObservation(1));

    expect(stateRequests(requests).map((request) => bodyOf(request).source)).toEqual([
      "workspace",
      "camera",
    ]);
  });

  it("coalesces a busy lane to its latest pending revision", async () => {
    const requests: CapturedRequest[] = [];
    let stateCall = 0;
    let releaseFirst!: (response: Response) => void;
    let signalFirstStarted!: () => void;
    const firstResponse = new Promise<Response>((resolve) => {
      releaseFirst = resolve;
    });
    const firstStarted = new Promise<void>((resolve) => {
      signalFirstStarted = resolve;
    });
    const forwarder = capturingForwarder({
      requests,
      stateResponse: () => {
        stateCall += 1;
        if (stateCall === 1) {
          signalFirstStarted();
          return firstResponse;
        }
        return new Response(null, { status: 204 });
      },
    });

    const first = forwarder.publish(workspaceObservation(1));
    await firstStarted;
    const second = forwarder.publish(workspaceObservation(2, "com.apple.Safari", "Safari"));
    const third = forwarder.publish(workspaceObservation(3, "com.apple.Terminal", "Terminal"));
    releaseFirst(new Response(null, { status: 204 }));
    await Promise.all([first, second, third]);

    expect(stateRequests(requests).map((request) => bodyOf(request).semanticKey)).toEqual([
      "com.microsoft.vscode",
      "com.apple.terminal",
    ]);
  });

  it("refreshes Accessibility at 2s without writing on every focused-value event", async () => {
    const requests: CapturedRequest[] = [];
    let now = 0;
    const forwarder = capturingForwarder({ requests, now: () => now });

    await forwarder.publish(accessibilityObservation(1));
    await forwarder.publish(accessibilityObservation(2, { focusedValue: "second value" }));
    now = 1_999;
    await forwarder.publish(accessibilityObservation(3, { focusedValue: "third value" }));
    expect(stateRequests(requests)).toHaveLength(1);

    now = 2_000;
    await forwarder.publish(accessibilityObservation(4, { focusedValue: "heartbeat value" }));
    expect(stateRequests(requests)).toHaveLength(2);

    now = 2_001;
    await forwarder.publish(accessibilityObservation(5, { windowTitle: "Another Tab" }));
    expect(stateRequests(requests)).toHaveLength(3);
    expect(bodyOf(stateRequests(requests)[2]!).semanticKey).toBe(
      "com.apple.safari:Another Tab:AXTextField",
    );
  });

  it("uses 5s camera and 30s workspace heartbeats", async () => {
    let now = 0;
    const cameraRequests: CapturedRequest[] = [];
    const cameraForwarder = capturingForwarder({
      requests: cameraRequests,
      now: () => now,
    });

    await cameraForwarder.publish(cameraObservation(1));
    now = 4_999;
    await cameraForwarder.publish(cameraObservation(2));
    expect(stateRequests(cameraRequests)).toHaveLength(1);
    now = 5_000;
    await cameraForwarder.publish(cameraObservation(3));
    expect(stateRequests(cameraRequests)).toHaveLength(2);

    now = 0;
    const workspaceRequests: CapturedRequest[] = [];
    const workspaceForwarder = capturingForwarder({
      requests: workspaceRequests,
      now: () => now,
    });
    await workspaceForwarder.publish(workspaceObservation(1));
    now = 29_999;
    await workspaceForwarder.publish(workspaceObservation(2));
    expect(stateRequests(workspaceRequests)).toHaveLength(1);
    now = 30_000;
    await workspaceForwarder.publish(workspaceObservation(3));
    expect(stateRequests(workspaceRequests)).toHaveLength(2);
  });

  it("drops observations while the API is unavailable and retries only on a fresh sample", async () => {
    const requests: CapturedRequest[] = [];
    let available = false;
    let failNextState = true;
    const errors: Error[] = [];
    const forwarder = capturingForwarder({
      requests,
      connection: async () => available ? { apiUrl: API_URL, apiKey: "secret" } : null,
      onError: (error) => errors.push(error),
      stateResponse: () => {
        if (failNextState) {
          failNextState = false;
          return new Response(null, { status: 503 });
        }
        return new Response(null, { status: 204 });
      },
    });

    await forwarder.publish(workspaceObservation(1));
    expect(requests).toHaveLength(0);

    available = true;
    await forwarder.publish(workspaceObservation(2));
    expect(stateRequests(requests)).toHaveLength(1);
    expect(errors).toHaveLength(1);
    await Promise.resolve();
    expect(stateRequests(requests)).toHaveLength(1);

    await forwarder.publish(workspaceObservation(3));
    expect(stateRequests(requests)).toHaveLength(2);
    expect(errors).toHaveLength(1);

    await forwarder.publish(workspaceObservation(4));
    expect(stateRequests(requests)).toHaveLength(2);
  });

  it("bounds a hung local API request without blocking later lanes forever", async () => {
    const requests: CapturedRequest[] = [];
    const errors: Error[] = [];
    let hangState = true;
    const forwarder = new ObservationForwarder({
      connection: async () => ({ apiUrl: API_URL, apiKey: "secret" }),
      requestTimeoutMs: 10,
      onError: (error) => errors.push(error),
      fetcher: async (input, init) => {
        const url = input.toString();
        requests.push({ url, init });
        if (url.endsWith("/v1/fishes/assistant")) {
          return Response.json({ fishId: FISH_ID });
        }
        if (!hangState) return new Response(null, { status: 204 });
        return await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          }, { once: true });
        });
      },
    });

    await forwarder.publish(workspaceObservation(1));
    expect(errors[0]?.name).toBe("AbortError");

    hangState = false;
    await forwarder.publish(cameraObservation(1));
    expect(stateRequests(requests)).toHaveLength(2);
  });

  it("strictly ignores unknown sources and malformed timestamps", async () => {
    const requests: CapturedRequest[] = [];
    const forwarder = capturingForwarder({ requests });

    await forwarder.publish({
      ...workspaceObservation(1),
      source: "future-untrusted-source",
    });
    await forwarder.publish({
      ...workspaceObservation(2),
      observedAt: Number.MAX_VALUE,
    });
    await forwarder.publish({ kind: "foreground_app_changed" });

    expect(requests).toHaveLength(0);
  });
});
