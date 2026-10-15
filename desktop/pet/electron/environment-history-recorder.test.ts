import { createHash } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  EnvironmentHistoryRecorder,
  type EnvironmentHistoryRecorderErrorCode,
  type EnvironmentHistoryRecorderOptions,
  type EnvironmentHistoryTarget,
  type EnvironmentHistoryWindow,
} from "./environment-history-recorder";
import type { EnvironmentFramePayload, SystemAudioWindow } from "./perception-manager";

const NOW = 1_788_200_000_000;
const SESSION = "11111111-1111-4111-8111-111111111111";
const OTHER_SESSION = "22222222-2222-4222-8222-222222222222";
const PCM = Buffer.alloc(320_000).toString("base64");
const TARGET: EnvironmentHistoryTarget = {
  apiUrl: "http://127.0.0.1:8123",
  apiKey: "local-key",
  fishId: "00000000-0000-0000-0000-000000000001",
};
const recorders: EnvironmentHistoryRecorder[] = [];

interface Request {
  url: string;
  body: EnvironmentHistoryWindow;
  headers: RequestInit["headers"];
  signal: AbortSignal;
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(NOW);
});

afterEach(() => {
  for (const recorder of recorders.splice(0)) recorder.stop();
  vi.useRealTimers();
});

describe("EnvironmentHistoryRecorder", () => {
  it("drops frames captured under the previous app exclusion filter", () => {
    const { recorder, requests } = createRecorder();
    recorder.handleFrame(frame(1, NOW + 800));
    recorder.clearCapturedWindow();
    recorder.handleFrame(frame(2, NOW + 1_800));
    recorder.handleAudioWindow(audioWindow());
    expect(requests[0]!.body.frames.map((item) => item.frameId)).toEqual([
      frame(2, NOW + 1_800).frameId,
    ]);
  });

  it("records visual windows without forwarding audio when only screen is enabled", () => {
    const { recorder, requests } = createRecorder();
    recorder.setPolicy({ operationEnabled: false, screenEnabled: true, audioEnabled: false });
    recorder.handleFrame(frame(1, NOW + 800));
    recorder.handleAudioWindow(audioWindow());
    expect(requests[0]!.body.frames).toHaveLength(1);
    expect(requests[0]!.body).toMatchObject({
      audioStatus: "unavailable", audioPcmBase64: null,
      audioCapturedSamples: 0, audioGapReason: "system_audio_disabled",
    });
  });

  it("records audio-only windows without retaining a screen frame", () => {
    const { recorder, requests } = createRecorder();
    recorder.setPolicy({ operationEnabled: false, screenEnabled: false, audioEnabled: true });
    recorder.handleFrame(frame(1, NOW + 800));
    recorder.handleAudioWindow(audioWindow());
    expect(requests[0]!.body.frames).toEqual([]);
    expect(requests[0]!.body.audioPcmBase64).toBe(audioWindow().audioPcmBase64);
    recorder.setPolicy({ operationEnabled: true, screenEnabled: false, audioEnabled: false });
    recorder.handleAudioWindow(audioWindow(1));
    expect(requests).toHaveLength(1);
  });
  it("aligns ten one-second frames with ten seconds of audio and accepts an early cut", async () => {
    const { recorder, requests } = createRecorder();
    for (let index = 0; index < 10; index += 1) recorder.handleFrame(frame(index + 1, NOW + index * 1_000));
    recorder.handleAudioWindow(audioWindow());
    expect(requests[0]!.body.frames.map((f) => f.observedAt)).toEqual(
      Array.from({ length: 10 }, (_, i) => NOW / 1_000 + i),
    );
    expect(Buffer.from(requests[0]!.body.audioPcmBase64!, "base64").length).toBe(320_000);
    for (let index = 0; index < 6; index += 1) recorder.handleFrame(frame(index + 11, NOW + (10 + index) * 1_000));
    recorder.handleAudioWindow(audioWindow(1, {
      startedAt: NOW / 1_000 + 10, endedAt: NOW / 1_000 + 16,
      audioCapturedSamples: 96_000, audioPcmBase64: Buffer.alloc(192_000).toString("base64"),
    }));
    await vi.advanceTimersByTimeAsync(0);
    expect(requests[1]!.body.frames.map((f) => f.observedAt)).toEqual(
      Array.from({ length: 6 }, (_, i) => NOW / 1_000 + 10 + i),
    );
  });

  it("sends one JPEG with ten distinct observations and a compact silent window", () => {
    const { recorder, requests } = createRecorder();
    const contentHash = createHash("sha256").update(Buffer.from("/9j//9k=", "base64")).digest("hex");
    for (let index = 0; index < 10; index += 1) {
      recorder.handleFrame({ ...frame(index + 1, NOW + index * 1000), contentHash });
    }
    recorder.handleAudioWindow({ ...audioWindow(), audioSilence: true, audioPcmBase64: null });
    const body = requests[0]!.body;
    expect(body.frames).toHaveLength(10);
    expect(body.frames.filter((f) => f.jpegBase64)).toHaveLength(1);
    expect(new Set(body.frames.map((f) => f.frameId)).size).toBe(10);
    expect(body.audioSilence).toBe(true);
    expect(body.audioPcmBase64).toBeNull();
  });
  it("pairs ordered observed frames with one native window and freezes their bytes", async () => {
    const { recorder, requests } = createRecorder();
    const first = frame(1, NOW + 800);
    recorder.handleFrame(frame(2, NOW + 9_000));
    recorder.handleFrame(frame(3, NOW + 5_200));
    recorder.handleFrame(first);
    recorder.handleFrame(frame(4, NOW + 4_600));
    recorder.handleFrame(frame(5, NOW + 10_000));
    first.jpegBase64 = "mutated-after-cache";

    recorder.handleAudioWindow(audioWindow());

    expect(requests).toHaveLength(1);
    expect(requests[0]!.url).toBe(`${TARGET.apiUrl}/internal/fishes/${TARGET.fishId}`
      + "/environment-perception/history/windows");
    expect(requests[0]!.headers).toEqual({
      Authorization: "Bearer local-key",
      "Content-Type": "application/json",
    });
    expect(requests[0]!.body.frames.map((value) => value.observedAt)).toEqual([
      (NOW + 800) / 1_000,
      (NOW + 4_600) / 1_000,
      (NOW + 5_200) / 1_000,
      (NOW + 9_000) / 1_000,
    ]);
    expect(requests[0]!.body.frames[0]!.jpegBase64).toBe("/9j//9k=");

    recorder.handleAudioWindow(unavailableWindow(1));
    await vi.advanceTimersByTimeAsync(0);
    expect(requests).toHaveLength(2);
    expect(requests[1]!.body.frames.map((value) => value.observedAt)).toEqual([
      (NOW + 10_000) / 1_000,
    ]);
  });

  it("never duplicates a lone frame to fill a grid", () => {
    const { recorder, requests } = createRecorder();
    recorder.handleFrame(frame(1, NOW + 4_000));
    recorder.handleAudioWindow(unavailableWindow());
    expect(requests[0]!.body.frames).toHaveLength(1);
  });

  it("submits audio-only and visual-only windows but skips a fully empty one", async () => {
    const { recorder, requests } = createRecorder();
    recorder.handleAudioWindow(audioWindow());
    recorder.handleFrame(frame(1, NOW + 10_000));
    recorder.handleAudioWindow(unavailableWindow(1));
    recorder.handleAudioWindow(unavailableWindow(2));
    await vi.advanceTimersByTimeAsync(0);

    expect(requests).toHaveLength(2);
    expect(requests[0]!.body.frames).toEqual([]);
    expect(requests[0]!.body.audioStatus).toBe("complete");
    expect(requests[1]!.body.audioStatus).toBe("unavailable");
    expect(requests[1]!.body.frames).toHaveLength(1);
  });

  it("marks an empty window seen so a duplicate cannot create it later", () => {
    const { recorder, requests } = createRecorder();
    const empty = unavailableWindow();
    recorder.handleAudioWindow(empty);
    recorder.handleFrame(frame(1, NOW + 1_000));
    recorder.handleAudioWindow(empty);
    recorder.handleFrame(frame(2, NOW + 10_000));
    recorder.handleAudioWindow(unavailableWindow(1));

    expect(requests).toHaveLength(1);
    expect(requests[0]!.body.windowIndex).toBe(1);
  });

  it("deduplicates exact session/index and accepts out-of-order and new-session windows", async () => {
    const { recorder, requests } = createRecorder();
    recorder.handleAudioWindow(audioWindow(1));
    recorder.handleAudioWindow(audioWindow(1));
    recorder.handleAudioWindow(audioWindow(0));
    recorder.handleAudioWindow(audioWindow(0, { captureSessionId: OTHER_SESSION }));
    await vi.advanceTimersByTimeAsync(0);

    expect(requests.map((request) => [
      request.body.captureSessionId,
      request.body.windowIndex,
    ])).toEqual([[SESSION, 1], [SESSION, 0], [OTHER_SESSION, 0]]);
  });

  it("bounds the recent frame deque while retaining the useful fifteen-second range", () => {
    const { recorder, requests } = createRecorder();
    for (let seconds = -20; seconds <= 0; seconds += 1) {
      recorder.handleFrame(frame(seconds + 21, NOW + seconds * 1_000));
    }
    recorder.handleAudioWindow(audioWindow(0, {
      startedAt: NOW / 1_000 - 10,
      endedAt: NOW / 1_000,
    }));

    const selected = requests[0]!.body.frames;
    expect(selected).toHaveLength(10);
    expect(selected[0]!.observedAt).toBe(NOW / 1_000 - 10);
    expect(selected.at(-1)!.observedAt).toBe(NOW / 1_000 - 1);
    expect(new Set(selected.map((item) => item.frameId)).size).toBe(10);
  });

  it("spreads ten slots across a speech burst without dropping the window tail", () => {
    const { recorder, requests } = createRecorder();
    for (let index = 0; index < 16; index += 1) {
      recorder.handleFrame(frame(index + 1, NOW + index * 600));
    }
    recorder.handleAudioWindow(audioWindow());
    const selected = requests[0]!.body.frames;
    expect(selected).toHaveLength(10);
    expect(selected[0]!.observedAt).toBe(NOW / 1_000);
    expect(selected.at(-1)!.observedAt).toBe((NOW + 9_000) / 1_000);
    expect(selected.every((item, index) => index === 0
      || item.observedAt > selected[index - 1]!.observedAt)).toBe(true);
  });

  it("rejects malformed native windows without exposing their payload", () => {
    const { recorder, requests, errors } = createRecorder();
    recorder.handleAudioWindow({ ...audioWindow(), endedAt: NOW / 1_000 + 9 });
    recorder.handleAudioWindow({ ...audioWindow(), audioPcmBase64: "private-audio" });
    recorder.handleAudioWindow({
      ...unavailableWindow(),
      audioCapturedSamples: 1,
    });

    expect(requests).toHaveLength(0);
    expect(errors).toEqual(["invalid_audio", "invalid_audio", "invalid_audio"]);
  });

  it("retries a frozen body without blocking a later native window", async () => {
    const { recorder, requests } = createRecorder({
      retryDelaysMs: [1_000],
      respond: async (request) => new Response(null, {
        status: request.body.windowIndex === 0 ? 503 : 202,
      }),
    });
    recorder.handleAudioWindow(audioWindow());
    recorder.handleAudioWindow(audioWindow(1));
    await vi.advanceTimersByTimeAsync(0);
    expect(requests.map((request) => request.body.windowIndex)).toEqual([0, 1]);
    const original = requests[0]!.body;

    await vi.advanceTimersByTimeAsync(1_000);
    const attempts = requests.filter((request) => request.body.windowIndex === 0);
    expect(attempts).toHaveLength(2);
    expect(attempts[1]!.body).toEqual(original);
  });

  it.each([429, 503, "network"] as const)("bounds retries for %s", async (failure) => {
    const { recorder, requests } = createRecorder({
      retryDelaysMs: [1, 2],
      respond: async () => {
        if (failure === "network") throw new Error("private-network-response");
        return new Response(null, { status: failure });
      },
    });
    recorder.handleAudioWindow(audioWindow());
    await vi.advanceTimersByTimeAsync(3);
    expect(requests).toHaveLength(3);
    expect(requests.map((request) => request.body)).toEqual([
      requests[0]!.body, requests[0]!.body, requests[0]!.body,
    ]);
  });

  it("bounds the frozen upload backlog to six including active work", async () => {
    const blocked = deferred<Response>();
    let first = true;
    const { recorder, requests, errors } = createRecorder({
      requestTimeoutMs: 1_000_000,
      respond: async () => {
        if (first) { first = false; return blocked.promise; }
        return new Response(null, { status: 202 });
      },
    });
    for (let index = 0; index < 8; index += 1) {
      recorder.handleAudioWindow(audioWindow(index));
    }
    expect(requests).toHaveLength(1);
    expect(errors).toEqual(["queue_overflow", "queue_overflow"]);

    blocked.resolve(new Response(null, { status: 202 }));
    await vi.advanceTimersByTimeAsync(0);
    expect(requests.map((request) => request.body.windowIndex)).toEqual([
      0, ...Array.from({ length: 5 }, (_value, index) => index + 3),
    ]);
  });

  it("buffers frames and windows while the same target is temporarily unknown", async () => {
    const blocked = deferred<Response>();
    let first = true;
    const { recorder, requests } = createRecorder({
      requestTimeoutMs: 60_000,
      respond: async () => {
        if (first) { first = false; return blocked.promise; }
        return new Response(null, { status: 202 });
      },
    });
    recorder.handleAudioWindow(audioWindow());
    recorder.suspendTarget();
    expect(requests[0]!.signal.aborted).toBe(true);
    recorder.handleFrame(frame(1, NOW + 10_000));
    recorder.handleAudioWindow(audioWindow(1));
    recorder.setTarget(TARGET);
    await vi.advanceTimersByTimeAsync(0);

    expect(requests).toHaveLength(3);
    expect(requests[1]!.body).toEqual(requests[0]!.body);
    expect(requests[2]!.body.windowIndex).toBe(1);
    expect(requests[2]!.body.frames).toHaveLength(1);
    blocked.resolve(new Response(null, { status: 503 }));
  });

  it("drops buffered media when a pending target resolves to a different fish", async () => {
    const { recorder, requests } = createRecorder();
    recorder.suspendTarget();
    recorder.handleFrame(frame(1, NOW));
    recorder.handleAudioWindow(audioWindow());
    recorder.setTarget({ ...TARGET, fishId: "00000000-0000-0000-0000-000000000002" });
    await vi.advanceTimersByTimeAsync(0);

    expect(requests).toHaveLength(0);
  });

  it.each(["disabled", "target_cleared", "stopped"] as const)(
    "drops transient media when %s",
    (reason) => {
      const { recorder, requests } = createRecorder();
      recorder.handleFrame(frame(1, NOW));
      if (reason === "disabled") recorder.setEnabled(false);
      else if (reason === "target_cleared") recorder.setTarget(null);
      else recorder.stop();
      recorder.handleAudioWindow(audioWindow());
      expect(requests).toHaveLength(0);
    },
  );
});

function createRecorder(options: EnvironmentHistoryRecorderOptions & {
  respond?: (request: Request) => Promise<Response>;
} = {}) {
  const requests: Request[] = [];
  const errors: EnvironmentHistoryRecorderErrorCode[] = [];
  const recorder = new EnvironmentHistoryRecorder({
    ...options,
    onErrorCode: (code) => errors.push(code),
    fetcher: async (input, init) => {
      const request: Request = {
        url: input.toString(),
        body: JSON.parse(init!.body as string) as EnvironmentHistoryWindow,
        headers: init!.headers,
        signal: init!.signal as AbortSignal,
      };
      requests.push(request);
      return options.respond?.(request) ?? new Response(null, { status: 202 });
    },
  });
  recorders.push(recorder);
  recorder.setEnabled(true);
  recorder.setTarget(TARGET);
  return { recorder, requests, errors };
}

function audioWindow(
  index = 0,
  overrides: Partial<SystemAudioWindow> = {},
): SystemAudioWindow {
  const startedAt = NOW / 1_000 + index * 10;
  return {
    captureSessionId: SESSION,
    windowIndex: index,
    startedAt,
    endedAt: startedAt + 10,
    audioStatus: "complete",
    audioPcmBase64: PCM,
    audioSampleRate: 16000,
    audioCapturedSamples: 160_000,
    audioGapReason: null,
    ...overrides,
  };
}

function unavailableWindow(index = 0): SystemAudioWindow {
  return audioWindow(index, {
    audioStatus: "unavailable",
    audioPcmBase64: null,
    audioCapturedSamples: 0,
    audioGapReason: "system_audio_unavailable",
  });
}

function frame(index: number, atMs: number): EnvironmentFramePayload {
  return {
    frameId: `00000000-0000-0000-0000-${String(index).padStart(12, "0")}`,
    revision: Math.max(1, index),
    observedAt: atMs / 1_000,
    expiresAt: atMs / 1_000 + 3,
    displayID: 1,
    pixelWidth: 960,
    pixelHeight: 540,
    byteLength: 5,
    mimeType: "image/jpeg",
    jpegBase64: "/9j//9k=",
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
