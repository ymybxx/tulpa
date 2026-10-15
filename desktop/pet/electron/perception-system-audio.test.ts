import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  parseSystemAudioWindow,
  PerceptionManager,
  type PerceptionHelperProcess,
  type SystemAudioWindow,
} from "./perception-manager";

const PCM = Buffer.alloc(320_000).toString("base64");

afterEach(() => vi.useRealTimers());

function window(overrides: Partial<SystemAudioWindow> = {}): SystemAudioWindow {
  return {
    captureSessionId: "11111111-1111-4111-8111-111111111111",
    windowIndex: 0,
    startedAt: 100,
    endedAt: 110,
    audioStatus: "complete",
    audioPcmBase64: PCM,
    audioSampleRate: 16000,
    audioCapturedSamples: 160_000,
    audioGapReason: null,
    ...overrides,
  };
}

class FakeAudioHelper extends EventEmitter implements PerceptionHelperProcess {
  readonly pid = 7;
  readonly stdin = new PassThrough();
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  kill(): boolean { return true; }
  send(event: string, body: unknown): void {
    this.stdout.write(`${JSON.stringify({
      protocolVersion: 1,
      sequence: 1,
      sentAt: 100,
      event,
      body,
    })}\n`);
  }
}

function harness(systemAudio = true) {
  vi.useFakeTimers();
  const helper = new FakeAudioHelper();
  const manager = new PerceptionManager({ helperPath: "/fake", spawnProcess: () => helper });
  const modules = {
    workspace: false,
    screen: true,
    accessibility: false,
    camera: false,
    systemAudio,
  };
  manager.start(modules);
  const capability = {
    supported: true,
    requested: false,
    state: "disabled",
    permission: "not_determined",
  };
  helper.send("hello", { capabilities: {
    workspace: capability,
    screen: capability,
    accessibility: capability,
    camera: capability,
    systemAudio: {
      ...capability,
      requested: true,
      state: "running",
      permission: "granted",
    },
  } });
  return { helper, manager, modules };
}

describe("private system audio window transport", () => {
  it.each([2, 5, 6, 9, 10])("accepts a %s-second cut only with aligned PCM", (seconds) => {
    const captured = window({ endedAt: 100 + seconds, audioCapturedSamples: seconds * 16_000,
      audioPcmBase64: Buffer.alloc(seconds * 32_000).toString("base64") });
    expect(parseSystemAudioWindow(captured)).toEqual(captured);
    expect(() => parseSystemAudioWindow({ ...captured, endedAt: captured.endedAt + 1 })).toThrow();
  });
  it("accepts compact captured silence and rejects gaps or conflicting PCM", () => {
    const quiet = window({ audioSilence: true, audioPcmBase64: null });
    expect(parseSystemAudioWindow(quiet)).toEqual(quiet);
    expect(JSON.stringify(quiet).length).toBeLessThan(500);
    for (const patch of [
      { audioStatus: "partial" }, { audioStatus: "unavailable" },
      { audioCapturedSamples: 0 }, { audioGapReason: "missing" },
      { audioPcmBase64: PCM }, { audioSilence: null }, { audioSilence: "true" },
    ]) expect(() => parseSystemAudioWindow({ ...quiet, ...patch })).toThrow();
  });
  it("validates the exact ten-second PCM and unavailable forms", () => {
    expect(parseSystemAudioWindow(window())).toEqual(window());
    const unavailable = window({
      audioStatus: "unavailable",
      audioPcmBase64: null,
      audioCapturedSamples: 0,
      audioGapReason: "system_audio_unavailable",
    });
    expect(parseSystemAudioWindow(unavailable)).toEqual(unavailable);

    // Swift's synthesized Codable omits nil optionals rather than emitting
    // explicit JSON nulls. Normalize both native wire forms at the boundary.
    const { audioGapReason: _gap, ...nativeComplete } = window();
    expect(parseSystemAudioWindow(nativeComplete)).toEqual(window());
    const {
      audioPcmBase64: _pcm,
      audioGapReason: _unavailableGap,
      ...nativeUnavailable
    } = unavailable;
    expect(parseSystemAudioWindow(nativeUnavailable)).toEqual({
      ...unavailable,
      audioGapReason: null,
    });

    for (const invalid of [
      { ...window(), endedAt: 109 },
      { ...window(), audioPcmBase64: "AA==" },
      { ...window(), audioSampleRate: 48_000 },
      { ...window(), windowIndex: -1 },
      { ...window(), audioCapturedSamples: 159_999 },
      { ...window(), audioCapturedSamples: 160_001 },
      { ...window(), microphone: true },
      { ...unavailable, audioCapturedSamples: 1 },
      { ...unavailable, audioPcmBase64: PCM },
    ]) expect(() => parseSystemAudioWindow(invalid)).toThrow();
  });

  it("routes a complete window only to the private listener", () => {
    const { helper, manager } = harness();
    const audio = vi.fn();
    const generic = vi.fn();
    const observations = vi.fn();
    manager.on("system-audio-window", audio);
    manager.on("event", generic);
    manager.on("observation", observations);
    helper.send("system_audio_window", window());

    expect(audio).toHaveBeenCalledTimes(1);
    expect(audio).toHaveBeenCalledWith(window());
    expect(generic).not.toHaveBeenCalled();
    expect(observations).not.toHaveBeenCalled();
    expect(manager.currentStatus().capabilities?.systemAudio?.state).toBe("running");
  });

  it("forwards audio with the screen input disabled", () => {
    const { helper, manager, modules } = harness();
    const audio = vi.fn();
    manager.on("system-audio-window", audio);
    manager.configure({ ...modules, screen: false, systemAudio: true });
    helper.send("system_audio_window", window());
    expect(audio).toHaveBeenCalledWith(window());
  });

  it("forwards the visual clock without forwarding disabled audio samples", () => {
    const { helper, manager } = harness(false);
    const audio = vi.fn();
    manager.on("system-audio-window", audio);
    const clockWindow = window({ audioStatus: "unavailable", audioPcmBase64: null,
      audioCapturedSamples: 0, audioGapReason: "system_audio_disabled" });
    helper.send("system_audio_window", clockWindow);
    helper.send("system_audio_window", window());
    expect(audio).toHaveBeenCalledTimes(1);
    expect(audio).toHaveBeenCalledWith(clockWindow);
  });

  it("keeps the native session intact across repeated configure calls", () => {
    const { helper, manager, modules } = harness();
    const audio = vi.fn();
    manager.on("system-audio-window", audio);
    helper.send("system_audio_window", window());
    manager.configure({ ...modules });
    manager.configure({ ...modules });
    helper.send("system_audio_window", window({
      windowIndex: 1,
      startedAt: 110,
      endedAt: 120,
    }));

    expect(audio.mock.calls.map(([value]) => [
      value.captureSessionId,
      value.windowIndex,
    ])).toEqual([
      [window().captureSessionId, 0],
      [window().captureSessionId, 1],
    ]);
  });

  it("drops windows while disabled without publishing private bytes generically", () => {
    const { helper, manager, modules } = harness(false);
    const audio = vi.fn();
    const generic = vi.fn();
    manager.on("system-audio-window", audio);
    manager.on("event", generic);
    helper.send("system_audio_window", window());
    manager.configure({ ...modules, systemAudio: true });
    helper.send("system_audio_window", window());
    manager.configure({ ...modules, systemAudio: false });
    helper.send("system_audio_window", window({ windowIndex: 1, startedAt: 110, endedAt: 120 }));

    expect(audio).toHaveBeenCalledTimes(1);
    expect(generic).not.toHaveBeenCalled();
  });

  it("rejects mutations with content-free diagnostics and discards legacy raw events", () => {
    const { helper, manager } = harness();
    const errors: Error[] = [];
    const generic = vi.fn();
    const audio = vi.fn();
    manager.on("protocol-error", (error: Error) => errors.push(error));
    manager.on("event", generic);
    manager.on("system-audio-window", audio);
    helper.send("system_audio_window", { ...window(), audioPcmBase64: "sensitive audio" });
    helper.send("system_audio_chunk", { private: "legacy bytes" });
    helper.send("system_audio_gap", { private: "legacy reason" });

    expect(errors.map((error) => error.message)).toEqual(["Invalid system audio payload"]);
    expect(audio).not.toHaveBeenCalled();
    expect(generic).not.toHaveBeenCalled();
  });
});
