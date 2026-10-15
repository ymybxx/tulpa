import { describe, expect, it } from "vitest";

import { applySenseAction, parseSenseAction, senseViews, type SenseInputs, type SenseKey, type SenseView } from "./pet-senses";

const inputs = (overrides: Partial<SenseInputs> = {}): SenseInputs => ({
  perception: {
    operationPerceptionEnabled: false, screenPerceptionEnabled: true, audioPerceptionEnabled: true,
    excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
  },
  microphoneEnabled: true,
  proactiveCompanionEnabled: true,
  audioSupported: true,
  permissions: { microphone: "granted", screen: "granted", systemAudio: "granted" },
  waiting: null,
  ...overrides,
});

const byKey = (input: SenseInputs): Record<SenseKey, SenseView> =>
  Object.fromEntries(senseViews(input).map((sense) => [sense.key, sense])) as Record<SenseKey, SenseView>;

describe("senseViews", () => {
  it("lists four senses in capsule order, online when switched on and permitted", () => {
    const views = senseViews(inputs());
    expect(views.map((view) => [view.key, view.status])).toEqual([
      ["screen", "on"], ["audio", "on"], ["microphone", "on"], ["inner", "on"],
    ]);
  });

  it("points the screen and computer sound at Screen Recording, which covers both", () => {
    const senses = byKey(inputs({ permissions: { microphone: "granted", screen: "denied", systemAudio: "not-determined" } }));
    expect(senses.screen).toMatchObject({ status: "noperm", fix: "screen" });
    expect(senses.audio).toMatchObject({ status: "noperm", fix: "screen" });
    expect(senses.microphone.status).toBe("on");
    const sound = byKey(inputs({ permissions: { microphone: "granted", screen: "granted", systemAudio: "denied" } }));
    expect(sound.audio).toMatchObject({ status: "noperm", fix: "systemAudio" });
    expect(sound.screen.status).toBe("on");
  });

  it("shows waiting while the user is in System Settings", () => {
    const senses = byKey(inputs({ permissions: { microphone: "denied", screen: "granted", systemAudio: "granted" }, waiting: "microphone" }));
    expect(senses.microphone).toMatchObject({ status: "waiting", fix: "microphone" });
  });

  it("marks computer sound as not in effect while 内心 is off", () => {
    const base = inputs();
    const senses = byKey(inputs({ perception: { ...base.perception, innerLifeEnabled: false } }));
    expect(senses.audio).toMatchObject({ status: "partial", label: "要打开内心", fix: null });
    expect(senses.inner.status).toBe("off");
  });

  it("asks for nothing when a sense is off or the platform does not ask", () => {
    const base = inputs();
    const off = byKey(inputs({ perception: { ...base.perception, screenPerceptionEnabled: false }, permissions: { microphone: "granted", screen: "denied", systemAudio: "granted" } }));
    expect(off.screen).toMatchObject({ status: "off", fix: null });
    expect(byKey(inputs({ permissions: { microphone: "granted", screen: "not-required", systemAudio: "not-required" } })).screen.status).toBe("on");
  });
});

describe("applySenseAction", () => {
  const current = { perception: inputs().perception, microphoneEnabled: true, proactiveCompanionEnabled: true };

  it("switches each sense on its own", () => {
    expect(applySenseAction(current, { type: "toggle", key: "microphone" }, true)).toMatchObject({
      microphoneEnabled: false, perception: { audioPerceptionEnabled: true, screenPerceptionEnabled: true },
    });
    const audioOff = applySenseAction(current, { type: "toggle", key: "audio" }, true);
    expect(audioOff).toMatchObject({ microphoneEnabled: true, perception: { audioPerceptionEnabled: false } });
    expect(applySenseAction(audioOff, { type: "toggle", key: "audio" }, false).perception.audioPerceptionEnabled).toBe(false);
  });

  it("pauses companionship and optionally closes every sense", () => {
    expect(applySenseAction(current, { type: "pause", closeSenses: false }, true))
      .toEqual({ ...current, proactiveCompanionEnabled: false });
    expect(applySenseAction(current, { type: "pause", closeSenses: true }, true)).toEqual({
      microphoneEnabled: false, proactiveCompanionEnabled: false,
      perception: { ...current.perception, screenPerceptionEnabled: false, audioPerceptionEnabled: false, innerLifeEnabled: false },
    });
  });

  it("resumes without reopening senses", () => {
    const paused = applySenseAction(current, { type: "pause", closeSenses: true }, true);
    expect(applySenseAction(paused, { type: "resume" }, true)).toEqual({ ...paused, proactiveCompanionEnabled: true });
  });
});

describe("parseSenseAction", () => {
  it("accepts only known actions from the HUD", () => {
    expect(parseSenseAction({ type: "toggle", key: "inner" })).toEqual({ type: "toggle", key: "inner" });
    expect(parseSenseAction({ type: "toggle", key: "nose" })).toBeNull();
    expect(parseSenseAction({ type: "toggle", key: "eye" })).toBeNull();
    expect(parseSenseAction({ type: "fix", kind: "camera" })).toBeNull();
    expect(parseSenseAction({ type: "pause" })).toBeNull();
    expect(parseSenseAction("settings")).toBeNull();
  });
});
