import { describe, expect, it, vi } from "vitest";

vi.mock("./cubism-model-instance", () => ({
  CubismModelInstance: class {},
}));
vi.mock("./cubism-runtime-host", () => ({
  CubismRuntimeHost: class {},
}));
vi.mock("./live2d-behavior-playback", () => ({
  Live2DBehaviorPlayback: class {},
}));
vi.mock("@cubism/rendering/cubismoffscreenmanager", () => ({
  CubismWebGLOffscreenManager: class {},
}));

import {
  GRAPHICS_HEALTH_SAMPLE_INTERVAL_MS,
  GraphicsHealthCheckCadence,
  Live2DRenderer,
} from "./live2d-renderer";
import { PET_STATES, type BehaviorCandidate } from "../shared/contracts";

describe("Live2D click routing", () => {
  it.each(PET_STATES)("plays the configured click independently of state %s", (state) => {
    const candidate = { id: "tap", motionAssetIds: ["wave"], expressionAssetIds: [] };
    const renderer = clickTarget(state, [candidate]);
    Live2DRenderer.prototype.playClick.call(renderer as unknown as Live2DRenderer);
    expect(renderer.behaviorPlayback.playClick.mock.calls).toEqual([[candidate]]);
    expect(renderer.state).toBe(state);
    expect(renderer.syncFrameRate).toHaveBeenCalledOnce();
  });

  it("does nothing when the click candidate list is empty", () => {
    const renderer = clickTarget("unbound", []);
    Live2DRenderer.prototype.playClick.call(renderer as unknown as Live2DRenderer);
    expect(renderer.behaviorPlayback.playClick).not.toHaveBeenCalled();
    expect(renderer.syncFrameRate).not.toHaveBeenCalled();
  });
});

function clickTarget(state: string, candidates: BehaviorCandidate[]) {
  return {
    state,
    currentPackage: { manifest: { behaviors: { states: { click: candidates } } } },
    pickCandidate: (_key: string, values: BehaviorCandidate[]) => values[0] ?? null,
    behaviorPlayback: { playClick: vi.fn() },
    syncFrameRate: vi.fn(),
  };
}

describe("wake state routing", () => {
  const candidate: BehaviorCandidate = { id: "wake", motionAssetIds: ["wave"], expressionAssetIds: [] };
  const target = () => ({
    state: "standby",
    activeState: "standby",
    model: { setSpeaking: vi.fn() },
    syncFrameRate: vi.fn(),
    selectStateCandidate: vi.fn(() => ({ candidate, rotate: true })),
    behaviorPlayback: {
      setState: vi.fn(), setSpeaking: vi.fn(), playAcknowledgement: vi.fn(),
      cancelAcknowledgement: vi.fn(),
    },
  });

  it("plays acknowledgement once rather than as a rotating state pool", () => {
    const renderer = target();
    for (let i = 0; i < 2; i++) {
      Live2DRenderer.prototype.setState.call(renderer as unknown as Live2DRenderer, "acknowledging");
    }
    expect(renderer.behaviorPlayback.playAcknowledgement.mock.calls).toEqual([[candidate]]);
    expect(renderer.behaviorPlayback.setState).not.toHaveBeenCalled();
  });

  it.each(["standby", "unbound"] as const)("releases a wake gesture when entering %s", (state) => {
    const renderer = target();
    renderer.activeState = "acknowledging";
    Live2DRenderer.prototype.setState.call(renderer as unknown as Live2DRenderer, state);
    expect(renderer.behaviorPlayback.cancelAcknowledgement).toHaveBeenCalledOnce();
  });

  it("updates listening immediately without cancelling the wake gesture", () => {
    const renderer = target();
    renderer.activeState = "acknowledging";
    Live2DRenderer.prototype.setState.call(renderer as unknown as Live2DRenderer, "listening");
    expect(renderer.state).toBe("listening");
    expect(renderer.behaviorPlayback.setState).toHaveBeenCalledOnce();
    expect(renderer.behaviorPlayback.cancelAcknowledgement).not.toHaveBeenCalled();
  });
});

describe("GraphicsHealthCheckCadence", () => {
  it("samples the render loop no more than once per second", () => {
    const cadence = new GraphicsHealthCheckCadence();

    expect(cadence.shouldCheck(0, "sampled")).toBe(true);
    expect(cadence.shouldCheck(16, "sampled")).toBe(false);
    expect(cadence.shouldCheck(
      GRAPHICS_HEALTH_SAMPLE_INTERVAL_MS - 1,
      "sampled",
    )).toBe(false);
    expect(cadence.shouldCheck(
      GRAPHICS_HEALTH_SAMPLE_INTERVAL_MS,
      "sampled",
    )).toBe(true);
  });

  it("checks immediately and starts the next sample interval from that check", () => {
    const cadence = new GraphicsHealthCheckCadence();

    expect(cadence.shouldCheck(0, "sampled")).toBe(true);
    expect(cadence.shouldCheck(250, "immediate")).toBe(true);
    expect(cadence.shouldCheck(1_249, "sampled")).toBe(false);
    expect(cadence.shouldCheck(1_250, "sampled")).toBe(true);
  });

  it("checks after a renderer timestamp reset", () => {
    const cadence = new GraphicsHealthCheckCadence();

    expect(cadence.shouldCheck(4_000, "sampled")).toBe(true);
    expect(cadence.shouldCheck(10, "sampled")).toBe(true);
  });
});

describe("explicit facial demo lifecycle", () => {
  it("ignores target writes until explicitly enabled, then restores editing preview on exit", () => {
    vi.stubGlobal("document", { hidden: false });
    try {
      const renderer = Object.create(Live2DRenderer.prototype) as Live2DRenderer;
      const model = {
        beginFacialDemo: vi.fn(), endFacialDemo: vi.fn(),
        setFacialDemoTargets: vi.fn(), setFacialDemoSpeechLevel: vi.fn(),
        setSpeaking: vi.fn(), setLipShape: vi.fn(), setGaze: vi.fn(),
      };
      const behaviorPlayback = { cancel: vi.fn(), playClick: vi.fn(), playTransient: vi.fn() };
      const frameScheduler = { resume: vi.fn(), pause: vi.fn() };
      const render = vi.fn();
      Object.assign(renderer, {
        model, behaviorPlayback, frameScheduler, render,
        destroyed: false, facialDemoActive: false, editingPreview: true,
        manuallyPaused: false, syncFrameRate: vi.fn(),
      });
      renderer.setFacialDemoTargets({ ParamAngleX: 20 });
      renderer.setFacialDemoSpeechLevel(1);
      expect(model.setFacialDemoTargets).not.toHaveBeenCalled();
      expect(model.setFacialDemoSpeechLevel).not.toHaveBeenCalled();
      renderer.beginFacialDemo();
      renderer.setFacialDemoTargets({ ParamAngleX: 20 });
      renderer.setFacialDemoSpeechLevel(0.6);
      expect(model.beginFacialDemo).toHaveBeenCalledOnce();
      expect(model.setFacialDemoTargets).toHaveBeenCalledWith({ ParamAngleX: 20 });
      expect(model.setFacialDemoSpeechLevel).toHaveBeenCalledWith(0.6);
      expect(frameScheduler.resume).toHaveBeenCalledOnce();
      renderer.setState("speaking");
      renderer.setEmotion("happy");
      renderer.playClick();
      renderer.setGaze(1, 1);
      renderer.setLipShape({ volume: 1, a: 1, e: 0, i: 0, o: 0, u: 0 });
      expect(behaviorPlayback.playClick).not.toHaveBeenCalled();
      expect(behaviorPlayback.playTransient).not.toHaveBeenCalled();
      expect(model.setGaze).not.toHaveBeenCalled();
      expect(model.setLipShape).not.toHaveBeenCalled();
      renderer.endFacialDemo();
      expect(model.endFacialDemo).toHaveBeenCalledOnce();
      expect(frameScheduler.pause).toHaveBeenCalledOnce();
      expect(renderer).toMatchObject({ editingPreview: true, facialDemoActive: false, state: "unbound" });
      expect(render).toHaveBeenLastCalledWith(0);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("facial speech overlay lifecycle", () => {
  it("suppresses legacy m assets while keeping facial targets and authored motion separate", () => {
    const renderer = Object.create(Live2DRenderer.prototype) as Live2DRenderer;
    const playback = { cancelEmotion: vi.fn(), playTransient: vi.fn() };
    Object.assign(renderer, { speechFacialEnabled: false, activeEmotion: "neutral",
      behaviorPlayback: playback, pickCandidate: vi.fn(() => ({})), syncFrameRate: vi.fn() });
    renderer.setSpeechFacialEnabled(true);
    renderer.setEmotion("happy");
    renderer.setEmotion("angry");
    renderer.setSpeechFacialEnabled(true);
    expect(playback.playTransient).not.toHaveBeenCalled();
    expect(playback.cancelEmotion).toHaveBeenCalledTimes(1);
    renderer.setSpeechFacialEnabled(false);
    expect(playback.playTransient).toHaveBeenCalledTimes(1);
  });
  it.each([false, true])("fades back to the existing preview without restarting it (static=%s)", (wasEditing) => {
    vi.stubGlobal("document", { hidden: false });
    try {
      let finished = false;
      const renderer = Object.create(Live2DRenderer.prototype) as Live2DRenderer;
      const model = {
        beginFacialOverlay: vi.fn(), endFacialOverlay: vi.fn(),
        hasFacialOverlay: () => !finished, update: vi.fn(), draw: vi.fn(),
      };
      const behaviorPlayback = { cancel: vi.fn(), setState: vi.fn() };
      const frameScheduler = { resume: vi.fn(), pause: vi.fn() };
      const reportFatal = vi.fn();
      Object.assign(renderer, {
        model, behaviorPlayback, frameScheduler, reportFatal,
        editingPreview: wasEditing, lastUpdateAt: 123, state: "listening",
        gl: { isContextLost: () => false }, clear: vi.fn(),
        checkGraphicsHealth: vi.fn(), syncFrameRate: vi.fn(),
      });
      const bindings = { neutralValues: { ParamAngleX: 0 }, gazeParameterIds: [], eyes: [], mouth: null };
      const draw = () => (renderer as unknown as { render(delta: number): void }).render(1 / 60);
      renderer.beginFacialOverlay(bindings);
      expect(model.beginFacialOverlay).toHaveBeenLastCalledWith(bindings, !wasEditing);
      if (!wasEditing) expect(renderer).toMatchObject({ lastUpdateAt: 123 });
      renderer.endFacialOverlay(); draw();
      expect(renderer).toMatchObject({ editingPreview: false });
      expect(frameScheduler.pause).not.toHaveBeenCalled();
      // Starting another sentence during fade retains the original preview mode.
      renderer.beginFacialOverlay(bindings);
      expect(model.beginFacialOverlay).toHaveBeenLastCalledWith(bindings, !wasEditing);
      renderer.endFacialOverlay();
      finished = true; draw();
      expect(renderer).toMatchObject({ editingPreview: wasEditing, facialOverlayState: null, state: "listening" });
      expect(frameScheduler.pause).toHaveBeenCalledTimes(wasEditing ? 1 : 0);
      expect(behaviorPlayback.cancel).not.toHaveBeenCalled();
      expect(behaviorPlayback.setState).not.toHaveBeenCalled();
      expect(reportFatal).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
