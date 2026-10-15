import { describe, expect, it, vi } from "vitest";
import { SpeechFacialRuntime } from "./speech-facial";
import { PetScheduler } from "./scheduler";
import type { PetRenderer } from "./pet-renderer";
import type { SpeechFacialApi, SpeechFacialProfile } from "../shared/speech-facial";
import type { PetControlEvent } from "../shared/contracts";

function fixture() {
  const renderer = {
    nativeParameters: () => [
      { id: "CustomHead", minimum: -2, maximum: 2, defaultValue: 0 },
      { id: "ParamEyeLOpen", minimum: 0, maximum: 1, defaultValue: 1 },
      { id: "ParamBreath", minimum: 0, maximum: 1, defaultValue: 0 },
    ],
    setSpeechFacialEnabled: vi.fn(), beginFacialOverlay: vi.fn(),
    endFacialOverlay: vi.fn(), setFacialDemoTargets: vi.fn(),
    setState: vi.fn(), setEmotion: vi.fn(), setLipShape: vi.fn(), cancelGeneration: vi.fn(), setGaze: vi.fn(),
  };
  let profile: SpeechFacialProfile | null = null;
  const api: SpeechFacialApi = {
    prepare: vi.fn(async id => ({ modelId: id, modelName: "模型", blushParameter: null, physicsOutputParameterIds: [],
      manualBindings: { ParamAngleX: [{ parameterId: "CustomHead", lowerValue: -2, neutralValue: 0, upperValue: 2 }] },
      parameterSettings: { ParamEyeLOpen: { enabled: false, multiplier: 1 }, ParamAngleX: { enabled: true, multiplier: 2 } },
    })),
    publish: vi.fn(async value => { profile = value; return { enabled: true }; }),
  };
  const face = new SpeechFacialRuntime(renderer as unknown as PetRenderer, api);
  return { renderer, api, face, profile: () => profile! };
}

describe("production speech expressions", () => {
  it("publishes binary JEV criteria and applies either calibrated endpoint", async () => {
    const f = fixture();
    const original = f.api.prepare;
    f.api.prepare = vi.fn(async id => ({ ...await original(id), criteria: {
      ParamAngleX: { mode: "binary" as const, falseDescription: "不转头", trueDescription: "转到最右" },
    } }));
    await f.face.configure("one");
    expect(f.profile().parameters).toEqual([expect.objectContaining({ id: "ParamAngleX", mode: "binary",
      falseDescription: "不转头", trueDescription: "转到最右" })]);
    f.face.apply({ profileId: f.profile().revision, targets: { ParamAngleX: 30 } });
    expect(f.renderer.setFacialDemoTargets).toHaveBeenLastCalledWith({ CustomHead: 2 });
    f.face.apply({ profileId: f.profile().revision, targets: { ParamAngleX: -30 } });
    expect(f.renderer.setFacialDemoTargets).toHaveBeenLastCalledWith({ CustomHead: -2 });
  });
  it("publishes editable and custom questions and applies their native targets", async () => {
    const f = fixture();
    const native = f.renderer.nativeParameters();
    f.renderer.nativeParameters = () => [...native, { id: "CustomEar", minimum: -1, maximum: 1, defaultValue: 0 }];
    f.api.prepare = vi.fn(async id => ({ modelId: id, modelName: "模型", blushParameter: null,
      physicsOutputParameterIds: [], descriptions: { ParamAngleX: "头部左右偏转" },
      manualBindings: { ParamAngleX: [{ parameterId: "CustomHead", lowerValue: -2, neutralValue: 0, upperValue: 2 }] },
      customParameters: [{ id: "CustomEar", label: "耳朵", description: "耳朵竖起程度",
        levels: ["垂下", "自然", "竖起"] as [string, string, string], enabled: true, multiplier: 1 }],
    }));
    await f.face.configure("one");
    expect(f.profile().parameters.map(value => value.id)).toEqual(["ParamAngleX", "ParamEyeLOpen", "CustomEar"]);
    expect(f.profile().parameters[0]).toMatchObject({ description: "头部左右偏转" });
    expect(f.profile().parameters[2]).toMatchObject({ description: "耳朵竖起程度", levels: ["垂下", "自然", "竖起"] });
    f.face.apply({ profileId: f.profile().revision, targets: { ParamEyeLOpen: 0.8, ParamAngleX: 15, CustomEar: 0.5 } });
    expect(f.renderer.setFacialDemoTargets).toHaveBeenCalledWith({ ParamEyeLOpen: 0.8, CustomHead: 1, CustomEar: 0.5 });
  });
  it("reuses manual bindings, enabled controls and gain without taking physics", async () => {
    const f = fixture();
    await f.face.configure("one");
    expect(f.profile().parameters.map(value => value.id)).toEqual(["ParamAngleX"]);
    // Each new turn picks its side at random: this one right, the next left.
    const draws = [0.9, 0.1];
    const random = vi.spyOn(Math, "random").mockImplementation(() => draws.shift() ?? 0.5);
    f.face.apply({ profileId: f.profile().revision, targets: { ParamAngleX: 15 } });
    expect(f.renderer.setFacialDemoTargets).toHaveBeenCalledWith({ CustomHead: 2 });
    expect(f.renderer.beginFacialOverlay).toHaveBeenCalledTimes(1);
    // Jev says how far to turn; after a rest the next turn picks its side again.
    f.face.apply({ profileId: f.profile().revision, targets: { ParamAngleX: 0 } });
    expect(f.renderer.setFacialDemoTargets).toHaveBeenLastCalledWith({ CustomHead: 0 });
    f.face.apply({ profileId: f.profile().revision, targets: { ParamAngleX: 15 } });
    expect(f.renderer.beginFacialOverlay).toHaveBeenCalledTimes(1);
    expect(f.renderer.setFacialDemoTargets).toHaveBeenLastCalledWith({ CustomHead: -2 });
    f.face.apply({ profileId: f.profile().revision, targets: { ParamAngleX: -15 } });
    expect(f.renderer.setFacialDemoTargets).toHaveBeenCalledTimes(3);
    expect(random).toHaveBeenCalledTimes(2);
    expect(f.renderer.endFacialOverlay).not.toHaveBeenCalled();
    f.face.release();
    expect(f.renderer.endFacialOverlay).toHaveBeenCalledTimes(1);
  });

  it("drops stale profiles and invalid values after switching models", async () => {
    const f = fixture();
    await f.face.configure("one");
    const old = f.profile().revision;
    await f.face.configure("two");
    f.face.apply({ profileId: old, targets: { ParamAngleX: 5 } });
    f.face.apply({ profileId: f.profile().revision, targets: { ParamAngleX: NaN } });
    f.face.apply({ profileId: f.profile().revision, targets: { ParamAngleX: 60 } });
    expect(f.renderer.setFacialDemoTargets).not.toHaveBeenCalled();
  });

  it("discards a preparation that finishes after model invalidation", async () => {
    const f = fixture();
    let resolve!: (value: Awaited<ReturnType<SpeechFacialApi["prepare"]>>) => void;
    f.api.prepare = () => new Promise(done => { resolve = done; });
    const pending = f.face.configure("old");
    f.face.clear();
    resolve({ modelId: "old", modelName: "old", blushParameter: null, physicsOutputParameterIds: [] });
    await pending;
    expect(f.api.publish).toHaveBeenCalledTimes(1);
    expect(f.api.publish).toHaveBeenCalledWith(null);
    expect(f.renderer.setSpeechFacialEnabled).not.toHaveBeenCalledWith(true);
  });

  it("accepts only the current playback generation and releases on stop", async () => {
    const f = fixture();
    await f.face.configure("one");
    const scheduler = new PetScheduler(f.renderer as unknown as PetRenderer, f.face);
    let sequence = 0;
    const event = (type: string, generation: string, payload: PetControlEvent["payload"]) =>
      ({ version: 1, timestampMs: 0, sequence: ++sequence, type, generation, payload });
    const cue = { profileId: f.profile().revision, targets: { ParamAngleX: 5 } };
    scheduler.apply(event("pet.state", "old", { state: "thinking" }));
    scheduler.apply(event("pet.state", "new", { state: "thinking" }));
    scheduler.apply(event("pet.facial", "old", cue));
    expect(f.renderer.setFacialDemoTargets).not.toHaveBeenCalled();
    scheduler.apply(event("pet.audio.start", "new", { sampleRate: 16000 }));
    scheduler.apply(event("pet.facial", "new", cue));
    scheduler.apply(event("pet.audio.pause", "new", {}));
    expect(f.renderer.endFacialOverlay).not.toHaveBeenCalled();
    scheduler.apply(event("pet.generation.cancel", "new", {}));
    scheduler.apply(event("pet.facial", "new", cue));
    expect(f.renderer.setFacialDemoTargets).toHaveBeenCalledTimes(1);
    expect(f.renderer.endFacialOverlay).toHaveBeenCalledTimes(1);
  });
});
