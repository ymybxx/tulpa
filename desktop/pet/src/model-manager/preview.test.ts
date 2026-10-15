import { afterEach, describe, expect, it, vi } from "vitest";
import { packageFixture } from "./test-fixture";

const renderer = vi.hoisted(() => ({
  setEditingPreview: vi.fn(), loadModel: vi.fn(async () => {}), displayedModelId: vi.fn(() => "girl"),
  setState: vi.fn(), setGaze: vi.fn(), setLipShape: vi.fn(), resetPreview: vi.fn(),
  configureDefaultAppearance: vi.fn(), previewMotionAsset: vi.fn(), previewExpressionAsset: vi.fn(),
  previewBehaviorCandidate: vi.fn(async () => {}),
  beginFacialOverlay: vi.fn(), endFacialOverlay: vi.fn(),
  beginFacialDemo: vi.fn(), endFacialDemo: vi.fn(), setFacialDemoTargets: vi.fn(),
  setFacialDemoSpeechLevel: vi.fn(), facialDemoSnapshot: vi.fn(() => ({ active: true, values: { ParamEyeLOpen: .5 }, targets: {} })),
  highlightLayers: vi.fn(), nativeLayers: vi.fn(() => []), pickLayers: vi.fn(() => ["Hat"]),
}));
vi.mock("../renderer/live2d-renderer", () => ({ Live2DRenderer: class { constructor() { return renderer; } } }));
import { ModelManagerPreview } from "./preview";

afterEach(() => { vi.clearAllMocks(); vi.useRealTimers(); });

describe("model-manager preview modes", () => {
  it("layers speech over the existing animation without resetting or restarting its state", () => {
    const preview = new ModelManagerPreview({} as HTMLCanvasElement, {} as HTMLElement);
    preview.playAwakeIdle();
    renderer.setState.mockClear(); renderer.setEditingPreview.mockClear(); renderer.resetPreview.mockClear();
    const bindings = { mouth: null, eyes: [], neutralValues: { ParamAngleX: 0 }, gazeParameterIds: [] };
    preview.beginFacialSpeech(bindings);
    preview.applyFacialTest({ ParamAngleX: 8 });
    preview.endFacialTest();
    expect(renderer.beginFacialOverlay).toHaveBeenCalledWith(bindings);
    expect(renderer.endFacialOverlay).toHaveBeenCalledOnce();
    expect(renderer.beginFacialDemo).not.toHaveBeenCalled();
    expect(renderer.setEditingPreview).not.toHaveBeenCalled();
    expect(renderer.setState).not.toHaveBeenCalled();
    expect(renderer.resetPreview).not.toHaveBeenCalled();
  });
  it("restores the previous static or animated mode after temporary JEV control", () => {
    const preview = new ModelManagerPreview({} as HTMLCanvasElement, {} as HTMLElement);
    preview.beginFacialTest();
    preview.applyFacialTest({ ParamEyeLOpen: .2 });
    preview.setFacialTestLip({ volume: .4, a: .4, e: 0, i: 0, o: 0, u: 0 });
    expect(renderer.setLipShape).toHaveBeenLastCalledWith({ volume: .4, a: .4, e: 0, i: 0, o: 0, u: 0 });
    expect(renderer.setFacialDemoTargets).toHaveBeenCalledWith({ ParamEyeLOpen: .2 });
    expect(preview.facialTestSnapshot().values.ParamEyeLOpen).toBe(.5);
    preview.endFacialTest();
    expect(renderer.endFacialDemo).toHaveBeenCalledOnce();
    expect(renderer.setEditingPreview).toHaveBeenLastCalledWith(true);
    preview.playAwakeIdle();
    preview.beginFacialTest();
    preview.endFacialTest();
    expect(renderer.setEditingPreview).toHaveBeenLastCalledWith(false);
    expect(renderer.setState).toHaveBeenLastCalledWith("listening");
    expect(renderer.setFacialDemoSpeechLevel).toHaveBeenLastCalledWith(0);
  });

  it("ends facial control before an existing preview action takes ownership", () => {
    const preview = new ModelManagerPreview({} as HTMLCanvasElement, {} as HTMLElement);
    preview.beginFacialTest(); preview.playMotion("wave");
    expect(renderer.endFacialDemo).toHaveBeenCalledOnce();
    expect(renderer.previewMotionAsset).toHaveBeenLastCalledWith("wave");
  });
  it("enables canvas layer selection only for the appearance editor and clears its highlight on exit", () => {
    const canvas = { addEventListener: vi.fn(), removeEventListener: vi.fn(), classList: { toggle: vi.fn() },
      getBoundingClientRect: () => ({ left: 20, top: 30 }) };
    const preview = new ModelManagerPreview(canvas as unknown as HTMLCanvasElement, {} as HTMLElement);
    const picked = vi.fn();
    preview.setLayerPickHandler(picked);
    const onClick = canvas.addEventListener.mock.calls.find((call) => call[0] === "click")![1] as (event: MouseEvent) => void;
    onClick({ clientX: 45, clientY: 70 } as MouseEvent);
    expect(renderer.pickLayers).toHaveBeenCalledWith(25, 40);
    expect(picked).toHaveBeenCalledWith(["Hat"], { x: 25, y: 40 });
    preview.setLayerPickHandler(null);
    expect(canvas.removeEventListener).toHaveBeenLastCalledWith("click", onClick);
    expect(renderer.highlightLayers).toHaveBeenLastCalledWith([]);
  });
  it("starts editing without a standby motion and enters animation only explicitly", async () => {
    const changed = vi.fn();
    const preview = new ModelManagerPreview({} as HTMLCanvasElement, {} as HTMLElement, undefined, changed);
    await preview.load({ model: packageFixture(), appearance: { modelId: "girl", streamId: "qa", revision: 0, patchSets: [], atlases: [] } });
    expect(renderer.setEditingPreview).toHaveBeenLastCalledWith(true);
    expect(renderer.setState).not.toHaveBeenCalled();
    preview.playAwakeIdle();
    expect(renderer.setEditingPreview).toHaveBeenLastCalledWith(false);
    expect(renderer.setState).toHaveBeenLastCalledWith("listening");
    expect(renderer.setState).not.toHaveBeenCalledWith("standby");
    const defaults = { expressionAssetIds: [], parameterValues: { Param95: 0.66 } };
    preview.configureDefaultAppearance(defaults);
    expect(renderer.setEditingPreview).toHaveBeenLastCalledWith(true);
    expect(renderer.configureDefaultAppearance).toHaveBeenLastCalledWith(defaults);
    expect(changed).toHaveBeenLastCalledWith(false);
  });

  it("cancels old demonstration timers when the user begins editing", () => {
    vi.useFakeTimers();
    const preview = new ModelManagerPreview({} as HTMLCanvasElement, {} as HTMLElement);
    preview.startValidationSequence();
    preview.configureDefaultAppearance({ expressionAssetIds: [], parameterValues: { Param95: 1 } });
    renderer.setState.mockClear(); renderer.setLipShape.mockClear();
    vi.runAllTimers();
    expect(renderer.setState).not.toHaveBeenCalled();
    expect(renderer.setLipShape).not.toHaveBeenCalled();
    expect(renderer.setEditingPreview).toHaveBeenLastCalledWith(true);
  });

  it("retains motion, combined-candidate, and mouth tests after static editing", async () => {
    const preview = new ModelManagerPreview({} as HTMLCanvasElement, {} as HTMLElement);
    preview.playMotion("wave");
    expect(renderer.setEditingPreview).toHaveBeenLastCalledWith(false);
    expect(renderer.previewMotionAsset).toHaveBeenLastCalledWith("wave");
    preview.reset();
    expect(renderer.setEditingPreview).toHaveBeenLastCalledWith(true);
    const candidate = { id: "combo", motionAssetIds: [], expressionAssetIds: ["smile"] };
    await preview.previewCandidate(candidate);
    expect(renderer.previewBehaviorCandidate).toHaveBeenLastCalledWith(candidate, "transient");
    expect(renderer.setEditingPreview).toHaveBeenLastCalledWith(false);
    preview.previewLipShape({ volume: 0.5, a: 0.5, e: 0, i: 0, o: 0, u: 0 });
    expect(renderer.setState).toHaveBeenLastCalledWith("speaking");
    preview.stopLipPreview();
    expect(renderer.setEditingPreview).toHaveBeenLastCalledWith(true);
  });

  it("starts an expression from a clean editing pose even after animated idle", () => {
    const preview = new ModelManagerPreview({} as HTMLCanvasElement, {} as HTMLElement);
    preview.playAwakeIdle();
    renderer.setEditingPreview.mockClear(); renderer.resetPreview.mockClear();
    preview.showExpression("love");
    expect(renderer.setEditingPreview.mock.calls).toEqual([[true], [false]]);
    expect(renderer.resetPreview).toHaveBeenCalledOnce();
    expect(renderer.previewExpressionAsset).toHaveBeenLastCalledWith("love");
  });

  it("clears an explicitly tested sleep candidate before returning to awake idle", async () => {
    const preview = new ModelManagerPreview({} as HTMLCanvasElement, {} as HTMLElement);
    await preview.previewCandidate({ id: "sleep", motionAssetIds: ["sleep"], expressionAssetIds: [] }, "state");
    renderer.setEditingPreview.mockClear(); renderer.resetPreview.mockClear(); renderer.setState.mockClear();
    preview.playAwakeIdle();
    expect(renderer.resetPreview).toHaveBeenCalledOnce();
    expect(renderer.setEditingPreview.mock.calls).toEqual([[true], [false]]);
    expect(renderer.setState.mock.calls).toEqual([["listening"]]);
  });
});
