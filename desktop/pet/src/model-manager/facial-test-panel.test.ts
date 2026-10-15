import { afterEach, describe, expect, it, vi } from "vitest";
import { packageFixture } from "./test-fixture";
import type { ModelManagerPreview } from "./preview";
import type { FacialTestModelInfo } from "../shared/facial-test";

const player = vi.hoisted(() => ({ start: vi.fn(async () => {}), stop: vi.fn(), destroy: vi.fn(), running: false }));
vi.mock("../facial-demo/jev-player", () => ({ JevTestPlayer: class { constructor() { return player; } } }));
import { FacialTestPanel } from "./facial-test-panel";

class Element {
  disabled = false; value = "这也太好笑了。"; textContent = ""; className = "";
  dataset: Record<string, string> = {};
  listeners = new Map<string, () => void>(); children: Element[] = [];
  attributes = new Map<string, string>(); title = "";
  classList = { toggle: (_name: string, _force?: boolean) => {} };
  setAttribute(name: string, value: string) { this.attributes.set(name, value); }
  getAttribute(name: string) { return this.attributes.get(name) ?? null; }
  addEventListener(event: string, callback: () => void) { this.listeners.set(event, callback); }
  replaceChildren(...children: Element[]) { this.children = children; }
  append(...children: Element[]) { this.children.push(...children); }
  focus() {}
}
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done; }); return { promise, resolve }; }
async function flush() { for (let count = 0; count < 6; count++) await Promise.resolve(); }
function fixture() {
  const nodes = new Map(["textarea", "[data-facial-run]", "[data-facial-stop]", "[data-facial-status]", "[data-facial-sentence]", "[data-facial-values]", "[data-facial-support]", "[data-facial-zoom]", "[data-facial-zoom-value]", "[data-facial-pan-reset]"].map(key => [key, new Element()]));
  const root = { querySelector: (key: string) => nodes.get(key) };
  const prepare = vi.fn(async (modelId: string): Promise<FacialTestModelInfo> => ({ modelId, modelName: modelId, blushParameter: null, physicsOutputParameterIds: [] as string[] }));
  vi.stubGlobal("window", { desktopPet: { facialTest: { prepare } } });
  vi.stubGlobal("document", { createElement: () => new Element() });
  const beforeStart = vi.fn(async () => {});
  const panel = new FacialTestPanel(root as unknown as HTMLElement, beforeStart);
  const model = (id: string) => ({ ...packageFixture(), id, name: id });
  const preview = (id: string, ids = ["ParamEyeLOpen"]) => ({ displayedModelId: () => id, setFacialTestZoom: vi.fn(),
    nativeParameters: () => ids.map(key => ({ id: key, minimum: 0, maximum: 1, defaultValue: 1 })) }) as unknown as ModelManagerPreview;
  return { panel, model, preview, nodes, prepare, beforeStart, click: () => nodes.get("[data-facial-run]")!.listeners.get("click")!() };
}
afterEach(() => { vi.clearAllMocks(); vi.unstubAllGlobals(); player.running = false; });

describe("model manager facial test lifecycle", () => {
  it("offers customization for already matched and missing JEV parameters, with automatic drivers preserved", async () => {
    const f = fixture(); f.panel.setContext(true, f.model("a"), f.preview("a")); await flush();
    const rows = f.nodes.get("[data-facial-support]")!.children;
    expect(rows.flatMap(row => row.children).filter(child => child.dataset.bindStandard)).toHaveLength(18);
    expect(rows.find(row => row.dataset.standardId === "ParamEyeLOpen")!.children.some(child => child.dataset.bindStandard === "ParamEyeLOpen")).toBe(true);
    // Parts the model animates itself (breathing, physics) are not listed at all.
    expect(rows.some(row => row.dataset.standardId === "ParamBreath")).toBe(false);
  });

  it("loads disabled generation and gain settings instead of silently re-enabling them", async () => {
    const f = fixture();
    f.prepare.mockResolvedValue({ modelId: "a", modelName: "a", blushParameter: null, physicsOutputParameterIds: [], parameterSettings: { ParamEyeLOpen: { enabled: false, multiplier: 2 } } });
    f.panel.setContext(true, f.model("a"), f.preview("a")); await flush();
    expect(f.nodes.get("[data-facial-run]")!.disabled).toBe(true);
    expect(f.nodes.get("[data-facial-status]")!.textContent).toContain("未开启任何参数");
    const eye = f.nodes.get("[data-facial-support]")!.children.find(row => row.dataset.standardId === "ParamEyeLOpen")!;
    expect(eye.children.find(child => child.className === "le-switch")!.getAttribute("aria-checked")).toBe("false");
    expect(eye.children.find(child => child.className.startsWith("le-param__scale"))!.textContent).toBe("2.0×");
  });
  it("switches a custom part on its own entry, not through the standard parts' settings", async () => {
    const f = fixture();
    const custom = { id: "ParamEyeEffect", label: "目 エフェクト", description: "星星眼", levels: ["无", "有"], mode: "score" as const, enabled: true, multiplier: 1 };
    f.prepare.mockResolvedValue({ modelId: "a", modelName: "a", blushParameter: null, physicsOutputParameterIds: [], customParameters: [custom] });
    const saveCustomParameter = vi.fn(async () => ({ modelId: "a", modelName: "a", blushParameter: null, physicsOutputParameterIds: [], customParameters: [{ ...custom, enabled: false }] }));
    const saveBinding = vi.fn();
    (window as unknown as { desktopPet: { facialTest: Record<string, unknown> } }).desktopPet.facialTest.saveCustomParameter = saveCustomParameter;
    (window as unknown as { desktopPet: { facialTest: Record<string, unknown> } }).desktopPet.facialTest.saveBinding = saveBinding;
    f.panel.setContext(true, f.model("a"), f.preview("a", ["ParamEyeLOpen", "ParamEyeEffect"])); await flush();
    const row = () => f.nodes.get("[data-facial-support]")!.children.find(item => item.dataset.standardId === "ParamEyeEffect")!;
    row().children.find(child => child.className === "le-switch")!.listeners.get("click")!(); await flush();
    expect(saveBinding).not.toHaveBeenCalled();
    expect(saveCustomParameter).toHaveBeenCalledWith("a", { ...custom, enabled: false });
    expect(row().children.find(child => child.className === "le-switch")!.getAttribute("aria-checked")).toBe("false");
  });

  it("cancels preparation on model switch and starts only the newly selected preview", async () => {
    const f = fixture();
    const a = f.preview("a"); f.panel.setContext(true, f.model("a"), a); await flush();
    const pending = deferred<{ modelId: string; modelName: string; blushParameter: null; physicsOutputParameterIds: string[] }>();
    f.prepare.mockImplementationOnce(() => pending.promise);
    f.click(); await flush();
    const b = f.preview("b", ["ParamEyeROpen"]); f.panel.setContext(true, f.model("b"), b);
    pending.resolve({ modelId: "a", modelName: "a", blushParameter: null, physicsOutputParameterIds: [] }); await flush();
    expect(player.start).not.toHaveBeenCalled();
    f.click(); await flush();
    expect(player.start).toHaveBeenCalledWith("b", "这也太好笑了。", expect.arrayContaining([expect.objectContaining({ id: "ParamEyeROpen" })]), b, undefined);
    expect(player.start).toHaveBeenCalledOnce();
  });

  it("does not start after the tab is left during preparation", async () => {
    const f = fixture(); const preview = f.preview("a");
    f.panel.setContext(true, f.model("a"), preview); await flush();
    // Entering shows the whole body, not a close-up of the face.
    expect(preview.setFacialTestZoom).toHaveBeenLastCalledWith(1);
    expect(preview.setFacialTestZoom).not.toHaveBeenCalledWith(3);
    const before = deferred<void>(); f.beforeStart.mockImplementationOnce(() => before.promise);
    f.click(); await flush(); f.panel.setContext(false, null, null);
    before.resolve(); await flush();
    expect(player.start).not.toHaveBeenCalled();
    expect(f.nodes.get("[data-facial-run]")!.disabled).toBe(true);
    expect(preview.setFacialTestZoom).toHaveBeenLastCalledWith(1);
  });

  it("disables unsupported models and avoids changing the model selection", async () => {
    const f = fixture(); f.panel.setContext(true, f.model("custom"), f.preview("custom", ["CustomEye"])); await flush();
    expect(f.nodes.get("[data-facial-run]")!.disabled).toBe(true);
    expect(f.nodes.get("[data-facial-status]")!.textContent).toContain("没有可直接控制的参数");
    expect(player.start).not.toHaveBeenCalled();
    // The 一键配置 row, 18 decision-model parts and the "添加自定义部位" row.
    expect(f.nodes.get("[data-facial-support]")!.children).toHaveLength(20);
    expect(f.nodes.get("[data-facial-support]")!.children[0]!.children.at(-1)!.dataset.autoSetupOpen).toBe("");
    expect(f.nodes.get("[data-facial-support]")!.children.at(-1)!.children.at(-1)!.dataset.addCustomParameter).toBe("");
  });
});
