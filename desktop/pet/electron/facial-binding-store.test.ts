import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import catalog from "../src/shared/facial-demo-parameters.json";
import { FacialBindingStore } from "./facial-binding-store";

const directories: string[] = [];
afterEach(async () => { await Promise.all(directories.splice(0).map(directory => rm(directory, { recursive: true, force: true }))); });
async function fixture() {
  const directory = await mkdtemp(path.join(os.tmpdir(), "facial-binding-test-")); directories.push(directory);
  return { directory, store: new FacialBindingStore(directory) };
}
const binding = { parameterId: "CustomEye", lowerValue: 80, neutralValue: 40, upperValue: 20 };
describe("local per-model facial bindings", () => {
  it("loads old bindings with default controls and persists independent controls atomically", async () => {
    const { directory, store } = await fixture();
    await writeFile(path.join(directory, "facial-test-bindings.json"), JSON.stringify({ version: 1, models: { a: { ParamEyeLOpen: binding } } }));
    expect(await store.getConfiguration("a")).toEqual({ manualBindings: { ParamEyeLOpen: [binding] }, parameterSettings: {}, descriptions: {}, criteria: {}, customParameters: [] });
    await Promise.all([
      store.save("a", "ParamEyeLOpen", binding, { enabled: false, multiplier: 2 }),
      store.save("a", "ParamMouthForm", null, { enabled: true, multiplier: .5 }),
    ]);
    const reopened = new FacialBindingStore(directory);
    expect(await reopened.getConfiguration("a")).toEqual({ manualBindings: { ParamEyeLOpen: [binding] }, parameterSettings: {
      ParamEyeLOpen: { enabled: false, multiplier: 2 }, ParamMouthForm: { enabled: true, multiplier: .5 },
    }, descriptions: {}, criteria: {}, customParameters: [] });
    expect(await reopened.getConfiguration("b")).toEqual({ manualBindings: {}, parameterSettings: {}, descriptions: {}, criteria: {}, customParameters: [] });
    // Clearing a custom mapping preserves its enable/gain setting unless explicitly reset.
    await reopened.save("a", "ParamEyeLOpen", null);
    expect((await reopened.getConfiguration("a")).parameterSettings.ParamEyeLOpen).toEqual({ enabled: false, multiplier: 2 });
    await reopened.save("a", "ParamEyeLOpen", null, { enabled: true, multiplier: 1 });
    expect((await reopened.getConfiguration("a")).parameterSettings).not.toHaveProperty("ParamEyeLOpen");
  });

  it("forgets settings of parts that left the standard set, and a custom part that joined it", async () => {
    const { directory, store } = await fixture();
    const tongue = { id: "ParamTongueOut", label: "吐舌", description: "吐舌头", levels: ["不吐", "吐"], enabled: true, multiplier: 1 };
    const own = { ...tongue, id: "Param43", label: "挥手" };
    await writeFile(path.join(directory, "facial-test-bindings.json"), JSON.stringify({ version: 1,
      models: { a: { ParamArmLA: binding, ParamEyeLOpen: binding } },
      parameterSettings: { a: { ParamShoulderY: { enabled: false, multiplier: 2 }, ParamAngleX: { enabled: false, multiplier: 1 } } },
      customParameters: { a: [tongue, own] } }));
    expect(await store.getConfiguration("a")).toEqual({ manualBindings: { ParamEyeLOpen: [binding] },
      parameterSettings: { ParamAngleX: { enabled: false, multiplier: 1 } }, descriptions: {}, criteria: {},
      customParameters: [{ ...own, mode: "score" }] });
  });

  it("keeps a part's own name with default controls and drops it when cleared", async () => {
    const { store } = await fixture();
    await store.save("a", "ParamAngleX", null, { enabled: true, multiplier: 1, label: "  歪头看你  " });
    expect((await store.getConfiguration("a")).parameterSettings.ParamAngleX).toEqual({ enabled: true, multiplier: 1, label: "歪头看你" });
    await store.save("a", "ParamAngleX", null, { enabled: true, multiplier: 1 });
    expect((await store.getConfiguration("a")).parameterSettings).not.toHaveProperty("ParamAngleX");
  });

  it("rejects malformed control settings without overwriting valid data", async () => {
    const { store } = await fixture();
    for (const multiplier of [-1, NaN, Infinity]) {
      await expect(store.save("a", "ParamEyeLOpen", binding, { enabled: true, multiplier })).rejects.toThrow("倍率");
    }
    await expect(store.save("a", "ParamEyeLOpen", null, { enabled: 1 as unknown as boolean, multiplier: 1 })).rejects.toThrow("开关");
    expect(await store.get("a")).toEqual({});
  });
  it("preserves concurrent independent edits, survives restart, and isolates models", async () => {
    const { store, directory } = await fixture();
    await Promise.all([
      store.save("a", "ParamEyeLOpen", binding),
      store.save("a", "ParamEyeROpen", binding),
      store.save("b", "ParamBrowLY", binding),
    ]);
    const reopened = new FacialBindingStore(directory);
    expect(await reopened.get("a")).toEqual({ ParamEyeLOpen: [binding], ParamEyeROpen: [binding] });
    expect(await reopened.get("b")).toEqual({ ParamBrowLY: [binding] });
    await reopened.save("a", "ParamEyeLOpen", null);
    expect(await store.get("a")).toEqual({ ParamEyeROpen: [binding] });
    expect(await store.get("missing")).toEqual({});
  });

  it("rejects invalid or automatic controls and recovers for the next valid save", async () => {
    const { store } = await fixture();
    await expect(store.save("a", "ParamBreath", binding)).rejects.toThrow("只能绑定");
    await expect(store.save("a", "Unknown", null)).rejects.toThrow("只能绑定");
    await expect(store.save("a", "ParamEyeLOpen", { ...binding, neutralValue: 99 })).rejects.toThrow("中性值");
    await expect(store.save("__proto__", "ParamEyeLOpen", binding)).rejects.toThrow("模型 ID");
    const original = { ...binding };
    const pending = store.save("a", "ParamEyeLOpen", original);
    original.parameterId = "Changed";
    await pending;
    expect(await store.get("a")).toEqual({ ParamEyeLOpen: [binding] });
  });

  it("keeps several parameters for one part, VTube Studio style, and refuses a parameter twice", async () => {
    const { store } = await fixture();
    const cheek = { parameterId: "CustomCheek", lowerValue: 0, neutralValue: 0, upperValue: 1 };
    await store.save("a", "ParamMouthForm", [{ ...binding, parameterId: "Smile", lowerValue: -1, neutralValue: 0, upperValue: 1 }, cheek]);
    expect((await store.get("a")).ParamMouthForm!.map(output => output.parameterId)).toEqual(["Smile", "CustomCheek"]);
    await expect(store.save("a", "ParamMouthForm", [cheek, cheek])).rejects.toThrow("只能出现一次");
    await store.save("a", "ParamMouthForm", []);
    expect(await store.get("a")).toEqual({});
  });

  it("never overwrites an unreadable existing configuration", async () => {
    const { store, directory } = await fixture();
    const file = path.join(directory, "facial-test-bindings.json");
    await writeFile(file, "broken-config");
    await expect(store.save("a", "ParamEyeLOpen", binding)).rejects.toThrow();
    expect(await readFile(file, "utf8")).toBe("broken-config");
    await writeFile(file, '{"version":1,"models":{}}');
    await store.save("a", "ParamEyeLOpen", binding);
    expect(await store.get("a")).toEqual({ ParamEyeLOpen: [binding] });
  });

  it("persists editable descriptions and native custom parameters per model", async () => {
    const { store, directory } = await fixture();
    const custom = { id: "CustomEar", label: "耳朵", description: "耳朵竖起程度",
      levels: ["耳朵垂下", "耳朵自然", "耳朵竖起"] as [string, string, string], enabled: true, multiplier: 1 };
    await store.save("a", "ParamAngleY", null, undefined, "  头轻轻上扬或低垂  ");
    await store.saveCustom("a", custom);
    const reopened = new FacialBindingStore(directory);
    expect(await reopened.getConfiguration("a")).toMatchObject({
      descriptions: { ParamAngleY: "头轻轻上扬或低垂" }, customParameters: [custom],
    });
    expect((await reopened.getConfiguration("b")).customParameters).toEqual([]);
    await reopened.save("a", "ParamAngleY", null, undefined, "");
    await reopened.saveCustom("a", null, "CustomEar");
    expect(await store.getConfiguration("a")).toMatchObject({ descriptions: {}, customParameters: [] });
    await expect(store.saveCustom("a", { ...custom, id: "ParamBreath" })).rejects.toThrow("非标准");
  });

  it("persists score gradients and binary decisions without changing other models", async () => {
    const { store, directory } = await fixture();
    await store.save("a", "ParamCheek", null, undefined, undefined,
      { mode: "score", levels: ["没有", "稍有", "中等", "明显"] });
    await store.save("a", "ParamAngleY", null, undefined, undefined,
      { mode: "binary", falseDescription: "保持低头", trueDescription: "抬到最高" });
    const reopened = new FacialBindingStore(directory);
    expect((await reopened.getConfiguration("a")).criteria).toEqual({
      ParamCheek: { mode: "score", levels: ["没有", "稍有", "中等", "明显"] },
      ParamAngleY: { mode: "binary", falseDescription: "保持低头", trueDescription: "抬到最高" },
    });
    expect((await reopened.getConfiguration("b")).criteria).toEqual({});
    await reopened.save("a", "ParamCheek", null, undefined, undefined,
      { mode: "score", levels: [...catalog.find(entry => entry.id === "ParamCheek")!.levels] });
    expect((await reopened.getConfiguration("a")).criteria).not.toHaveProperty("ParamCheek");
    await expect(reopened.save("a", "ParamAngleY", null, undefined, undefined,
      { mode: "binary", falseDescription: "一样", trueDescription: "一样" })).rejects.toThrow("是/否描述");
  });
});
