import { describe, expect, it } from "vitest";
import type { NativeModelParameter } from "../src/shared/contracts";
import { autoSetupCandidates, readCheck, readSorted, sortRequest, standardBinding } from "./facial-auto-setup";

const parameter = (id: string, name = id, minimum = 0, maximum = 1, defaultValue = 0): NativeModelParameter => ({ id, name, minimum, maximum, defaultValue });
const box = { x: 0.4, y: 0.1, width: 0.2, height: 0.1 };
const survey = (id: string, changed = 0.02, found: typeof box | null = box) => ({ id, parameter: parameter(id), changed, box: found });

describe("一键配置 candidates", () => {
  it("keeps visible, free parameters and leaves out standard, automatic, physics and used ones", () => {
    const native = [parameter("ParamAngleX"), parameter("ParamBreath"), parameter("Hair1"), parameter("Used"), parameter("Angry", "生气"),
      parameter("Still"), parameter("Fixed", "固定", 0, 0)];
    const { candidates, invisible } = autoSetupCandidates(native,
      [survey("Hair1"), survey("Used"), survey("Angry"), survey("Still", 0, null)],
      new Map([["Angry", "Exp"]]), { used: ["Used"], open: [], customSlots: 16 }, new Set(["Hair1"]));
    expect(candidates.map(candidate => [candidate.parameter.id, candidate.group])).toEqual([["Angry", "Exp"]]);
    expect(invisible.map(item => item.id)).toEqual(["Still"]);
  });

  it("tells the sorter where each parameter changes and which standard parts are still open", () => {
    const { candidates } = autoSetupCandidates([parameter("ParamBrowRY7", "右眉　上下", -1, 1, 0)], [survey("ParamBrowRY7")],
      new Map([["ParamBrowRY7", "Brow"]]), { used: [], open: ["ParamBrowRY"], customSlots: 4 }, new Set());
    const request = sortRequest(candidates, ["ParamBrowRY"], 4);
    expect(request.system).toContain("最多 4 个");
    expect(request.user).toContain("ParamBrowRY · 右眉上下");
    expect(request.user).toMatch(/ParamBrowRY7 · 右眉　上下 · 分组：Brow · 范围 -1～1（默认 0）· 变化：x 0\.40–0\.60，y 0\.10–0\.20/u);
  });
});

describe("一键配置 sorting", () => {
  const candidates = ["A", "B", "C", "D", "E"].map(id => ({ parameter: parameter(id), group: "", box, changed: 0.02 }));

  it("holds the AI to open standard parts once each, the custom slots and every candidate", () => {
    const sorted = readSorted({ items: [
      { id: "A", use: "standard", standardId: "ParamBrowRY" },
      { id: "B", use: "standard", standardId: "ParamBrowRY" },
      { id: "C", use: "standard", standardId: "ParamAngleY" },
      { id: "D", use: "custom", kind: "motion", mode: "binary", motion: "pulse" },
      { id: "E", use: "custom", mode: "weird", motion: "dance" },
      { id: "Z", use: "custom" },
    ] }, candidates, ["ParamBrowRY"], 1);
    expect(sorted).toEqual([
      { id: "A", use: "standard", standardId: "ParamBrowRY" },
      { id: "B", use: "skip", reason: "和其他参数重复" },
      { id: "C", use: "skip", reason: "对应的标准部位已经有参数了" },
      { id: "D", use: "custom", kind: "motion", mode: "binary", motion: "pulse" },
      { id: "E", use: "skip", reason: "自定义部位的名额已满" },
    ]);
  });

  it("marks candidates the AI left out", () => {
    expect(readSorted({ items: [] }, candidates.slice(0, 1), [], 16)).toEqual([{ id: "A", use: "skip", reason: "AI 没有给出判断" }]);
    expect(() => readSorted({}, candidates, [], 16)).toThrow("items");
  });
});

describe("一键配置 standard parts", () => {
  it("binds the model's range, flipped when the strong end is the parameter's minimum", () => {
    const native = parameter("Param43", "左眉 角度", -1, 1, 0);
    expect(standardBinding("ParamBrowLAngle", native, "maximum")).toEqual({ parameterId: "Param43", lowerValue: -1, neutralValue: 0, upperValue: 1 });
    expect(standardBinding("ParamBrowLAngle", native, "minimum")).toEqual({ parameterId: "Param43", lowerValue: 1, neutralValue: 0, upperValue: -1 });
    // A part at rest at its weak end keeps rest on the weak end after flipping.
    const blush = parameter("ParamBlush", "脸红", 0, 1, 0);
    expect(standardBinding("ParamCheek", blush, "minimum")).toEqual({ parameterId: "ParamBlush", lowerValue: 1, neutralValue: 1, upperValue: 0 });
  });

  it("reads the picture check", () => {
    expect(readCheck({ match: true, strongAt: "minimum", note: "向上" })).toEqual({ match: true, strongAt: "minimum", note: "向上" });
    expect(readCheck({ match: false, note: "动的是头发" })).toMatchObject({ match: false });
    expect(() => readCheck({ match: true, strongAt: "left" })).toThrow("strongAt");
  });
});

describe("一键配置 physics-smoothed parts", () => {
  it("gives an open standard part the free input that leads it, one part per input", async () => {
    const { physicsDrivenParts } = await import("./facial-auto-setup");
    const candidates = ["ParamBrowLYY", "ParamMouthPuff"].map(id => ({ parameter: parameter(id), group: "", box, changed: 0.001 }));
    const links = new Map([
      ["ParamBrowLYY", [{ output: "ParamBrowLY", weight: 80 }, { output: "ParamBrowLX", weight: 80 }, { output: "ParamBrowLAngle", weight: 60 }]],
      ["ParamMouthPuff", [{ output: "ParamBrowLAngle", weight: 20 }]],
      ["ParamMouthForm", [{ output: "ParamEyeLSmile", weight: 100 }]],
    ]);
    const parts = physicsDrivenParts(candidates, ["ParamEyeLSmile", "ParamBrowLY", "ParamBrowLX", "ParamBrowLAngle"], links);
    // MouthForm is already Jev's; the puff only nudges the brow.
    expect([...parts]).toEqual([["ParamBrowLY", "ParamBrowLYY"]]);
  });
});
