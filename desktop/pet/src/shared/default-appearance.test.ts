import { describe, expect, it } from "vitest";
import { normalizeDefaultAppearance } from "./default-appearance";

describe("native default appearance values", () => {
  it("preserves canonical layer visibility and groups alongside expressions and parameters", () => {
    const input = { expressionAssetIds: ["coat"], parameterValues: { P: 1 }, hiddenDrawableIds: ["B", "A", "B", "", 1],
      layerGroups: [{ id: "hat", name: " 帽子 ", drawableIds: ["B", "A", "B"] }, { id: "hat", name: "重复", drawableIds: ["C"] }, { id: "empty", name: "", drawableIds: ["A"] }] };
    const assets = { coat: { id: "coat" } } as never;
    const result = normalizeDefaultAppearance(input, assets);
    expect(result).toEqual({ expressionAssetIds: ["coat"], parameterValues: { P: 1 }, hiddenDrawableIds: ["A", "B"],
      layerGroups: [{ id: "hat", name: "帽子", drawableIds: ["A", "B"] }] });
    result.layerGroups![0]!.drawableIds.push("C");
    expect(input.layerGroups[0]!.drawableIds).toEqual(["B", "A", "B"]);
    expect(normalizeDefaultAppearance({ hiddenDrawableIds: {}, layerGroups: [null, {}, { id: "x", name: "x", drawableIds: [] }] }, {})).toEqual({ expressionAssetIds: [] });
  });
  it("keeps finite explicit values without coercing malformed settings", () => {
    const input = { parameterValues: { Watermark: 1, Angle: -12.5, Zero: 0, Blank: "", Bool: true, Nan: NaN, Infinity, "": 1 } };
    expect(normalizeDefaultAppearance(input, {})).toEqual({
      expressionAssetIds: [], parameterValues: { Angle: -12.5, Watermark: 1, Zero: 0 },
    });
    expect(normalizeDefaultAppearance({ parameterValues: [1, 2] }, {})).toEqual({ expressionAssetIds: [] });
  });

  it("canonicalizes key order for appearance fingerprints and isolates edits", () => {
    const input = { parameterValues: { B: 2, A: 1 } };
    const a = normalizeDefaultAppearance(input, {});
    const b = normalizeDefaultAppearance({ parameterValues: { A: 1, B: 2 } }, {});
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    a.parameterValues!.A = 5;
    expect(input.parameterValues.A).toBe(1);
    expect(normalizeDefaultAppearance({ parameterValues: {} }, {})).toEqual({ expressionAssetIds: [] });
  });
});
