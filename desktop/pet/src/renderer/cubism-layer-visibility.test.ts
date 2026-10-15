import { describe, expect, it } from "vitest";
import type { CubismModel } from "@cubism/model/cubismmodel";
import { CubismLayerVisibility } from "./cubism-layer-visibility";

function fixture() {
  const opacity = [0.8, 0.35, 0];
  const screen = [{ r: 0.2, g: 0.1, b: 0, a: 1 }, { r: 0, g: 0, b: 0, a: 1 }, { r: 0, g: 0, b: 0, a: 1 }];
  const enabled = [true, false, false];
  const multiply = [{ r: 1, g: 1, b: 1, a: 1 }, { r: 0.9, g: 0.8, b: 0.7, a: 1 }, { r: 1, g: 1, b: 1, a: 1 }];
  const multiplyEnabled = [false, true, false];
  const colors = {
    getDrawableMultiplyColor: (index: number) => multiply[index],
    getDrawableMultiplyColorEnabled: (index: number) => multiplyEnabled[index],
    setDrawableMultiplyColorEnabled: (index: number, value: boolean) => { multiplyEnabled[index] = value; },
    setDrawableMultiplyColorByRGBA: (index: number, r: number, g: number, b: number, a: number) => { multiply[index] = { r, g, b, a }; },
    setDrawableMultiplyColorByTextureColor: (index: number, value: typeof multiply[number]) => { multiply[index] = { ...value }; },
    getDrawableScreenColor: (index: number) => screen[index],
    getDrawableScreenColorEnabled: (index: number) => enabled[index],
    setDrawableScreenColorEnabled: (index: number, value: boolean) => { enabled[index] = value; },
    setDrawableScreenColorByRGBA: (index: number, r: number, g: number, b: number, a: number) => { screen[index] = { r, g, b, a }; },
    setDrawableScreenColorByTextureColor: (index: number, value: typeof screen[number]) => { screen[index] = { ...value }; },
  };
  const model = { getModel: () => ({ drawables: { ids: ["Hat", "Hair", "Alternate"] } }),
    getDrawableOpacity: (index: number) => opacity[index]!, getOverrideMultiplyAndScreenColor: () => colors,
    getDrawableDynamicFlagIsVisible: (index: number) => opacity[index]! > 0 } as unknown as CubismModel;
  const original = model.getDrawableOpacity;
  return { model, opacity, screen, enabled, multiply, multiplyEnabled, original, layer: new CubismLayerVisibility(model) };
}

describe("draw-time layer visibility", () => {
  it("holds a hide through animation and restores the current authored opacity", () => {
    const s = fixture();
    s.layer.configure(["Hat", "missing", "Hat"]);
    for (const value of [0.2, 1, 0.6]) {
      s.opacity[0] = value;
      expect(s.model.getDrawableOpacity(0)).toBe(0);
      expect(s.opacity[0]).toBe(value);
      expect(s.layer.authoredOpacity(0)).toBe(value);
      expect(s.model.getDrawableOpacity(1)).toBe(0.35);
    }
    s.layer.configure([]);
    expect(s.model.getDrawableOpacity(0)).toBe(0.6);
    expect(s.model.getDrawableOpacity).toBe(s.original);
  });
  it("keeps previews temporary: the pieces keep their look, the rest darkens but stays solid", () => {
    const s = fixture();
    const colors = s.model.getOverrideMultiplyAndScreenColor();
    const plainMultiply = colors.getDrawableMultiplyColor;
    s.layer.configure(["Hat"]);
    s.layer.highlight(["Hat", "missing", "Hat"]);
    // A hidden piece being looked at shows; other layers keep their opacity and turn dark.
    expect(s.model.getDrawableOpacity(0)).toBe(0.8);
    expect(s.model.getDrawableOpacity(1)).toBe(0.35);
    expect(colors.getDrawableMultiplyColor(0)).toEqual({ r: 1, g: 1, b: 1, a: 1 });
    expect(colors.getDrawableMultiplyColor(1).r).toBeCloseTo(0.9 * 0.32);
    expect(colors.getDrawableMultiplyColor(1).a).toBe(1);
    expect(s.multiply[1]).toEqual({ r: 0.9, g: 0.8, b: 0.7, a: 1 });
    s.layer.highlight([]);
    expect(s.model.getDrawableOpacity(0)).toBe(0);
    expect(s.model.getDrawableOpacity(1)).toBe(0.35);
    expect(colors.getDrawableMultiplyColor).toBe(plainMultiply);
    s.layer.highlight(["Hair"]); s.layer.release();
    expect(s.model.getDrawableOpacity).toBe(s.original);
  });
  it("draws only the pieces for thumbnails", () => {
    const s = fixture();
    const colors = s.model.getOverrideMultiplyAndScreenColor();
    const plainMultiply = colors.getDrawableMultiplyColor;
    const plainVisible = s.model.getDrawableDynamicFlagIsVisible;
    s.layer.highlight(["Hair", "Alternate"], "solo");
    expect(s.model.getDrawableOpacity(0)).toBe(0);
    expect(s.model.getDrawableOpacity(1)).toBe(0.35);
    expect(colors.getDrawableMultiplyColor).toBe(plainMultiply);
    // A piece of a switched-off hairstyle is drawn too.
    expect(s.model.getDrawableOpacity(2)).toBe(1);
    expect(s.model.getDrawableDynamicFlagIsVisible(2)).toBe(true);
    s.layer.highlight([]);
    expect(s.model.getDrawableOpacity).toBe(s.original);
    expect(s.model.getDrawableDynamicFlagIsVisible).toBe(plainVisible);
  });
  it("has no draw-path override for empty or stale-only settings and never leaves authored hidden layers on", () => {
    const s = fixture();
    s.layer.configure(["gone"]);
    expect(s.model.getDrawableOpacity).toBe(s.original);
    s.layer.highlight([]);
    expect(s.model.getDrawableOpacity).toBe(s.original);
    // Looking at a piece this pose hides shows it only while it is looked at.
    s.layer.highlight(["Alternate"]);
    expect(s.model.getDrawableOpacity(2)).toBe(1);
    s.layer.release();
    expect(s.model.getDrawableOpacity(2)).toBe(0);
  });
});
