import { describe, expect, it } from "vitest";
import { idlePresetSettings, normalizeNaturalIdleSettings } from "./natural-idle-settings";
describe("saved idle profiles", () => {
  it("preserves the old opt-in contract without enabling a new driver", () => {
    expect(normalizeNaturalIdleSettings(undefined)).toEqual({ enabled: false, intensity: 1, speed: 1 });
    expect(normalizeNaturalIdleSettings({ enabled: true, intensity: 2, speed: 1.5 })).toEqual({ enabled: true, intensity: 2, speed: 1.5 });
  });
  it("round-trips a customized preset without sharing mutable preset state", () => {
    const value = idlePresetSettings("active");
    value.expressive!.bodyIntensity = 1.4;
    value.expressive!.bindings = { bodyX: "CustomBody", eyeX: null };
    value.expressive!.bodyDirections = { x: -1 };
    expect(normalizeNaturalIdleSettings(JSON.parse(JSON.stringify(value)))).toEqual(value);
    expect(idlePresetSettings("active").expressive!.bodyIntensity).toBe(1);
  });
  it("bounds imported settings and drops invalid mappings", () => {
    const result = normalizeNaturalIdleSettings({ enabled: true, expressive: { preset: "active", bodyIntensity: Infinity, largeMotionProbability: 10,
        reverseInterval: -10, yawRange: 999, bindings: { headX: 123, bodyX: "", eyeX: null, wrong: "x" }, bodyDirections: { x: 3, y: -1 } } });
    expect(result.expressive).toMatchObject({ bodyIntensity: 1, largeMotionProbability: 1, reverseInterval: 0, yawRange: 28, bindings: { eyeX: null }, bodyDirections: { y: -1 } });
  });
});
