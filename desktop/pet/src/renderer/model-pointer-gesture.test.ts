import { describe, expect, it } from "vitest";

import {
  modelPointerCompletesClick,
  modelPointerMoved,
  isPetScaleWheel,
  petScaleWheelStep,
  modelCursorForHover,
  normalizePetModelScale,
  petModelScaleFromPinch,
  petWindowScale,
  petContentZoom,
} from "./model-pointer-gesture";

describe("modelPointerMoved", () => {
  it("keeps a short press as a model click", () => {
    expect(modelPointerMoved(
      { x: 100, y: 200 },
      { x: 103, y: 202 },
    )).toBe(false);
  });

  it("starts dragging once pointer movement reaches the threshold", () => {
    expect(modelPointerMoved(
      { x: 100, y: 200 },
      { x: 104, y: 200 },
    )).toBe(true);
  });
});

describe("model click completion", () => {
  it("fires only for an uncancelled short press", () => {
    expect(modelPointerCompletesClick(false, false)).toBe(true);
    expect(modelPointerCompletesClick(true, false)).toBe(false);
    expect(modelPointerCompletesClick(false, true)).toBe(false);
  });
});

describe("model hover cursor", () => {
  it("uses a pointer cursor while the model is hoverable", () => {
    expect(modelCursorForHover(true)).toBe("pointer");
    expect(modelCursorForHover(false)).toBe("default");
  });
});

describe("pet scale input", () => {
  it("accepts trackpad pinch and physical mouse wheel", () => {
    expect(isPetScaleWheel({ ctrlKey: true, deltaMode: 0, deltaY: 2 })).toBe(true);
    expect(isPetScaleWheel({ ctrlKey: false, deltaMode: 1, deltaY: 3 })).toBe(true);
    expect(isPetScaleWheel({ ctrlKey: false, deltaMode: 0, deltaY: 100 })).toBe(true);
  });

  it("ignores ordinary two-finger trackpad scrolling", () => {
    expect(isPetScaleWheel({ ctrlKey: false, deltaMode: 0, deltaY: 12 })).toBe(false);
  });

  it("scales the desktop pet with a two-finger scroll or a mouse wheel, never a pinch", () => {
    expect(petScaleWheelStep({ ctrlKey: true, deltaMode: 0, deltaY: 2 })).toBe(0);
    expect(petScaleWheelStep({ ctrlKey: false, deltaMode: 1, deltaY: 3 })).toBe(12);
    expect(petScaleWheelStep({ ctrlKey: false, deltaMode: 0, deltaY: -120 })).toBe(-12);
    expect(petScaleWheelStep({ ctrlKey: false, deltaMode: 0, deltaY: 10 })).toBeCloseTo(3.5);
    expect(petScaleWheelStep({ ctrlKey: false, deltaMode: 0, deltaY: 0 })).toBe(0);
  });
});

describe("pet model scale", () => {
  it("allows magnification beyond the old 3x window limit", () => {
    expect(normalizePetModelScale(0.01)).toBe(0.2);
    expect(normalizePetModelScale(4)).toBe(4);
    expect(normalizePetModelScale(100)).toBe(64);
    expect(normalizePetModelScale(1.25)).toBe(1.25);
    expect(normalizePetModelScale(Number.NaN)).toBe(1);
    expect(petWindowScale(8)).toBe(3);
    expect(petContentZoom(8)).toBeCloseTo(8 / 3);
    expect(petContentZoom(2)).toBe(1);
  });

  it("uses pinch direction to zoom in and out within the same range", () => {
    expect(petModelScaleFromPinch(1, -20)).toBeGreaterThan(1);
    expect(petModelScaleFromPinch(1, 20)).toBeLessThan(1);
    expect(petModelScaleFromPinch(3, -100)).toBeGreaterThan(3);
    expect(petModelScaleFromPinch(64, -100)).toBe(64);
    expect(petModelScaleFromPinch(64, -100_000)).toBe(64);
    expect(petModelScaleFromPinch(0.2, 100)).toBe(0.2);
  });
});
