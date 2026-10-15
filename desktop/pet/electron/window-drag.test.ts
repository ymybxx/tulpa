import { describe, expect, it } from "vitest";

import { petScaleShownBy, petWindowSize, placePet, resizePetBounds } from "./window-drag";
import { petModelScaleFromPinch, petWindowScale } from "../src/shared/pet-view";

const area = { x: 0, y: 0, width: 1_000, height: 800 };
const panel = { width: 500, height: 600 };
const model = { x: 100, y: 100, width: 300, height: 400 };

/** Screen extent of the model that the display actually shows. */
function shown(placement: ReturnType<typeof placePet>, pet = model, display = area) {
  const x = placement.window.x + placement.offset.x + pet.x;
  const y = placement.window.y + placement.offset.y + pet.y;
  return {
    width: Math.min(x + pet.width, display.x + display.width) - Math.max(x, display.x),
    height: Math.min(y + pet.height, display.y + display.height) - Math.max(y, display.y),
  };
}

describe("placePet", () => {
  it("moves only the panel while it fits on the display", () => {
    expect(placePet({ x: 300, y: 150 }, panel, model, area))
      .toEqual({ window: { x: 300, y: 150 }, offset: { x: 0, y: 0 } });
  });

  it("lets the model hang half-way past an edge once the panel stops", () => {
    const placement = placePet({ x: 100, y: 500 }, panel, model, area);
    expect(placement).toEqual({ window: { x: 100, y: 200 }, offset: { x: 0, y: 300 } });
    expect(shown(placement).height).toBe(200);
  });

  it("always keeps part of the model visible to grab it again", () => {
    const far = placePet({ x: 5_000, y: -5_000 }, panel, model, area);
    expect(far.window).toEqual({ x: 500, y: 0 });
    expect(shown(far)).toEqual({ width: 96, height: 96 });
    const tall = { x: 0, y: 0, width: 300, height: 1_000 };
    expect(shown(placePet({ x: 0, y: 5_000 }, panel, tall, area), tall).height).toBe(200);
  });

  it("unwinds the offset when dragged back, restoring the whole pet", () => {
    for (const distance of [400, -400]) {
      const out = placePet({ x: 100, y: 100 + distance }, panel, model, area);
      const back = placePet({
        x: out.window.x + out.offset.x,
        y: out.window.y + out.offset.y - distance,
      }, panel, model, area);
      expect(back).toEqual({ window: { x: 100, y: 100 }, offset: { x: 0, y: 0 } });
    }
  });

  it("centres a panel larger than the display and pans the model inside it", () => {
    const display = { x: 0, y: 33, width: 1_512, height: 949 };
    expect(placePet({ x: 692, y: 242 }, { width: 1_800, height: 1_100 }, null, display))
      .toEqual({ window: { x: -144, y: -42 }, offset: { x: 836, y: 284 } });
  });

  it("lets a magnified model pan past the edge but keeps part of it visible", () => {
    const full = { width: 800, height: 600 };
    const display = { x: 0, y: 0, ...full };
    const large = { x: -300, y: -200, width: 1_600, height: 1_200 };
    expect(placePet({ x: 400, y: 400 }, full, large, display))
      .toEqual({ window: { x: 0, y: 0 }, offset: { x: 400, y: 400 } });
    const far = placePet({ x: 2_600, y: 2_600 }, full, large, display);
    expect(far.offset).toEqual({ x: 780, y: 560 });
    expect(shown(far, large, display)).toEqual({ width: 320, height: 240 });
  });

  it("places only the panel while the model size is unknown", () => {
    expect(placePet({ x: 2_000, y: -100 }, panel, null, area))
      .toEqual({ window: { x: 500, y: 0 }, offset: { x: 1_500, y: -100 } });
  });
});

describe("pet window scale", () => {
  it("keeps the pointer anchor fixed while resizing the panel", () => {
    expect(resizePetBounds(
      { x: 100, y: 100, width: 480, height: 720 },
      petWindowSize(2), { x: 220, y: 280 },
    )).toEqual({ x: -20, y: -80, width: 960, height: 1_440 });
  });

  it("uses the window center for scale changes from settings", () => {
    expect(resizePetBounds(
      { x: 100, y: 100, width: 480, height: 720 }, petWindowSize(0.5),
    )).toEqual({ x: 220, y: 280, width: 240, height: 360 });
  });

  it("keeps shrinking through a slow pinch made of tiny steps", () => {
    let scale = 1;
    let bounds = { x: 0, y: 0, ...petWindowSize(1) };
    for (let step = 0; step < 2_000; step += 1) {
      scale = petModelScaleFromPinch(scale, 0.1);
      bounds = resizePetBounds(bounds, petWindowSize(petWindowScale(scale)));
    }
    expect(scale).toBeLessThan(0.2 + 1e-9);
    expect(bounds).toMatchObject({ width: 96, height: 144 });
  });

  it("takes the scale a drifted saved panel actually shows", () => {
    expect(petScaleShownBy({ width: 479, height: 544 }, 0.2, 0.2)).toBeCloseTo(544 / 720);
    expect(petScaleShownBy({ width: 1_018, height: 638 }, 2.05, 2.05)).toBeCloseTo(638 / 720);
    expect(petScaleShownBy({ width: 1_200, height: 1_800 }, 10, 3)).toBeCloseTo(10 * 2.5 / 3);
    expect(petScaleShownBy({ width: 240, height: 360 }, 0.5, 0.5)).toBe(0.5);
  });
});


describe("a squeezed pet panel", () => {
  it("keeps the saved scale instead of reading a sliver as a tiny pet", () => {
    expect(petScaleShownBy({ width: 24, height: 266 }, 0.37, 0.37)).toBe(0.37);
    expect(petScaleShownBy({ width: 240, height: 360 }, 1, 1)).toBeCloseTo(0.5);
  });
});
