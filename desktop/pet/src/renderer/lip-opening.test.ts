import { describe, expect, it } from "vitest";

import { mouthOpening, mouthParts } from "./lip-opening";

const shape = (volume: number, vowels: Partial<Record<"a" | "e" | "i" | "o" | "u", number>> = {}) =>
  ({ volume, a: 0, e: 0, i: 0, o: 0, u: 0, ...vowels });

describe("mouth opening", () => {
  it("scales loudness by the vowel heard, so a loud i stays narrow", () => {
    expect(mouthOpening(shape(0.8, { a: 0.8 }))).toBeCloseTo(0.8);
    expect(mouthOpening(shape(0.8, { i: 0.8 }))).toBeCloseTo(0.36);
    expect(mouthOpening(shape(0.8, { u: 0.8 }))).toBeLessThan(mouthOpening(shape(0.8, { o: 0.8 })));
  });

  it("opens a vowel preview by that vowel's amount, as speech does", () => {
    expect(mouthOpening(shape(0, { i: 1 }))).toBeCloseTo(mouthOpening(shape(1, { i: 1 })));
    expect(mouthOpening(shape(0, { a: 0.5 }))).toBeCloseTo(0.5);
  });

  it("uses loudness alone until a vowel is heard, without a jump as one fades in", () => {
    expect(mouthOpening(shape(0.6))).toBeCloseTo(0.6);
    expect(mouthOpening(shape(0.6, { i: 0.002 }))).toBeCloseTo(0.6, 2);
    expect(mouthOpening(shape(0.6, { i: 0.3 }))).toBeCloseTo(0.6 * (0.5 + 0.5 * 0.45));
  });
});

describe("VBridger mouth parts", () => {
  it("drops the jaw for a, puckers and funnels o and u, widens i", () => {
    const a = mouthParts(shape(1, { a: 1 }));
    const i = mouthParts(shape(1, { i: 1 }));
    const u = mouthParts(shape(1, { u: 1 }));
    expect(a.jaw).toBeGreaterThan(i.jaw);
    expect(i.puckerWiden).toBeGreaterThan(0);
    expect(u.puckerWiden).toBe(-1);
    expect(u.funnel).toBeGreaterThan(a.funnel);
  });

  it("follows how strongly the vowel is heard and rests at zero in silence", () => {
    expect(mouthParts(shape(0.5, { o: 0.5 })).funnel).toBeCloseTo(mouthParts(shape(1, { o: 1 })).funnel / 2);
    expect(mouthParts(shape(0))).toEqual({ jaw: 0, puckerWiden: 0, funnel: 0 });
  });
});
