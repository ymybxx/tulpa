import { describe, expect, it } from "vitest";

import { standardMouthTarget } from "./mouth-shape-mapping";

describe("standardMouthTarget", () => {
  it.each([
    ["a", { open: 1, form: 0 }],
    ["e", { open: 0.65, form: 0.55 }],
    ["i", { open: 0.45, form: 0.55 }],
    ["o", { open: 0.85, form: -0.3 }],
    ["u", { open: 0.5, form: -0.3 }],
  ] as const)("maps %s into standard open/form dimensions", (vowel, expected) => {
    expect(standardMouthTarget({
      volume: 0,
      a: vowel === "a" ? 1 : 0,
      e: vowel === "e" ? 1 : 0,
      i: vowel === "i" ? 1 : 0,
      o: vowel === "o" ? 1 : 0,
      u: vowel === "u" ? 1 : 0,
    }, 1)).toEqual(expected);
  });

  it("applies the per-model mouth intensity", () => {
    expect(standardMouthTarget({ volume: 1, a: 0, e: 0, i: 0, o: 0, u: 0 }, 0.7))
      .toEqual({ open: 0.7, form: 0 });
  });
});
