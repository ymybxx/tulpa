import { describe, expect, it } from "vitest";

import {
  petTextureMaximumOrDefault,
  suggestedPetTextureMaximum,
} from "./texture-quality";

const gib = 1024 ** 3;

describe("initial pet texture ceiling", () => {
  it("suggests an appropriate ceiling from installed memory", () => {
    expect(suggestedPetTextureMaximum(8 * gib)).toBe(2048);
    expect(suggestedPetTextureMaximum(16 * gib)).toBe(4096);
    expect(suggestedPetTextureMaximum(32 * gib)).toBe(8192);
  });

  it("keeps a saved choice instead of detecting again", () => {
    expect(petTextureMaximumOrDefault(2048, 8192)).toBe(2048);
    expect(petTextureMaximumOrDefault(8192, 2048)).toBe(8192);
    expect(petTextureMaximumOrDefault(16384, 4096)).toBe(4096);
  });
});
