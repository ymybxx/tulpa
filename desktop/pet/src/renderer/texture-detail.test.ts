import { describe, expect, it } from "vitest";

import {
  adaptiveTextureDimension,
  supportedTextureDimensions,
  textureDetailRequestUrl,
} from "./texture-detail";

describe("adaptive Live2D texture detail", () => {
  it("uses hysteresis around the 2048/4096 boundary", () => {
    expect(adaptiveTextureDimension(1400, 2048)).toBe(2048);
    expect(adaptiveTextureDimension(1900, 2048)).toBe(4096);
    expect(adaptiveTextureDimension(1700, 4096)).toBe(4096);
    expect(adaptiveTextureDimension(1400, 4096)).toBe(2048);
  });

  it("reaches the 8192 original at large zoom and steps back when shrunk", () => {
    expect(adaptiveTextureDimension(8000, 2048)).toBe(8192);
    expect(adaptiveTextureDimension(3800, 4096)).toBe(8192);
    expect(adaptiveTextureDimension(3100, 8192)).toBe(8192);
    expect(adaptiveTextureDimension(2800, 8192)).toBe(4096);
    expect(adaptiveTextureDimension(1400, 8192)).toBe(2048);
  });

  it("respects the user ceiling and the graphics device hard limit", () => {
    expect(supportedTextureDimensions(8192, 16384)).toEqual([2048, 4096, 8192]);
    expect(supportedTextureDimensions(4096, 16384)).toEqual([2048, 4096]);
    expect(supportedTextureDimensions(8192, 4096)).toEqual([2048, 4096]);
  });

  it("adds a cacheable detail request only to model resources", () => {
    expect(textureDetailRequestUrl("pet-model://girl/texture.png?revision=2", 2048))
      .toBe("pet-model://girl/texture.png?revision=2&textureMaxDimension=2048");
    expect(textureDetailRequestUrl("https://example.com/texture.png", 2048))
      .toBe("https://example.com/texture.png");
  });
});
