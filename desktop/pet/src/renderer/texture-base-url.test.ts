import { describe, expect, it } from "vitest";

import { immutableTextureBaseUrl } from "./texture-base-url";

describe("immutableTextureBaseUrl", () => {
  it("marks pet-model atlas reads without dropping content identity", () => {
    expect(immutableTextureBaseUrl("pet-model://mao/textures/atlas.png?content=abc"))
      .toBe("pet-model://mao/textures/atlas.png?content=abc&textureBase=1");
  });

  it("does not alter third-party resource URLs", () => {
    expect(immutableTextureBaseUrl("https://example.test/atlas.png?version=2"))
      .toBe("https://example.test/atlas.png?version=2");
  });
});
