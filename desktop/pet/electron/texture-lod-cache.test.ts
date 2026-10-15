import { describe, expect, it, vi } from "vitest";

import { requestedTextureDimension } from "./texture-lod-cache";

vi.mock("electron", () => ({ nativeImage: {} }));

describe("texture LOD cache requests", () => {
  it("accepts only the application detail levels", () => {
    expect(requestedTextureDimension("2048")).toBe(2048);
    expect(requestedTextureDimension("4096")).toBe(4096);
    expect(requestedTextureDimension("8192")).toBe(8192);
    expect(requestedTextureDimension("1024")).toBeNull();
    expect(requestedTextureDimension("invalid")).toBeNull();
  });
});
