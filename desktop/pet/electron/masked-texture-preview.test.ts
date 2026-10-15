import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";

import { createMaskedTexturePreview } from "./masked-texture-preview";

describe("createMaskedTexturePreview", () => {
  it("hides unrelated pixels from the crop without changing their RGB", () => {
    const patch = new PNG({ width: 2, height: 1 });
    patch.data.set([
      240, 240, 240, 255,
      255, 128, 32, 255,
    ]);
    const mask = new PNG({ width: 2, height: 1 });
    mask.data.set([
      255, 255, 255, 255,
      255, 255, 255, 0,
    ]);

    const preview = createMaskedTexturePreview(patch, mask);

    expect([...preview.data]).toEqual([
      240, 240, 240, 255,
      255, 128, 32, 0,
    ]);
    expect([...patch.data]).toEqual([
      240, 240, 240, 255,
      255, 128, 32, 255,
    ]);
  });

  it("rejects a stale mask with different dimensions", () => {
    const patch = new PNG({ width: 2, height: 1 });
    const mask = new PNG({ width: 1, height: 1 });

    expect(() => createMaskedTexturePreview(patch, mask)).toThrow(
      "纹理版本与 UV 蒙版尺寸不一致",
    );
  });
});
