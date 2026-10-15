import { access } from "node:fs/promises";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { presetPreviewFile } from "./preset-previews";

describe("presetPreviewFile", () => {
  it("maps the bundled sample to its shipped thumbnail", () => {
    expect(presetPreviewFile("pet-model://mao-1/__revision-2/Mao.model3.json", "/app", false))
      .toBe(path.join("/app", "public-runtime", "presets", "mao-preview.png"));
    expect(presetPreviewFile("pet-model://mao-1/Mao.model3.json", "/app.asar", true))
      .toBe(path.join("/app.asar", "dist-renderer", "presets", "mao-preview.png"));
  });

  it("leaves imported models to be captured at import time", () => {
    expect(presetPreviewFile("pet-model://custom/Hiyori.model3.json", "/app", false)).toBeNull();
  });

  it("points at a thumbnail that exists in the source tree", async () => {
    const file = presetPreviewFile("pet-model://x/Mao.model3.json", path.resolve(__dirname, ".."), false)!;
    await expect(access(file)).resolves.toBeUndefined();
  });
});
