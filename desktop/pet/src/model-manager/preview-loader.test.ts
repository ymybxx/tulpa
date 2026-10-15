import { describe, expect, it, vi } from "vitest";

import { loadPreviewConstructor } from "./preview-loader";

describe("loadPreviewConstructor", () => {
  it("does not import the Live2D renderer before Cubism Core exists", async () => {
    const load = vi.fn();

    await expect(loadPreviewConstructor(false, load)).rejects.toThrow(/尚未准备/);

    expect(load).not.toHaveBeenCalled();
  });

  it("loads the preview lazily once Cubism Core is available", async () => {
    class Preview {}
    const load = vi.fn(async () => ({ ModelManagerPreview: Preview }));

    await expect(loadPreviewConstructor(true, load)).resolves.toBe(Preview);
  });
});
