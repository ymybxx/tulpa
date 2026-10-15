import { describe, expect, it } from "vitest";

import { modelDialogOptions, referenceImageDialogOptions } from "./model-dialogs";

describe("Live2D model dialogs", () => {
  it("keeps full-model folder and ZIP choices separate", () => {
    expect(modelDialogOptions("folder").properties).toEqual(["openDirectory"]);
    expect(modelDialogOptions("zip").filters).toEqual([
      { name: "Live2D 模型 ZIP", extensions: ["zip"] },
    ]);
  });

  it("takes a folder or a ZIP in one picker", () => {
    const options = modelDialogOptions("any");
    expect(options.properties).toEqual(["openFile", "openDirectory"]);
    expect(options.filters?.[0]?.extensions).toEqual(["zip"]);
  });

  it("uses English dialog text without changing accepted file types", () => {
    expect(modelDialogOptions("any", "en").title).toBe("Choose a Live2D model folder or .zip");
    expect(modelDialogOptions("zip", "en").filters).toEqual([{ name: "Live2D model ZIP", extensions: ["zip"] }]);
    expect(referenceImageDialogOptions("en").filters).toEqual([{ name: "Reference image", extensions: ["png", "jpg", "jpeg", "webp"] }]);
  });
});
