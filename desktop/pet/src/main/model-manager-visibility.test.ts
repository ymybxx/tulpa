import { describe, expect, it } from "vitest";

import { shouldActivateModelManager } from "./model-manager-visibility";

describe("model manager window visibility", () => {
  it("does not keep Live2D GPU resources when the BrowserWindow is hidden", () => {
    expect(shouldActivateModelManager("models", false, false)).toBe(false);
  });

  it("keeps the selected preview while the visible model window is merely occluded", () => {
    expect(shouldActivateModelManager("models", false, true)).toBe(true);
    expect(shouldActivateModelManager("models", true, true)).toBe(true);
  });

  it("releases the preview after leaving models or hiding the main window", () => {
    expect(shouldActivateModelManager("chat", false, true)).toBe(false);
    expect(shouldActivateModelManager("settings", false, true)).toBe(false);
    expect(shouldActivateModelManager("models", false, false)).toBe(false);
  });
});
