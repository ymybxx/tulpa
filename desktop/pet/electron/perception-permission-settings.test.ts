import { describe, expect, it } from "vitest";

import { perceptionPermissionSettingsUrl } from "./perception-permission-settings";

describe("perceptionPermissionSettingsUrl", () => {
  it("maps only the two supported macOS privacy panels", () => {
    expect(perceptionPermissionSettingsUrl("accessibility", "darwin")).toBe(
      "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    );
    expect(perceptionPermissionSettingsUrl("screen", "darwin")).toBe(
      "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    );
  });

  it("opens Windows privacy settings without a macOS URL", () => {
    expect(perceptionPermissionSettingsUrl("screen", "win32")).toBe("ms-settings:privacy");
    expect(perceptionPermissionSettingsUrl("camera", "win32")).toBeNull();
  });

  it("rejects unknown renderer input", () => {
    expect(perceptionPermissionSettingsUrl("camera")).toBeNull();
    expect(perceptionPermissionSettingsUrl({ kind: "screen" })).toBeNull();
  });
});
