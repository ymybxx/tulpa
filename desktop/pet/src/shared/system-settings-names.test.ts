import { describe, expect, it } from "vitest";

import { permissionPlace, systemSettingsNames } from "./system-settings-names";

describe("systemSettingsNames", () => {
  it("names Windows privacy settings in either language without a macOS audio pane", () => {
    expect(systemSettingsNames(0, "en", "win32")).toEqual({ privacy: "Privacy & security", screen: "Screenshots and apps", audioOnly: null });
    expect(permissionPlace("microphone", systemSettingsNames(0, "zh-CN", "win32"), "en")).toBe("Privacy & security › Microphone");
  });
  it("names the screen list the way each macOS does", () => {
    expect(systemSettingsNames(24)).toEqual({ privacy: "隐私与安全性", screen: "录屏与系统录音", audioOnly: "仅系统录音" });
    expect(systemSettingsNames(23)).toEqual({ privacy: "隐私与安全性", screen: "屏幕录制", audioOnly: null });
    expect(systemSettingsNames(21)).toEqual({ privacy: "安全性与隐私 › 隐私", screen: "屏幕录制", audioOnly: null });
  });

  it("points to the audio-only list only where it exists", () => {
    expect(permissionPlace("systemAudio", systemSettingsNames(24))).toBe("隐私与安全性 › 录屏与系统录音 › 仅系统录音");
    expect(permissionPlace("systemAudio", systemSettingsNames(23))).toBe("隐私与安全性 › 屏幕录制");
    expect(permissionPlace("microphone", systemSettingsNames(21))).toBe("安全性与隐私 › 隐私 › 麦克风");
  });
});
