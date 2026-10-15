import { describe, expect, it, vi } from "vitest";

import {
  currentSystemPermission,
  requestSystemPermission,
  settingsAppName,
  systemPermissionKind,
  systemPermissionState,
  type SystemPermissionDeps,
} from "./system-permissions";

function deps(overrides: Partial<SystemPermissionDeps> & { statuses?: string[] } = {}): SystemPermissionDeps {
  const statuses = overrides.statuses ?? ["not-determined"];
  let index = 0;
  return {
    platform: "darwin",
    mediaAccessStatus: () => statuses[Math.min(index++, statuses.length - 1)]!,
    askForMicrophone: vi.fn(async () => true),
    primeScreenCapture: vi.fn(async () => undefined),
    openExternal: vi.fn(async () => undefined),
    ...overrides,
  };
}

describe("systemPermissionState", () => {
  it("maps macOS media access values", () => {
    expect(systemPermissionState("screen", deps({ statuses: ["granted"] }))).toBe("granted");
    expect(systemPermissionState("screen", deps({ statuses: ["restricted"] }))).toBe("denied");
    expect(systemPermissionState("microphone", deps({ statuses: ["not-determined"] }))).toBe("not-determined");
  });

  it("does not ask for screen permission on Windows", () => {
    expect(systemPermissionState("screen", deps({ platform: "win32", statuses: ["denied"] }))).toBe("not-required");
    expect(systemPermissionState("microphone", deps({ platform: "win32", statuses: ["granted"] }))).toBe("granted");
  });
  it("never turns unknown Windows microphone access into a grant", () => {
    expect(systemPermissionState("microphone", deps({ platform: "win32", statuses: ["unknown"] }))).toBe("not-determined");
  });
});

describe("settingsAppName", () => {
  it("uses the bundle name shown in System Settings", () => {
    expect(settingsAppName("/Applications/tulpa.app/Contents/MacOS/tulpa")).toBe("tulpa");
    expect(settingsAppName("/x/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron")).toBe("Electron");
    expect(settingsAppName("C:\\Program Files\\tulpa\\tulpa.exe")).toBe("tulpa");
  });
});

describe("currentSystemPermission", () => {
  it("trusts a fresh helper when the running app still reports a permission as denied", async () => {
    await expect(currentSystemPermission("screen", deps({ statuses: ["denied"], probe: async () => true }))).resolves.toBe("granted");
    await expect(currentSystemPermission("microphone", deps({ statuses: ["denied"], probe: async () => true }))).resolves.toBe("granted");
  });

  it("keeps the in-process answer when the probe fails or says no", async () => {
    await expect(currentSystemPermission("screen", deps({ statuses: ["denied"], probe: async () => null }))).resolves.toBe("denied");
    await expect(currentSystemPermission("microphone", deps({ statuses: ["denied"], probe: async () => false }))).resolves.toBe("denied");
  });

  it("does not probe an already granted permission or outside macOS", async () => {
    const probe = vi.fn(async () => true);
    await currentSystemPermission("screen", deps({ statuses: ["granted"], probe }));
    await currentSystemPermission("microphone", deps({ platform: "win32", statuses: ["denied"], probe }));
    expect(probe).not.toHaveBeenCalled();
  });
});

describe("requestSystemPermission", () => {
  it("opens Windows microphone privacy settings without calling the macOS prompt", async () => {
    const value = deps({ platform: "win32", statuses: ["denied"] });
    expect(await requestSystemPermission("microphone", value)).toBe("denied");
    expect(value.askForMicrophone).not.toHaveBeenCalled();
    expect(value.openExternal).toHaveBeenCalledWith("ms-settings:privacy-microphone");
  });
  it("asks for the microphone in place and stops when answered", async () => {
    const value = deps({ statuses: ["not-determined", "granted"] });
    await expect(requestSystemPermission("microphone", value)).resolves.toBe("granted");
    expect(value.askForMicrophone).toHaveBeenCalledOnce();
    expect(value.openExternal).not.toHaveBeenCalled();
  });

  it("opens the microphone pane after an earlier denial", async () => {
    const value = deps({ statuses: ["denied"] });
    await expect(requestSystemPermission("microphone", value)).resolves.toBe("denied");
    expect(value.askForMicrophone).not.toHaveBeenCalled();
    expect(value.openExternal).toHaveBeenCalledWith(expect.stringContaining("Privacy_Microphone"));
  });

  it("registers the app before opening the Screen Recording pane", async () => {
    const value = deps({ statuses: ["not-determined"] });
    await expect(requestSystemPermission("screen", value)).resolves.toBe("not-determined");
    expect(value.primeScreenCapture).toHaveBeenCalledOnce();
    expect(value.openExternal).toHaveBeenCalledWith(expect.stringContaining("Privacy_ScreenCapture"));
  });

  it("does nothing when already granted", async () => {
    const value = deps({ statuses: ["granted"] });
    await expect(requestSystemPermission("screen", value)).resolves.toBe("granted");
    expect(value.primeScreenCapture).not.toHaveBeenCalled();
  });

  it("validates the permission kind", () => {
    expect(systemPermissionKind("screen")).toBe("screen");
    expect(() => systemPermissionKind("camera")).toThrow();
  });
});

describe("system audio", () => {
  it("lets macOS ask first and opens System Settings only once it was refused", async () => {
    const openExternal = vi.fn(async () => undefined);
    const ask = vi.fn(async () => "denied" as const);
    const unasked = { status: () => "not-determined" as const, refresh: async () => "not-determined" as const, ask };
    // The question may still be on screen: do not open System Settings over it.
    expect(await requestSystemPermission("systemAudio", deps({ openExternal, systemAudio: unasked }))).toBe("denied");
    expect(ask).toHaveBeenCalledTimes(1);
    expect(openExternal).not.toHaveBeenCalled();

    const refused = { ...unasked, refresh: async () => "denied" as const, ask: vi.fn(async () => "denied" as const) };
    expect(await requestSystemPermission("systemAudio", deps({ openExternal, systemAudio: refused }))).toBe("denied");
    expect(refused.ask).not.toHaveBeenCalled();
    expect(openExternal).toHaveBeenCalledWith(expect.stringContaining("Privacy_ScreenCapture"));
    expect(systemPermissionKind("systemAudio")).toBe("systemAudio");
  });
});
