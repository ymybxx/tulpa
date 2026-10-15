import path from "node:path";
import { describe, expect, it } from "vitest";
import { frozenRuntimeExecutable, packagedRuntimeExecutable, platformPerceptionExecutable, runtimeAssetsRoot } from "./runtime-paths";
import { supportsSystemAudio } from "../system-audio-support";

describe("Windows runtime layout", () => {
  const common = { resourcesPath: "/resources", projectRoot: "/project" };
  it("uses platform binaries in the development and packaged layouts", () => {
    expect(runtimeAssetsRoot({ ...common, packaged: false, platform: "win32", arch: "x64" }))
      .toBe(path.join("/project", "desktop/pet/runtime/win32-x64"));
    expect(packagedRuntimeExecutable("/resources", "win32"))
      .toBe(path.join("/resources", "runtime/fish-memory-runtime/fish-memory-runtime.exe"));
    expect(platformPerceptionExecutable({ ...common, packaged: true, platform: "win32", arch: "x64" }))
      .toBe(path.join("/resources", "perception/fish-memory-perception-helper.exe"));
    expect(platformPerceptionExecutable({ ...common, packaged: false, platform: "darwin", arch: "arm64" }))
      .toBe(path.join("/project", "desktop/pet/runtime/mac-arm64/FishMemoryPerceptionHelper.app/Contents/MacOS/fish-memory-perception-helper"));
  });
  it("shares the frozen backend on Windows while retaining source development on macOS", () => {
    expect(frozenRuntimeExecutable({ ...common, packaged: false, platform: "win32", arch: "x64" }))
      .toBe(path.join("/project", "desktop/pet/runtime/win32-x64/fish-memory-runtime/fish-memory-runtime.exe"));
    expect(frozenRuntimeExecutable({ ...common, packaged: true, platform: "win32", arch: "x64" }))
      .toBe(packagedRuntimeExecutable(common.resourcesPath, "win32"));
    expect(frozenRuntimeExecutable({ ...common, packaged: false, platform: "darwin", arch: "arm64" })).toBeNull();
    expect(frozenRuntimeExecutable({ ...common, packaged: true, platform: "darwin", arch: "arm64" }))
      .toBe(packagedRuntimeExecutable(common.resourcesPath, "darwin"));
  });
  it("exposes system audio on Windows 11 but not the unimplemented Windows 10 target", () => {
    expect(supportsSystemAudio("win32", "10.0.22000")).toBe(true);
    expect(supportsSystemAudio("win32", "10.0.26100")).toBe(true);
    expect(supportsSystemAudio("win32", "10.0.19045")).toBe(false);
    expect(supportsSystemAudio("win32", "unknown")).toBe(false);
    expect(supportsSystemAudio("darwin", "22.0.0")).toBe(true);
    expect(supportsSystemAudio("darwin", "21.0.0")).toBe(false);
  });
});
