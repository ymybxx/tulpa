import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { EventEmitter } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RuntimeManager } from "./runtime-manager";

vi.mock("node:child_process", async importOriginal => ({
  ...await importOriginal<typeof import("node:child_process")>(), spawn: vi.fn(),
}));

afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks(); });

describe("desktop backend launch", () => {
  it.each([
    ["darwin", false], ["darwin", true], ["win32", false], ["win32", true],
  ] as const)("uses the correct backend with platform=%s packaged=%s", async (platform, packaged) => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "tulpa-runtime-launch-"));
    vi.stubGlobal("process", { ...process, platform, arch: platform === "win32" ? "x64" : "arm64" });
    const child = Object.assign(new EventEmitter(), {
      stdout: new EventEmitter(), stderr: new EventEmitter(), exitCode: 0,
    }) as ChildProcess;
    vi.mocked(spawn).mockReturnValue(child);
    const unexpected = async (): Promise<never> => { throw new Error("Unexpected configuration access"); };
    const manager = new RuntimeManager({
      configStore: { load: unexpected, save: unexpected, runtimeSecrets: unexpected },
      userDataPath: userData, projectRoot: "/project", resourcesPath: "/resources", packaged,
      authenticateBrowser: async () => {},
    });
    try {
      manager["spawnRole"]("api", { FISH_MEMORY_PORT: "12345" });
      const frozen = packaged || platform === "win32";
      const executable = packaged
        ? path.join("/resources", "runtime/fish-memory-runtime", `fish-memory-runtime${platform === "win32" ? ".exe" : ""}`)
        : platform === "win32"
          ? path.join("/project", "desktop/pet/runtime/win32-x64/fish-memory-runtime/fish-memory-runtime.exe")
          : "uv";
      expect(spawn).toHaveBeenCalledWith(executable,
        frozen ? ["api"] : ["run", "--extra", "voice", "python", "-m", "fish_memory.desktop_runtime", "api"],
        expect.objectContaining({ cwd: frozen ? userData : "/project", windowsHide: true }));
    } finally {
      await manager["runtimeLogWrite"];
      await rm(userData, { recursive: true, force: true });
    }
  });
});
