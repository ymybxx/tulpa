import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  matchesPackagedRuntime,
  matchesPackagedRuntimeRole,
  matchesKnownPackagedRuntime,
  parseProcessList,
  parseWindowsProcessList,
  RuntimeProcessRegistry,
  type ProcessSnapshot,
  type RuntimeProcessController,
} from "./runtime-process-registry";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { recursive: true, force: true }),
    ),
  );
});

describe("runtime process ownership", () => {
  it("reads CIM records and matches only the Windows executable and role", () => {
    const executable = "C:\\Users\\A User\\tulpa\\fish-memory-runtime.exe";
    const command = `"${executable}" api --port 1234`;
    expect(parseWindowsProcessList(JSON.stringify([
      { ProcessId: 123, ParentProcessId: 100, CommandLine: command },
      { ProcessId: 456, ParentProcessId: 0, CommandLine: null },
    ]))).toEqual([{ pid: 123, parentPid: 100, command }]);
    expect(matchesPackagedRuntimeRole(command, executable, "api")).toBe(true);
    expect(matchesPackagedRuntimeRole(command, executable, "companion")).toBe(false);
    expect(matchesPackagedRuntimeRole(`"${executable}.other" api`, executable, "api")).toBe(false);
    expect(matchesPackagedRuntimeRole(`"${executable}" api-other`, executable, "api")).toBe(false);
  });
  it("parses the macOS process table without relying on localized headers", () => {
    expect(parseProcessList(
      "  42     1 /Applications/tulpa.app/Contents/runtime api\n" +
      " 100    42 helper --flag\n",
    )).toEqual([
      { pid: 42, parentPid: 1, command: "/Applications/tulpa.app/Contents/runtime api" },
      { pid: 100, parentPid: 42, command: "helper --flag" },
    ]);
  });

  it("only matches the exact packaged runtime executable and role", () => {
    const executable = "/Applications/tulpa.app/Contents/Resources/runtime/" +
      "fish-memory-runtime/fish-memory-runtime";
    expect(matchesPackagedRuntime(`${executable} api`, executable)).toBe(true);
    expect(matchesPackagedRuntime(`${executable} companion --log-level INFO`, executable)).toBe(true);
    expect(matchesPackagedRuntime("python -m fish_memory.desktop_runtime api", executable)).toBe(false);
    expect(matchesPackagedRuntime(`${executable}-other api`, executable)).toBe(false);
    expect(matchesPackagedRuntimeRole(`${executable} companion`, executable, "companion")).toBe(
      true,
    );
    expect(matchesPackagedRuntimeRole(`${executable} api`, executable, "companion")).toBe(false);
    expect(matchesKnownPackagedRuntime(`${executable} api`)).toBe(true);
    expect(matchesKnownPackagedRuntime(`${executable} owner-voice --action enroll`)).toBe(true);
    expect(matchesPackagedRuntime(`${executable} owner-voice --action enroll`, executable)).toBe(true);
    expect(matchesKnownPackagedRuntime(`${executable}-other api`)).toBe(false);
  });

  it("stops every owned companion while preserving the API and unrelated processes", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "fish-memory-processes-"));
    temporaryDirectories.push(directory);
    const file = path.join(directory, "runtime-processes.json");
    const executable = "/Applications/tulpa.app/Contents/runtime/fish-memory-runtime";
    const controller = fakeController([
      { pid: 51, parentPid: process.pid, command: `${executable} api` },
      { pid: 52, parentPid: process.pid, command: `${executable} companion` },
      { pid: 53, parentPid: process.pid, command: "python unrelated.py" },
      { pid: 54, parentPid: 8888, command: `${executable} companion` },
    ]);
    const registry = new RuntimeProcessRegistry(file, executable, controller);
    registry.track(51, "api", executable, ["api"]);
    registry.track(52, "companion", executable, ["companion"]);

    await expect(registry.stopRole("companion")).resolves.toEqual([52]);

    expect(controller.stopped).toEqual([52]);
    expect(JSON.parse(await readFile(file, "utf8"))).toMatchObject({
      processes: [{ pid: 51, role: "api" }],
    });
  });

  it("cleans this profile's recorded orphans and preserves unrecorded runtime processes", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "fish-memory-processes-"));
    temporaryDirectories.push(directory);
    const file = path.join(directory, "runtime-processes.json");
    const executable = "/Applications/tulpa.app/Contents/runtime/fish-memory-runtime";
    await writeFile(file, JSON.stringify({
      version: 1,
      ownerPid: 9999,
      processes: [
        { pid: 41, role: "api", signature: `${executable} api` },
        { pid: 42, role: "companion", signature: `${executable} companion` },
      ],
    }));
    const controller = fakeController([
      { pid: 41, parentPid: 9999, command: `${executable} api` },
      { pid: 42, parentPid: 9999, command: `${executable} companion` },
      { pid: 43, parentPid: 1, command: "python unrelated.py" },
      { pid: 44, parentPid: 1, command: `${executable} companion` },
    ]);
    const registry = new RuntimeProcessRegistry(file, executable, controller);

    await expect(registry.beginSession()).resolves.toEqual([41, 42]);
    expect(controller.stopped).toEqual([41, 42]);
    expect(JSON.parse(await readFile(file, "utf8"))).toMatchObject({
      version: 1,
      ownerPid: process.pid,
      processes: [],
    });
  });

  it("recovers an owner-voice process recorded before an interrupted enrollment", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "fish-memory-processes-"));
    temporaryDirectories.push(directory);
    const file = path.join(directory, "runtime-processes.json");
    const command = "uv run --extra voice python -m fish_memory.desktop_runtime owner-voice --action enroll";
    await writeFile(file, JSON.stringify({
      version: 1, ownerPid: 9999,
      processes: [{ pid: 45, role: "owner-voice", signature: command }],
    }));
    const controller = fakeController([{ pid: 45, parentPid: 9999, command }]);
    const registry = new RuntimeProcessRegistry(file, null, controller);
    expect(await registry.beginSession()).toEqual([45]);
    expect(controller.stopped).toEqual([45]);
  });

  it.each(["live-owner", "reused-pid", "changed-command"])("preserves a stale record with %s", async (condition) => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "fish-memory-processes-"));
    temporaryDirectories.push(directory);
    const file = path.join(directory, "runtime-processes.json");
    const executable = "E:\\Apps\\tulpa\\fish-memory-runtime.exe";
    await writeFile(file, JSON.stringify({ version: 1, ownerPid: 9999,
      processes: [{ pid: 41, role: "api", executable, signature: `${executable} api` }],
    }));
    const controller = fakeController([
      { pid: 41, parentPid: condition === "reused-pid" ? 8888 : 9999,
        command: `"${executable}" ${condition === "changed-command" ? "companion" : "api"}` },
      ...(condition === "live-owner" ? [{ pid: 9999, parentPid: 1, command: "tulpa.exe" }] : []),
    ]);
    expect(await new RuntimeProcessRegistry(file, executable, controller).beginSession()).toEqual([]);
    expect(controller.stopped).toEqual([]);
  });
});

function fakeController(processes: ProcessSnapshot[]): RuntimeProcessController & { stopped: number[] } {
  const stopped: number[] = [];
  return {
    stopped,
    list: async () => processes,
    stop: async (pid) => {
      stopped.push(pid);
    },
  };
}
