import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import type { ChildProcess } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";
import { attachProcessJob, configureProcessJobs, stopProcessTree } from "./process-tree";

const taskkill = vi.hoisted(() => vi.fn());
vi.mock("node:child_process", () => ({ execFile: (...args: unknown[]) => {
  taskkill(...args.slice(0, 3));
  (args[3] as () => void)();
} }));
afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks(); });

function windowsChild() {
  vi.stubGlobal("process", { ...process, platform: "win32" });
  return Object.assign(new EventEmitter(), { pid: 321, exitCode: null as number | null, stdin: new PassThrough() }) as unknown as ChildProcess;
}

describe("Windows runtime lifecycle", () => {
  it("reaps remaining job descendants when the parent role exits", () => {
    const child = windowsChild();
    const jobs = { attachProcessToJob: vi.fn(() => true), terminateProcessJob: vi.fn(() => true) };
    configureProcessJobs(jobs);
    attachProcessJob(child);
    child.emit("exit", 0);
    expect(jobs.attachProcessToJob).toHaveBeenCalledWith(321);
    expect(jobs.terminateProcessJob).toHaveBeenCalledWith(321);
  });
  it("sends graceful shutdown through stdin before terminating the job", async () => {
    const child = windowsChild();
    const terminate = vi.fn(() => true);
    configureProcessJobs({ attachProcessToJob: () => true, terminateProcessJob: terminate });
    const input: string[] = [];
    (child.stdin as PassThrough).on("data", (data) => input.push(String(data)));
    (child.stdin as PassThrough).on("finish", () => { child.exitCode = 0; child.emit("exit", 0); });
    await stopProcessTree(child, 50);
    expect(input.join("")).toBe("__FISH_MEMORY_SHUTDOWN__\n");
    expect(terminate).not.toHaveBeenCalled();
    expect(taskkill).not.toHaveBeenCalled();
  });
  it("forces the whole job when graceful shutdown times out", async () => {
    const child = windowsChild();
    const terminate = vi.fn(() => { child.exitCode = 1; child.emit("exit", 1); return true; });
    configureProcessJobs({ attachProcessToJob: () => true, terminateProcessJob: terminate });
    await stopProcessTree(child, 1);
    expect(terminate).toHaveBeenCalledWith(321);
    expect(taskkill).not.toHaveBeenCalled();
  });
  it("fails startup and cleans the process tree if job assignment fails", async () => {
    const child = windowsChild();
    configureProcessJobs({ attachProcessToJob: () => false, terminateProcessJob: () => false });
    expect(() => attachProcessJob(child)).toThrow("无法管理 Windows 子进程");
    expect(taskkill).toHaveBeenCalledWith("taskkill.exe", ["/PID", "321", "/T", "/F"], { windowsHide: true });
  });
});
