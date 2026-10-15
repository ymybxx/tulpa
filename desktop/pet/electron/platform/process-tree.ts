import { execFile, type ChildProcess } from "node:child_process";

export interface NativeProcessJobs {
  attachProcessToJob(pid: number): boolean;
  terminateProcessJob(pid: number): boolean | void;
}

let nativeJobs: NativeProcessJobs | null = null;

export function configureProcessJobs(jobs: NativeProcessJobs): void {
  nativeJobs = jobs;
}

/** The native handles close on host failure, including when JS cleanup cannot run. */
export function attachProcessJob(child: ChildProcess): void {
  if (process.platform !== "win32" || !child.pid) return;
  const pid = child.pid;
  if (!nativeJobs?.attachProcessToJob(pid)) {
    void terminateWindowsProcessTree(pid);
    throw new Error("无法管理 Windows 子进程，已停止启动以避免遗留后台服务。");
  }
  child.once("exit", () => nativeJobs?.terminateProcessJob(pid));
}

export async function terminateWindowsProcessTree(pid: number): Promise<void> {
  if (!Number.isSafeInteger(pid) || pid <= 0) return;
  if (nativeJobs?.terminateProcessJob(pid)) return;
  await new Promise<void>((resolve) => {
    execFile("taskkill.exe", ["/PID", String(pid), "/T", "/F"], { windowsHide: true }, () => resolve());
  });
}

export async function stopProcessTree(
  child: ChildProcess | null,
  graceMs = 5_000,
): Promise<void> {
  if (!child?.pid || child.exitCode !== null) return;
  if (process.platform === "win32") {
    // Windows has no SIGTERM handler. Runtime roles drain their own resources
    // after this private parent-control message, then the job reaps descendants.
    if (child.stdin && !child.stdin.destroyed) {
      child.stdin.on("error", () => { /* The runtime may already have exited. */ });
      child.stdin.end("__FISH_MEMORY_SHUTDOWN__\n");
    }
    if (await waitForExit(child, graceMs)) return;
    await terminateWindowsProcessTree(child.pid);
    await waitForExit(child, 1_000);
    return;
  }
  signalGroup(child.pid, "SIGTERM");
  if (await waitForExit(child, graceMs)) return;
  signalGroup(child.pid, "SIGKILL");
  await waitForExit(child, 1_000);
}

function signalGroup(pid: number, signal: NodeJS.Signals): void {
  try {
    process.kill(-pid, signal);
  } catch {
    try {
      process.kill(pid, signal);
    } catch {
      // The child already exited between the state check and signal.
    }
  }
}

function waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (child.exitCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timeout = setTimeout(() => finish(false), timeoutMs);
    const onExit = () => finish(true);
    const finish = (value: boolean) => {
      clearTimeout(timeout);
      child.off("exit", onExit);
      resolve(value);
    };
    child.once("exit", onExit);
  });
}
