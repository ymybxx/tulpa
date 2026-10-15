import { execFile } from "node:child_process";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { terminateWindowsProcessTree } from "./platform/process-tree";

const RUNTIME_ROLES = ["api", "companion", "environment-asr", "audio-devices", "owner-voice"] as const;
export type RuntimeProcessRole = typeof RUNTIME_ROLES[number];

interface RuntimeProcessRecord {
  pid: number;
  role: RuntimeProcessRole;
  signature: string;
  executable?: string;
}

interface RuntimeProcessState {
  version: 1;
  ownerPid: number;
  processes: RuntimeProcessRecord[];
}

export interface ProcessSnapshot {
  pid: number;
  parentPid: number;
  command: string;
}

export interface RuntimeProcessController {
  list(): Promise<ProcessSnapshot[]>;
  stop(pid: number): Promise<void>;
}

const execFileAsync = promisify(execFile);

export class RuntimeProcessRegistry {
  private state: RuntimeProcessState = currentState();
  private writeOperation: Promise<void> = Promise.resolve();

  constructor(
    private readonly file: string,
    _packagedExecutable: string | null,
    private readonly controller: RuntimeProcessController = systemProcessController,
  ) {}

  async beginSession(): Promise<number[]> {
    const previous = await loadState(this.file);
    const processes = await this.controller.list();
    const byPid = new Map(processes.map((item) => [item.pid, item]));
    const stale = new Set<number>();

    if (previous && previous.ownerPid !== process.pid && !byPid.has(previous.ownerPid)) {
      for (const record of previous?.processes ?? []) {
        const running = byPid.get(record.pid);
        // Windows retains the original parent PID after it exits. Requiring
        // that owner as well as the command prevents a reused runtime PID in
        // another profile from being mistaken for this session's orphan.
        const parentMatches = running && (running.parentPid === previous.ownerPid
          || (process.platform !== "win32" && running.parentPid === 1));
        if (parentMatches && matchesRecordedRuntime(running.command, record)) {
          stale.add(record.pid);
        }
      }
    }

    for (const pid of stale) await this.controller.stop(pid);
    this.state = currentState();
    await this.persist();
    return [...stale];
  }

  track(
    pid: number | undefined,
    role: RuntimeProcessRole,
    command: string,
    args: string[],
  ): void {
    if (!pid) return;
    this.state.processes = this.state.processes.filter((item) => item.pid !== pid);
    this.state.processes.push({ pid, role, signature: [command, ...args].join(" "), executable: command });
    void this.persistQueued();
  }

  forget(pid: number | undefined): void {
    if (!pid) return;
    const remaining = this.state.processes.filter((item) => item.pid !== pid);
    if (remaining.length === this.state.processes.length) return;
    this.state.processes = remaining;
    void this.persistQueued();
  }

  async clear(): Promise<void> {
    this.state = currentState();
    await this.persistQueued();
  }

  async stopRole(role: RuntimeProcessRole): Promise<number[]> {
    await this.writeOperation.catch(() => undefined);
    const processes = await this.controller.list();
    const recorded = new Map(
      this.state.processes
        .filter((item) => item.role === role)
        .map((item) => [item.pid, item]),
    );
    const targets = new Set<number>();
    for (const process of processes) {
      const record = recorded.get(process.pid);
      // The executable is shared by every profile. Only this registry's
      // recorded children belong to this manager.
      if (record && process.parentPid === this.state.ownerPid
        && matchesRecordedRuntime(process.command, record)) {
        targets.add(process.pid);
      }
    }
    for (const pid of targets) await this.controller.stop(pid);
    this.state.processes = this.state.processes.filter((item) => item.role !== role);
    await this.persistQueued();
    return [...targets];
  }

  private persistQueued(): Promise<void> {
    const next = this.writeOperation.then(() => this.persist());
    this.writeOperation = next.catch(() => undefined);
    return next;
  }

  private async persist(): Promise<void> {
    await mkdir(path.dirname(this.file), { recursive: true });
    const temporary = `${this.file}.${process.pid}.tmp`;
    await writeFile(temporary, JSON.stringify(this.state, null, 2), { mode: 0o600 });
    await rename(temporary, this.file);
  }
}

export function parseWindowsProcessList(value: string): ProcessSnapshot[] {
  const decoded: unknown = JSON.parse(value.trim().replace(/^\uFEFF/, "") || "[]");
  const rows = Array.isArray(decoded) ? decoded : [decoded];
  return rows.flatMap((row) => {
    if (!row || !Number.isInteger(row.ProcessId) || row.ProcessId <= 0
      || !Number.isInteger(row.ParentProcessId) || typeof row.CommandLine !== "string") return [];
    return [{ pid: row.ProcessId, parentPid: row.ParentProcessId, command: row.CommandLine }];
  });
}

export function parseProcessList(value: string): ProcessSnapshot[] {
  const result: ProcessSnapshot[] = [];
  for (const line of value.split(/\r?\n/)) {
    const match = line.match(/^\s*(\d+)\s+(\d+)\s+(.+)$/);
    if (!match) continue;
    result.push({
      pid: Number(match[1]),
      parentPid: Number(match[2]),
      command: match[3]!.trim(),
    });
  }
  return result;
}

export function matchesPackagedRuntime(command: string, executable: string): boolean {
  return RUNTIME_ROLES.some((role) =>
    matchesPackagedRuntimeRole(command, executable, role)
  );
}

export function matchesPackagedRuntimeRole(
  command: string,
  executable: string,
  role: RuntimeProcessRole,
): boolean {
  if (/^[a-z]:[\\/]/i.test(executable)) {
    const prefix = command.startsWith('"') ? `"${executable}"` : executable;
    const expected = `${prefix} ${role}`.toLowerCase();
    const actual = command.toLowerCase();
    return actual === expected || actual.startsWith(`${expected} `);
  }
  return command === `${executable} ${role}` || command.startsWith(`${executable} ${role} `);
}

export function matchesKnownPackagedRuntime(command: string): boolean {
  const marker = "/tulpa.app/Contents/Resources/runtime/" +
    "fish-memory-runtime/fish-memory-runtime";
  const executableEnd = command.indexOf(marker);
  if (executableEnd < 0) return false;
  const remainder = command.slice(executableEnd + marker.length);
  return RUNTIME_ROLES.some(
    (role) => remainder === ` ${role}` || remainder.startsWith(` ${role} `),
  );
}

function matchesRecordedRuntime(
  command: string,
  record: RuntimeProcessRecord,
): boolean {
  if (record.executable && /^[a-z]:[\\/]/i.test(record.executable)) {
    // CIM includes quotes around executable paths with spaces; a PID alone is
    // insufficient because Windows reuses PIDs after the previous process exits.
    const normalized = command.startsWith(`"${record.executable}"`)
      ? record.executable + command.slice(record.executable.length + 2) : command;
    return normalized.toLowerCase() === record.signature.toLowerCase();
  }
  if (command === record.signature || command.startsWith(`${record.signature} `)) return true;
  return command.includes(
    `run --extra voice python -m fish_memory.desktop_runtime ${record.role}`,
  );
}

function currentState(): RuntimeProcessState {
  return { version: 1, ownerPid: process.pid, processes: [] };
}

async function loadState(file: string): Promise<RuntimeProcessState | null> {
  try {
    const value = JSON.parse(await readFile(file, "utf8")) as Partial<RuntimeProcessState>;
    if (value.version !== 1 || !Number.isInteger(value.ownerPid) || !Array.isArray(value.processes)) {
      return null;
    }
    return {
      version: 1,
      ownerPid: value.ownerPid!,
      processes: value.processes.filter(isProcessRecord),
    };
  } catch {
    return null;
  }
}

function isProcessRecord(value: unknown): value is RuntimeProcessRecord {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<RuntimeProcessRecord>;
  return Number.isInteger(candidate.pid) &&
    RUNTIME_ROLES.some((role) => role === candidate.role) &&
    (candidate.executable === undefined || typeof candidate.executable === "string") &&
    typeof candidate.signature === "string";
}

const systemProcessController: RuntimeProcessController = {
  async list() {
    try {
      if (process.platform === "win32") {
        const { stdout } = await execFileAsync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command",
          "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"],
        { windowsHide: true, maxBuffer: 8 * 1024 * 1024 });
        return parseWindowsProcessList(stdout);
      }
      const { stdout } = await execFileAsync("/bin/ps", ["-axo", "pid=,ppid=,command="]);
      return parseProcessList(stdout);
    } catch {
      return [];
    }
  },
  async stop(pid) {
    if (process.platform === "win32") {
      await terminateWindowsProcessTree(pid);
      await waitUntilStopped(pid, 1_000);
      return;
    }
    signalGroup(pid, "SIGTERM");
    if (await waitUntilStopped(pid, 5_000)) return;
    signalGroup(pid, "SIGKILL");
    await waitUntilStopped(pid, 1_000);
  },
};

function signalGroup(pid: number, signal: NodeJS.Signals): void {
  try {
    process.kill(-pid, signal);
  } catch {
    try {
      process.kill(pid, signal);
    } catch {
      // The recorded process has already exited.
    }
  }
}

async function waitUntilStopped(pid: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!isRunning(pid)) return true;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return !isRunning(pid);
}

function isRunning(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
