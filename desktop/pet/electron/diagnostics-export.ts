import { open, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import type { DesktopRuntimeStatus, ModelReadiness } from "../src/shared/contracts";

/** How much of the end of runtime.log goes into the report; startup failures are at the end. */
export const DIAGNOSTICS_LOG_TAIL_BYTES = 256 * 1024;

export interface DiagnosticsFacts {
  appName: string;
  appVersion: string;
  platform: string;
  arch: string;
  systemVersion: string;
  createdAt: Date;
  runtime: Pick<DesktopRuntimeStatus, "phase" | "detail" | "apiUrl">;
  readiness: ModelReadiness | null;
  runtimeLog: string | null;
}

const SECRET_PATTERNS: RegExp[] = [
  /(authorization\s*[:=]\s*["']?(?:bearer|basic)\s+)[^\s"',]+/gi,
  /(\bbearer\s+)[A-Za-z0-9._~+/=-]{12,}/gi,
  /((?:api[_-]?key|secret|token|password|access[_-]?key)["']?\s*[:=]\s*["']?)[^\s"',}]{6,}/gi,
  /()\bsk-[A-Za-z0-9_-]{12,}/g,
];

/** Keys and tokens never leave the machine, even if a log line happened to carry one. */
export function redactSecrets(text: string): string {
  return SECRET_PATTERNS.reduce((value, pattern) => value.replace(pattern, "$1[已隐藏]"), text);
}

/** One plain-text file the user can send as is. */
export function buildDiagnosticsReport(facts: DiagnosticsFacts): string {
  const lines = [
    `${facts.appName} 诊断信息`,
    `生成时间: ${facts.createdAt.toISOString()}`,
    `版本: ${facts.appVersion}`,
    `系统: ${facts.platform} ${facts.systemVersion} (${facts.arch})`,
    `本地服务: ${facts.runtime.phase}${facts.runtime.detail ? ` · ${facts.runtime.detail}` : ""}`,
  ];
  if (facts.readiness && facts.readiness.state !== "ready") {
    lines.push(`准备情况: ${facts.readiness.text}${facts.readiness.detail ? ` · ${facts.readiness.detail}` : ""}`);
  }
  for (const item of facts.readiness?.items ?? []) {
    lines.push(`  - ${item.label}: ${item.state}${item.error ? ` · ${item.error}` : ""}`);
  }
  lines.push("", "---- runtime.log（最近部分）----", facts.runtimeLog ?? "（还没有运行日志）");
  return redactSecrets(`${lines.join("\n")}\n`);
}

/** The end of the log, starting at a line boundary. */
export async function readLogTail(logPath: string, maxBytes = DIAGNOSTICS_LOG_TAIL_BYTES): Promise<string | null> {
  let size: number;
  try { size = (await stat(logPath)).size; } catch { return null; }
  const start = Math.max(0, size - maxBytes);
  const handle = await open(logPath, "r");
  try {
    const buffer = Buffer.alloc(size - start);
    await handle.read(buffer, 0, buffer.length, start);
    const text = buffer.toString("utf8");
    return start > 0 ? text.slice(text.indexOf("\n") + 1) : text;
  } finally {
    await handle.close();
  }
}

export function diagnosticsFileName(appName: string, at: Date): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  const stamp = `${at.getFullYear()}${pad(at.getMonth() + 1)}${pad(at.getDate())}-${pad(at.getHours())}${pad(at.getMinutes())}${pad(at.getSeconds())}`;
  return `${appName}-诊断信息-${stamp}.txt`;
}

/** Writes the report into `directory` and returns its path. */
export async function exportDiagnostics(
  directory: string,
  logPath: string,
  facts: Omit<DiagnosticsFacts, "runtimeLog">,
): Promise<string> {
  const report = buildDiagnosticsReport({ ...facts, runtimeLog: await readLogTail(logPath) });
  const target = path.join(directory, diagnosticsFileName(facts.appName, facts.createdAt));
  await writeFile(target, report, "utf8");
  return target;
}
