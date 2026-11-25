import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { buildDiagnosticsReport, diagnosticsFileName, exportDiagnostics, readLogTail, redactSecrets } from "./diagnostics-export";

const facts = {
  appName: "tulpa",
  appVersion: "0.1.1",
  platform: "darwin",
  arch: "arm64",
  systemVersion: "12.7.6",
  createdAt: new Date("2026-11-25T08:00:00Z"),
  runtime: { phase: "failed" as const, detail: "本地记忆服务提前退出 (3)", apiUrl: null },
  readiness: {
    state: "failed" as const, text: "TA 的服务未能就绪", detail: "点一下重新启动并重试",
    items: [{ component: "asr", label: "语音识别", state: "failed" as const, error: "download failed" }],
  },
};

let dir: string | null = null;
afterEach(async () => { if (dir) await rm(dir, { recursive: true, force: true }); dir = null; });

describe("diagnostics export", () => {
  it("says which version, system and failure, then the end of the log", () => {
    const report = buildDiagnosticsReport({ ...facts, runtimeLog: "Traceback\nImportError: x\n" });
    expect(report).toContain("版本: 0.1.1");
    expect(report).toContain("系统: darwin 12.7.6 (arm64)");
    expect(report).toContain("本地服务: failed · 本地记忆服务提前退出 (3)");
    expect(report).toContain("  - 语音识别: failed · download failed");
    expect(report.trimEnd().endsWith("ImportError: x")).toBe(true);
  });

  it("hides keys and tokens that slipped into a log line", () => {
    const text = redactSecrets([
      "Authorization: Bearer abcdefghijklmnop123",
      "api_key=abcdef123456789",
      "\"apiKey\": \"zzzzzzzzzzzz\"",
      "using sk-abcdefghijklmnopqrstu",
      "ready on port 52011",
    ].join("\n"));
    for (const secret of ["abcdefghijklmnop123", "abcdef123456789", "zzzzzzzzzzzz", "sk-abcdefghijklmnopqrstu"]) {
      expect(text).not.toContain(secret);
    }
    expect(text).toContain("ready on port 52011");
  });

  it("keeps only whole lines from the end of a long log", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "diag-"));
    const log = path.join(dir, "runtime.log");
    await writeFile(log, `${"x".repeat(50)}\nline two\nline three\n`);
    expect(await readLogTail(log, 13)).toBe("line three\n");
    expect(await readLogTail(path.join(dir, "missing.log"))).toBeNull();
  });

  it("writes one dated text file", async () => {
    dir = await mkdtemp(path.join(tmpdir(), "diag-"));
    const target = await exportDiagnostics(dir, path.join(dir, "missing.log"), facts);
    expect(path.basename(target)).toBe(diagnosticsFileName("tulpa", facts.createdAt));
    expect(await readFile(target, "utf8")).toContain("（还没有运行日志）");
  });
});
