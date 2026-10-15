import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it, vi } from "vitest";

import { DownloadClock, ModelReadinessPoller, clearModelStatus, readModelStatus, summarizeReadiness } from "./model-readiness";
import type { ModelStatusItem } from "../src/shared/contracts";

describe("model readiness", () => {
  it("coalesces slow reads, retries after failure, and never publishes after stop", async () => {
    let active = 0;
    let maximum = 0;
    const reads: Array<{ resolve(items: ModelStatusItem[]): void; reject(error: Error): void }> = [];
    const read = vi.fn(async () => {
      maximum = Math.max(maximum, ++active);
      try {
        return await new Promise<ModelStatusItem[]>((resolve, reject) => reads.push({ resolve, reject }));
      } finally { active -= 1; }
    });
    const publish = vi.fn();
    const poller = new ModelReadinessPoller(read, publish);
    const first = poller.refresh();
    await Promise.resolve();
    for (let tick = 0; tick < 10; tick += 1) expect(poller.refresh()).toBe(first);
    expect(read).toHaveBeenCalledOnce();
    const ready: ModelStatusItem[] = [{ component: "asr", label: "听力", state: "ready" }];
    reads[0]!.resolve(ready);
    await first;
    expect(publish).toHaveBeenLastCalledWith(ready);
    expect(read).toHaveBeenCalledOnce(); // No accumulated follow-up work.

    const failed = poller.refresh();
    const rejection = expect(failed).rejects.toThrow("slow disk failed");
    await Promise.resolve();
    reads[1]!.reject(new Error("slow disk failed"));
    await rejection;

    const retry = poller.refresh();
    await Promise.resolve();
    expect(read).toHaveBeenCalledTimes(3);
    poller.stop();
    reads[2]!.resolve([]);
    await retry;
    await poller.refresh();
    expect(read).toHaveBeenCalledTimes(3);
    expect(publish).toHaveBeenCalledOnce();
    expect(maximum).toBe(1);
    expect(active).toBe(0);
  });

  it("reads valid status files and ignores broken ones", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "model-status-"));
    await writeFile(path.join(directory, "tts.json"), JSON.stringify({ component: "tts", label: "语音合成", state: "downloading", done: 10, total: 100 }));
    await writeFile(path.join(directory, "asr.json"), "{broken");
    await writeFile(path.join(directory, "kws.json"), JSON.stringify({ component: "kws", label: "唤醒词", state: "weird" }));
    expect(await readModelStatus(directory)).toEqual([{ component: "tts", label: "语音合成", state: "downloading", done: 10, total: 100, error: undefined }]);
    await clearModelStatus(directory);
    expect(await readModelStatus(directory)).toEqual([]);
    await mkdir(directory, { recursive: true });
  });

  it("says how far along the first download is and how long is left", () => {
    const items = (done: number) => [
      { component: "asr", label: "听力", state: "downloading" as const, done, total: 200_000_000 },
      { component: "tts", label: "声音", state: "downloading" as const, done, total: 200_000_000 },
      { component: "kws", label: "唤醒词", state: "ready" as const },
    ];
    const clock = new DownloadClock();
    expect(summarizeReadiness(items(0), null, clock, 0)).toMatchObject({ state: "preparing", text: "第一次下载 0%", percent: 0 });
    const later = summarizeReadiness(items(100_000_000), null, clock, 10_000);
    // 200 MB in 10 s leaves 200 MB, about 10 s more.
    expect(later).toMatchObject({ text: "第一次下载 50% · 不到 1 分钟", percent: 50 });
    expect(later.detail).toContain("TA 的听力、声音要先下载，只有第一次需要");
    expect(summarizeReadiness(items(101_000_000), null, clock, 610_000).text).toBe("第一次下载 50% · 约 10 分钟");
    expect(summarizeReadiness([{ component: "asr", label: "听力", state: "preparing" }], null, new DownloadClock()))
      .toMatchObject({ text: "正在准备 TA 的听力…" });
  });

  it("keeps one download going forward while models finish one after another", () => {
    const clock = new DownloadClock();
    const first = summarizeReadiness([
      { component: "asr", label: "听力", state: "downloading", done: 41, total: 100 },
    ], null, clock, 0);
    expect(first.percent).toBe(41);
    // 听力 is done and 声音 starts: the bar does not fall back.
    const next = summarizeReadiness([
      { component: "asr", label: "听力", state: "ready", done: 100, total: 100 },
      { component: "tts", label: "声音", state: "downloading", done: 10, total: 300 },
    ], null, clock, 1_000);
    expect(next.percent).toBe(41);
    // Held above the real progress, it gives no time estimate.
    expect(next.text).toBe("第一次下载 41%");
    // A pause between downloads still reads as the same download.
    expect(summarizeReadiness([{ component: "tts", label: "声音", state: "preparing" }], null, clock, 2_000))
      .toMatchObject({ text: "第一次下载 41%", percent: 41 });
    expect(summarizeReadiness([
      { component: "asr", label: "听力", state: "ready", done: 100, total: 100 },
      { component: "tts", label: "声音", state: "downloading", done: 250, total: 300 },
    ], null, clock, 3_000).percent).toBe(87);
  });

  it("puts failures first and is quiet once everything is ready", () => {
    expect(summarizeReadiness([
      { component: "asr", label: "语音识别", state: "failed", error: "网络不通" },
      { component: "tts", label: "语音合成", state: "downloading" },
    ], null)).toMatchObject({ state: "failed", text: "TA 的语音识别没准备好" });
    expect(summarizeReadiness([{ component: "tts", label: "语音合成", state: "ready" }], null)).toMatchObject({ state: "ready", text: "" });
    expect(summarizeReadiness([], "正在启动语音服务…")).toMatchObject({ state: "preparing", text: "正在启动语音服务…" });
  });

  it("switches preparation messages without resetting download progress", () => {
    const clock = new DownloadClock();
    const items = [{ component: "asr", label: "听力", state: "downloading" as const, done: 50, total: 100 }];
    expect(summarizeReadiness(items, null, clock, 0).text).toBe("第一次下载 50%");
    const english = summarizeReadiness(items, null, clock, 1_000, "en");
    expect(english.text).toBe("First download 50%");
    expect(english.detail).toContain("speech recognition need a one-time download");
    expect(summarizeReadiness([{ ...items[0]!, state: "failed" }], null, clock, 2_000, "en").text)
      .toBe("Could not prepare speech recognition");
    expect(summarizeReadiness([], "Starting voice…", null, 0, "en").detail).toBe("Almost ready");
  });

  it.each(["zh-CN", "en"] as const)("does not promise chat while memory or the API is unavailable (%s)", (locale) => {
    const runtime = { phase: "starting-api" as const, apiUrl: "http://127.0.0.1:49152", detail: "Starting" };
    const downloading = summarizeReadiness([
      { component: "embeddings", label: "记忆", state: "downloading", done: 20, total: 100 },
    ], "Starting", new DownloadClock(), 0, locale, runtime);
    const failed = summarizeReadiness([
      { component: "embeddings", label: "记忆", state: "failed", error: "offline" },
    ], null, null, 0, locale, { ...runtime, phase: "failed" });
    expect(downloading.detail).not.toMatch(/不受影响|可以继续|still (?:works|available)/u);
    expect(failed.detail).not.toMatch(/不受影响|可以继续|still (?:works|available)/u);
    expect(failed.detail).toMatch(locale === "en" ? /chat is unavailable/u : /暂时不能聊天/u);
    if (locale === "en") expect(failed.text).toBe("Could not prepare memory");
  });

  it("keeps failed startup retryable even without model reports", () => {
    const summary = summarizeReadiness([], null, null, 0, "en", {
      phase: "failed", apiUrl: null, detail: "API process exited",
    });
    expect(summary).toMatchObject({ state: "failed", text: "Your companion's service could not start" });
    expect(summary.detail).toContain("Click to restart and retry");
    // A leftover download report must not hide the stopped core service.
    expect(summarizeReadiness([{ component: "tts", label: "声音", state: "downloading" }], null, null, 0, "en", {
      phase: "failed", apiUrl: null, detail: "API process exited",
    }).state).toBe("failed");
  });

  it("only promises text chat when an authenticated API phase is available", () => {
    const models = [{ component: "asr", label: "听力", state: "failed" as const }];
    for (const phase of ["ready", "starting-companion"] as const) {
      const status = { phase, apiUrl: "http://127.0.0.1:49152", detail: "Voice unavailable" };
      expect(summarizeReadiness(models, null, null, 0, "en", status).detail).toContain("Text chat is still available");
    }
    expect(summarizeReadiness(models, null, null, 0, "en", {
      phase: "ready", apiUrl: null, detail: "Unavailable",
    }).detail).toContain("chat is unavailable");
  });
});
