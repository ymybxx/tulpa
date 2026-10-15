import { readdir, readFile, rm } from "node:fs/promises";
import path from "node:path";

import type { DesktopRuntimeStatus, DesktopUiLocale, ModelReadiness, ModelStatusItem } from "../src/shared/contracts";

const STATES = new Set(["preparing", "downloading", "ready", "failed"]);

/** Timer and explicit refreshes share one read; no slow-I/O request queue. */
export class ModelReadinessPoller {
  private pending: Promise<void> | null = null;
  private stopped = false;

  constructor(
    private readonly read: () => Promise<ModelStatusItem[]>,
    private readonly publish: (items: ModelStatusItem[]) => void,
  ) {}

  refresh(): Promise<void> {
    if (this.stopped) return Promise.resolve();
    if (this.pending) return this.pending;
    const pending = Promise.resolve().then(async () => {
      if (this.stopped) return;
      const items = await this.read();
      if (!this.stopped) this.publish(items);
    }).finally(() => { if (this.pending === pending) this.pending = null; });
    this.pending = pending;
    return pending;
  }

  stop(): void {
    this.stopped = true;
  }
}

/** Reads the `<component>.json` files the runtime processes write while preparing models. */
export async function readModelStatus(directory: string): Promise<ModelStatusItem[]> {
  let names: string[];
  try {
    names = await readdir(directory);
  } catch {
    return [];
  }
  const items: ModelStatusItem[] = [];
  for (const name of names.filter((value) => value.endsWith(".json")).sort()) {
    try {
      const value = JSON.parse(await readFile(path.join(directory, name), "utf8")) as Record<string, unknown>;
      if (typeof value.component !== "string" || typeof value.label !== "string" || !STATES.has(String(value.state))) continue;
      items.push({
        component: value.component,
        label: value.label,
        state: value.state as ModelStatusItem["state"],
        done: typeof value.done === "number" ? value.done : undefined,
        total: typeof value.total === "number" ? value.total : undefined,
        error: typeof value.error === "string" ? value.error : undefined,
      });
    } catch { /* a file being replaced is read on the next poll */ }
  }
  return items;
}

/** Old results would mislead after a restart; each process reports afresh. */
export async function clearModelStatus(directory: string): Promise<void> {
  await rm(directory, { recursive: true, force: true });
}

/**
 * Average download speed since the current download began; null until it is
 * measurable. It also keeps the shown percent from going backwards while one
 * model after another downloads, so the user sees a single first download.
 */
export class DownloadClock {
  private start: { at: number; done: number } | null = null;
  private shown: number | null = null;

  /** The highest percent shown so far in this round of preparing. */
  steady(percent: number | null): number | null {
    if (percent !== null) this.shown = Math.max(this.shown ?? 0, percent);
    return this.shown;
  }

  get downloading(): boolean {
    return this.shown !== null;
  }

  remainingSeconds(done: number, total: number, now: number): number | null {
    if (!this.start || done < this.start.done) this.start = { at: now, done };
    const elapsed = (now - this.start.at) / 1000;
    const received = done - this.start.done;
    if (elapsed < 3 || received <= 0) return null;
    return (total - done) / (received / elapsed);
  }

  reset(): void {
    this.start = null;
    this.shown = null;
  }
}

function remaining(seconds: number | null, locale: DesktopUiLocale): string {
  if (seconds === null) return "";
  if (locale === "en") return seconds < 60 ? " · under 1 minute" : ` · about ${Math.ceil(seconds / 60)} minutes`;
  return seconds < 60 ? " · 不到 1 分钟" : ` · 约 ${Math.ceil(seconds / 60)} 分钟`;
}

/**
 * One line for the user: what is still being prepared, how far along, or what
 * failed. `runtimeStarting` covers the time before any model reports.
 */
export function summarizeReadiness(
  items: readonly ModelStatusItem[],
  runtimeStarting: string | null,
  clock: DownloadClock | null = null,
  now = Date.now(),
  locale: DesktopUiLocale = "zh-CN",
  runtime?: Pick<DesktopRuntimeStatus, "phase" | "apiUrl" | "detail">,
): ModelReadiness {
  const t = (zh: string, en: string): string => locale === "en" ? en : zh;
  // A port is assigned before the API is ready. Only these phases follow
  // successful API preparation and browser authentication.
  const chatReady = Boolean(runtime?.apiUrl && ["ready", "starting-companion"].includes(runtime.phase));
  const labels = (values: readonly ModelStatusItem[]): string => values.map((item) => {
    if (locale !== "en") return item.label;
    return ({ embeddings: "memory", asr: "speech recognition", tts: "voice", kws: "wake word", vad: "voice activity detection", speaker: "voiceprint recognition" } as Record<string, string>)[item.component] ?? item.label;
  }).join(locale === "en" ? ", " : "、");
  const failed = items.filter((item) => item.state === "failed");
  if (failed.length) {
    clock?.reset();
    return {
      state: "failed",
      text: t(`TA 的${labels(failed)}没准备好`, `Could not prepare ${labels(failed)}`),
      detail: chatReady
        ? t("点一下重试，打字聊天可以继续", "Click to retry. Text chat is still available.")
        : t("本地服务尚未就绪，暂时不能聊天。点一下重新启动并重试", "The local service is not ready, so chat is unavailable. Click to restart and retry."),
      items: [...items],
    };
  }
  // Startup can fail before writing any model report (or after reports were
  // cleared for a retry). Keep the recovery action visible in that case too.
  if (runtime?.phase === "failed") {
    clock?.reset();
    return {
      state: "failed",
      text: t("TA 的服务未能就绪", "Your companion's service could not start"),
      detail: [runtime.detail, t("点一下重新启动并重试", "Click to restart and retry.")].filter(Boolean).join(" · "),
      items: [...items],
    };
  }
  const pending = items.filter((item) => item.state === "preparing" || item.state === "downloading");
  if (pending.length) {
    // Finished models still count, so one model finishing does not restart the bar.
    const measured = items.filter((item) => item.total && item.total > 0);
    const done = measured.reduce((sum, item) => sum + (item.state === "ready" ? item.total! : item.done ?? 0), 0);
    const total = measured.reduce((sum, item) => sum + (item.total ?? 0), 0);
    const raw = total > 0 ? Math.min(99, Math.floor((done / total) * 100)) : null;
    const active = pending.some((item) => item.state === "downloading");
    // Between two downloads the line stays on the same download.
    const downloading = active || Boolean(clock?.downloading);
    const percent = active && clock ? clock.steady(raw) : clock?.downloading ? clock.steady(null) : raw;
    const pendingLabels = labels(pending);
    // While the bar is held above the real progress, a time estimate would contradict it.
    const holding = percent !== null && raw !== null && percent > raw;
    const eta = active && !holding && total > 0 && clock ? clock.remainingSeconds(done, total, now) : null;
    return {
      state: "preparing",
      text: downloading
        ? t(`第一次下载${percent === null ? "中…" : ` ${percent}%`}${remaining(eta, locale)}`, `First download${percent === null ? "…" : ` ${percent}%`}${remaining(eta, locale)}`)
        : t(`正在准备 TA 的${pendingLabels}…`, `Preparing ${pendingLabels}…`),
      detail: downloading
        ? t(`TA 的${pendingLabels}要先下载，只有第一次需要。`, `${pendingLabels} need a one-time download. `)
          + (chatReady
            ? t("好之前 TA 可能听不到或说不出声，打字聊天可以继续", "Until it finishes, voice may be unavailable; text chat is still available.")
            : t("本地服务准备好之前，暂时不能聊天", "Chat will be available once the local service is ready."))
        : chatReady
          ? t("正在准备，打字聊天可以继续", "Preparing; text chat is still available.")
          : t("正在准备本地服务，完成后就能聊天", "Preparing the local service. Chat will be available when it is ready."),
      percent,
      items: [...items],
    };
  }
  clock?.reset();
  if (runtimeStarting) {
    return { state: "preparing", text: runtimeStarting, detail: t("马上就好", "Almost ready"), percent: null, items: [...items] };
  }
  return { state: "ready", text: "", detail: "", items: [...items] };
}
