import type { ChildProcess } from "node:child_process";
import { EventEmitter } from "node:events";
import { StringDecoder } from "node:string_decoder";

import type {
  DesktopConfiguration,
  OwnerVoiceEnrollmentState,
  OwnerVoiceStatus,
} from "../src/shared/contracts";
import { stopProcessTree } from "./platform/process-tree";

export type OwnerVoiceAction = "status" | "prepare" | "enroll" | "append" | "delete";
/** What the owner-voice process reports before its result. */
export type OwnerVoiceProgress =
  | { kind: "download"; received: number; total: number }
  | { kind: "recording" }
  | { kind: "processing" };
const EMPTY_STATUS: OwnerVoiceStatus = {
  enrolled: false, modelReady: false, modelId: "cam++", createdAt: null, detail: "尚未录入主人声纹",
};

export function parseOwnerVoiceStatus(value: unknown): OwnerVoiceStatus {
  if (typeof value !== "object" || value === null) throw new Error("声纹状态返回格式不正确");
  const item = value as Record<string, unknown>;
  if (
    typeof item.enrolled !== "boolean" || typeof item.modelReady !== "boolean"
    || typeof item.modelId !== "string" || !item.modelId
    || !(item.createdAt === null || typeof item.createdAt === "string")
    || typeof item.detail !== "string"
    || !(item.enrollmentCount === undefined || (Number.isInteger(item.enrollmentCount) && (item.enrollmentCount as number) >= 0))
    || ![undefined, null, "sounddevice", "livekit"].includes(item.audioBackend as never)
  ) throw new Error("声纹状态返回格式不正确");
  return {
    enrolled: item.enrolled, modelReady: item.modelReady, modelId: item.modelId,
    createdAt: item.createdAt, detail: item.detail,
    ...(typeof item.enrollmentCount === "number" ? { enrollmentCount: item.enrollmentCount } : {}),
    ...(Array.isArray(item.enrollments) ? { enrollments: enrollmentList(item.enrollments) } : {}),
    ...(item.audioBackend !== undefined ? { audioBackend: item.audioBackend as OwnerVoiceStatus["audioBackend"] } : {}),
  };
}

/** A voiceprint's speech file name, as the owner-voice service writes it. */
export const OWNER_VOICE_CLIP = /^[0-9a-f]{32}\.wav$/;

function enrollmentList(value: unknown[]): NonNullable<OwnerVoiceStatus["enrollments"]> {
  if (value.length > 20) throw new Error("声纹状态返回格式不正确");
  return value.map((item) => {
    const entry = item as Record<string, unknown>;
    if (typeof entry?.createdAt !== "string" || entry.createdAt.length > 64
      || (entry.source !== "recorded" && entry.source !== "added")
      || !(entry.clip === undefined || entry.clip === null || (typeof entry.clip === "string" && OWNER_VOICE_CLIP.test(entry.clip)))) {
      throw new Error("声纹状态返回格式不正确");
    }
    return { createdAt: entry.createdAt, source: entry.source, clip: typeof entry.clip === "string" ? entry.clip : null };
  });
}

/** Parse only bounded public status JSONL; never forward raw process output to a renderer. */
export function runOwnerVoiceCommand(
  child: ChildProcess,
  action: OwnerVoiceAction,
  signal: AbortSignal,
  onProgress: (progress: OwnerVoiceProgress) => void = () => {},
  options: { timeoutMs?: number; stop?: typeof stopProcessTree } = {},
): Promise<OwnerVoiceStatus> {
  // A download may be slow but must keep moving: each progress report restarts the clock.
  const timeoutMs = options.timeoutMs ?? (action === "prepare" ? 60_000 : action === "enroll" || action === "append" ? 60_000 : 15_000);
  const stop = options.stop ?? stopProcessTree;
  return new Promise((resolve, reject) => {
    const decoder = new StringDecoder("utf8");
    let buffer = "";
    let bytes = 0;
    let result: OwnerVoiceStatus | undefined;
    let failure: Error | undefined;
    let settled = false;
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      child.stdout?.off("data", data);
      child.off("error", processError);
      child.off("close", closed);
      if (error) reject(error);
      else if (result) resolve(result);
      else reject(new Error("声纹进程未返回有效结果"));
    };
    const cancel = (error: Error) => {
      if (failure || settled) return;
      failure = error;
      void stop(child).then(() => finish(error), () => finish(error));
    };
    const line = (raw: string) => {
      if (!raw.trim()) return;
      const event: unknown = JSON.parse(raw);
      if (typeof event !== "object" || event === null) throw new Error("声纹进程返回无效 JSONL");
      const item = event as Record<string, unknown>;
      if (item.event === "result") result = parseOwnerVoiceStatus(item.status);
      else if (item.event === "download" && Number.isSafeInteger(item.received) && Number.isSafeInteger(item.total)
        && (item.total as number) > 0 && (item.received as number) >= 0 && (item.received as number) <= (item.total as number)) {
        arm();
        onProgress({ kind: "download", received: item.received as number, total: item.total as number });
      }
      else if (item.event === "recording" && (action === "enroll" || action === "append") && item.durationSeconds === 15) {
        arm();
        onProgress({ kind: "recording" });
      }
      else if (item.event === "processing" && (action === "enroll" || action === "append")) {
        arm();
        onProgress({ kind: "processing" });
      }
      else if (item.event === "error") throw new Error(typeof item.error === "string" ? item.error : "声纹处理失败");
      else throw new Error("声纹进程返回未知事件");
    };
    const data = (value: Buffer | string) => {
      if (failure) return;
      bytes += Buffer.byteLength(value);
      if (bytes > 64 * 1024) { cancel(new Error("声纹进程返回内容过大")); return; }
      buffer += typeof value === "string" ? value : decoder.write(value);
      try {
        let end;
        while ((end = buffer.indexOf("\n")) >= 0) {
          line(buffer.slice(0, end));
          buffer = buffer.slice(end + 1);
        }
      } catch (error) { cancel(error instanceof Error ? error : new Error("声纹协议错误")); }
    };
    const processError = () => finish(new Error("无法启动本地主人声纹进程"));
    const closed = (code: number | null) => {
      if (failure) return; // Cancellation resolves only after the process tree is stopped.
      try { line(buffer + decoder.end()); } catch (error) { finish(error instanceof Error ? error : new Error("声纹协议错误")); return; }
      finish(code === 0 ? undefined : new Error(`本地主人声纹处理失败 (${code ?? "terminated"})`));
    };
    const abort = () => cancel(new Error("主人声纹录入已取消"));
    let timer: NodeJS.Timeout;
    function arm(): void {
      clearTimeout(timer);
      timer = setTimeout(() => cancel(new Error("主人声纹处理超时，请检查网络后重试")), timeoutMs);
    }
    arm();
    child.stdout?.on("data", data);
    child.once("error", processError);
    child.once("close", closed);
    if (signal.aborted) abort();
    else signal.addEventListener("abort", abort, { once: true });
  });
}

interface OwnerVoiceManagerOptions {
  configuration: () => Promise<DesktopConfiguration>;
  exclusive: <T>(operation: () => Promise<T>) => Promise<T>;
  command: (action: OwnerVoiceAction, configuration: DesktopConfiguration, signal: AbortSignal, onProgress?: (progress: OwnerVoiceProgress) => void, createdAt?: string) => Promise<OwnerVoiceStatus>;
  suspendCompanion: () => Promise<() => Promise<void>>;
  disableRecognition: () => Promise<void>;
}

export class OwnerVoiceManager extends EventEmitter {
  private active: { controller: AbortController; promise: Promise<OwnerVoiceStatus> } | null = null;
  private readonly inspections = new Map<AbortController, Promise<OwnerVoiceStatus>>();
  private state: OwnerVoiceEnrollmentState = { phase: "idle", status: null, detail: "尚未录入主人声纹" };

  constructor(private readonly options: OwnerVoiceManagerOptions) { super(); }

  get busy(): boolean { return this.active !== null; }
  currentState(): OwnerVoiceEnrollmentState { return structuredClone(this.state); }
  assertIdle(): void { if (this.busy) throw new Error("主人声纹正在处理中，请先完成或取消录入"); }

  async status(configuration?: DesktopConfiguration): Promise<OwnerVoiceStatus> {
    if (this.busy) return this.state.status ?? { ...EMPTY_STATUS };
    const controller = new AbortController();
    const pending = Promise.resolve().then(async () => {
      const current = configuration ?? await this.options.configuration();
      controller.signal.throwIfAborted();
      return this.options.command("status", current, controller.signal);
    });
    this.inspections.set(controller, pending);
    try {
      const result = await pending;
      this.state = { ...this.state, status: result };
      return result;
    } finally { this.inspections.delete(controller); }
  }

  async validate(configuration: DesktopConfiguration): Promise<void> {
    this.assertIdle();
    const status = await this.status(configuration);
    if (!status.enrolled || !status.modelReady) throw new Error(status.detail || "请先准备模型并录入主人声纹");
  }

  enroll(append = false): Promise<OwnerVoiceStatus> { return this.begin(append ? "append" : "enroll"); }
  delete(createdAt?: string): Promise<OwnerVoiceStatus> { return this.begin("delete", createdAt); }

  async cancel(): Promise<void> {
    const inspections = [...this.inspections];
    for (const [controller] of inspections) controller.abort();
    const active = this.active;
    active?.controller.abort();
    await Promise.allSettled([...inspections.map(([, promise]) => promise), ...(active ? [active.promise] : [])]);
  }

  private begin(action: "enroll" | "append" | "delete", createdAt?: string): Promise<OwnerVoiceStatus> {
    const records = action !== "delete";
    this.assertIdle();
    const controller = new AbortController();
    const promise = Promise.resolve().then(() => this.options.exclusive(async () => {
      let restore: (() => Promise<void>) | undefined;
      let result: OwnerVoiceStatus | undefined;
      let failure: unknown;
      let restoreFailed = false;
      try {
        const configuration = await this.options.configuration();
        controller.signal.throwIfAborted();
        if (records) {
          const prepared = await this.options.command("prepare", configuration, controller.signal, (progress) => {
            if (progress.kind === "download") this.publishDownload(progress.received, progress.total);
          });
          this.publish("preparing", "正在准备麦克风…", prepared);
          if (!prepared.modelReady) throw new Error("主人识别模型尚未准备完成");
        }
        controller.signal.throwIfAborted();
        restore = await this.options.suspendCompanion();
        controller.signal.throwIfAborted();
        // Removing everything turns 只听我的声音 off first; removing one keeps it,
        // unless that was the last voiceprint.
        if (action === "delete" && !createdAt) await this.options.disableRecognition();
        result = await this.options.command(action, configuration, controller.signal, (progress) => {
          if (progress.kind === "recording") this.publish("recording", "请在安静环境正常说话 15 秒", undefined, 15);
          else if (progress.kind === "processing") this.publish("processing", "正在生成你的声纹…");
        }, createdAt);
        if (action === "delete" && createdAt && !result.enrolled) await this.options.disableRecognition();
        if (records && (!result.enrolled || !result.modelReady)) {
          throw new Error(result.detail || "声纹录入未通过校验，请重试");
        }
      } catch (error) { failure = error; }
      finally {
        if (restore) {
          this.publish("restoring", "正在恢复之前的语音状态…", result);
          try { await restore(); } catch (error) { restoreFailed = true; failure = error; }
        }
      }
      if (failure) {
        const reason = failure instanceof Error ? failure.message : String(failure);
        const detail = restoreFailed
          ? `声纹处理已结束，但恢复语音失败：${reason}`
          : controller.signal.aborted
            ? restore ? "录入已取消，已恢复之前的语音状态" : "录入已取消"
            : reason;
        this.publish(controller.signal.aborted && !restoreFailed ? "idle" : "failed", detail);
        throw new Error(detail);
      }
      if (!result) throw new Error("声纹处理未返回结果");
      this.publish("idle", result.detail, result);
      return result;
    })).finally(() => { if (this.active?.controller === controller) this.active = null; });
    this.active = { controller, promise };
    this.publish("preparing", records ? "正在准备声纹模型…" : "正在删除主人声纹…");
    return promise;
  }

  private publish(phase: OwnerVoiceEnrollmentState["phase"], detail: string, status = this.state.status, durationSeconds?: number): void {
    this.state = { phase, detail, status: status ?? null, ...(durationSeconds ? { durationSeconds } : {}) };
    this.emit("state", this.currentState());
  }

  private publishDownload(received: number, total: number): void {
    const mb = (bytes: number) => (bytes / 1024 / 1024).toFixed(1);
    this.state = {
      phase: "preparing", status: this.state.status, download: { received, total },
      detail: `第一次录入要先下载声纹模型：${mb(received)} / ${mb(total)} MB`,
    };
    this.emit("state", this.currentState());
  }
}
