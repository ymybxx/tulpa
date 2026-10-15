import { spawn, type ChildProcessWithoutNullStreams, type SpawnOptionsWithoutStdio } from "node:child_process";
import path from "node:path";
import { StringDecoder } from "node:string_decoder";

import type { DemoEvent } from "../src/shared/facial-demo";
import type { FacialTestStartRequest } from "../src/shared/facial-test";
import standardParameters from "../src/shared/facial-demo-parameters.json";
import { redactSensitiveText } from "./sensitive-text";
import { frozenRuntimeExecutable } from "./platform/runtime-paths";
import { attachProcessJob } from "./platform/process-tree";

export type FacialTestRequest = FacialTestStartRequest;
export type FacialTestParameter = FacialTestStartRequest["parameters"][number];

export type SpawnFacialTest = (
  command: string,
  args: string[],
  options: SpawnOptionsWithoutStdio & { stdio: ["pipe", "pipe", "pipe"] },
) => ChildProcessWithoutNullStreams;

export interface FacialTestServiceOptions {
  userData: string;
  projectRoot: string;
  resourcesPath: string;
  packaged: boolean;
  resolveModel(modelId: string): Promise<{ modelId: string; modelName: string }>;
  /** Current desktop API; resolves the character's own voice for this run. */
  speechConnection?(): Promise<{ apiUrl: string; apiKey: string }>;
  emit(event: DemoEvent): void;
  spawnProcess?: SpawnFacialTest;
}

interface Run {
  generation: number;
  id: string;
  request: FacialTestRequest;
  child: ChildProcessWithoutNullStreams | null;
  ready: boolean;
  terminal: boolean;
  closing: boolean;
  closed: boolean;
  buffer: string;
  totalBytes: number;
  decoder: StringDecoder;
  startupTimer?: NodeJS.Timeout;
  shutdown?: Promise<void>;
  closedPromise: Promise<void>;
  resolveClosed(): void;
  cleanup?: () => void;
}

const MAX_LINE_BYTES = 8 * 1024 * 1024;
const MAX_RUN_BYTES = 64 * 1024 * 1024;
const STARTUP_TIMEOUT_MS = 30_000;
const STOP_GRACE_MS = 500;
const EVENTS = new Set(["started", "segment", "face", "audio", "audio_end", "error", "done"]);
const STANDARD_IDS = new Set(standardParameters.filter(parameter => parameter.owner === "jev").map(parameter => parameter.id));
const ALL_STANDARD_IDS = new Set(standardParameters.map(parameter => parameter.id));

/** One isolated provider process per test; never starts Companion or its microphone. */
export class FacialTestService {
  private generation = 0;
  private current: Run | null = null;
  private disposed = false;
  private readonly runs = new Set<Run>();
  private readonly spawnProcess: SpawnFacialTest;

  constructor(private readonly options: FacialTestServiceOptions) {
    this.spawnProcess = options.spawnProcess ?? ((command, args, settings) => {
      const child = spawn(command, args, settings);
      attachProcessJob(child);
      return child;
    });
  }

  async start(value: FacialTestRequest): Promise<void> {
    if (this.disposed) throw new Error("表情测试已关闭");
    const request = validateRequest(value);
    const previous = this.current;
    const generation = ++this.generation;
    let resolveClosed!: () => void;
    const run: Run = {
      generation, id: request.id, request, child: null, ready: false, terminal: false,
      closing: false, closed: false, buffer: "", totalBytes: 0, decoder: new StringDecoder("utf8"),
      closedPromise: new Promise<void>(resolve => { resolveClosed = resolve; }), resolveClosed: () => resolveClosed(),
    };
    this.current = run;
    this.runs.add(run);
    if (previous) this.cancel(previous);

    try {
      const model = await this.options.resolveModel(request.modelId);
      if (!this.isCurrent(run)) return;
      if (model.modelId !== request.modelId || typeof model.modelName !== "string" || !model.modelName.trim()) {
        throw new Error("Invalid model");
      }
      const connection = this.options.speechConnection ? await this.options.speechConnection() : undefined;
      if (!this.isCurrent(run)) return;
      const env: NodeJS.ProcessEnv = { ...process.env, PYTHONUNBUFFERED: "1", PYTHONUTF8: "1" };
      // This subprocess has its own JSON control protocol on stdin.
      delete env.FISH_MEMORY_PARENT_CONTROL;
      // Only this run's resolved service may be used; legacy callers must not
      // accidentally reuse a connection inherited from another process.
      delete env.FISH_MEMORY_FACIAL_TEST_API_URL;
      delete env.FISH_MEMORY_FACIAL_TEST_API_KEY;
      if (connection) {
        env.FISH_MEMORY_FACIAL_TEST_API_URL = connection.apiUrl;
        env.FISH_MEMORY_FACIAL_TEST_API_KEY = connection.apiKey;
      }
      // Paths and operation are selected here, never copied from renderer input.
      const executable = frozenRuntimeExecutable(this.options);
      const command = executable ?? path.join(this.options.projectRoot, ".venv", "bin", "python");
      const args = [
        ...(executable ? [] : ["-u", "-m", "fish_memory.desktop_runtime"]),
        "facial-test", "--config-directory", this.options.userData,
        "--output", path.join(this.options.userData, "logs", "facial-tests"),
      ];
      const child = this.spawnProcess(command, args, {
        cwd: executable ? this.options.userData : this.options.projectRoot,
        stdio: ["pipe", "pipe", "pipe"], shell: false, windowsHide: true,
        env,
      });
      run.child = child;
      this.bind(run, model.modelName);
      run.startupTimer = setTimeout(() => this.fail(run, "表情测试进程启动超时，请重试"), STARTUP_TIMEOUT_MS);
      run.startupTimer.unref();
    } catch {
      if (!this.isCurrent(run)) return;
      this.fail(run, "无法启动表情测试，请确认模型及运行环境可用");
    }
  }

  async stop(id?: string): Promise<void> {
    const run = this.current;
    if (!run || (id !== undefined && id !== run.id)) return;
    this.current = null;
    ++this.generation;
    this.cancel(run);
    await run.shutdown;
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    await this.stop();
    await Promise.all([...this.runs].map(run => this.shutdown(run, true)));
  }

  private isCurrent(run: Run): boolean {
    return this.current === run && run.generation === this.generation && !this.disposed && !run.closing;
  }

  private bind(run: Run, character: string): void {
    const child = run.child!;
    const onData = (value: Buffer | string) => {
      if (!this.isCurrent(run) || run.terminal) return;
      const bytes = Buffer.isBuffer(value) ? value : Buffer.from(value);
      run.totalBytes += bytes.byteLength;
      if (run.totalBytes > MAX_RUN_BYTES) { this.fail(run, "表情测试返回数据超出限制"); return; }
      run.buffer += run.decoder.write(bytes);
      let newline: number;
      while (!run.terminal && !run.closing && (newline = run.buffer.indexOf("\n")) >= 0) {
        const line = run.buffer.slice(0, newline);
        run.buffer = run.buffer.slice(newline + 1);
        this.receive(run, line, character);
      }
      if (Buffer.byteLength(run.buffer) > MAX_LINE_BYTES) this.fail(run, "表情测试单条返回数据超出限制");
    };
    const onClose = () => {
      if (this.isCurrent(run) && !run.terminal) {
        run.buffer += run.decoder.end();
        if (run.buffer.trim()) this.receive(run, run.buffer, character);
        if (!run.terminal) this.fail(run, "表情测试进程意外退出，请重试");
      }
      this.closed(run);
    };
    const onError = () => this.fail(run, "表情测试进程无法运行，请检查运行环境");
    const onInputError = () => {
      if (!run.closing && !run.terminal) this.fail(run, "表情测试连接已关闭，请重试");
    };
    const ignoreStderr = () => { /* Only sanitized JSONL events are exposed, never stderr. */ };
    child.stdout.on("data", onData);
    child.stdout.on("error", onError);
    child.stderr.on("data", ignoreStderr);
    child.stderr.on("error", onError);
    child.stdin.on("error", onInputError);
    child.on("error", onError);
    child.once("close", onClose);
    run.cleanup = () => {
      child.stdout.off("data", onData);
      child.stdout.off("error", onError);
      child.stderr.off("data", ignoreStderr);
      child.stderr.off("error", onError);
      child.stdin.off("error", onInputError);
      child.off("error", onError);
      child.off("close", onClose);
      // Destroying pipes may still schedule an error; it must not become uncaught.
      for (const stream of [child.stdin, child.stdout, child.stderr]) {
        stream.on("error", () => {});
        stream.destroy();
      }
      child.on("error", () => {});
    };
  }

  private receive(run: Run, line: string, character: string): void {
    if (!this.isCurrent(run) || run.terminal || !line.trim()) return;
    if (Buffer.byteLength(line) > MAX_LINE_BYTES) { this.fail(run, "表情测试单条返回数据超出限制"); return; }
    let event: DemoEvent;
    try {
      const parsed: unknown = JSON.parse(line);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed) || typeof (parsed as DemoEvent).type !== "string") throw new Error();
      event = parsed as DemoEvent;
    } catch { this.fail(run, "表情测试返回格式异常，请重试"); return; }
    if (event.type === "ready") {
      if (run.ready) return;
      run.ready = true;
      clearTimeout(run.startupTimer);
      this.options.emit({ type: "ready", id: run.id });
      this.write(run, {
        operation: "run", id: run.id, mode: "jev", character,
        text: run.request.text, parameters: run.request.parameters,
      });
      return;
    }
    if (event.type === "fatal" || (event.type === "error" && !event.id)) {
      this.fail(run, "表情测试初始化失败，请检查决策模型和语音配置"); return;
    }
    if (!run.ready || event.id !== run.id || !EVENTS.has(event.type)) return;
    // Provider failures may contain diagnostics; suppress raw data and redact the
    // user-facing message rather than exposing credentials from a remote error.
    if (event.type === "error") {
      event = { ...event, message: redactSensitiveText(String(event.message ?? "表情测试请求失败"), 500) };
      delete event.raw;
    }
    if (event.type === "done") {
      run.terminal = true;
      this.options.emit(event);
      void this.shutdown(run, false);
    } else {
      this.options.emit(event);
    }
  }

  private write(run: Run, event: Record<string, unknown>): void {
    if (!run.child || run.child.stdin.destroyed || run.child.stdin.writableEnded) return;
    try { run.child.stdin.write(`${JSON.stringify(event)}\n`); }
    catch { if (!run.terminal && !run.closing) this.fail(run, "表情测试连接已关闭，请重试"); }
  }

  private fail(run: Run, message: string): void {
    if (!this.isCurrent(run) || run.terminal) return;
    run.terminal = true;
    this.options.emit({ type: "error", id: run.id, stage: "process", message });
    this.options.emit({ type: "done", id: run.id, status: "failed" });
    void this.shutdown(run, true);
  }

  private cancel(run: Run): void {
    if (!run.terminal) {
      run.terminal = true;
      this.options.emit({ type: "done", id: run.id, status: "stopped" });
    }
    void this.shutdown(run, true);
  }

  private shutdown(run: Run, cancel: boolean): Promise<void> {
    if (run.shutdown) return run.shutdown;
    run.closing = true;
    clearTimeout(run.startupTimer);
    run.buffer = "";
    run.shutdown = (async () => {
      const child = run.child;
      if (!child) { this.closed(run); return; }
      if (cancel && run.ready) this.write(run, { operation: "stop", id: run.id });
      try { child.stdin.end(); } catch { /* Broken stdin is already terminal. */ }
      if (await closedWithin(run, STOP_GRACE_MS)) return;
      try { child.kill("SIGTERM"); } catch { /* May have exited after the deadline. */ }
      if (await closedWithin(run, STOP_GRACE_MS)) return;
      try { child.kill("SIGKILL"); } catch { /* May have exited after the deadline. */ }
      await closedWithin(run, STOP_GRACE_MS);
      this.closed(run);
    })();
    return run.shutdown;
  }

  private closed(run: Run): void {
    if (run.closed) return;
    run.closed = true;
    clearTimeout(run.startupTimer);
    run.cleanup?.();
    this.runs.delete(run);
    if (this.current === run) this.current = null;
    run.resolveClosed();
  }
}

function closedWithin(run: Run, milliseconds: number): Promise<boolean> {
  if (run.closed) return Promise.resolve(true);
  return new Promise(resolve => {
    const timer = setTimeout(() => resolve(false), milliseconds);
    run.closedPromise.then(() => { clearTimeout(timer); resolve(true); });
  });
}

function validateRequest(value: FacialTestRequest): FacialTestRequest {
  if (!value || typeof value !== "object" || !validIdentifier(value.id) || !validIdentifier(value.modelId)) throw new Error("表情测试请求无效");
  if (typeof value.text !== "string" || !value.text.trim() || value.text.length > 2000) throw new Error("请输入 1–2000 字的朗读文字");
  if (!Array.isArray(value.parameters) || !value.parameters.length || value.parameters.length > 49) throw new Error("当前模型没有可测试的参数");
  const seen = new Set<string>();
  const parameters = value.parameters.map(parameter => {
    if (!parameter || !validIdentifier(parameter.id) || seen.has(parameter.id)
      || ![parameter.minimum, parameter.maximum, parameter.defaultValue].every(Number.isFinite)
      || parameter.minimum >= parameter.maximum || parameter.defaultValue < parameter.minimum || parameter.defaultValue > parameter.maximum) {
      throw new Error("表情测试参数无效");
    }
    const custom = !STANDARD_IDS.has(parameter.id);
    if (custom && ALL_STANDARD_IDS.has(parameter.id)) throw new Error("自动参数不能交给决策模型");
    const mode = parameter.mode ?? "score";
    if (mode !== "score" && mode !== "binary") throw new Error("参数判断模式无效");
    if (mode === "score") {
      if ((custom || parameter.levels !== undefined)
        && (!Array.isArray(parameter.levels) || parameter.levels.length < 2 || parameter.levels.length > 10
          || !parameter.levels.every(level => typeof level === "string" && !!level.trim() && level.length <= 100)
          || new Set(parameter.levels.map(level => level.trim())).size !== parameter.levels.length)) {
        throw new Error("渐进式参数需要 2–10 个不同的评分描述");
      }
      if (parameter.anchors !== undefined && (!Array.isArray(parameter.anchors)
        || parameter.anchors.length !== (parameter.levels?.length ?? 0)
        || !parameter.anchors.every(Number.isFinite)
        || parameter.anchors.some((value, index) => index > 0 && value <= parameter.anchors![index - 1]!))) {
        throw new Error("评分锚点无效");
      }
      if (parameter.falseDescription !== undefined || parameter.trueDescription !== undefined) throw new Error("渐进式参数不能包含选择式描述");
    } else if (typeof parameter.falseDescription !== "string" || !parameter.falseDescription.trim()
      || parameter.falseDescription.length > 100 || typeof parameter.trueDescription !== "string"
      || !parameter.trueDescription.trim() || parameter.trueDescription.length > 100
      || parameter.falseDescription.trim() === parameter.trueDescription.trim()
      || parameter.levels !== undefined || parameter.anchors !== undefined) throw new Error("选择式参数需要不同的是/否描述");
    if (parameter.description !== undefined && (typeof parameter.description !== "string"
      || !parameter.description.trim() || parameter.description.length > 500)) throw new Error("参数描述无效");
    if (custom && !parameter.description) throw new Error("自定义参数需要描述");
    seen.add(parameter.id);
    return { id: parameter.id, minimum: parameter.minimum, maximum: parameter.maximum, defaultValue: parameter.defaultValue,
      ...(parameter.description ? { description: parameter.description } : {}),
      mode,
      ...(mode === "score" ? {
        ...(parameter.levels ? { levels: parameter.levels.map(level => level.trim()) } : {}),
        ...(parameter.anchors ? { anchors: [...parameter.anchors] } : {}),
      } : { falseDescription: parameter.falseDescription!.trim(), trueDescription: parameter.trueDescription!.trim() }) };
  });
  return { id: value.id, modelId: value.modelId, text: value.text.trim(), parameters };
}

function validIdentifier(value: unknown): value is string {
  return typeof value === "string" && !!value.trim() && value.length <= 120 && !/[\r\n\0]/u.test(value);
}
