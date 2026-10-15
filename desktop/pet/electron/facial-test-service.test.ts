import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";
import path from "node:path";
import { PassThrough, Writable } from "node:stream";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { DemoEvent } from "../src/shared/facial-demo";
import { FacialTestService, type FacialTestRequest, type FacialTestServiceOptions, type SpawnFacialTest } from "./facial-test-service";

const services: FacialTestService[] = [];
afterEach(async () => {
  for (const service of services.splice(0)) await service.dispose();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("FacialTestService", () => {
  it("forwards a validated native custom question to the isolated JEV process", async () => {
    const { service, children } = harness();
    const parameter = { id: "CustomEar", minimum: -1, maximum: 1, defaultValue: 0,
      description: "耳朵竖起程度", levels: ["耳朵垂下", "耳朵自然", "耳朵竖起"] };
    await service.start({ ...request(), parameters: [parameter] });
    children[0]!.send({ type: "ready" });
    expect(children[0]!.messages[0]?.parameters).toEqual([{ ...parameter, mode: "score" }]);
    await expect(service.start({ ...request(), parameters: [{ ...parameter, levels: undefined }] })).rejects.toThrow("2–10");
  });
  it("accepts binary endpoints and rejects malformed mixed rubrics", async () => {
    const { service, children } = harness();
    const binary = { id: "ParamCheek", minimum: 0, maximum: 1, defaultValue: 0,
      mode: "binary" as const, description: "是否显示红晕",
      falseDescription: "不显示", trueDescription: "显示" };
    await service.start({ ...request(), parameters: [binary] });
    children[0]!.send({ type: "ready" });
    expect(children[0]!.messages[0]?.parameters).toEqual([binary]);
    await expect(service.start({ ...request(), parameters: [{ ...binary, levels: ["无", "有"] }] })).rejects.toThrow("选择式");
    await expect(service.start({ ...request(), parameters: [{ ...binary, falseDescription: "显示" }] })).rejects.toThrow("选择式");
  });
  it("uses the verified frozen backend for Windows development facial tests", async () => {
    vi.stubGlobal("process", { ...process, platform: "win32", arch: "x64" });
    const { service, spawnProcess } = harness({ packaged: false });
    await service.start(request());
    expect(spawnProcess.mock.calls[0]).toEqual([
      path.join("/repo", "desktop/pet/runtime/win32-x64/fish-memory-runtime/fish-memory-runtime.exe"),
      ["facial-test", "--config-directory", "/user data", "--output", path.join("/user data", "logs/facial-tests")],
      expect.objectContaining({ cwd: "/user data", shell: false, stdio: ["pipe", "pipe", "pipe"] }),
    ]);
    expect(spawnProcess.mock.calls[0]![2].env).not.toHaveProperty("FISH_MEMORY_PARENT_CONTROL");
  });

  it.each([false, true])("uses server-owned executable paths and JEV-only commands (packaged=%s)", async packaged => {
    vi.stubGlobal("process", { ...process, platform: "darwin", arch: "arm64" });
    const { service, children, spawnProcess, events } = harness({ packaged });
    await service.start({ ...request(), mode: "compare", character: "fake", operation: "other", output: "/tmp/renderer-path" } as FacialTestRequest);
    const child = children[0]!;
    expect(child.messages).toEqual([]);
    child.send({ type: "ready", cases: [{ text: "ignored" }] });

    expect(spawnProcess.mock.calls[0]).toEqual([
      packaged ? path.join("/app/resources", "runtime/fish-memory-runtime/fish-memory-runtime") : path.join("/repo", ".venv/bin/python"),
      [
        ...(packaged ? [] : ["-u", "-m", "fish_memory.desktop_runtime"]),
        "facial-test", "--config-directory", "/user data", "--output", path.join("/user data", "logs/facial-tests"),
      ],
      expect.objectContaining({ cwd: packaged ? "/user data" : "/repo", shell: false, stdio: ["pipe", "pipe", "pipe"] }),
    ]);
    expect(child.messages).toEqual([{
      operation: "run", id: "test-1", mode: "jev", character: "模型甲",
      text: "今天真开心。", parameters: request().parameters.map(parameter => ({ ...parameter, mode: "score" })),
    }]);
    expect(events).toEqual([{ type: "ready", id: "test-1" }]);
  });

  it("passes the current desktop speech connection only through the child environment", async () => {
    const connection = { apiUrl: "http://127.0.0.1:54321", apiKey: "private-speech-connection" };
    const speechConnection = vi.fn(async () => connection);
    const { service, children, spawnProcess, events } = harness({ speechConnection });
    await service.start(request());
    children[0]!.send({ type: "ready" });

    expect(speechConnection).toHaveBeenCalledOnce();
    const [command, args, options] = spawnProcess.mock.calls[0]!;
    expect(options.env).toMatchObject({
      FISH_MEMORY_FACIAL_TEST_API_URL: connection.apiUrl,
      FISH_MEMORY_FACIAL_TEST_API_KEY: connection.apiKey,
    });
    const publicOutput = JSON.stringify([command, args, children[0]!.messages, events]);
    expect(publicOutput).not.toContain(connection.apiKey);
    expect(publicOutput).not.toContain(connection.apiUrl);
  });

  it("does not inherit a stale speech connection when the legacy caller has no resolver", async () => {
    vi.stubEnv("FISH_MEMORY_FACIAL_TEST_API_URL", "http://127.0.0.1:12345");
    vi.stubEnv("FISH_MEMORY_FACIAL_TEST_API_KEY", "stale-speech-connection");
    const { service, spawnProcess } = harness();
    await service.start(request());
    const env = spawnProcess.mock.calls[0]![2].env!;
    expect(env).not.toHaveProperty("FISH_MEMORY_FACIAL_TEST_API_URL");
    expect(env).not.toHaveProperty("FISH_MEMORY_FACIAL_TEST_API_KEY");
  });

  it("refreshes the desktop speech connection for each new run", async () => {
    const speechConnection = vi.fn()
      .mockResolvedValueOnce({ apiUrl: "http://127.0.0.1:54321", apiKey: "first-connection" })
      .mockResolvedValueOnce({ apiUrl: "http://127.0.0.1:54322", apiKey: "second-connection" });
    const { service, spawnProcess } = harness({ speechConnection });
    await service.start(request());
    await service.stop("test-1");
    await service.start({ ...request(), id: "test-2" });
    expect(speechConnection).toHaveBeenCalledTimes(2);
    expect(spawnProcess.mock.calls.map(([, , options]) => [
      options.env?.FISH_MEMORY_FACIAL_TEST_API_URL,
      options.env?.FISH_MEMORY_FACIAL_TEST_API_KEY,
    ])).toEqual([
      ["http://127.0.0.1:54321", "first-connection"],
      ["http://127.0.0.1:54322", "second-connection"],
    ]);
  });

  it("forwards split UTF-8 and cached audio, then ends stdin without a false crash", async () => {
    const { service, children, events } = harness();
    await service.start(request());
    const child = children[0]!;
    child.send({ type: "ready" });
    const event = { type: "segment", id: "test-1", text: "开心", segmentIndex: 0 };
    const bytes = Buffer.from(JSON.stringify(event) + "\n");
    const split = bytes.indexOf(Buffer.from("开")) + 1;
    child.stdout.write(bytes.subarray(0, split));
    child.stdout.write(bytes.subarray(split));
    child.send({ type: "audio", id: "test-1", pcm: "AAABAA==" });
    child.send({ type: "done", id: "test-1", status: "completed" });
    await flush();

    expect(events).toContainEqual(event);
    expect(events).toContainEqual({ type: "audio", id: "test-1", pcm: "AAABAA==" });
    expect(events.filter(event => event.type === "done")).toEqual([{ type: "done", id: "test-1", status: "completed" }]);
    expect(events.some(event => event.type === "error" || event.type === "fatal")).toBe(false);
    expect(child.stdin.writableEnded).toBe(true);
    expect(child.kill).not.toHaveBeenCalled();
    expect(child.stdout.listenerCount("data")).toBe(0);
    expect(child.listenerCount("close")).toBe(0);
  });

  it("isolates late output and same-id reuse from replaced processes", async () => {
    const { service, children, events } = harness();
    await service.start(request());
    const first = children[0]!;
    first.closeOnEnd = false;
    first.send({ type: "ready" });
    await service.start({ ...request(), modelId: "model-b" });
    const second = children[1]!;
    second.send({ type: "ready" });
    first.send({ type: "face", id: "test-1", targets: { ParamEyeLOpen: 0.9 } });
    first.send({ type: "done", id: "test-1", status: "completed" });
    second.send({ type: "face", id: "wrong-id", targets: { ParamEyeLOpen: 0.8 } });
    second.send({ type: "face", id: "test-1", targets: { ParamEyeLOpen: 0.3 } });
    first.close();
    expect(events.filter(event => event.type === "face")).toEqual([{ type: "face", id: "test-1", targets: { ParamEyeLOpen: 0.3 } }]);
    expect(first.messages.at(-1)).toEqual({ operation: "stop", id: "test-1" });
    expect(second.messages[0]?.character).toBe("模型乙");
  });

  it("does not resurrect a run stopped during model validation", async () => {
    const model = deferred<{ modelId: string; modelName: string }>();
    const { service, spawnProcess, events } = harness({ resolveModel: () => model.promise });
    const pending = service.start(request());
    await service.stop("test-1");
    model.resolve({ modelId: "model-a", modelName: "模型甲" });
    await pending;
    expect(spawnProcess).not.toHaveBeenCalled();
    expect(events).toEqual([{ type: "done", id: "test-1", status: "stopped" }]);
  });

  it("does not spawn after a run is stopped while resolving the desktop speech connection", async () => {
    const connection = deferred<{ apiUrl: string; apiKey: string }>();
    const speechConnection = vi.fn(() => connection.promise);
    const { service, spawnProcess, events } = harness({ speechConnection });
    const pending = service.start(request());
    await flush();
    expect(speechConnection).toHaveBeenCalledOnce();
    await service.stop("test-1");
    connection.resolve({ apiUrl: "http://127.0.0.1:54321", apiKey: "cancelled-connection" });
    await pending;
    expect(spawnProcess).not.toHaveBeenCalled();
    expect(events).toEqual([{ type: "done", id: "test-1", status: "stopped" }]);
  });

  it("stops before ready and ignores late ready/face feedback", async () => {
    const { service, children, events } = harness();
    await service.start(request());
    const child = children[0]!;
    await service.stop("wrong-id");
    expect(child.stdin.writableEnded).toBe(false);
    const stopped = service.stop("test-1");
    child.send({ type: "ready" });
    await stopped;
    expect(child.messages).toEqual([]);
    expect(events).toEqual([{ type: "done", id: "test-1", status: "stopped" }]);
  });

  it("strips legacy renderer state from the provider command", async () => {
    const { service, children } = harness();
    await service.start({ ...request(), currentFace: { ParamEyeLOpen: 0.2 }, currentFaceSource: "initial", currentFaceObservedAtMs: 10 } as FacialTestRequest);
    children[0]!.send({ type: "ready" });
    const command = children[0]!.messages[0]!;
    expect(command).not.toHaveProperty("currentFace");
    expect(command).not.toHaveProperty("currentFaceSource");
    expect(command).not.toHaveProperty("currentFaceObservedAtMs");
  });

  it.each(["malformed", "closed", "spawn-error", "stdin-error", "startup-error"])("reports %s once with no stderr/credential leakage", async failure => {
    const { service, children, events } = harness();
    await service.start(request());
    const child = children[0]!;
    child.stderr.write("Bearer private-token-must-not-leak\n");
    if (failure !== "startup-error") child.send({ type: "ready" });
    if (failure === "malformed") child.stdout.write("not JSON\n");
    if (failure === "closed") child.close();
    if (failure === "spawn-error") child.emit("error", new Error("Bearer private-token-must-not-leak"));
    if (failure === "stdin-error") child.stdin.emit("error", new Error("private-token-must-not-leak"));
    if (failure === "startup-error") child.send({ type: "error", stage: "startup", message: "Bearer private-token-must-not-leak" });
    await flush();
    expect(events.filter(event => event.type === "error")).toHaveLength(1);
    expect(events.filter(event => event.type === "done")).toEqual([{ type: "done", id: "test-1", status: "failed" }]);
    expect(events.every(event => event.id === "test-1")).toBe(true);
    expect(JSON.stringify(events)).not.toContain("private-token-must-not-leak");
    expect(child.stdout.listenerCount("data")).toBe(0);
  });

  it("redacts provider errors and removes raw failure payloads", async () => {
    const { service, children, events } = harness();
    await service.start(request());
    const child = children[0]!;
    child.send({ type: "ready" });
    child.send({ type: "error", id: "test-1", stage: "jev", message: "Authorization: Bearer private-token-1234", raw: { secret: "sensitive" } });
    expect(events.at(-1)).toEqual({ type: "error", id: "test-1", stage: "jev", message: "Authorization=[REDACTED]" });
  });

  it("kills unresponsive processes after stop and cleans their pipes", async () => {
    vi.useFakeTimers();
    const { service, children, events } = harness();
    await service.start(request());
    const child = children[0]!;
    child.closeOnEnd = false;
    child.closeOnKill = false;
    child.send({ type: "ready" });
    const stopped = service.stop("test-1");
    expect(events.at(-1)).toEqual({ type: "done", id: "test-1", status: "stopped" });
    expect(child.stdin.writableEnded).toBe(true);
    await vi.advanceTimersByTimeAsync(1500);
    await stopped;
    expect(child.kill.mock.calls).toEqual([["SIGTERM"], ["SIGKILL"]]);
    expect(child.stdout.destroyed).toBe(true);
    expect(child.stderr.destroyed).toBe(true);
    expect(child.listenerCount("close")).toBe(0);
  });

  it("times out a process that never becomes ready", async () => {
    vi.useFakeTimers();
    const { service, events } = harness();
    await service.start(request());
    await vi.advanceTimersByTimeAsync(30_001);
    expect(events).toEqual([
      { type: "error", id: "test-1", stage: "process", message: "表情测试进程启动超时，请重试" },
      { type: "done", id: "test-1", status: "failed" },
    ]);
  });

  it("bounds unterminated JSONL data instead of retaining unlimited output", async () => {
    const { service, children, events } = harness();
    await service.start(request());
    children[0]!.stdout.write("x".repeat(8 * 1024 * 1024 + 1));
    await flush();
    expect(events.at(-1)).toEqual({ type: "done", id: "test-1", status: "failed" });
    expect(events[0]?.message).toContain("超出限制");
  });

  it("does not spawn for missing models or invalid input, and rejects after dispose", async () => {
    const { service, spawnProcess, events } = harness({ resolveModel: async () => { throw new Error("not found"); } });
    await service.start(request());
    expect(events.at(-1)).toEqual({ type: "done", id: "test-1", status: "failed" });
    await expect(service.start({ ...request(), text: " " })).rejects.toThrow("朗读文字");
    await expect(service.start({ ...request(), parameters: [{ ...request().parameters[0]!, id: "ParamBreath" }] })).rejects.toThrow("自动参数");
    expect(spawnProcess).not.toHaveBeenCalled();
    await service.dispose();
    await expect(service.start(request())).rejects.toThrow("已关闭");
  });
});

function request(): FacialTestRequest {
  return {
    id: "test-1", modelId: "model-a", text: "今天真开心。",
    parameters: [{ id: "ParamEyeLOpen", minimum: 0, maximum: 1, defaultValue: 1 }],
  };
}

function harness(overrides: Partial<FacialTestServiceOptions> = {}) {
  const children: FakeChild[] = [];
  const events: DemoEvent[] = [];
  const spawnProcess = vi.fn<SpawnFacialTest>(() => {
    const child = new FakeChild();
    children.push(child);
    return child as unknown as ChildProcessWithoutNullStreams;
  });
  const service = new FacialTestService({
    userData: "/user data", projectRoot: "/repo", resourcesPath: "/app/resources", packaged: false,
    resolveModel: async modelId => ({ modelId, modelName: modelId === "model-a" ? "模型甲" : "模型乙" }),
    emit: event => events.push(event), spawnProcess, ...overrides,
  });
  services.push(service);
  return { service, children, spawnProcess, events };
}

class FakeChild extends EventEmitter {
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  readonly messages: Record<string, unknown>[] = [];
  closeOnEnd = true;
  closeOnKill = true;
  exitCode: number | null = null;
  readonly stdin = new Writable({
    write: (chunk, _encoding, callback) => {
      for (const line of String(chunk).trim().split("\n")) this.messages.push(JSON.parse(line));
      callback();
    },
    final: callback => {
      callback();
      if (this.closeOnEnd) queueMicrotask(() => this.close());
    },
  });
  readonly kill = vi.fn((_signal?: NodeJS.Signals) => {
    if (this.closeOnKill) queueMicrotask(() => this.close());
    return true;
  });

  send(event: Record<string, unknown>): void { this.stdout.write(JSON.stringify(event) + "\n"); }
  close(): void {
    if (this.exitCode !== null) return;
    this.exitCode = 0;
    this.emit("exit", 0, null);
    this.emit("close", 0, null);
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  return { promise: new Promise<T>(done => { resolve = done; }), resolve: (value: T) => resolve(value) };
}

async function flush(): Promise<void> { await Promise.resolve(); await Promise.resolve(); }
