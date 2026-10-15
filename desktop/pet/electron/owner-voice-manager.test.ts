import { EventEmitter } from "node:events";
import type { ChildProcess } from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import type { DesktopConfiguration, OwnerVoiceStatus } from "../src/shared/contracts";
import { OwnerVoiceManager, parseOwnerVoiceStatus, runOwnerVoiceCommand, type OwnerVoiceProgress } from "./owner-voice-manager";

const status: OwnerVoiceStatus = {
  enrolled: true, modelReady: true, modelId: "cam++", createdAt: "2026-09-03T00:00:00Z", detail: "已录入主人声纹", audioBackend: "livekit",
};
const configuration = { advanced: { audioBackend: "livekit" } } as DesktopConfiguration;

function child() {
  return Object.assign(new EventEmitter(), { stdout: new EventEmitter(), stderr: new EventEmitter() }) as unknown as ChildProcess;
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

describe("owner voice CLI protocol", () => {
  it("accepts only a sanitized public status", () => {
    expect(parseOwnerVoiceStatus({ ...status, embedding: [1, 2, 3] })).toEqual(status);
    expect(() => parseOwnerVoiceStatus({ ...status, enrolled: "true" })).toThrow("格式");
    expect(() => parseOwnerVoiceStatus({ ...status, audioBackend: "remote" })).toThrow("格式");
  });

  it("waits for recording and a complete JSONL result, including split UTF-8", async () => {
    const process = child();
    const recording = vi.fn();
    const result = runOwnerVoiceCommand(process, "enroll", new AbortController().signal, recording);
    process.stdout!.emit("data", Buffer.from('{"event":"download","received":1,"total":2}\n'));
    process.stdout!.emit("data", Buffer.from('{"event":"recording","durationSeconds":15}\n'));
    process.stdout!.emit("data", Buffer.from('{"event":"processing"}\n'));
    const bytes = Buffer.from(JSON.stringify({ event: "result", status }));
    const split = bytes.indexOf(Buffer.from("已")) + 1;
    process.stdout!.emit("data", bytes.subarray(0, split));
    process.stdout!.emit("data", bytes.subarray(split));
    process.emit("close", 0);
    expect(await result).toEqual(status);
    expect(recording.mock.calls).toEqual([
      [{ kind: "download", received: 1, total: 2 }], [{ kind: "recording" }], [{ kind: "processing" }],
    ]);
  });

  it.each(["not json\n", '{"event":"result","status":{}}\n', '{"event":"unknown"}\n'])(
    "terminates invalid protocol output: %s", async (line) => {
      const process = child();
      const stop = vi.fn(async () => {});
      const result = runOwnerVoiceCommand(process, "status", new AbortController().signal, undefined, { stop });
      const rejected = expect(result).rejects.toThrow();
      process.stdout!.emit("data", Buffer.from(line));
      await rejected;
      expect(stop).toHaveBeenCalledWith(process);
    },
  );

  it.each([0, 1])("rejects missing results and nonzero exit (%i)", async (code) => {
    const process = child();
    const result = runOwnerVoiceCommand(process, "status", new AbortController().signal);
    const rejected = expect(result).rejects.toThrow();
    process.emit("close", code);
    await rejected;
  });

  it.each(["timeout", "cancel"])("stops the process tree on %s", async (reason) => {
    const process = child();
    const controller = new AbortController();
    const stop = vi.fn(async () => {});
    const result = runOwnerVoiceCommand(process, "enroll", controller.signal, undefined, { timeoutMs: 5, stop });
    const rejected = expect(result).rejects.toThrow(reason === "timeout" ? "超时" : "取消");
    if (reason === "cancel") controller.abort();
    await rejected;
    expect(stop).toHaveBeenCalledOnce();
  });
});

describe("owner voice enrollment lifecycle", () => {
  function harness(command = vi.fn(async () => status)) {
    const restore = vi.fn(async () => {});
    const suspend = vi.fn(async () => restore);
    const disable = vi.fn(async () => {});
    const manager = new OwnerVoiceManager({
      configuration: async () => configuration,
      exclusive: async (operation) => operation(),
      command, suspendCompanion: suspend, disableRecognition: disable,
    });
    return { manager, command, restore, suspend, disable };
  }

  it("removing one voiceprint keeps 只听我的声音 on unless none is left", async () => {
    const remaining = { ...status, enrolled: true, enrollmentCount: 1 };
    const one = harness(vi.fn(async () => remaining));
    await one.manager.delete("2026-09-28T12:40:24+00:00");
    expect(one.command).toHaveBeenCalledWith("delete", configuration, expect.anything(), expect.anything(), "2026-09-28T12:40:24+00:00");
    expect(one.disable).not.toHaveBeenCalled();
    const last = harness(vi.fn(async () => ({ ...status, enrolled: false, enrollmentCount: 0 })));
    await last.manager.delete("2026-09-28T12:40:24+00:00");
    expect(last.disable).toHaveBeenCalledOnce();
    const all = harness(vi.fn(async () => ({ ...status, enrolled: false })));
    await all.manager.delete();
    expect(all.disable).toHaveBeenCalledOnce();
  });

  it("prepares before touching Companion, records only on explicit enroll, then restores", async () => {
    const ready = deferred<OwnerVoiceStatus>();
    const order: string[] = [];
    const phases: string[] = [];
    const command = vi.fn(async (action: string, _configuration: DesktopConfiguration, _signal: AbortSignal,
      progress?: (value: OwnerVoiceProgress) => void) => {
      order.push(action);
      if (action === "prepare") {
        progress?.({ kind: "download", received: 14 * 1024 * 1024, total: 28 * 1024 * 1024 });
        return ready.promise;
      }
      progress?.({ kind: "recording" });
      progress?.({ kind: "processing" });
      return status;
    });
    const { manager, suspend, restore } = harness(command);
    manager.on("state", (state) => phases.push(state.download ? `${state.phase}:${state.detail}` : state.phase));
    expect(command).not.toHaveBeenCalled();
    const result = manager.enroll();
    await vi.waitFor(() => expect(command).toHaveBeenCalledOnce());
    expect(suspend).not.toHaveBeenCalled();
    expect(() => manager.enroll()).toThrow("处理中");
    ready.resolve(status);
    expect(await result).toEqual(status);
    expect(order).toEqual(["prepare", "enroll"]);
    expect(phases).toEqual(["preparing", "preparing:第一次录入要先下载声纹模型：14.0 / 28.0 MB",
      "preparing", "recording", "processing", "restoring", "idle"]);
    expect(suspend).toHaveBeenCalledOnce();
    expect(restore).toHaveBeenCalledOnce();
    expect(manager.currentState().phase).toBe("idle");
    expect(manager.busy).toBe(false);
  });

  it.each(["prepare", "enroll"])("cancels %s through a separate signal and restores when suspended", async (stage) => {
    const started = deferred<void>();
    const command = vi.fn(async (action: string, _configuration: DesktopConfiguration, signal: AbortSignal) => {
      if (action !== stage) return status;
      started.resolve();
      return new Promise<OwnerVoiceStatus>((_resolve, reject) => {
        signal.addEventListener("abort", () => reject(new Error("canceled")), { once: true });
      });
    });
    const { manager, restore } = harness(command);
    const result = manager.enroll();
    const rejected = expect(result).rejects.toThrow("取消");
    await started.promise;
    await manager.cancel();
    await rejected;
    expect(restore).toHaveBeenCalledTimes(stage === "enroll" ? 1 : 0);
    expect(manager.busy).toBe(false);
  });

  it("restores Companion after enrollment failure without automatically enabling recognition", async () => {
    const command = vi.fn(async (action: string) => {
      if (action === "enroll") throw new Error("sample rejected");
      return status;
    });
    const { manager, restore, disable } = harness(command);
    await expect(manager.enroll()).rejects.toThrow("sample rejected");
    expect(restore).toHaveBeenCalledOnce();
    expect(disable).not.toHaveBeenCalled();
    expect(manager.currentState().phase).toBe("failed");
  });

  it("does not claim restoration succeeded when cancellation is followed by a restore failure", async () => {
    const started = deferred<void>();
    const command = vi.fn(async (action: string, _configuration: DesktopConfiguration, signal: AbortSignal) => {
      if (action !== "enroll") return status;
      started.resolve();
      return new Promise<OwnerVoiceStatus>((_resolve, reject) => {
        signal.addEventListener("abort", () => reject(new Error("canceled")), { once: true });
      });
    });
    const { manager, restore } = harness(command);
    restore.mockRejectedValueOnce(new Error("audio restart rejected"));
    const enrollment = manager.enroll();
    const rejected = expect(enrollment).rejects.toThrow("恢复语音失败");
    await started.promise;
    await manager.cancel();
    await rejected;
    expect(manager.currentState().phase).toBe("failed");
    expect(manager.currentState().detail).not.toContain("已恢复");
  });

  it("disables recognition when deleting the profile and restores prior runtime intent", async () => {
    const { manager, command, restore, disable } = harness();
    await manager.delete();
    expect(command.mock.calls.map((call) => call[0])).toEqual(["delete"]);
    expect(disable).toHaveBeenCalledOnce();
    expect(restore).toHaveBeenCalledOnce();
  });

  it("refuses enabling a missing or wrong-backend profile", async () => {
    const { manager } = harness(vi.fn(async () => ({ ...status, enrolled: false, detail: "音频后端不同，请重新录入" })));
    await expect(manager.validate(configuration)).rejects.toThrow("重新录入");
  });
});
