import { EventEmitter } from "node:events";

import { describe, expect, it, vi } from "vitest";

import {
  UtilityProcessAtlasCompactionRunner,
  type AtlasCompactionUtilityProcess,
} from "./atlas-compaction-utility-runner";
import type { AtlasCompactionRequest } from "./atlas-compaction-coordinator";

vi.mock("electron", () => ({ utilityProcess: { fork: vi.fn(() => { throw new Error("Unexpected native fork in unit test"); }) } }));

describe("UtilityProcessAtlasCompactionRunner", () => {
  it("forks one short-lived worker, validates its response, and terminates it", async () => {
    const child = new FakeUtilityProcess();
    child.onPostMessage = (message) => {
      queueMicrotask(() => child.emit("message", {
        type: "atlas-compaction-result",
        requestId: message.requestId,
        modelId: "mao",
        revision: 4,
        contentKeys: [CONTENT_KEY],
        built: 1,
        reused: 0,
      }));
    };
    const fork = vi.fn(() => child);
    const runner = new UtilityProcessAtlasCompactionRunner("/app/atlas-compaction-worker.js", {
      fork,
      timeoutMs: 1_000,
    });

    const pending = runner.compact(request());
    child.emit("spawn");
    await pending;

    expect(fork).toHaveBeenCalledWith(
      "/app/atlas-compaction-worker.js",
      [],
      expect.objectContaining({ serviceName: "Live2D atlas compactor" }),
    );
    expect(child.messages).toHaveLength(1);
    expect(child.kill).toHaveBeenCalledOnce();
  });

  it("terminates and rejects a worker that returns a stale revision", async () => {
    const child = new FakeUtilityProcess();
    child.onPostMessage = (message) => queueMicrotask(() => child.emit("message", {
      type: "atlas-compaction-result",
      requestId: message.requestId,
      modelId: "mao",
      revision: 3,
      contentKeys: [CONTENT_KEY],
      built: 1,
      reused: 0,
    }));
    const runner = new UtilityProcessAtlasCompactionRunner("/worker.js", {
      fork: () => child,
      timeoutMs: 1_000,
    });

    const pending = runner.compact(request());
    child.emit("spawn");

    await expect(pending).rejects.toThrow("响应与请求不匹配");
    expect(child.kill).toHaveBeenCalledOnce();
  });

  it("terminates a worker that exceeds the compaction deadline", async () => {
    vi.useFakeTimers();
    try {
      const child = new FakeUtilityProcess();
      const runner = new UtilityProcessAtlasCompactionRunner("/worker.js", {
        fork: () => child,
        timeoutMs: 50,
      });
      const pending = runner.compact(request());
      child.emit("spawn");
      const rejection = expect(pending).rejects.toThrow("压实进程超时");

      await vi.advanceTimersByTimeAsync(51);

      await rejection;
      expect(child.kill).toHaveBeenCalledOnce();
    } finally {
      vi.useRealTimers();
    }
  });

  it("terminates and forwards a worker-side error", async () => {
    const child = new FakeUtilityProcess();
    child.onPostMessage = (message) => queueMicrotask(() => child.emit("message", {
      type: "atlas-compaction-error",
      requestId: message.requestId,
      modelId: "mao",
      revision: 4,
      message: "PNG 损坏",
    }));
    const runner = new UtilityProcessAtlasCompactionRunner("/worker.js", {
      fork: () => child,
      timeoutMs: 1_000,
    });

    const pending = runner.compact(request());
    child.emit("spawn");

    await expect(pending).rejects.toThrow("PNG 损坏");
    expect(child.kill).toHaveBeenCalledOnce();
  });
});

class FakeUtilityProcess extends EventEmitter implements AtlasCompactionUtilityProcess {
  readonly messages: unknown[] = [];
  readonly kill = vi.fn(() => true);
  onPostMessage?: (message: any) => void;

  postMessage(message: unknown): void {
    this.messages.push(message);
    this.onPostMessage?.(message);
  }
}

const CONTENT_KEY = "b".repeat(64);

function request(): AtlasCompactionRequest {
  return {
    modelId: "mao",
    revision: 4,
    allowedRoots: ["/models/mao"],
    atlases: [{
      contentKey: CONTENT_KEY,
      atlasFile: "textures/atlas.png",
      sourcePath: "/models/mao/textures/atlas.png",
      targetPath: `/models/mao/.appearance/compiled/${CONTENT_KEY}/textures/atlas.png`,
      edits: [{
        textureUnitId: "leg",
        atlasIndex: 0,
        bounds: { x: 0, y: 0, width: 1, height: 1 },
        maskPath: "/models/mao/masks/leg.png",
        imagePath: "/models/mao/materials/leg.png",
      }],
    }],
  };
}
