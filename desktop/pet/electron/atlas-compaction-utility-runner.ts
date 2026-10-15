import { randomUUID } from "node:crypto";

import { utilityProcess } from "electron";

import type {
  AtlasCompactionRequest,
  AtlasCompactionRunner,
} from "./atlas-compaction-coordinator";
import { assertAtlasCompactionRequest } from "./atlas-compaction-job";
import {
  isAtlasCompactionWorkerResponse,
  type AtlasCompactionWorkerRequest,
} from "./atlas-compaction-protocol";

export interface AtlasCompactionUtilityProcess {
  on(event: "spawn", listener: () => void): this;
  on(event: "message", listener: (message: unknown) => void): this;
  on(event: "exit", listener: (code: number) => void): this;
  off(event: "spawn", listener: () => void): this;
  off(event: "message", listener: (message: unknown) => void): this;
  off(event: "exit", listener: (code: number) => void): this;
  postMessage(message: unknown): void;
  kill(): boolean;
}

export type AtlasCompactionUtilityFork = (
  modulePath: string,
  args: string[],
  options: Electron.ForkOptions,
) => AtlasCompactionUtilityProcess;

export interface UtilityProcessAtlasCompactionRunnerOptions {
  fork?: AtlasCompactionUtilityFork;
  timeoutMs?: number;
}

/** Runs exactly one compaction request in one disposable Electron utility process. */
export class UtilityProcessAtlasCompactionRunner implements AtlasCompactionRunner {
  private readonly fork: AtlasCompactionUtilityFork;
  private readonly timeoutMs: number;

  constructor(
    private readonly workerPath: string,
    options: UtilityProcessAtlasCompactionRunnerOptions = {},
  ) {
    this.fork = options.fork ?? ((modulePath, args, forkOptions) =>
      utilityProcess.fork(modulePath, args, forkOptions));
    this.timeoutMs = options.timeoutMs ?? 120_000;
    if (!Number.isSafeInteger(this.timeoutMs) || this.timeoutMs <= 0) {
      throw new Error("图集压实超时时间无效");
    }
  }

  compact(request: AtlasCompactionRequest): Promise<void> {
    assertAtlasCompactionRequest(request);
    const requestId = randomUUID();
    const envelope: AtlasCompactionWorkerRequest = {
      type: "atlas-compaction-request",
      requestId,
      request,
    };
    return new Promise<void>((resolve, reject) => {
      let child: AtlasCompactionUtilityProcess;
      try {
        child = this.fork(this.workerPath, [], {
          serviceName: "Live2D atlas compactor",
          stdio: "ignore",
        });
      } catch (error) {
        reject(asError(error));
        return;
      }
      let finished = false;
      const timer = setTimeout(() => {
        finish(new Error(`图集压实进程超时（${this.timeoutMs}ms）`));
      }, this.timeoutMs);
      const onSpawn = () => {
        try {
          child.postMessage(envelope);
        } catch (error) {
          finish(asError(error));
        }
      };
      const onMessage = (message: unknown) => {
        if (!isAtlasCompactionWorkerResponse(message)) {
          finish(new Error("图集压实进程返回了无效响应"));
          return;
        }
        if (
          message.requestId !== requestId ||
          message.modelId !== request.modelId ||
          message.revision !== request.revision
        ) {
          finish(new Error("图集压实进程响应与请求不匹配"));
          return;
        }
        if (message.type === "atlas-compaction-error") {
          finish(new Error(message.message || "图集压实失败"));
          return;
        }
        const expectedKeys = request.atlases.map((atlas) => atlas.contentKey).sort();
        const actualKeys = [...message.contentKeys].sort();
        if (
          !sameStrings(expectedKeys, actualKeys) ||
          message.built < 0 ||
          message.reused < 0 ||
          message.built + message.reused !== request.atlases.length
        ) {
          finish(new Error("图集压实进程返回了错误的内容键"));
          return;
        }
        finish();
      };
      const onExit = (code: number) => {
        finish(new Error(`图集压实进程提前退出（code=${code}）`));
      };
      const finish = (error?: Error) => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        child.off("spawn", onSpawn);
        child.off("message", onMessage);
        child.off("exit", onExit);
        try {
          child.kill();
        } catch {
          // The process may already have exited; the request result remains valid.
        }
        if (error) reject(error);
        else resolve();
      };
      child.on("spawn", onSpawn);
      child.on("message", onMessage);
      child.on("exit", onExit);
    });
  }
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function asError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}
