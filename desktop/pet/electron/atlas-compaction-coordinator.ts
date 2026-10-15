import type { TextureAtlasEdit } from "./texture-atlas-composer";

export interface AtlasCompactionRequest {
  modelId: string;
  revision: number;
  allowedRoots: string[];
  atlases: Array<{
    contentKey: string;
    atlasFile: string;
    sourcePath: string;
    targetPath: string;
    edits: TextureAtlasEdit[];
  }>;
}

export interface AtlasCompactionRunner {
  compact(request: AtlasCompactionRequest): Promise<void>;
}

export interface AtlasCompactionCoordinatorOptions {
  delay?: (milliseconds: number) => Promise<void>;
  onWarning?: (request: AtlasCompactionRequest, error: Error) => void;
}

const RETRY_DELAYS_MS = [250, 1_000, 3_000] as const;

export class AtlasCompactionCoordinator {
  private readonly pending = new Map<string, AtlasCompactionRequest>();
  private readonly retryWakeups = new Map<string, {
    revision: number;
    wake: () => void;
  }>();
  private draining = false;
  private readonly idleWaiters: Array<() => void> = [];
  private readonly delay: (milliseconds: number) => Promise<void>;
  private readonly onWarning: (request: AtlasCompactionRequest, error: Error) => void;

  constructor(
    private readonly runner: AtlasCompactionRunner,
    options: AtlasCompactionCoordinatorOptions = {},
  ) {
    this.delay = options.delay ?? ((milliseconds) => new Promise((resolve) => {
      setTimeout(resolve, milliseconds);
    }));
    this.onWarning = options.onWarning ?? (() => undefined);
  }

  schedule(request: AtlasCompactionRequest): void {
    const existing = this.pending.get(request.modelId);
    if (!existing || request.revision > existing.revision) {
      this.pending.set(request.modelId, request);
    } else if (request.revision === existing.revision) {
      this.pending.set(request.modelId, mergeRequests(existing, request));
    }
    const retry = this.retryWakeups.get(request.modelId);
    if (retry && request.revision > retry.revision) retry.wake();
    if (!this.draining) void this.drain();
  }

  async ensure(request: AtlasCompactionRequest): Promise<void> {
    this.schedule(request);
    while (this.draining || this.pending.size > 0) await this.waitUntilIdle();
  }

  async flush(timeoutMs: number): Promise<boolean> {
    if (!this.draining && this.pending.size === 0) return true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<false>((resolve) => {
      timer = setTimeout(() => resolve(false), Math.max(0, timeoutMs));
    });
    const idle = this.waitUntilIdle().then(() => true as const);
    const result = await Promise.race([idle, timeout]);
    if (timer) clearTimeout(timer);
    return result;
  }

  private async drain(): Promise<void> {
    this.draining = true;
    try {
      while (this.pending.size > 0) {
        const entry = this.pending.entries().next().value as
          | [string, AtlasCompactionRequest]
          | undefined;
        if (!entry) break;
        const [modelId, request] = entry;
        this.pending.delete(modelId);
        await this.compactWithRetry(request);
      }
    } finally {
      this.draining = false;
      for (const resolve of this.idleWaiters.splice(0)) resolve();
      if (this.pending.size > 0) void this.drain();
    }
  }

  private async compactWithRetry(request: AtlasCompactionRequest): Promise<void> {
    for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
      try {
        await this.runner.compact(request);
        return;
      } catch (error) {
        this.onWarning(request, asError(error));
        const replacement = this.pending.get(request.modelId);
        if (replacement && replacement.revision > request.revision) return;
        const delay = RETRY_DELAYS_MS[attempt];
        if (delay === undefined) return;
        if (await this.waitForRetryOrReplacement(request, delay)) return;
      }
    }
  }

  private async waitForRetryOrReplacement(
    request: AtlasCompactionRequest,
    milliseconds: number,
  ): Promise<boolean> {
    let wake!: () => void;
    const replaced = new Promise<true>((resolve) => {
      wake = () => resolve(true);
    });
    const token = { revision: request.revision, wake };
    this.retryWakeups.set(request.modelId, token);
    try {
      return await Promise.race([
        this.delay(milliseconds).then(() => false as const),
        replaced,
      ]);
    } finally {
      if (this.retryWakeups.get(request.modelId) === token) {
        this.retryWakeups.delete(request.modelId);
      }
    }
  }

  private waitUntilIdle(): Promise<void> {
    if (!this.draining && this.pending.size === 0) return Promise.resolve();
    return new Promise((resolve) => this.idleWaiters.push(resolve));
  }
}

function mergeRequests(
  left: AtlasCompactionRequest,
  right: AtlasCompactionRequest,
): AtlasCompactionRequest {
  const atlases = new Map(left.atlases.map((atlas) => [atlas.atlasFile, atlas]));
  for (const atlas of right.atlases) atlases.set(atlas.atlasFile, atlas);
  return {
    ...right,
    allowedRoots: [...new Set([...left.allowedRoots, ...right.allowedRoots])],
    atlases: [...atlases.values()].sort((a, b) => a.atlasFile.localeCompare(b.atlasFile)),
  };
}

function asError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}
