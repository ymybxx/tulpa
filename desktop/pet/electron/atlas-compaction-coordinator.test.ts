import { describe, expect, it, vi } from "vitest";

import {
  AtlasCompactionCoordinator,
  type AtlasCompactionRequest,
} from "./atlas-compaction-coordinator";

describe("AtlasCompactionCoordinator", () => {
  it("schedules without awaiting a blocked compactor and coalesces to the newest revision", async () => {
    let releaseFirst!: () => void;
    let firstStarted!: () => void;
    const blocked = new Promise<void>((resolve) => { releaseFirst = resolve; });
    const started = new Promise<void>((resolve) => { firstStarted = resolve; });
    const revisions: number[] = [];
    const compact = vi.fn(async (request: AtlasCompactionRequest) => {
      revisions.push(request.revision);
      if (request.revision === 1) {
        firstStarted();
        await blocked;
      }
    });
    const coordinator = new AtlasCompactionCoordinator({ compact });

    coordinator.schedule(request(1));
    await started;
    coordinator.schedule(request(2));
    coordinator.schedule(request(3));
    releaseFirst();

    expect(await coordinator.flush(1_000)).toBe(true);
    expect(revisions).toEqual([1, 3]);
  });

  it("retries with the fixed backoff without rolling state back", async () => {
    const delays: number[] = [];
    const warnings: string[] = [];
    let attempts = 0;
    const coordinator = new AtlasCompactionCoordinator({
      compact: async () => {
        attempts += 1;
        if (attempts < 4) throw new Error(`failure-${attempts}`);
      },
    }, {
      delay: async (milliseconds) => { delays.push(milliseconds); },
      onWarning: (_request, error) => warnings.push(error.message),
    });

    coordinator.schedule(request(8));

    expect(await coordinator.flush(1_000)).toBe(true);
    expect(attempts).toBe(4);
    expect(delays).toEqual([250, 1_000, 3_000]);
    expect(warnings).toEqual(["failure-1", "failure-2", "failure-3"]);
  });

  it("returns false when the flush deadline expires", async () => {
    const coordinator = new AtlasCompactionCoordinator({
      compact: async () => new Promise<void>(() => undefined),
    });
    coordinator.schedule(request(1));

    expect(await coordinator.flush(5)).toBe(false);
  });

  it("unions affected atlases submitted for the same pending revision", async () => {
    let releaseFirst!: () => void;
    let firstStarted!: () => void;
    const blocked = new Promise<void>((resolve) => { releaseFirst = resolve; });
    const started = new Promise<void>((resolve) => { firstStarted = resolve; });
    const received: AtlasCompactionRequest[] = [];
    const coordinator = new AtlasCompactionCoordinator({
      compact: async (value) => {
        received.push(value);
        if (value.revision === 1) {
          firstStarted();
          await blocked;
        }
      },
    });
    coordinator.schedule(request(1));
    await started;
    coordinator.schedule(request(2, "textures/atlas-1.png"));
    coordinator.schedule(request(2, "textures/atlas-0.png"));
    releaseFirst();

    expect(await coordinator.flush(1_000)).toBe(true);
    expect(received[1]?.atlases.map((atlas) => atlas.atlasFile)).toEqual([
      "textures/atlas-0.png",
      "textures/atlas-1.png",
    ]);
  });

  it("cancels an obsolete retry delay when a newer revision arrives", async () => {
    let firstAttempt!: () => void;
    let delayStarted!: () => void;
    const attempted = new Promise<void>((resolve) => { firstAttempt = resolve; });
    const delaying = new Promise<void>((resolve) => { delayStarted = resolve; });
    const revisions: number[] = [];
    const coordinator = new AtlasCompactionCoordinator({
      compact: async (value) => {
        revisions.push(value.revision);
        if (value.revision === 1) {
          firstAttempt();
          throw new Error("obsolete");
        }
      },
    }, {
      delay: () => {
        delayStarted();
        return new Promise<void>(() => undefined);
      },
    });
    coordinator.schedule(request(1));
    await attempted;
    await delaying;
    coordinator.schedule(request(2));

    expect(await coordinator.flush(1_000)).toBe(true);
    expect(revisions).toEqual([1, 2]);
  });
});

function request(
  revision: number,
  atlasFile = "textures/atlas.png",
): AtlasCompactionRequest {
  return {
    modelId: "mao",
    revision,
    allowedRoots: ["/models/mao"],
    atlases: [{
      contentKey: `key-${revision}`,
      atlasFile,
      sourcePath: `/models/mao/${atlasFile}`,
      targetPath: `/models/mao/.appearance/compiled/key-${revision}/${atlasFile}`,
      edits: [],
    }],
  };
}
