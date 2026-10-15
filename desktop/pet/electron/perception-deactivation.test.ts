import { describe, expect, it, vi } from "vitest";

import { deactivatePerception } from "./perception-deactivation";

describe("deactivatePerception", () => {
  it("starts native shutdown first and waits for all cleanup operations in parallel", async () => {
    const calls: string[] = [];
    const perception = deferred();
    const computerHistory = deferred();
    const environmentFrames = deferred();
    const stopPerception = vi.fn((detail: string) => {
      calls.push(`perception:${detail}`);
      return perception.promise;
    });

    const deactivation = deactivatePerception({
      stopPerception,
      disableComputerHistory: () => {
        calls.push("computer-history");
        return computerHistory.promise;
      },
      disableEnvironmentFrames: () => {
        calls.push("environment-frames");
        return environmentFrames.promise;
      },
    }, "感知服务未启用");

    expect(calls).toEqual([
      "perception:感知服务未启用",
      "computer-history",
      "environment-frames",
    ]);
    expect(stopPerception).toHaveBeenCalledOnce();

    computerHistory.resolve();
    environmentFrames.resolve();
    await expect(pending(deactivation)).resolves.toBe(true);

    perception.resolve();
    await expect(deactivation).resolves.toBeUndefined();
  });

  it("starts remaining cleanup when an operation throws synchronously", async () => {
    const calls: string[] = [];
    const error = new Error("stop failed");

    const deactivation = deactivatePerception({
      stopPerception: () => {
        calls.push("perception");
        throw error;
      },
      disableComputerHistory: () => {
        calls.push("computer-history");
      },
      disableEnvironmentFrames: () => {
        calls.push("environment-frames");
      },
    }, "disabled");

    expect(calls).toEqual(["perception", "computer-history", "environment-frames"]);
    await expect(deactivation).rejects.toBe(error);
  });
});

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function pending(promise: Promise<void>): Promise<boolean> {
  const sentinel = Symbol("pending");
  return Promise.race([promise, Promise.resolve(sentinel)]).then(
    (value) => value === sentinel,
  );
}
