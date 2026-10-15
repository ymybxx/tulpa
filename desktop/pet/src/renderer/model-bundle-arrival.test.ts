import { describe, expect, it } from "vitest";

import {
  ModelBundleArrivalGate,
  runCurrentModelBundleArrival,
} from "./model-bundle-arrival";

describe("ModelBundleArrivalGate", () => {
  it("rejects an initial snapshot that resolves after a pushed selection", async () => {
    const gate = new ModelBundleArrivalGate();
    const initial = deferred<string>();
    const loaded: string[] = [];
    const initialEpoch = gate.capture();

    const initialBundle = runCurrentModelBundleArrival(
      gate,
      initialEpoch,
      () => initial.promise,
    );

    const pushedEpoch = gate.advance();
    const pushedLoad = runCurrentModelBundleArrival(gate, pushedEpoch, async () => {
      loaded.push("B");
    });
    await expect(pushedLoad).resolves.toEqual({ current: true, value: undefined });
    initial.resolve("A");
    await expect(initialBundle).resolves.toEqual({ current: false });

    expect(loaded).toEqual(["B"]);
  });

  it("rejects completion side effects when an in-flight load is superseded", async () => {
    const gate = new ModelBundleArrivalGate();
    const load = deferred<void>();
    const first = gate.advance();
    const firstLoad = runCurrentModelBundleArrival(gate, first, () => load.promise);

    const second = gate.advance();
    expect(gate.isCurrent(first)).toBe(false);
    expect(gate.isCurrent(second)).toBe(true);
    load.resolve();
    await expect(firstLoad).resolves.toEqual({ current: false });
  });
});

function deferred<T>(): { promise: Promise<T>; resolve(value: T): void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
