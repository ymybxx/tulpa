import { describe, expect, it } from "vitest";

import { loadCurrentModelSelection } from "./model-selection-load";

describe("loadCurrentModelSelection", () => {
  it("does not let a late A bundle roll the preview back after B is selected", async () => {
    const a = deferred<{ id: string }>();
    const b = deferred<{ id: string }>();
    const loaded: string[] = [];
    let epoch = 1;
    let selectedId = "A";

    const start = (id: string, requestEpoch: number, bundle: Promise<{ id: string }>) =>
      loadCurrentModelSelection({
        loadBundle: () => bundle,
        loadPreview: async (value) => {
          loaded.push(value.id);
          return value.id;
        },
        isCurrent: () => epoch === requestEpoch && selectedId === id,
      });

    const first = start("A", 1, a.promise);
    epoch = 2;
    selectedId = "B";
    const second = start("B", 2, b.promise);

    b.resolve({ id: "B" });
    await expect(second).resolves.toEqual({ current: true, result: "B" });
    a.resolve({ id: "A" });
    await expect(first).resolves.toEqual({ current: false });
    expect(loaded).toEqual(["B"]);
  });
});

function deferred<T>(): { promise: Promise<T>; resolve(value: T): void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
