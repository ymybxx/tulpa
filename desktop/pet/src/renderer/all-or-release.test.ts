import { describe, expect, it } from "vitest";

import { allOrRelease, allSettledOrThrow } from "./all-or-release";

describe("allOrRelease", () => {
  it("keeps all resources when every preparation succeeds", async () => {
    const released: string[] = [];
    await expect(allOrRelease(
      [Promise.resolve("atlas-0"), Promise.resolve("atlas-1")],
      (value) => released.push(value),
    )).resolves.toEqual(["atlas-0", "atlas-1"]);
    expect(released).toEqual([]);
  });

  it("releases every fulfilled resource when a sibling fails", async () => {
    const released: string[] = [];
    await expect(allOrRelease(
      [Promise.resolve("atlas-0"), Promise.reject(new Error("decode failed"))],
      (value) => released.push(value),
    )).rejects.toThrow("decode failed");
    expect(released).toEqual(["atlas-0"]);
  });
});

describe("allSettledOrThrow", () => {
  it("does not release a shared owner until every sibling has settled", async () => {
    let finishSibling!: () => void;
    const sibling = new Promise<void>((resolve) => { finishSibling = resolve; });
    const operation = allSettledOrThrow([
      Promise.reject(new Error("first failed")),
      sibling,
    ]);
    let rejected = false;
    void operation.catch(() => { rejected = true; });

    await Promise.resolve();
    expect(rejected).toBe(false);
    finishSibling();
    await expect(operation).rejects.toThrow("first failed");
  });
});
