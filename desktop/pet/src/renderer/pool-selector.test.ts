import { describe, expect, it } from "vitest";

import { PoolSelector } from "./pool-selector";

describe("PoolSelector", () => {
  it("excludes the previous item when alternatives exist", () => {
    const selector = new PoolSelector(() => 0);

    expect(selector.pick("happy", ["a", "b"])).toBe("a");
    expect(selector.pick("happy", ["a", "b"])).toBe("b");
  });

  it("repeats the only available item and ignores missing ids", () => {
    const selector = new PoolSelector(() => 0.9);

    expect(selector.pick("wave", ["only"])).toBe("only");
    expect(selector.pick("wave", ["only"])).toBe("only");
    expect(selector.pick("empty", [])).toBeNull();
  });

  it("clears one history key without disturbing the others", () => {
    const selector = new PoolSelector(() => 0);
    selector.pick("happy", ["a", "b"]);
    selector.pick("wave", ["x", "y"]);

    selector.clear("happy");

    expect(selector.pick("happy", ["a", "b"])).toBe("a");
    expect(selector.pick("wave", ["x", "y"])).toBe("y");
  });
});
