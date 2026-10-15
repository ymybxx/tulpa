import { describe, expect, it } from "vitest";

import { sceneFor } from "./scene";

describe("sceneFor", () => {
  it("follows the local hour", () => {
    expect(sceneFor(9, false)).toBe("day");
    expect(sceneFor(17, false)).toBe("dusk");
    expect(sceneFor(19, false)).toBe("night");
    expect(sceneFor(5, false)).toBe("night");
    expect(sceneFor(6, false)).toBe("day");
  });

  it("uses night whenever the system is in dark mode", () => {
    expect(sceneFor(12, true)).toBe("night");
  });
});
