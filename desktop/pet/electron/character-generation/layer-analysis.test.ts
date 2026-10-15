import { describe, expect, it } from "vitest";

import { validateRequiredSlots } from "./layer-analysis";

describe("validateRequiredSlots", () => {
  it("reports only required slots without an assigned texture", () => {
    expect(validateRequiredSlots([
      { id: "body", name: "Body", required: true },
      { id: "hat", name: "Hat", required: false },
    ], { hat: "/hat.png" })).toEqual(["body"]);
  });
});
