import { describe, expect, it } from "vitest";

import { runModelLoadTransaction } from "./model-load-transaction";

describe("runModelLoadTransaction", () => {
  it("validates the fully prepared candidate before committing it", async () => {
    const events: string[] = [];
    const result = await runModelLoadTransaction({
      prepare: async () => { events.push("prepare-with-appearance"); return "candidate"; },
      validate: () => { events.push("validate-first-draw"); },
      commit: () => { events.push("swap-visible-model"); },
      release: () => events.push("release-candidate"),
      isCurrent: () => true,
    });

    expect(result).toBe("candidate");
    expect(events).toEqual([
      "prepare-with-appearance",
      "validate-first-draw",
      "swap-visible-model",
    ]);
  });

  it("releases candidates that become stale or fail validation", async () => {
    const staleEvents: string[] = [];
    expect(await runModelLoadTransaction({
      prepare: async () => "candidate",
      validate: () => { staleEvents.push("validate"); },
      commit: () => { staleEvents.push("commit"); },
      release: () => staleEvents.push("release"),
      isCurrent: () => false,
    })).toBeNull();
    expect(staleEvents).toEqual(["release"]);

    const failedEvents: string[] = [];
    await expect(runModelLoadTransaction({
      prepare: async () => "candidate",
      validate: () => { throw new Error("bad draw"); },
      commit: () => { failedEvents.push("commit"); },
      release: () => failedEvents.push("release"),
      isCurrent: () => true,
    })).rejects.toThrow("bad draw");
    expect(failedEvents).toEqual(["release"]);
  });
});
