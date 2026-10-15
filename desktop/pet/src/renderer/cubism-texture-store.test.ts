import { describe, expect, it } from "vitest";

import { commitTextureCandidate, commitTextureCandidates } from "./cubism-texture-store";

describe("commitTextureCandidate", () => {
  it("releases the previous resource only after the candidate validates", () => {
    const events: string[] = [];
    const result = commitTextureCandidate({
      previous: "previous",
      candidate: "candidate",
      isCurrent: () => true,
      bind: (value) => events.push(`bind:${value}`),
      validate: () => events.push("validate"),
      release: (value) => events.push(`release:${value}`),
    });

    expect(result).toBe("candidate");
    expect(events).toEqual([
      "bind:candidate",
      "validate",
      "release:previous",
    ]);
  });

  it("rolls back and releases the candidate when validation fails", () => {
    const events: string[] = [];

    expect(() => commitTextureCandidate({
      previous: "previous",
      candidate: "candidate",
      isCurrent: () => true,
      bind: (value) => events.push(`bind:${value}`),
      validate: () => { throw new Error("invalid candidate"); },
      release: (value) => events.push(`release:${value}`),
    })).toThrow("invalid candidate");
    expect(events).toEqual([
      "bind:candidate",
      "bind:previous",
      "release:candidate",
    ]);
  });

  it("releases a stale candidate without binding it", () => {
    const events: string[] = [];
    const result = commitTextureCandidate({
      previous: "previous",
      candidate: "candidate",
      isCurrent: () => false,
      bind: (value) => events.push(`bind:${value}`),
      validate: () => events.push("validate"),
      release: (value) => events.push(`release:${value}`),
    });

    expect(result).toBe("previous");
    expect(events).toEqual(["release:candidate"]);
  });
});

describe("commitTextureCandidates", () => {
  it("commits every atlas after one validation", () => {
    const events: string[] = [];
    const candidates = [
      { atlasIndex: 0, previous: "old-0", candidate: "new-0" },
      { atlasIndex: 1, previous: "old-1", candidate: "new-1" },
    ];
    expect(commitTextureCandidates({
      candidates,
      isCurrent: () => true,
      bind: (item, value) => events.push(`bind:${item.atlasIndex}:${value}`),
      validate: () => events.push("validate"),
      release: (value) => events.push(`release:${value}`),
    })).toBe(true);
    expect(events).toEqual([
      "bind:0:new-0",
      "bind:1:new-1",
      "validate",
      "release:old-0",
      "release:old-1",
    ]);
  });

  it("rolls every atlas back when snapshot validation fails", () => {
    const events: string[] = [];
    const candidates = [
      { atlasIndex: 0, previous: "old-0", candidate: "new-0" },
      { atlasIndex: 1, previous: "old-1", candidate: "new-1" },
    ];
    expect(() => commitTextureCandidates({
      candidates,
      isCurrent: () => true,
      bind: (item, value) => events.push(`bind:${item.atlasIndex}:${value}`),
      validate: () => { throw new Error("bad draw"); },
      release: (value) => events.push(`release:${value}`),
    })).toThrow("bad draw");
    expect(events).toEqual([
      "bind:0:new-0",
      "bind:1:new-1",
      "bind:0:old-0",
      "bind:1:old-1",
      "release:new-0",
      "release:new-1",
    ]);
  });
});
