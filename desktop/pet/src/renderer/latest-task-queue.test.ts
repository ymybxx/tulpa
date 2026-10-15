import { describe, expect, it } from "vitest";

import { LatestTaskQueue } from "./latest-task-queue";

describe("LatestTaskQueue", () => {
  it("skips work replaced before it starts", async () => {
    const queue = new LatestTaskQueue<string>();
    const values: string[] = [];

    const first = queue.enqueue("first", async (value) => {
      values.push(value);
    });
    const second = queue.enqueue("second", async (value) => {
      values.push(value);
    });

    await Promise.all([first, second]);
    expect(values).toEqual(["second"]);
  });

  it("never overlaps a stale task with its replacement", async () => {
    const queue = new LatestTaskQueue<string>();
    const events: string[] = [];
    let active = 0;
    let maxActive = 0;
    let releaseFirst!: () => void;
    let markStarted!: () => void;
    const firstStarted = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const firstRelease = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });

    const first = queue.enqueue("first", async (value, isCurrent) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      events.push(`${value}:start`);
      markStarted();
      await firstRelease;
      events.push(`${value}:${isCurrent() ? "current" : "stale"}`);
      active -= 1;
    });
    await firstStarted;
    const second = queue.enqueue("second", async (value, isCurrent) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      events.push(`${value}:start`);
      events.push(`${value}:${isCurrent() ? "current" : "stale"}`);
      active -= 1;
    });
    releaseFirst();

    await Promise.all([first, second]);
    expect(maxActive).toBe(1);
    expect(events).toEqual([
      "first:start",
      "first:stale",
      "second:start",
      "second:current",
    ]);
  });

  it("exposes a settlement barrier for deferred resource cleanup", async () => {
    const queue = new LatestTaskQueue<string>();
    let release!: () => void;
    const blocked = new Promise<void>((resolve) => { release = resolve; });
    let settled = false;
    void queue.enqueue("model", async () => blocked);
    void queue.whenIdle().then(() => { settled = true; });
    await Promise.resolve();
    expect(settled).toBe(false);
    release();
    await queue.whenIdle();
    expect(settled).toBe(true);
  });
});
