import { describe, expect, it } from "vitest";

import {
  TargetFrameScheduler,
  type TargetFrameSchedulerClock,
} from "./target-frame-scheduler";

class FakeClock implements TargetFrameSchedulerClock {
  nowMs = 0;
  nextHandle = 1;
  readonly timers = new Map<number, { callback: () => void; delayMs: number }>();
  readonly frames = new Map<number, (timestamp: number) => void>();

  now(): number {
    return this.nowMs;
  }

  requestAnimationFrame(callback: (timestamp: number) => void): number {
    const handle = this.nextHandle++;
    this.frames.set(handle, callback);
    return handle;
  }

  cancelAnimationFrame(handle: number): void {
    this.frames.delete(handle);
  }

  setTimeout(callback: () => void, delayMs: number): number {
    const handle = this.nextHandle++;
    this.timers.set(handle, { callback, delayMs });
    return handle;
  }

  clearTimeout(handle: number): void {
    this.timers.delete(handle);
  }

  fireFrame(timestamp: number): void {
    const entry = this.frames.entries().next().value as
      | [number, (timestamp: number) => void]
      | undefined;
    if (!entry) throw new Error("No animation frame is pending");
    this.frames.delete(entry[0]);
    this.nowMs = timestamp;
    entry[1](timestamp);
  }

  fireTimer(): void {
    const entry = this.timers.entries().next().value as
      | [number, { callback: () => void; delayMs: number }]
      | undefined;
    if (!entry) throw new Error("No timer is pending");
    this.timers.delete(entry[0]);
    this.nowMs += entry[1].delayMs;
    entry[1].callback();
  }

  onlyTimerDelay(): number {
    expect(this.timers.size).toBe(1);
    return [...this.timers.values()][0]!.delayMs;
  }
}

describe("TargetFrameScheduler", () => {
  it("sleeps between 30 fps frames instead of polling every 120 Hz display frame", () => {
    const clock = new FakeClock();
    const drawnAt: number[] = [];
    const scheduler = new TargetFrameScheduler(30, (timestamp) => drawnAt.push(timestamp), clock);

    scheduler.resume();
    expect(clock.frames.size).toBe(1);
    expect(clock.timers.size).toBe(0);

    clock.fireFrame(0);
    expect(drawnAt).toEqual([0]);
    expect(clock.frames.size).toBe(0);
    expect(clock.onlyTimerDelay()).toBeCloseTo(1000 / 30 - 14, 5);

    clock.fireTimer();
    expect(clock.frames.size).toBe(1);
    clock.fireFrame(1000 / 30);
    expect(drawnAt).toEqual([0, 1000 / 30]);
    expect(clock.frames.size).toBe(0);
    expect(clock.timers.size).toBe(1);
  });

  it("runs speaking animation at a 60 fps deadline", () => {
    const clock = new FakeClock();
    const drawnAt: number[] = [];
    const scheduler = new TargetFrameScheduler(60, (timestamp) => drawnAt.push(timestamp), clock);

    scheduler.resume();
    clock.fireFrame(0);
    expect(clock.onlyTimerDelay()).toBeCloseTo(1000 / 60 - 14, 5);
    clock.fireTimer();
    clock.fireFrame(1000 / 60);

    expect(drawnAt).toEqual([0, 1000 / 60]);
  });

  it("reschedules immediately when switching between 30 and 60 fps", () => {
    const clock = new FakeClock();
    const drawnAt: number[] = [];
    const scheduler = new TargetFrameScheduler(30, (timestamp) => drawnAt.push(timestamp), clock);

    scheduler.resume();
    clock.fireFrame(0);
    const oldTimerHandle = [...clock.timers.keys()][0]!;
    clock.nowMs = 10;
    scheduler.setFrameRate(60);

    expect(clock.timers.has(oldTimerHandle)).toBe(false);
    expect(clock.onlyTimerDelay()).toBeCloseTo(0, 5);
    clock.fireTimer();
    clock.fireFrame(1000 / 60);
    expect(drawnAt).toEqual([0, 1000 / 60]);

    clock.nowMs = 20;
    scheduler.setFrameRate(30);
    expect(clock.frames.size).toBe(0);
    expect(clock.onlyTimerDelay()).toBeCloseTo(1000 / 30 - (20 - 1000 / 60) - 14, 5);
  });

  it("pauses at zero fps and resumes once without duplicate work", () => {
    const clock = new FakeClock();
    const drawnAt: number[] = [];
    const scheduler = new TargetFrameScheduler(30, (timestamp) => drawnAt.push(timestamp), clock);

    scheduler.resume();
    scheduler.resume();
    expect(clock.frames.size).toBe(1);
    scheduler.pause();
    expect(clock.frames.size).toBe(0);
    expect(clock.timers.size).toBe(0);

    scheduler.resume();
    scheduler.resume();
    expect(clock.frames.size).toBe(1);
    clock.fireFrame(80);
    expect(drawnAt).toEqual([80]);
    expect(clock.timers.size).toBe(1);

    scheduler.destroy();
    expect(clock.frames.size).toBe(0);
    expect(clock.timers.size).toBe(0);
  });

  it("does not draw an early rAF and still keeps only one pending callback", () => {
    const clock = new FakeClock();
    const drawnAt: number[] = [];
    const scheduler = new TargetFrameScheduler(30, (timestamp) => drawnAt.push(timestamp), clock);

    scheduler.resume();
    clock.fireFrame(0);
    clock.fireTimer();
    // Well before due it sleeps again; near due it waits on the display.
    clock.fireFrame(10);
    expect(drawnAt).toEqual([0]);
    expect(clock.frames.size).toBe(0);
    expect(clock.timers.size).toBe(1);
    clock.fireTimer();
    clock.fireFrame(25);
    expect(drawnAt).toEqual([0]);
    expect(clock.frames.size).toBe(1);
    expect(clock.timers.size).toBe(0);
  });

  it("keeps an even 30 fps on a 120 Hz display even when the timer wakes late", () => {
    const clock = new FakeClock();
    const drawnAt: number[] = [];
    const scheduler = new TargetFrameScheduler(30, (timestamp) => drawnAt.push(timestamp), clock);
    const refresh = 1000 / 120;
    scheduler.resume();
    clock.fireFrame(0);
    for (let frame = 0; frame < 60; frame += 1) {
      // The timer wakes up to 10 ms late; rAF then fires on the next display refresh.
      const timer = [...clock.timers.values()][0];
      if (timer) { clock.timers.clear(); clock.nowMs += timer.delayMs + (frame % 3) * 5; timer.callback(); }
      while (drawnAt.length < frame + 2) clock.fireFrame(Math.ceil((clock.nowMs + 0.01) / refresh) * refresh);
    }
    const gaps = drawnAt.slice(1).map((at, index) => at - drawnAt[index]!);
    for (const gap of gaps) expect(gap).toBeCloseTo(1000 / 30, 0);
  });
});
