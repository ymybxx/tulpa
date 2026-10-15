import { describe, expect, it, vi } from "vitest";

import type { BehaviorCandidate } from "../shared/contracts";
import {
  Live2DBehaviorPlayback,
  type BehaviorRuntimeStartOptions,
  type Live2DBehaviorRuntime,
} from "./live2d-behavior-playback";

const idle: BehaviorCandidate = {
  id: "idle",
  motionAssetIds: ["idle-motion"],
  expressionAssetIds: ["neutral-expression"],
};
const happy: BehaviorCandidate = {
  id: "happy",
  motionAssetIds: ["happy-a", "happy-b"],
  expressionAssetIds: ["smile-a", "smile-b"],
};
const alternateIdle: BehaviorCandidate = {
  id: "idle-alternate",
  motionAssetIds: ["idle-motion-alternate"],
  expressionAssetIds: [],
};

class Runtime implements Live2DBehaviorRuntime {
  starts: Array<{ candidate: BehaviorCandidate; options: BehaviorRuntimeStartOptions }> = [];
  stops = 0;
  fail = false;
  persistentStarts: string[][] = [];
  persistentStops = 0;
  poseBegins = 0;
  poseRestores = 0;
  readonly lifecycle: string[] = [];
  private readonly faceOnly = new Set(["smile-a", "smile-b"]);
  private poseCaptured = false;

  startBehavior(candidate: BehaviorCandidate, options: BehaviorRuntimeStartOptions): void {
    if (this.fail) throw new Error("load failed");
    this.starts.push({ candidate, options });
    this.lifecycle.push(`start:${candidate.id}`);
  }

  stopBehavior(): void {
    this.stops += 1;
    this.lifecycle.push("stop");
  }

  beginTransientPose(): void {
    if (this.poseCaptured) return;
    this.poseCaptured = true;
    this.poseBegins += 1;
    this.lifecycle.push("pose:begin");
  }

  restoreTransientPose(): void {
    if (!this.poseCaptured) return;
    this.poseCaptured = false;
    this.poseRestores += 1;
    this.lifecycle.push("pose:restore");
  }

  isFaceOnlyExpressionAsset(id: string): boolean {
    return this.faceOnly.has(id);
  }

  startPersistentExpressions(ids: readonly string[]): void {
    this.persistentStarts.push([...ids]);
  }

  stopPersistentExpressions(): void {
    this.persistentStops += 1;
  }

  finishLast(): void {
    this.starts.at(-1)?.options.onFinished?.();
  }
}

/** Models virtual part selectors, which ordinary stopBehavior does not reset. */
class PoseRuntime extends Runtime {
  pose = "authored";
  private statePose: string | null = null;
  private transientPose: string | null = null;

  override startBehavior(candidate: BehaviorCandidate, options: BehaviorRuntimeStartOptions): void {
    super.startBehavior(candidate, options);
    if (candidate.motionAssetIds.length) this.pose = candidate.motionAssetIds[0]!;
  }

  beginStatePose(): void {
    this.statePose ??= this.pose;
  }

  restoreStatePose(): void {
    if (this.statePose === null) return;
    this.pose = this.statePose;
    this.statePose = null;
  }

  override beginTransientPose(): void {
    super.beginTransientPose();
    this.transientPose ??= this.pose;
  }

  override restoreTransientPose(): void {
    super.restoreTransientPose();
    if (this.transientPose === null) return;
    this.pose = this.transientPose;
    this.transientPose = null;
  }
}

describe("state pose lifecycle", () => {
  const face: BehaviorCandidate = { id: "face", motionAssetIds: [], expressionAssetIds: ["smile-a"] };

  it.each([null, face])("clears the previous motion pose when the next state has no motion: %j", (next) => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    expect(runtime.pose).toBe("idle-motion");
    playback.setState(next);
    expect(runtime.pose).toBe("authored");
  });

  it("restores the baseline between rotating state candidates", () => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle, () => ({ candidate: face, rotate: true }));
    runtime.finishLast();
    expect(runtime.starts.at(-1)?.candidate).toEqual(face);
    expect(runtime.pose).toBe("authored");
    playback.setState(alternateIdle);
    playback.setState(null);
    expect(runtime.pose).toBe("authored");
  });

  it("restores the configured state after a transient and its baseline when leaving the state", () => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    playback.playTransient(happy, "happy");
    expect(runtime.pose).toBe("happy-a");
    runtime.finishLast();
    expect(runtime.pose).toBe("idle-motion");
    playback.setSpeaking(true);
    expect(runtime.pose).toBe("authored");
  });

  it.each([null, face])("uses the new state after a deferred transition: %j", (next) => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    playback.playTransient(happy, "happy");
    playback.setState(next);
    expect(runtime.pose).toBe("happy-a");
    runtime.finishLast();
    expect(runtime.pose).toBe("authored");
  });

  it("lets the transient finish after speech begins, then restores the default body", () => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    playback.playTransient(happy, "happy");
    playback.setSpeaking(true);
    expect(runtime.pose).toBe("happy-a");
    runtime.finishLast();
    expect(runtime.pose).toBe("authored");
    expect(runtime.persistentStarts).toEqual([["smile-a", "smile-b"]]);
    playback.setSpeaking(false);
    expect(runtime.pose).toBe("idle-motion");
  });

  it("does not revive a stale state pose after a face-only emotion replaces a body transient", () => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    playback.playTransient(happy, "happy");
    playback.setSpeaking(true);
    playback.playTransient(face, "face");
    expect(runtime.pose).toBe("authored");
    expect(runtime.persistentStarts.at(-1)).toEqual(["smile-a"]);
    playback.setState(null);
    playback.setSpeaking(false);
    expect(runtime.pose).toBe("authored");
  });

  it("clears state and transient pose snapshots on cancellation", () => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    playback.playTransient(happy, "happy");
    playback.cancel();
    runtime.finishLast();
    expect(runtime.pose).toBe("authored");
    playback.setState(alternateIdle);
    playback.cancel();
    expect(runtime.pose).toBe("authored");
  });

  it.each([false, true])("restores the baseline on startup failure during speech=%s", (speaking) => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime, vi.fn());
    playback.setState(idle);
    playback.playTransient(happy, "happy");
    playback.setSpeaking(speaking);
    runtime.fail = true;
    expect(playback.playTransient(happy, "broken")).toBe(false);
    expect(runtime.pose).toBe("authored");
    runtime.fail = false;
    playback.setState(null);
    playback.setSpeaking(false);
    expect(runtime.pose).toBe("authored");
  });
});

describe("wake acknowledgement playback", () => {
  const wake: BehaviorCandidate = { id: "wake", motionAssetIds: ["wave"], expressionAssetIds: [] };
  const listening: BehaviorCandidate = { id: "listening", motionAssetIds: [], expressionAssetIds: ["smile-a"] };

  it("finishes one gesture while listening takes effect and voice emotion is cleared", () => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    playback.playAcknowledgement(wake);
    const count = runtime.starts.length;
    expect(runtime.starts.at(-1)?.options.loopMotions).toBe(false);

    playback.cancelEmotion();
    playback.setState(listening);
    playback.setSpeaking(false);
    playback.cancelEmotion();
    expect(runtime.starts).toHaveLength(count);
    expect(playback.hasTransient()).toBe(true);

    runtime.finishLast();
    expect(runtime.starts.at(-1)?.candidate).toEqual(listening);
    expect(runtime.pose).toBe("authored");
    expect(playback.hasTransient()).toBe(false);
    expect(runtime.starts.filter(({ candidate }) => candidate.id === "wake")).toHaveLength(1);
  });

  it("does not interrupt speech and restores the latest emotion after the gesture", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.playAcknowledgement(wake);
    const count = runtime.starts.length;
    playback.setState(listening);
    playback.setSpeaking(true);
    playback.playTransient(happy, "happy");
    expect(runtime.starts).toHaveLength(count);
    runtime.finishLast();
    expect(runtime.starts.at(-1)?.candidate.motionAssetIds).toEqual(happy.motionAssetIds);
    expect(runtime.persistentStarts.at(-1)).toEqual(happy.expressionAssetIds);
  });

  it("can cancel acknowledgement explicitly without reviving it on late completion", () => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    playback.playAcknowledgement(wake);
    const complete = runtime.starts.at(-1)!.options.onFinished!;
    playback.setState(null);
    playback.cancelAcknowledgement();
    complete();
    expect(playback.hasTransient()).toBe(false);
    expect(runtime.pose).toBe("authored");
  });

  it("does not cancel a click that replaced the acknowledgement", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.playAcknowledgement(wake);
    const complete = runtime.starts.at(-1)!.options.onFinished!;
    playback.playClick(happy);
    const count = runtime.starts.length;
    playback.cancelAcknowledgement();
    complete();
    expect(runtime.starts).toHaveLength(count);
    expect(playback.hasTransient()).toBe(true);
  });
});

describe("independent click playback", () => {
  const click: BehaviorCandidate = { id: "click", motionAssetIds: ["tap-motion"], expressionAssetIds: ["smile-b"] };
  const face: BehaviorCandidate = { id: "click-face", motionAssetIds: [], expressionAssetIds: ["smile-b"] };
  const emotion: BehaviorCandidate = { id: "emotion", motionAssetIds: ["emotion-motion"], expressionAssetIds: ["smile-a"] };

  it.each([false, true])("plays a bounded click and recovers without a state candidate, speaking=%s", (speaking) => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setSpeaking(speaking);
    expect(playback.playClick(click)).toBe(true);
    expect(runtime.starts.at(-1)).toMatchObject({ candidate: click, options: { loopMotions: false } });
    expect(runtime.pose).toBe("tap-motion");
    runtime.finishLast();
    expect(runtime.pose).toBe("authored");
    expect(playback.hasTransient()).toBe(false);
    expect(runtime.persistentStarts).toEqual([]);
  });

  it("gives a pure click expression two seconds instead of promoting it for the whole speech", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setSpeaking(true);
    playback.playClick(face);
    expect(runtime.starts.at(-1)).toMatchObject({ candidate: face, options: { expressionOnlyDurationMs: 2_000 } });
    expect(playback.hasTransient()).toBe(true);
    runtime.finishLast();
    expect(playback.hasTransient()).toBe(false);
    expect(runtime.persistentStarts).toEqual([]);
  });

  it("finishes a click after speech begins without retaining its expression", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.playClick(click);
    playback.setSpeaking(true);
    runtime.finishLast();
    expect(runtime.persistentStarts).toEqual([]);
    expect(playback.hasSpeakingEmotion()).toBe(false);
  });

  it("restores the latest state after state changes and voice cancellation during a click", () => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    playback.playClick(click);
    const finishClick = runtime.starts.at(-1)!.options.onFinished!;
    playback.cancelEmotion();
    playback.setState(alternateIdle);
    playback.setSpeaking(true);
    playback.setSpeaking(false);
    expect(runtime.pose).toBe("tap-motion");
    expect(playback.hasTransient()).toBe(true);
    finishClick();
    expect(runtime.pose).toBe("idle-motion-alternate");
  });

  it("restores a speaking face after a click without replaying the old emotion motion", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setSpeaking(true);
    playback.playTransient(emotion, "emotion");
    runtime.finishLast();
    playback.playClick(click);
    expect(runtime.persistentStops).toBe(1);
    runtime.finishLast();
    expect(runtime.persistentStarts).toEqual([["smile-a"], ["smile-a"]]);
    expect(runtime.starts.map((value) => value.candidate.id)).toEqual(["emotion", "click"]);
  });

  it("invalidates the interrupted emotion callback when a click takes over", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.playTransient(emotion, "emotion");
    const finishEmotion = runtime.starts.at(-1)!.options.onFinished!;
    playback.playClick(click);
    finishEmotion();
    expect(playback.hasTransient()).toBe(true);
    expect(runtime.starts.at(-1)!.candidate.id).toBe("click");
    runtime.finishLast();
    expect(playback.hasTransient()).toBe(false);
  });

  it("retains only the newest deferred emotion across repeated clicks", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.playClick(click);
    const finishOldClick = runtime.starts.at(-1)!.options.onFinished!;
    playback.playTransient(happy, "old-emotion");
    playback.playClick(face);
    const newest = structuredClone(emotion);
    playback.playTransient(newest, "new-emotion");
    newest.motionAssetIds.push("mutated-after-submission");
    finishOldClick();
    expect(runtime.starts.map((value) => value.candidate.id)).toEqual(["click", "click-face"]);
    runtime.finishLast();
    expect(runtime.starts.map((value) => value.candidate.id)).toEqual(["click", "click-face", "emotion"]);
    expect(runtime.starts.at(-1)!.candidate.motionAssetIds).toEqual(["emotion-motion"]);
  });

  it("drops deferred and prior speech emotions when their voice generation is cancelled", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setSpeaking(true);
    playback.playTransient(emotion, "emotion");
    playback.playClick(click);
    playback.playTransient(happy, "deferred");
    playback.cancelEmotion();
    expect(playback.hasTransient()).toBe(true);
    const startsBeforeFinish = runtime.starts.length;
    runtime.finishLast();
    expect(runtime.starts).toHaveLength(startsBeforeFinish);
    expect(runtime.persistentStarts).toEqual([["smile-a"]]);
    expect(playback.hasSpeakingEmotion()).toBe(false);
  });

  it.each(["reset", "cancel", "destroy"] as const)("clears clicks and deferred emotions on explicit %s", (method) => {
    const runtime = new PoseRuntime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.playClick(click);
    const finishClick = runtime.starts.at(-1)!.options.onFinished!;
    playback.playTransient(emotion, "deferred");
    playback[method]();
    finishClick();
    expect(runtime.pose).toBe("authored");
    expect(playback.hasTransient()).toBe(false);
    expect(runtime.starts).toHaveLength(1);
  });

  it("recovers the speaking face if a replacement click cannot start", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime, vi.fn());
    playback.setSpeaking(true);
    playback.playTransient(emotion, "emotion");
    playback.playClick(click);
    const staleFinish = runtime.starts.at(-1)!.options.onFinished!;
    runtime.fail = true;
    expect(playback.playClick(face)).toBe(false);
    staleFinish();
    expect(playback.hasTransient()).toBe(false);
    expect(runtime.persistentStarts.at(-1)).toEqual(["smile-a"]);
  });
});

describe("Live2DBehaviorPlayback", () => {
  it("reports state preview startup failures instead of silently confirming them", () => {
    const runtime = new Runtime(); const warn = vi.fn();
    const playback = new Live2DBehaviorPlayback(() => runtime, warn);
    runtime.fail = true;
    expect(playback.setState(idle)).toBe(false);
    expect(warn).toHaveBeenCalledOnce();
    runtime.fail = false;
    expect(playback.setState(idle)).toBe(true);
    expect(runtime.starts.at(-1)?.options.loopMotions).toBe(true);
  });
  it("restores the exact state candidate after a transient group", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    const selectNext = vi.fn(() => ({ candidate: alternateIdle, rotate: true }));
    playback.setState(idle, selectNext);
    playback.playTransient(happy, "happy");

    expect(runtime.starts.map((value) => value.candidate.id)).toEqual(["idle", "happy"]);
    expect(runtime.starts[0]!.options.loopMotions).toBe(false);
    expect(runtime.starts[1]!.options.loopMotions).toBe(false);

    runtime.finishLast();
    expect(runtime.starts.map((value) => value.candidate.id)).toEqual(["idle", "happy", "idle"]);
    expect(runtime.starts.at(-1)!.candidate).toEqual(idle);
    expect(selectNext).not.toHaveBeenCalled();
    expect(runtime.poseBegins).toBe(1);
    expect(runtime.poseRestores).toBe(1);
    expect(runtime.lifecycle).toContain("pose:restore");
    expect(runtime.lifecycle.indexOf("pose:restore")).toBeLessThan(
      runtime.lifecycle.lastIndexOf("start:idle"),
    );
  });

  it("restores pose selectors when a transient is cancelled", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    playback.playTransient(happy, "happy");

    playback.cancelTransient();

    expect(runtime.poseBegins).toBe(1);
    expect(runtime.poseRestores).toBe(1);
    expect(runtime.starts.at(-1)!.candidate.id).toBe("idle");
  });

  it("rotates a multi-candidate state after one complete loop", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle, () => ({ candidate: alternateIdle, rotate: true }));

    expect(runtime.starts[0]!.options).toMatchObject({
      loopMotions: false,
      finishAfterOneCycle: true,
    });
    runtime.finishLast();

    expect(runtime.starts.map((value) => value.candidate.id)).toEqual([
      "idle",
      "idle-alternate",
    ]);
    expect(runtime.starts.at(-1)!.options.finishAfterOneCycle).toBe(true);
  });

  it("keeps state rotation armed when there is no transient to cancel", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    const selectNext = vi.fn(() => ({ candidate: alternateIdle, rotate: true }));
    playback.setState(idle, selectNext);

    playback.cancelTransient();
    runtime.finishLast();

    expect(selectNext).toHaveBeenCalledTimes(1);
    expect(runtime.starts.map((value) => value.candidate.id)).toEqual([
      "idle",
      "idle-alternate",
    ]);
  });

  it("keeps the default pose after an emotion ends while speaking", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setState(idle);
    expect(runtime.starts[0]!.options).toMatchObject({
      loopMotions: true,
      finishAfterOneCycle: false,
    });
    playback.setSpeaking(true);
    playback.playTransient(happy, "happy");
    expect(runtime.persistentStarts).toEqual([["smile-a", "smile-b"]]);
    expect(runtime.starts[1]!.candidate.expressionAssetIds).toEqual([]);
    runtime.finishLast();
    expect(runtime.persistentStops).toBe(0);

    expect(runtime.starts.map((value) => value.candidate.id)).toEqual(["idle", "happy"]);
    expect(runtime.stops).toBeGreaterThanOrEqual(2);

    playback.setSpeaking(false);
    expect(runtime.persistentStops).toBe(1);
    expect(runtime.poseRestores).toBe(1);
    expect(runtime.starts.at(-1)!.candidate.id).toBe("idle");
  });

  it("keeps a pure face expression alive until speaking ends", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setSpeaking(true);

    expect(playback.playTransient({
      id: "face-only",
      motionAssetIds: [],
      expressionAssetIds: ["smile-a"],
    }, "face-only")).toBe(true);
    expect(playback.hasTransient()).toBe(false);
    expect(runtime.starts).toHaveLength(0);
    expect(runtime.persistentStarts).toEqual([["smile-a"]]);

    playback.setSpeaking(false);
    expect(runtime.persistentStops).toBe(1);
  });

  it("migrates an emotion started before speaking into the face overlay", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.playTransient(happy, "happy");
    expect(runtime.persistentStarts).toEqual([]);

    playback.setSpeaking(true);

    expect(runtime.persistentStarts).toEqual([]);
    expect(runtime.starts).toHaveLength(1);

    runtime.finishLast();
    expect(runtime.persistentStarts).toEqual([["smile-a", "smile-b"]]);
    expect(runtime.starts.at(-1)!.candidate.expressionAssetIds).toEqual([
      "smile-a", "smile-b",
    ]);
  });

  it("does not retain unclassified expressions during speaking", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.setSpeaking(true);
    playback.playTransient({
      id: "unknown-expression",
      motionAssetIds: ["happy-a"],
      expressionAssetIds: ["unknown-expression"],
    }, "unknown-expression");

    expect(runtime.persistentStarts).toEqual([]);
    expect(runtime.starts.at(-1)!.candidate.expressionAssetIds).toEqual([
      "unknown-expression",
    ]);
    runtime.finishLast();
    expect(runtime.persistentStarts).toEqual([]);
  });

  it("does not revive a pending face expression after speaking ends early", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.playTransient(happy, "happy");
    playback.setSpeaking(true);
    playback.setSpeaking(false);
    expect(runtime.poseRestores).toBe(0);

    runtime.finishLast();

    expect(runtime.persistentStarts).toEqual([]);
    expect(runtime.poseRestores).toBe(1);
  });

  it("uses a fixed two-second lifetime for an expression-only candidate", () => {
    const runtime = new Runtime();
    const playback = new Live2DBehaviorPlayback(() => runtime);
    playback.playTransient({ id: "expression", motionAssetIds: [], expressionAssetIds: ["smile"] }, "expression");
    expect(runtime.starts[0]!.options.expressionOnlyDurationMs).toBe(2_000);
  });

  it("atomically recovers the state when a transient cannot start", () => {
    const runtime = new Runtime();
    const warn = vi.fn();
    const playback = new Live2DBehaviorPlayback(() => runtime, warn);
    playback.setState(idle);
    runtime.fail = true;

    expect(playback.playTransient(happy, "happy")).toBe(false);
    expect(warn).toHaveBeenCalled();
    expect(runtime.stops).toBeGreaterThan(0);
    expect(runtime.poseBegins).toBe(1);
    expect(runtime.poseRestores).toBe(1);
  });
});
