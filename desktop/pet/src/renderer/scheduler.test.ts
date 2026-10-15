import { describe, expect, it } from "vitest";

import { PET_STATES, type PetControlEvent } from "../shared/contracts";
import type { LipShape, PetRenderer } from "./pet-renderer";
import { PetScheduler } from "./scheduler";

class RecordingRenderer implements PetRenderer {
  calls: string[] = [];
  loadModel = async () => undefined;
  updateTextureAtlases = async () => undefined;
  setState = (value: string) => this.calls.push(`state:${value}`);
  setEmotion = (value: string) => this.calls.push(`emotion:${value}`);
  playClick = () => this.calls.push("click");
  setLipShape = (value: LipShape) => this.calls.push(`mouth:${value.volume}`);
  setGaze = () => undefined;
  getModelBounds = () => null;
  isPointInDragArea = () => false;
  cancelGeneration = () => this.calls.push("cancel");
  resize = () => undefined;
  setPaused = () => undefined;
  destroy = () => undefined;
}

describe("PetScheduler", () => {
  it("drops old generation expression and audio after a new turn", () => {
    const renderer = new RecordingRenderer();
    const scheduler = new PetScheduler(renderer);
    scheduler.apply(event(1, "pet.state", "old", { state: "thinking" }));
    scheduler.apply(event(2, "pet.state", "new", { state: "listening" }));
    scheduler.apply(event(3, "pet.expression", "old", { emotion: "angry" }));

    expect(scheduler.acceptsAudio("old")).toBe(false);
    expect(scheduler.acceptsAudio("new")).toBe(true);
    expect(renderer.calls).not.toContain("emotion:angry");
    expect(renderer.calls).toContain("cancel");
  });

  it("keeps listening active while allowing click", () => {
    const renderer = new RecordingRenderer();
    const scheduler = new PetScheduler(renderer);
    scheduler.apply(event(1, "pet.state", "turn", { state: "listening" }));
    scheduler.interact();

    expect(renderer.calls).toContain("mouth:0");
    expect(renderer.calls).toContain("click");
    expect(scheduler.currentGeneration()).toBe("turn");
  });

  it("allows click while preserving the speaking generation and emotion", () => {
    const renderer = new RecordingRenderer();
    const scheduler = new PetScheduler(renderer);
    scheduler.apply(event(1, "pet.state", "turn", { state: "speaking" }));
    scheduler.apply(event(2, "pet.expression", "turn", { emotion: "happy" }));
    scheduler.interact();

    expect(renderer.calls).toContain("emotion:happy");
    expect(renderer.calls).toContain("click");
    expect(scheduler.acceptsAudio("turn")).toBe(true);
    expect(renderer.calls).not.toContain("cancel");
  });

  it("normalizes removed emotion values to the supported neutral emotion", () => {
    const renderer = new RecordingRenderer();
    const scheduler = new PetScheduler(renderer);
    scheduler.apply(event(1, "pet.expression", "turn", { emotion: "sleepy" }));

    expect(renderer.calls).toContain("emotion:neutral");
  });

  it.each(PET_STATES)("triggers click without changing state %s", (state) => {
    const renderer = new RecordingRenderer();
    const scheduler = new PetScheduler(renderer);
    scheduler.apply(event(1, "pet.state", null, { state }));
    renderer.calls = [];
    scheduler.interact();

    expect(renderer.calls).toEqual(["click"]);
    scheduler.replay();
    expect(renderer.calls).toContain(`state:${state}`);
  });

  it("allows click without any companion state events", () => {
    const renderer = new RecordingRenderer();
    const scheduler = new PetScheduler(renderer);
    scheduler.apply({
      ...event(0, "pet.snapshot", null, {}),
      type: "pet.snapshot",
      payload: { publisherConnected: false, events: [] },
    });
    scheduler.interact();
    expect(renderer.calls).toEqual(["click"]);
    expect(scheduler.currentGeneration()).toBeNull();
  });

  it("retires a completed generation before returning to listening", () => {
    const renderer = new RecordingRenderer();
    const scheduler = new PetScheduler(renderer);
    scheduler.apply(event(1, "pet.state", "done", { state: "speaking" }));
    scheduler.apply(event(2, "pet.audio.stop", "done", {}));
    scheduler.apply(event(3, "pet.state", null, { state: "listening" }));
    scheduler.apply(event(4, "pet.expression", "done", {
      emotion: "angry",
    }));

    expect(scheduler.currentGeneration()).toBeNull();
    expect(renderer.calls).not.toContain("emotion:angry");
  });

  it("accepts a lower sequence snapshot after the API restarts", () => {
    const renderer = new RecordingRenderer();
    const scheduler = new PetScheduler(renderer);
    scheduler.apply(event(50, "pet.state", "old", { state: "thinking" }));
    scheduler.apply({
      ...event(1, "pet.snapshot", "new", {}),
      type: "pet.snapshot",
      payload: {
        publisherConnected: true,
        events: [event(1, "pet.state", "new", { state: "listening" })],
      },
    });

    expect(scheduler.currentGeneration()).toBe("new");
    expect(renderer.calls).toContain("state:listening");
  });
});

function event(
  sequence: number,
  type: string,
  generation: string | null,
  payload: Record<string, unknown>,
): PetControlEvent {
  return { version: 1, sequence, type, generation, timestampMs: 0, payload };
}
