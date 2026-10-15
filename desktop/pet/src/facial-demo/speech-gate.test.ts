import { describe, expect, it } from "vitest";
import { SpeechGate } from "./speech-gate";

describe("SpeechGate", () => {
  it("holds cached audio that arrives before its face is ready", () => {
    const gate = new SpeechGate<Uint8Array>();
    const pcm = new Uint8Array([1, 2]);
    gate.push(0, pcm);
    expect(gate.drain()).toEqual([]);
    expect(gate.hasPending()).toBe(true);
    expect(gate.pendingSegments()).toEqual([0]);
    gate.ready(0);
    const released = gate.drain();
    expect(released).toEqual([{ segment: 0, chunk: pcm }]);
    expect(released[0]!.chunk).toBe(pcm);
    expect(gate.hasPending()).toBe(false);
    expect(gate.drain()).toEqual([]);
  });

  it("does not let a ready future sentence pass an earlier blocked sentence", () => {
    const gate = new SpeechGate<string>();
    gate.push(0, "first-a");
    gate.push(0, "first-b");
    gate.push(1, "second");
    gate.ready(1);
    expect(gate.drain()).toEqual([]);
    expect(gate.pendingSegments()).toEqual([0, 1]);
    gate.ready(0);
    expect(gate.drain()).toEqual([
      { segment: 0, chunk: "first-a" },
      { segment: 0, chunk: "first-b" },
      { segment: 1, chunk: "second" },
    ]);
    expect(gate.pendingSegments()).toEqual([]);
  });

  it("releases only the ready prefix and keeps the remaining FIFO order", () => {
    const gate = new SpeechGate<string>();
    gate.push(0, "first");
    gate.push(1, "second");
    gate.push(2, "third");
    gate.ready(0);
    gate.ready(2);
    expect(gate.drain()).toEqual([{ segment: 0, chunk: "first" }]);
    expect(gate.pendingSegments()).toEqual([1, 2]);
    gate.ready(1);
    expect(gate.drain()).toEqual([{ segment: 1, chunk: "second" }, { segment: 2, chunk: "third" }]);
  });

  it("remembers an early face and releases later streaming chunks of the same sentence", () => {
    const gate = new SpeechGate<string>();
    gate.ready(0);
    expect(gate.hasPending()).toBe(false);
    expect(gate.pendingSegments()).toEqual([]);
    gate.push(0, "first-chunk");
    expect(gate.drain()).toEqual([{ segment: 0, chunk: "first-chunk" }]);
    gate.push(0, "later-chunk");
    expect(gate.drain()).toEqual([{ segment: 0, chunk: "later-chunk" }]);
  });

  it("clears both queued chunks and ready state before the next run", () => {
    const gate = new SpeechGate<string>();
    gate.ready(0);
    gate.ready(1);
    gate.push(0, "old-run");
    gate.clear();
    expect(gate.hasPending()).toBe(false);
    expect(gate.pendingSegments()).toEqual([]);
    expect(gate.drain()).toEqual([]);
    gate.push(0, "new-run");
    expect(gate.drain()).toEqual([]);
    gate.ready(0);
    expect(gate.drain()).toEqual([{ segment: 0, chunk: "new-run" }]);
    gate.push(1, "new-run-next");
    expect(gate.drain()).toEqual([]);
  });
});
