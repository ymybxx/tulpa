import { describe, expect, it } from "vitest";

import { petStateEvent, SleepWatch } from "./sleep-watch";

describe("noticing TA fell asleep", () => {
  it("says so once TA first settles, and after each voice conversation", () => {
    const watch = new SleepWatch();
    expect(watch.next("unbound")).toBe(false);
    expect(watch.next("standby")).toBe(true);
    expect(watch.next("standby")).toBe(false);
    for (const state of ["acknowledging", "listening", "thinking", "speaking", "listening"]) expect(watch.next(state)).toBe(false);
    expect(watch.next("standby")).toBe(true);
  });

  it("stays quiet when TA spoke up by itself or answered typing", () => {
    const watch = new SleepWatch();
    watch.next("standby");
    for (const state of ["thinking", "speaking"]) watch.next(state);
    expect(watch.next("standby")).toBe(false);
  });

  it("reads the state from an event or a reconnect's replay", () => {
    expect(petStateEvent({ type: "pet.state", payload: { state: "standby" } })?.payload.state).toBe("standby");
    expect(petStateEvent({ type: "pet.snapshot", payload: { events: [
      { type: "pet.state", payload: { state: "listening" } }, { type: "pet.microphone.state", payload: {} },
    ] } })?.payload.state).toBe("listening");
    expect(petStateEvent({ type: "pet.snapshot", payload: { events: [] } })).toBeUndefined();
    expect(petStateEvent({ type: "pet.voice.hearing", payload: {} })).toBeUndefined();
  });
});
