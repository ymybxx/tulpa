import { describe, expect, it } from "vitest";

import {
  ACTIVE_RENDER_FPS,
  ENGAGED_RENDER_FPS,
  IDLE_RENDER_FPS,
  live2DFrameRate,
} from "./frame-rate";

describe("live2DFrameRate", () => {
  it("keeps procedural idle within the existing non-speaking frame budget", () => {
    for (const state of ["standby", "unbound", "acknowledging", "listening", "thinking"] as const) {
      expect(live2DFrameRate(state, 0, false, true)).toBe(live2DFrameRate(state, 0));
      expect(live2DFrameRate(state, 0, true, true)).toBe(ENGAGED_RENDER_FPS);
    }
    expect(live2DFrameRate("speaking", 0, false, true)).toBe(ACTIVE_RENDER_FPS);
    expect(live2DFrameRate("thinking", 0.4, false, true)).toBe(ACTIVE_RENDER_FPS);
  });
  it("renders fully idle states at 30 fps", () => {
    expect(IDLE_RENDER_FPS).toBe(30);
    expect(live2DFrameRate("standby", 0)).toBe(IDLE_RENDER_FPS);
    expect(live2DFrameRate("unbound", 0)).toBe(IDLE_RENDER_FPS);
  });

  it("keeps interactive states and transient motions at 30 fps", () => {
    expect(ENGAGED_RENDER_FPS).toBe(30);
    expect(live2DFrameRate("acknowledging", 0)).toBe(ENGAGED_RENDER_FPS);
    expect(live2DFrameRate("listening", 0)).toBe(ENGAGED_RENDER_FPS);
    expect(live2DFrameRate("thinking", 0)).toBe(ENGAGED_RENDER_FPS);
    expect(live2DFrameRate("standby", 0, true)).toBe(ENGAGED_RENDER_FPS);
  });

  it("returns to 60 fps while speaking or receiving active lip frames", () => {
    expect(ACTIVE_RENDER_FPS).toBe(60);
    expect(live2DFrameRate("speaking", 0)).toBe(ACTIVE_RENDER_FPS);
    expect(live2DFrameRate("thinking", 0.4)).toBe(ACTIVE_RENDER_FPS);
  });
});
