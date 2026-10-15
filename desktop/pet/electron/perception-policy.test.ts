import { describe, expect, it } from "vitest";

import { perceptionFramesEnabled, perceptionHistoryEnabled, perceptionModules, perceptionPolicy } from "./perception-policy";
import { supportsSystemAudio } from "./system-audio-support";

describe("independent perception policy", () => {
  it("disables unsupported audio without enabling screen capture", () => {
    const policy = perceptionPolicy({
      operationPerceptionEnabled: false, screenPerceptionEnabled: false, audioPerceptionEnabled: true,
      excludeFishMemoryFromScreenCapture: false,
    }, false);
    expect(policy).toEqual({ operationEnabled: false, screenEnabled: false, audioEnabled: false, audioSupported: false });
    expect(perceptionHistoryEnabled(policy)).toBe(false);
    expect(perceptionModules(policy)).toMatchObject({ screen: false, systemAudio: false });
    expect(perceptionHistoryEnabled({ ...policy, screenEnabled: true })).toBe(true);
  });

  it("keeps the eyes in the moment but stops summaries and computer sound when 内心 is off", () => {
    const policy = perceptionPolicy({
      operationPerceptionEnabled: false, screenPerceptionEnabled: true, audioPerceptionEnabled: true,
      excludeFishMemoryFromScreenCapture: false, innerLifeEnabled: false,
    });
    expect(policy).toEqual({ operationEnabled: false, screenEnabled: true, audioEnabled: false, summariesEnabled: false });
    expect(perceptionFramesEnabled(policy)).toBe(true);
    expect(perceptionHistoryEnabled(policy)).toBe(false);
    expect(perceptionModules(policy)).toMatchObject({ screen: true, systemAudio: false });
  });

  it.each([
    ["darwin", "21.6.0", false], ["darwin", "22.0.0", true],
    ["darwin", "25.0.0", true], ["linux", "6.0.0", false],
  ])("detects audio support on %s %s", (platform, version, supported) => {
    expect(supportsSystemAudio(platform as string, version as string)).toBe(supported);
  });
  it.each(Array.from({ length: 8 }, (_value, mask) => mask))(
    "maps combination %i without changing the explicit screen choice", (mask) => {
      const configuration = {
        operationPerceptionEnabled: Boolean(mask & 1),
        screenPerceptionEnabled: Boolean(mask & 2),
        audioPerceptionEnabled: Boolean(mask & 4),
        excludeFishMemoryFromScreenCapture: false,
      };
      const original = structuredClone(configuration);
      const policy = perceptionPolicy(configuration);
      const frames = Boolean(mask & 2);
      expect(policy).toEqual({
        operationEnabled: Boolean(mask & 1),
        screenEnabled: Boolean(mask & 2),
        audioEnabled: Boolean(mask & 4),
      });
      expect(perceptionFramesEnabled(policy)).toBe(frames);
      expect(perceptionHistoryEnabled(policy)).toBe(Boolean(mask & 6));
      expect(perceptionModules(policy)).toEqual({
        workspace: false, screen: frames, accessibility: false, camera: false,
        enhancedVision: false, computerHistory: Boolean(mask & 1),
        systemAudio: Boolean(mask & 4), legacyScreenAnalysis: false,
        excludeHostApplicationFromCapture: false,
      });
      expect(configuration).toEqual(original);
    },
  );

  it("forwards the independent Fish Memory exclusion only to screen capture", () => {
    const policy = perceptionPolicy({
      operationPerceptionEnabled: true,
      screenPerceptionEnabled: true,
      audioPerceptionEnabled: true,
      excludeFishMemoryFromScreenCapture: true,
    });
    expect(perceptionModules(policy, true)).toMatchObject({
      screen: true,
      computerHistory: true,
      systemAudio: true,
      legacyScreenAnalysis: false,
      excludeHostApplicationFromCapture: true,
    });
    expect(perceptionModules(policy, false).excludeHostApplicationFromCapture).toBe(false);
  });
});
