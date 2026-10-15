import { describe, expect, it } from "vitest";

import type { PerceptionManagerStatus } from "./perception-manager";
import { perceptionStatusDiagnostic } from "./perception-status-diagnostic";

describe("perception status diagnostics", () => {
  it("keeps requested permission/state and numeric native error without private text", () => {
    const status = fixture();
    const encoded = perceptionStatusDiagnostic(status);
    const value = JSON.parse(encoded);
    expect(Object.keys(value)).toEqual(["event", "phase", "modules", "screen", "systemAudio"]);
    expect(value.screen).toEqual({
      requested: true,
      state: "unavailable",
      permission: "not_determined",
      reason: "screen_capture_start_failed",
      nativeErrorCode: -3801,
    });
    expect(value.systemAudio.reason).toBe("screen_capture_unavailable");
    expect(value.modules).toMatchObject({
      screen: true, systemAudio: true, computerHistory: false,
      excludeHostApplicationFromCapture: false,
    });
    expect(encoded).not.toContain("private");
  });

  it("never copies an unknown reason string into the runtime log", () => {
    const status = fixture();
    status.capabilities!.screen.reason = "private_window_title";
    const encoded = perceptionStatusDiagnostic(status);
    expect(JSON.parse(encoded).screen.reason).toBe("unclassified");
    expect(encoded).not.toContain("private");
  });

  it("keeps process-tap permission and fallback reasons visible", () => {
    const status = fixture();
    status.capabilities!.systemAudio!.reason = "core_audio_process_tap_permission_required";
    expect(JSON.parse(perceptionStatusDiagnostic(status)).systemAudio.reason)
      .toBe("core_audio_process_tap_permission_required");

    status.capabilities!.systemAudio!.state = "running";
    status.capabilities!.systemAudio!.reason = "screen_capture_kit_fallback";
    expect(JSON.parse(perceptionStatusDiagnostic(status)).systemAudio.reason)
      .toBe("screen_capture_kit_fallback");

    status.capabilities!.systemAudio!.state = "failed";
    status.capabilities!.systemAudio!.reason = "system_audio_cleanup_failed";
    expect(JSON.parse(perceptionStatusDiagnostic(status)).systemAudio.reason)
      .toBe("system_audio_cleanup_failed");
  });
});

function fixture(): PerceptionManagerStatus {
  const disabled = {
    supported: true, requested: false, state: "disabled", permission: "not_required",
  } as const;
  return {
    phase: "degraded", detail: "private desktop detail", helperPid: 123, crashCount: 0,
    modules: {
      workspace: false, screen: true, accessibility: false, camera: false,
      computerHistory: false, systemAudio: true, legacyScreenAnalysis: false,
    },
    capabilities: {
      workspace: disabled, accessibility: disabled, camera: disabled, computerHistory: disabled,
      screen: {
        supported: true, requested: true, state: "unavailable", permission: "not_determined",
        reason: "screen_capture_start_failed: Error Domain=ScreenCaptureKit Code=-3801 private window content",
      },
      systemAudio: {
        supported: true, requested: true, state: "unavailable", permission: "not_determined",
        reason: "screen_capture_unavailable",
      },
    },
  };
}
