import type { PerceptionCapabilityReport, PerceptionManagerStatus } from "./perception-manager";

const REASONS = new Set([
  "screen_recording_permission_required", "display_not_found", "screen_capture_start_failed",
  "screen_capture_stop_failed", "screen_capture_stopped", "screen_capture_unavailable",
  "screen_capture_filter_update_failed",
  "system_audio_starting", "system_audio_capture_start_failed", "system_audio_capture_stopped",
  "system_audio_filter_update_failed", "system_audio_waiting_for_cleanup",
  "system_audio_cleanup_failed", "core_audio_process_tap_start_failed",
  "core_audio_process_tap_cleanup_failed",
  "core_audio_process_tap_permission_required", "core_audio_process_filter_update_failed",
  "core_audio_aggregate_device_stopped", "screen_capture_kit_fallback",
  "macos_13_required", "macos_12_3_required",
]);

export function perceptionStatusDiagnostic(status: PerceptionManagerStatus): string {
  return JSON.stringify({
    event: "perception_status",
    phase: status.phase,
    modules: {
      workspace: status.modules.workspace,
      screen: status.modules.screen,
      accessibility: status.modules.accessibility,
      camera: status.modules.camera,
      enhancedVision: status.modules.enhancedVision === true,
      computerHistory: status.modules.computerHistory === true,
      systemAudio: status.modules.systemAudio === true,
      legacyScreenAnalysis: status.modules.legacyScreenAnalysis !== false,
      excludeHostApplicationFromCapture:
        status.modules.excludeHostApplicationFromCapture === true,
    },
    screen: capability(status.capabilities?.screen),
    systemAudio: capability(status.capabilities?.systemAudio),
  });
}

function capability(report: PerceptionCapabilityReport | undefined) {
  if (!report) return null;
  const prefix = report.reason?.split(":", 1)[0];
  const knownReason = prefix !== undefined && REASONS.has(prefix);
  const nativeCode = knownReason ? report.reason?.match(/\bCode=(-?\d+)\b/u)?.[1] : undefined;
  return {
    requested: report.requested,
    state: report.state,
    permission: report.permission,
    reason: prefix === undefined ? null : knownReason ? prefix : "unclassified",
    ...(nativeCode === undefined ? {} : { nativeErrorCode: Number(nativeCode) }),
  };
}
