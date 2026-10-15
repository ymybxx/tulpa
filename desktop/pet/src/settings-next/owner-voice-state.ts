import type { DesktopAudioBackend, OwnerVoiceEnrollmentState, OwnerVoiceStatus } from "../shared/contracts";

export function ownerVoiceControls(
  status: OwnerVoiceStatus | null,
  enrollment: OwnerVoiceEnrollmentState,
  selectedBackend: DesktopAudioBackend,
  savedBackend: DesktopAudioBackend,
  pending = false,
  enabled = false,
  saving = false,
) {
  const busy = pending || ["preparing", "recording", "restoring"].includes(enrollment.phase);
  const matchesBackend = selectedBackend === savedBackend
    && (!status?.audioBackend || status.audioBackend === selectedBackend);
  const canEnable = !busy && matchesBackend && Boolean(status?.enrolled && status.modelReady);
  return {
    busy,
    canEnable,
    // Missing/corrupt models must never trap an existing opt-in in the on state.
    checkboxDisabled: busy || saving || (!enabled && !canEnable),
    canDelete: !busy && Boolean(status?.createdAt || status?.enrolled),
    label: ({ preparing: "准备中", recording: "录入中", restoring: "正在恢复语音", failed: "失败" } as Record<string, string>)[enrollment.phase]
      ?? (status?.enrolled ? "已录入" : "未录入"),
    enrollLabel: status?.createdAt || status?.enrolled ? "重新录入" : "录入主人声纹",
  };
}
