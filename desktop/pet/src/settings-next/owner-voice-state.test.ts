import { describe, expect, it } from "vitest";
import type { OwnerVoiceEnrollmentState, OwnerVoiceStatus } from "../shared/contracts";
import { ownerVoiceControls } from "./owner-voice-state";

const status: OwnerVoiceStatus = { enrolled: true, modelReady: true, modelId: "cam++", createdAt: "2026-09-03", detail: "ready", audioBackend: "livekit" };
const idle: OwnerVoiceEnrollmentState = { phase: "idle", status, detail: "ready" };
describe("owner voice settings state", () => {
  it("requires a ready model and matching enrolled backend before enabling", () => {
    expect(ownerVoiceControls(null, idle, "livekit", "livekit").canEnable).toBe(false);
    expect(ownerVoiceControls({ ...status, modelReady: false }, idle, "livekit", "livekit").canEnable).toBe(false);
    expect(ownerVoiceControls(status, idle, "sounddevice", "sounddevice").canEnable).toBe(false);
    expect(ownerVoiceControls(status, idle, "livekit", "sounddevice").canEnable).toBe(false);
    expect(ownerVoiceControls(status, idle, "livekit", "livekit").canEnable).toBe(true);
  });
  it.each(["preparing", "recording", "restoring"] as const)("locks mutation while %s", (phase) => {
    const controls = ownerVoiceControls(status, { ...idle, phase }, "livekit", "livekit");
    expect(controls.busy).toBe(true);
    expect(controls.canEnable).toBe(false);
    expect(controls.canDelete).toBe(false);
  });
  it("offers re-enrollment and deletion for an incompatible retained profile", () => {
    const controls = ownerVoiceControls({ ...status, enrolled: false }, idle, "sounddevice", "sounddevice");
    expect(controls.enrollLabel).toBe("重新录入");
    expect(controls.canDelete).toBe(true);
  });
  it.each([
    null,
    { ...status, modelReady: false },
    { ...status, enrolled: false },
    { ...status, audioBackend: "sounddevice" as const },
  ])("allows disabling an existing opt-in even when its model/profile is unavailable", (unavailable) => {
    const enabled = ownerVoiceControls(unavailable, idle, "livekit", "livekit", false, true);
    expect(enabled.canEnable).toBe(false);
    expect(enabled.checkboxDisabled).toBe(false);
    // After the user turns it off, the change handler must prohibit enabling it again.
    const disabled = ownerVoiceControls(unavailable, idle, "livekit", "livekit", false, false);
    expect(disabled.checkboxDisabled).toBe(true);
  });
  it("still locks an enabled checkbox while recording or saving", () => {
    expect(ownerVoiceControls(null, { ...idle, phase: "recording" }, "livekit", "livekit", false, true).checkboxDisabled).toBe(true);
    expect(ownerVoiceControls(null, idle, "livekit", "livekit", false, true, true).checkboxDisabled).toBe(true);
  });
});
