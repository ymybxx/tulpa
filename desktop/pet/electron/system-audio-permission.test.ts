import { describe, expect, it, vi } from "vitest";

import { SYSTEM_AUDIO_PERMISSION_REQUIRED, SystemAudioPermission } from "./system-audio-permission";
import type { PerceptionCapabilityReport } from "./perception-manager";

function report(state: PerceptionCapabilityReport["state"], reason?: string): PerceptionCapabilityReport {
  return { supported: true, requested: true, state, permission: "not_determined", ...(reason ? { reason } : {}) } as PerceptionCapabilityReport;
}

describe("SystemAudioPermission", () => {
  it("follows Screen Recording until a capture says otherwise: it covers computer sound too", () => {
    const permission = new SystemAudioPermission({ supported: () => true, screenGranted: () => true, capability: () => undefined, probe: vi.fn() });
    expect(permission.status()).toBe("granted");
  });

  it("reads the running helper: recording means allowed, a refused tap means denied", () => {
    let capability: PerceptionCapabilityReport | undefined = report("running");
    const permission = new SystemAudioPermission({ supported: () => true, screenGranted: () => false, capability: () => capability, probe: vi.fn() });
    expect(permission.status()).toBe("granted");
    capability = report("unavailable", SYSTEM_AUDIO_PERMISSION_REQUIRED);
    expect(permission.status()).toBe("denied");
    capability = undefined;
    expect(permission.status()).toBe("not-determined");
  });

  it("asks with a short recording and remembers the answer", async () => {
    const probe = vi.fn(async () => false);
    const permission = new SystemAudioPermission({ supported: () => true, screenGranted: () => false, capability: () => undefined, probe });
    expect(await permission.ask()).toBe("denied");
    probe.mockResolvedValue(true);
    expect(await permission.ask()).toBe("granted");
    expect(permission.status()).toBe("granted");
  });

  it("re-checks a refusal every few seconds while nothing else is recording", async () => {
    let now = 0;
    const probe = vi.fn(async () => false);
    const permission = new SystemAudioPermission({ supported: () => true, screenGranted: () => false, capability: () => undefined, probe, now: () => now });
    await permission.ask();
    expect(await permission.refresh()).toBe("denied");
    expect(probe).toHaveBeenCalledTimes(1);
    now = 4_000;
    probe.mockResolvedValue(true);
    // Switched on in System Settings meanwhile.
    expect(await permission.refresh()).toBe("granted");
    expect(probe).toHaveBeenCalledTimes(2);
  });

  it("is not needed where the system cannot record its own audio", async () => {
    const probe = vi.fn();
    const permission = new SystemAudioPermission({ supported: () => false, screenGranted: () => false, capability: () => undefined, probe });
    expect(permission.status()).toBe("not-required");
    expect(await permission.ask()).toBe("not-required");
    expect(probe).not.toHaveBeenCalled();
  });
});
