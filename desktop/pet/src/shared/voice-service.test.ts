import { describe, expect, it } from "vitest";
import { cloudVoiceKey } from "../../electron/model-configuration";
import type { DesktopConfiguration } from "./contracts";
import { cloudVoiceService } from "./voice-service";

const configuration = (voice: DesktopConfiguration["models"]["voice"], speech: string) =>
  ({ models: { text: [], voice }, modelAssignments: { chat: "t", reasoning: "t", speech } }) as unknown as DesktopConfiguration;

describe("the 通义 key that characters speak with", () => {
  const voices = [
    { id: "local", name: "本机", provider: "local" as const },
    { id: "tongyi", name: "通义", provider: "dashscope" as const },
  ];

  it("reaches the voice service even when the default is 本机", () => {
    expect(cloudVoiceKey(configuration(voices, "local"), { tongyi: "sk-1" })).toBe("sk-1");
    expect(cloudVoiceKey(configuration(voices, "local"), {})).toBeUndefined();
    // A local service's key is never used for 通义.
    expect(cloudVoiceKey(configuration(voices, "local"), { local: "stray" })).toBeUndefined();
  });

  it("is stored on the existing 通义 service, or on a new one", () => {
    expect(cloudVoiceService(configuration(voices, "local"))).toMatchObject({ id: "tongyi" });
    const created = cloudVoiceService(configuration([voices[0]!], "local"));
    expect(created.id).toBe("voice-dashscope");
    expect(created.models.voice.map((item) => item.provider)).toEqual(["local", "dashscope"]);
  });
});
