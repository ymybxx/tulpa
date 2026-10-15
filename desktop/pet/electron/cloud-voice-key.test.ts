import { describe, expect, it } from "vitest";

import { cloudVoiceKey } from "./model-configuration";
import type { DesktopConfiguration } from "../src/shared/contracts";

function configuration(baseUrl: string): DesktopConfiguration {
  // Only the models and assignments matter here.
  return {
    models: {
      text: [{ id: "text-default", name: "对话", baseUrl, model: "m" }],
      voice: [{ id: "voice-default", name: "默认语音服务", provider: "auto" }],
    },
    modelAssignments: { chat: "text-default", reasoning: "text-default", speech: "voice-default" },
  } as unknown as DesktopConfiguration;
}

describe("cloudVoiceKey", () => {
  it("uses the 千问 chat key for 通义 voices: it is the same 阿里云百炼 key", () => {
    const qwen = configuration("https://dashscope.aliyuncs.com/compatible-mode/v1");
    expect(cloudVoiceKey(qwen, { "text-default": "sk-bailian" })).toBe("sk-bailian");
    // A key set for the voice service itself still wins.
    expect(cloudVoiceKey(qwen, { "text-default": "sk-bailian", "voice-default": "sk-voice" })).toBe("sk-voice");
  });

  it("does not send another provider's key to 通义", () => {
    const doubao = configuration("https://ark.cn-beijing.volces.com/api/v3");
    expect(cloudVoiceKey(doubao, { "text-default": "sk-ark" })).toBeUndefined();
  });
});
