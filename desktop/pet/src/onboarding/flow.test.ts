import { describe, expect, it } from "vitest";

import type { DesktopConfigurationView } from "../shared/contracts";
import {
  AI_PROVIDERS,
  aiServiceRequest,
  describeSetting,
  onboardingConfiguration,
  onboardingSteps,
  pendingCharacter,
  readableError,
  toggleTrait,
  traitBlock,
  traitNote,
  wakeWordFor,
} from "./flow";

const current = {
  version: 2,
  configured: false,
  voiceEnabled: true,
  proactiveCompanionEnabled: false,
  launchAtLogin: false,
  models: {
    text: [{ id: "old", name: "旧模型", baseUrl: "https://old.example/v1", model: "old" }],
    voice: [{ id: "voice-default", name: "默认语音服务", provider: "auto" }],
  },
  modelAssignments: { chat: "old", reasoning: "old", speech: "voice-default" },
  imageEditing: { openaiCompatibleBaseUrl: null },
  homeAssistant: { enabled: false, url: "http://127.0.0.1:8123" },
  perception: {
    operationPerceptionEnabled: true,
    screenPerceptionEnabled: false,
    audioPerceptionEnabled: false,
    excludeFishMemoryFromScreenCapture: true,
  },
  rendering: { maxTextureDimension: 4096 },
  audio: { wakeWord: "小鱼", inputDevice: null, outputDevice: null, ownerRecognitionEnabled: false },
  advanced: {} as DesktopConfigurationView["advanced"],
  secrets: {
    modelApiKeys: {}, homeAssistantToken: false, openaiCompatibleImageApiKey: false, falApiKey: false, agentApiKey: false,
  },
} as DesktopConfigurationView;

const qwen = AI_PROVIDERS[0]!;
const custom = AI_PROVIDERS.find((provider) => provider.id === "custom")!;

describe("onboardingSteps", () => {
  it("adds the permission step only when the chosen level needs it", () => {
    expect(onboardingSteps("text", true)).toEqual(["look", "soul", "brain", "companion"]);
    expect(onboardingSteps("full", true)).toEqual(["look", "soul", "brain", "companion", "permissions"]);
    expect(onboardingSteps("full", false)).toEqual(["look", "soul", "brain", "companion"]);
  });
});

describe("soul answers", () => {
  it("never drops a chosen trait on its own and keeps opposites apart", () => {
    expect(toggleTrait(["温柔", "可靠", "傲娇"], "天然呆")).toEqual(["温柔", "可靠", "傲娇"]);
    expect(traitBlock(["温柔", "可靠", "傲娇"], "天然呆")).toBe("已经选了 3 个，想换就先点掉一个");
    expect(toggleTrait(["安静"], "话痨")).toEqual(["安静"]);
    expect(traitBlock(["安静"], "元气")).toBe("「安静」和「元气」不太搭，只能选一个");
    expect(traitBlock(["安静"], "安静")).toBeNull();
    expect(toggleTrait(["温柔", "元气"], "温柔")).toEqual(["元气"]);
    expect(toggleTrait(["温柔"], "可靠")).toEqual(["温柔", "可靠"]);
  });

  it("names every greyed-out trait under the chips", () => {
    const options = ["温柔", "毒舌", "安静", "话痨", "元气"];
    expect(traitNote([], options)).toBe("选 1 到 3 个");
    expect(traitNote(["温柔", "安静"], options)).toBe("「毒舌」和「温柔」、「话痨」和「安静」、「元气」和「安静」不太搭，想选就先点掉后面那个");
    expect(traitNote(["温柔", "可靠", "傲娇"], options)).toBe("已选 3 个，想换就先点掉一个");
  });

  it("prefers the user's own words for the setting", () => {
    expect(describeSetting({ traits: ["温柔", "元气"], customSetting: "" })).toBe("性格温柔、元气，说话自然、简短。");
    // A 话痨 is not told to keep it short.
    expect(describeSetting({ traits: ["元气", "话痨"], customSetting: "" })).toBe("性格元气、话痨，说话自然。");
    expect(describeSetting({ traits: ["温柔"], customSetting: " 爱吐槽但很靠谱 " })).toBe("爱吐槽但很靠谱");
  });

  it("builds the pending character with its appearance", () => {
    expect(pendingCharacter({ name: " 小鱼 ", relation: "青梅竹马", traits: ["傲娇"], customSetting: "" }, "mao")).toEqual({
      name: "小鱼", role: "和用户是青梅竹马关系", setting: "性格傲娇，说话自然、简短。", modelId: "mao",
    });
  });

  it("uses Chinese names as the wake word and keeps the old one otherwise", () => {
    expect(wakeWordFor("阿澄", "小鱼")).toBe("阿澄");
    expect(wakeWordFor("Luna", "小鱼")).toBe("小鱼");
    expect(wakeWordFor("鱼", "小鱼")).toBe("小鱼");
  });
});

describe("onboardingConfiguration", () => {
  const service = { provider: qwen, baseUrl: "", model: "", apiKey: " sk-test " };

  it("uses one AI service for chat and reasoning and stores its key", () => {
    const value = onboardingConfiguration(current, { service, level: "text", name: "阿澄", audioSupported: true });
    expect(value.configured).toBe(true);
    expect(value.models.text).toEqual([{ id: "text-default", name: "千问", baseUrl: qwen.baseUrl, model: "qwen3.7-flash" }]);
    expect(value.modelAssignments).toEqual({ chat: "text-default", reasoning: "text-default", speech: "voice-default" });
    // The 千问 key also speaks: 通义 voices use the same 阿里云百炼 key.
    expect(value.secrets).toEqual({ modelApiKeys: { "text-default": "sk-test", "voice-default": "sk-test" } });
    expect(value.audio.wakeWord).toBe("阿澄");
    const doubao = AI_PROVIDERS.find((item) => item.id === "doubao")!;
    const other = onboardingConfiguration(current, { service: { ...service, provider: doubao }, level: "text", name: "阿澄", audioSupported: true });
    expect(other.secrets).toEqual({ modelApiKeys: { "text-default": "sk-test" } });
    expect("secrets" in value && (value as { secrets: unknown }).secrets).not.toHaveProperty("homeAssistantToken");
  });

  it("connects the decision model it was given: 百炼 through 千问's key, otherwise Jev", () => {
    const bailian = onboardingConfiguration(current, { service, level: "full", name: "小鱼", audioSupported: true,
      bailianWorkspaceId: " ws-1234 ", jevApiKey: "jev-1" });
    expect(bailian.experimentalCompanion).toMatchObject({ decisionProvider: "bailian", bailianWorkspaceId: "ws-1234" });
    expect(bailian.secrets?.jevApiKey).toBeUndefined();
    const doubao = AI_PROVIDERS.find((item) => item.id === "doubao")!;
    const jev = onboardingConfiguration(current, { service: { ...service, provider: doubao }, level: "full", name: "小鱼",
      audioSupported: true, bailianWorkspaceId: "ws-1234", jevApiKey: "jev-1" });
    expect(jev.experimentalCompanion).toMatchObject({ decisionProvider: "jev" });
    expect(jev.experimentalCompanion?.bailianWorkspaceId).toBeUndefined();
    expect(jev.secrets?.jevApiKey).toBe("jev-1");
    const voice = onboardingConfiguration(current, { service, level: "voice", name: "小鱼", audioSupported: true, bailianWorkspaceId: "ws-1234" });
    expect(voice.experimentalCompanion?.decisionProvider).toBeUndefined();
  });

  it("maps companion levels to voice and senses, never enabling operation perception", () => {
    const text = onboardingConfiguration(current, { service, level: "text", name: "小鱼", audioSupported: true });
    expect([text.voiceEnabled, text.proactiveCompanionEnabled, text.perception.screenPerceptionEnabled]).toEqual([false, false, false]);

    const voice = onboardingConfiguration(current, { service, level: "voice", name: "小鱼", audioSupported: true });
    expect([voice.voiceEnabled, voice.microphoneEnabled, voice.proactiveCompanionEnabled]).toEqual([true, true, false]);

    expect(voice.secrets?.jevApiKey).toBeUndefined();
    const full = onboardingConfiguration(current, { service, level: "full", name: "小鱼", audioSupported: false, jevApiKey: " jev-1 " });
    expect(full.proactiveCompanionEnabled).toBe(true);
    expect(full.secrets?.jevApiKey).toBe("jev-1");
    expect(full.perception).toEqual({
      operationPerceptionEnabled: false,
      screenPerceptionEnabled: true,
      audioPerceptionEnabled: false,
      excludeFishMemoryFromScreenCapture: true,
    });
  });

  it("takes address and model from the form for a custom service", () => {
    expect(aiServiceRequest({ provider: custom, baseUrl: " https://api.example/v1 ", model: " gpt ", apiKey: "k" }))
      .toEqual({ baseUrl: "https://api.example/v1", model: "gpt", apiKey: "k" });
    const value = onboardingConfiguration(current, {
      service: { provider: custom, baseUrl: "https://api.example/v1", model: "gpt", apiKey: "k" },
      level: "text", name: "小鱼", audioSupported: true,
    });
    expect(value.models.text[0]).toMatchObject({ name: "AI 服务", baseUrl: "https://api.example/v1", model: "gpt" });
  });
});

describe("readableError", () => {
  it("drops Electron's IPC wrapper", () => {
    expect(readableError(new Error("Error invoking remote method 'desktop:config:save': Error: 请填写“千问”的 API Key")))
      .toBe("请填写“千问”的 API Key");
    expect(readableError("网络断了")).toBe("网络断了");
  });
});
