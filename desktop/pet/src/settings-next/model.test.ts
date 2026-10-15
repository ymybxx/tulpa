import { describe, expect, it } from "vitest";

import type { DesktopConfigurationView } from "../shared/contracts";
import { AI_PROVIDERS } from "../onboarding/flow";
import {
  blockedFeatures, configuredModels, conversationService, modelLabel, withAssignedModel, withChanges,
  withRemovedModel, withSavedModel,
} from "./model";

const view = {
  version: 2, configured: true, voiceEnabled: true, microphoneEnabled: true, playbackEnabled: true,
  proactiveCompanionEnabled: false, launchAtLogin: false,
  models: {
    text: [
      { id: "chat", name: "千问", baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1", model: "qwen3.7-flash" },
      { id: "deep", name: "推理", baseUrl: "https://example.com/v1", model: "deep" },
    ],
    voice: [{ id: "voice-default", name: "默认语音服务", provider: "auto" }],
  },
  modelAssignments: { chat: "chat", reasoning: "deep", speech: "voice-default" },
  perception: {
    operationPerceptionEnabled: false, screenPerceptionEnabled: true, audioPerceptionEnabled: true,
    excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
  },
  secrets: { modelApiKeys: { chat: true, deep: true }, jevApiKey: false },
} as unknown as DesktopConfigurationView;

describe("conversationService", () => {
  it("recognises the provider and a separate reasoning model", () => {
    expect(conversationService(view)).toMatchObject({ provider: { id: "qwen" }, split: true, hasKey: true });
    expect("secrets" in withChanges(view, { launchAtLogin: true })).toBe(false);
  });
});

describe("saved models", () => {
  const shared = { ...view, modelAssignments: { ...view.modelAssignments, reasoning: "chat" } } as DesktopConfigurationView;

  it("lists the models saved with a key and what uses them", () => {
    expect(configuredModels(view).map(({ model, provider, usedBy }) => [model.id, provider.id, usedBy]))
      .toEqual([["chat", "qwen", ["对话大脑"]], ["deep", "custom", ["推理大脑"]]]);
    expect(modelLabel(view.models.text[0]!)).toBe("千问 · qwen3.7-flash");
    expect(modelLabel(view.models.text[1]!)).toBe("deep · example.com");
  });

  it("adds a new model without changing what is in use, and edits one in place", () => {
    const added = withSavedModel(view, { provider: AI_PROVIDERS[1]!, baseUrl: "", model: "", apiKey: " ark " });
    expect(added.models.text.map((model) => model.id)).toEqual(["chat", "deep", "text-3"]);
    expect(added.models.text[2]).toMatchObject({ name: "豆包", model: "doubao-seed-character-260628" });
    expect(added.modelAssignments).toEqual(view.modelAssignments);
    expect(added.secrets).toEqual({ modelApiKeys: { "text-3": "ark" } });

    const edited = withSavedModel(view, { provider: AI_PROVIDERS[0]!, baseUrl: "", model: "", apiKey: "q" }, { id: "deep" });
    expect(edited.models.text[1]).toMatchObject({ id: "deep", name: "千问", model: "qwen3.7-flash" });

    const used = withSavedModel(view, { provider: AI_PROVIDERS[1]!, baseUrl: "", model: "", apiKey: "a" }, { use: "reasoning" });
    expect(used.modelAssignments).toMatchObject({ chat: "chat", reasoning: "text-3" });
  });

  it("makes the first saved model both brains", () => {
    const empty = { ...view, secrets: { ...view.secrets, modelApiKeys: {} } } as DesktopConfigurationView;
    const input = withSavedModel(empty, { provider: AI_PROVIDERS[0]!, baseUrl: "", model: "", apiKey: "q" });
    expect(input.modelAssignments).toMatchObject({ chat: "text-3", reasoning: "text-3" });
  });

  it("moves reasoning left on the chat model along with it, and not a separate one", () => {
    expect(withAssignedModel(shared, "chat", "deep").modelAssignments).toMatchObject({ chat: "deep", reasoning: "deep" });
    expect(withAssignedModel(view, "chat", "deep").modelAssignments).toMatchObject({ chat: "deep", reasoning: "deep" });
    expect(withAssignedModel(view, "reasoning", "chat").modelAssignments).toMatchObject({ chat: "chat", reasoning: "chat" });
  });

  it("removes a model and moves what used it, but keeps the last one", () => {
    const input = withRemovedModel(view, "chat")!;
    expect(input.models.text.map((model) => model.id)).toEqual(["deep"]);
    expect(input.modelAssignments).toMatchObject({ chat: "deep", reasoning: "deep" });
    expect(withRemovedModel(view, "deep")!.modelAssignments).toMatchObject({ chat: "chat", reasoning: "chat" });
    expect(withRemovedModel({ ...view, models: { ...view.models, text: [view.models.text[0]!] } } as DesktopConfigurationView, "chat")).toBeNull();
  });
});

describe("blockedFeatures", () => {
  it("lists switched-on features waiting for a permission", () => {
    // Screen Recording covers computer sound as well.
    expect(blockedFeatures(view, { screen: "denied", systemAudio: "not-determined", microphone: "granted" }))
      .toEqual({ screen: ["屏幕感知", "电脑音频感知"], systemAudio: [], microphone: [] });
    // Only once the screen is allowed can computer sound still be refused on its own (仅系统录音).
    expect(blockedFeatures(view, { screen: "granted", systemAudio: "denied", microphone: "granted" }))
      .toEqual({ screen: [], systemAudio: ["电脑音频感知"], microphone: [] });
    expect(blockedFeatures({ ...view, perception: { ...view.perception, innerLifeEnabled: false } }, { screen: "denied", systemAudio: "denied", microphone: "denied" }))
      .toEqual({ screen: ["屏幕感知"], systemAudio: [], microphone: ["麦克风收音"] });
    expect(blockedFeatures(view, { screen: "not-required", systemAudio: "not-required", microphone: "granted" }))
      .toEqual({ screen: [], systemAudio: [], microphone: [] });
  });
});

