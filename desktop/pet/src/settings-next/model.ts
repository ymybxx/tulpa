import { t } from "../i18n";
import type {
  DesktopConfiguration,
  DesktopConfigurationInput,
  DesktopConfigurationView,
  DesktopTextModel,
  SystemPermissionKind,
  SystemPermissionState,
} from "../shared/contracts";
import { AI_PROVIDERS, aiServiceRequest, type AiProvider, type AiServiceChoice } from "../onboarding/flow";

export type SettingsSection = "ai" | "jev" | "companion" | "usage" | "extensions" | "permissions" | "general" | "advanced";

export const SETTINGS_SECTIONS: ReadonlyArray<{
  id: SettingsSection; group: string; label: string; latin: string; subtitle: string; footer: string;
}> = [
  // By what it is about: TA itself, its brains and what they cost, other
  // services, the app. A group starts where `group` changes.
  { id: "companion", group: "ta", label: "陪伴", latin: "COMPANION", subtitle: "TA 能感知什么、怎么和你说话", footer: "屏幕画面交给你设置的 AI 服务理解；请求和图片保存在本机日志，按保留天数自动删除" },
  { id: "permissions", group: "ta", label: "权限", latin: "PERMISSIONS", subtitle: "系统允许 TA 使用哪些设备", footer: "权限随时可以在系统设置里收回" },
  { id: "ai", group: "brain", label: "AI 服务", latin: "SERVICES", subtitle: "TA 思考和说话用到的服务", footer: "Key 只保存在这台电脑上，不会上传" },
  { id: "jev", group: "brain", label: "决策模型", latin: "DECISION", subtitle: "接上决策模型（百炼或 Jev）以后，这些判断交给它：更准、更快、更省", footer: "Key 只保存在这台电脑上，不会上传" },
  { id: "usage", group: "brain", label: "用量", latin: "USAGE", subtitle: "TA 用模型花了多少 token，都用在哪", footer: "只统计这台电脑上的调用记录" },
  { id: "extensions", group: "services", label: "帮你做事", latin: "HELP", subtitle: "TA 能替你做的事，和别的程序托 TA 转告你的消息", footer: "地址、Key 和令牌只保存在这台电脑上，不会上传" },
  { id: "general", group: "app", label: "通用", latin: "GENERAL", subtitle: "外观、启动和资源", footer: "改动立即生效" },
  { id: "advanced", group: "app", label: "高级", latin: "ADVANCED", subtitle: "语音识别和调试参数", footer: "改动立即生效，语音服务会自动重启" },
];

/** The configuration without secret flags, ready to be saved with changes. */
export function editable(view: DesktopConfigurationView): DesktopConfiguration {
  const { secrets: _secrets, ...configuration } = view;
  return configuration;
}

export function withChanges(
  view: DesktopConfigurationView,
  change: Partial<DesktopConfiguration>,
  secrets?: DesktopConfigurationInput["secrets"],
): DesktopConfigurationInput {
  return { ...editable(view), ...change, ...(secrets ? { secrets } : {}) };
}

function assigned(view: DesktopConfigurationView, module: "chat" | "reasoning"): DesktopTextModel | undefined {
  return view.models.text.find((model) => model.id === view.modelAssignments[module]) ?? view.models.text[0];
}

export function providerOf(model: DesktopTextModel | undefined): AiProvider {
  return AI_PROVIDERS.find((item) => item.id !== "custom" && item.baseUrl === model?.baseUrl.replace(/\/$/, ""))
    ?? AI_PROVIDERS.find((item) => item.id === "custom")!;
}

/** 对话大脑: the chat model, and whether reasoning uses a different one. */
export function conversationService(view: DesktopConfigurationView): {
  model: DesktopTextModel | undefined;
  provider: AiProvider;
  split: boolean;
  hasKey: boolean;
} {
  const model = assigned(view, "chat");
  const reasoning = assigned(view, "reasoning");
  return {
    model,
    provider: providerOf(model),
    split: Boolean(model && reasoning && model.id !== reasoning.id),
    hasKey: Boolean(model && view.secrets.modelApiKeys[model.id]),
  };
}

/** The text models saved with a key, which 对话大脑 and 推理大脑 pick from. */
export function configuredModels(view: DesktopConfigurationView): Array<{
  model: DesktopTextModel;
  provider: AiProvider;
  usedBy: Array<"对话大脑" | "推理大脑">;
}> {
  return view.models.text
    .filter((model) => view.secrets.modelApiKeys[model.id])
    .map((model) => ({
      model,
      provider: providerOf(model),
      usedBy: [
        ...(view.modelAssignments.chat === model.id ? ["对话大脑" as const] : []),
        ...(view.modelAssignments.reasoning === model.id ? ["推理大脑" as const] : []),
      ],
    }));
}

/** How a saved model reads in a list or picker. */
export function modelLabel(model: DesktopTextModel): string {
  const provider = providerOf(model);
  return provider.id === "custom"
    ? `${model.model} · ${hostOf(model.baseUrl)}`
    : `${t(provider.label)} · ${model.model}`;
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

function serviceModel(id: string, choice: AiServiceChoice): { model: DesktopTextModel; apiKey: string } {
  const request = aiServiceRequest(choice);
  return {
    model: {
      id,
      name: choice.provider.id === "custom" ? "AI 服务" : choice.provider.label,
      baseUrl: request.baseUrl,
      model: request.model,
    },
    apiKey: request.apiKey,
  };
}

/**
 * Saves a verified service as a model: a new entry, or `id` edited in place.
 * `use` puts it to work straight away; the first model saved is both brains.
 */
export function withSavedModel(
  view: DesktopConfigurationView,
  choice: AiServiceChoice,
  { id = newModelId(view), use }: { id?: string; use?: "chat" | "reasoning" } = {},
): DesktopConfigurationInput {
  const { model: updated, apiKey } = serviceModel(id, choice);
  const exists = view.models.text.some((model) => model.id === id);
  const text = exists
    ? view.models.text.map((model) => model.id === id ? updated : model)
    : [...view.models.text, updated];
  const saved = withChanges(view, { models: { ...view.models, text } }, { modelApiKeys: { [id]: apiKey } });
  if (configuredModels(view).length === 0) {
    return { ...saved, modelAssignments: { ...view.modelAssignments, chat: id, reasoning: id } };
  }
  return use ? { ...saved, modelAssignments: withAssignedModel(view, use, id).modelAssignments } : saved;
}

function newModelId(view: DesktopConfigurationView): string {
  const taken = new Set(view.models.text.map((model) => model.id));
  let index = view.models.text.length + 1;
  while (taken.has(`text-${index}`)) index += 1;
  return `text-${index}`;
}

/**
 * Chooses 对话大脑's or 推理大脑's model. 推理大脑 left on 对话大脑's model
 * moves with it.
 */
export function withAssignedModel(
  view: DesktopConfigurationView,
  module: "chat" | "reasoning",
  id: string,
): DesktopConfigurationInput {
  const { chat, reasoning } = view.modelAssignments;
  return withChanges(view, {
    modelAssignments: module === "chat"
      ? { ...view.modelAssignments, chat: id, reasoning: reasoning === chat ? id : reasoning }
      : { ...view.modelAssignments, reasoning: id },
  });
}

/**
 * Removes a saved model; whatever used it moves to another saved model.
 * Null when it is the last one, which 对话大脑 cannot do without.
 */
export function withRemovedModel(view: DesktopConfigurationView, id: string): DesktopConfigurationInput | null {
  const others = configuredModels(view).filter((item) => item.model.id !== id);
  if (!others.length) return null;
  const fallback = others[0]!.model.id;
  const chat = view.modelAssignments.chat === id ? fallback : view.modelAssignments.chat;
  const reasoning = view.modelAssignments.reasoning === id ? chat : view.modelAssignments.reasoning;
  return withChanges(view, {
    models: { ...view.models, text: view.models.text.filter((model) => model.id !== id) },
    modelAssignments: { ...view.modelAssignments, chat, reasoning },
  });
}

export const TEXTURE_OPTIONS = [
  { id: "2048", label: "流畅" },
  { id: "4096", label: "清晰" },
  { id: "8192", label: "超清" },
] as const;

type Permissions = Record<SystemPermissionKind, SystemPermissionState> | null;
const usable = (state: SystemPermissionState | undefined): boolean => state === "granted" || state === "not-required";

/** Switched-on features the system has not allowed yet, grouped by the permission they wait for. */
export function blockedFeatures(view: DesktopConfigurationView, permissions: Permissions): {
  screen: string[];
  systemAudio: string[];
  microphone: string[];
} {
  if (!permissions) return { screen: [], systemAudio: [], microphone: [] };
  const listening = view.perception.audioPerceptionEnabled && view.perception.innerLifeEnabled !== false;
  const screenMissing = !usable(permissions.screen);
  // Screen Recording covers computer sound too; its own switch matters only once the screen is allowed.
  const screen = [
    ...(view.perception.screenPerceptionEnabled && screenMissing ? ["屏幕感知"] : []),
    ...(listening && screenMissing ? ["电脑音频感知"] : []),
  ];
  const systemAudio = listening && !screenMissing && !usable(permissions.systemAudio) ? ["电脑音频感知"] : [];
  const microphoneOn = view.microphoneEnabled ?? view.voiceEnabled;
  return { screen, systemAudio, microphone: microphoneOn && !usable(permissions.microphone) ? ["麦克风收音"] : [] };
}

/** Tuning values the 高级 page resets to. Mirrors the desktop defaults. */
export const ADVANCED_DEFAULTS = {
  asrThreads: 2,
  vadThreshold: 0.25,
  vadMinSilenceMs: 350,
  vadMinSpeechMs: 80,
  smartTurnEnabled: true,
  smartTurnThreshold: 0.5,
  voiceDraftEnabled: true,
  diagnosticsEnabled: false,
} as const;
