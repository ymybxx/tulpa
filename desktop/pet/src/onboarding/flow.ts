import { getLocale, t } from "../i18n";
import type {
  DesktopConfigurationInput,
  DesktopConfigurationView,
  PendingCharacter,
  SystemPermissionKind,
} from "../shared/contracts";

export type OnboardingStep = "look" | "soul" | "brain" | "companion" | "permissions";
export type CompanionLevel = "text" | "voice" | "full";
export type AiProviderId = "qwen" | "doubao" | "custom";

export interface AiProvider {
  id: AiProviderId;
  label: string;
  hint: string;
  baseUrl: string;
  model: string;
  keyUrl: string | null;
  keySite: string | null;
  recommended?: boolean;
  /** A step some services need before the key works, e.g. switching the model on. */
  activate?: string;
}

export const AI_PROVIDERS: readonly AiProvider[] = [
  {
    id: "qwen", label: "千问", hint: "反应快，日常聊天够用",
    baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1", model: "qwen3.7-flash",
    keyUrl: "https://bailian.console.aliyun.com/", keySite: "阿里云百炼", recommended: true,
  },
  {
    id: "doubao", label: "豆包", hint: "更会演角色",
    baseUrl: "https://ark.cn-beijing.volces.com/api/v3", model: "doubao-seed-character-260628",
    keyUrl: "https://console.volcengine.com/ark", keySite: "火山方舟",
    activate: "在左侧「开通管理」里找到 Doubao-Seed-Character，点「开通」",
  },
  { id: "custom", label: "其他", hint: "自己填服务地址", baseUrl: "", model: "", keyUrl: null, keySite: null },
];

export interface CompanionOption {
  id: CompanionLevel;
  title: string;
  description: string;
  permissions: readonly SystemPermissionKind[];
  recommended?: boolean;
}

export const COMPANION_OPTIONS: readonly CompanionOption[] = [
  { id: "text", title: "打字聊天", description: "点 TA 打开聊天窗口，打字交流", permissions: [] },
  { id: "voice", title: "语音聊天", description: "打字之外，还能喊名字和 TA 说话，随时打断 TA", permissions: ["microphone"] },
  {
    id: "full", title: "主动陪伴", description: "打字、语音都能用；TA 还会看看你在忙什么，懂你的心情，主动找你聊",
    permissions: ["microphone", "screen"], recommended: true,
  },
];

export const PERMISSION_LABELS: Record<SystemPermissionKind, { title: string; why: string }> = {
  microphone: { title: "麦克风", why: "听见你喊 TA、和 TA 说话" },
  // One switch in System Settings covers both: it records "屏幕内容和音频".
  screen: { title: "屏幕与系统录音", why: "看屏幕内容，一起听电脑里在放的视频、音乐" },
  systemAudio: { title: "电脑声音", why: "一起听电脑里在放的视频、音乐" },
};

export const RELATIONS = ["朋友", "家人", "恋人", "工作搭档", "宠物"] as const;
export const TRAITS = ["温柔", "元气", "毒舌", "傲娇", "安静", "话痨", "可靠", "天然呆"] as const;
export const MAX_TRAITS = 3;

/** Steps in order; permissions only appear when the companion level needs them. */
export function onboardingSteps(level: CompanionLevel, permissionsNeeded: boolean): OnboardingStep[] {
  const steps: OnboardingStep[] = ["look", "soul", "brain", "companion"];
  if (permissionsNeeded && COMPANION_OPTIONS.find((option) => option.id === level)!.permissions.length > 0) {
    steps.push("permissions");
  }
  return steps;
}

/** Traits that pull a character in opposite directions; only one of each pair fits. */
const TRAIT_CONFLICTS: readonly (readonly [string, string])[] = [["安静", "话痨"], ["安静", "元气"], ["温柔", "毒舌"]];

/** The chosen traits that rule `trait` out. */
export function traitConflicts(traits: readonly string[], trait: string): string[] {
  return TRAIT_CONFLICTS.flatMap(([a, b]) => (trait === a ? [b] : trait === b ? [a] : []))
    .filter((value) => traits.includes(value));
}

function traitLabel(value: string): string {
  return TRAITS.some((trait) => trait === value) ? t(value) : value;
}

/** Why `trait` cannot be picked right now, or null when it can. */
export function traitBlock(traits: readonly string[], trait: string): string | null {
  if (traits.includes(trait)) return null;
  const conflict = traitConflicts(traits, trait)[0];
  if (conflict) return t("「{{first}}」和「{{second}}」不太搭，只能选一个", { first: traitLabel(conflict), second: traitLabel(trait) });
  if (traits.length >= MAX_TRAITS) return t("已经选了 {{max}} 个，想换就先点掉一个", { max: MAX_TRAITS });
  return null;
}

/** One line under the trait chips: why some are greyed out, or how many to pick. */
export function traitNote(traits: readonly string[], options: readonly string[]): string {
  if (traits.length >= MAX_TRAITS) return t("已选 {{max}} 个，想换就先点掉一个", { max: MAX_TRAITS });
  // Every greyed-out trait is named, not just the first.
  const clashing = options.filter((value) => traitConflicts(traits, value).length > 0 && !traits.includes(value));
  if (clashing.length) {
    const pairs = clashing.map((value) => t("「{{first}}」和「{{second}}」", { first: traitLabel(value), second: traitLabel(traitConflicts(traits, value)[0]!) })).join(getLocale() === "en" ? "; " : "、");
    return t("{{pairs}}不太搭，想选就先点掉后面那个", { pairs });
  }
  return t("选 1 到 {{max}} 个", { max: MAX_TRAITS });
}

/** Picks or unpicks a trait; a blocked pick changes nothing (the chip says why). */
export function toggleTrait(traits: readonly string[], trait: string): string[] {
  if (traits.includes(trait)) return traits.filter((value) => value !== trait);
  if (traitBlock(traits, trait)) return [...traits];
  return [...traits, trait];
}

export interface SoulChoice {
  name: string;
  relation: string;
  traits: readonly string[];
  /** Free text written by the user; replaces the generated setting when present. */
  customSetting: string;
}

export function describeSetting(choice: Pick<SoulChoice, "traits" | "customSetting">): string {
  const custom = choice.customSetting.trim();
  if (custom) return custom;
  return traitsSetting(choice.traits);
}

/** A 话痨 is not asked to keep it short; everyone else is. */
export function traitsSetting(traits: readonly string[]): string {
  const named = traits.length ? traits.join("、") : "平和";
  return `性格${named}，说话${traits.includes("话痨") ? "自然" : "自然、简短"}。`;
}

export function pendingCharacter(choice: SoulChoice, modelId: string | null): PendingCharacter {
  const relation = choice.relation.trim();
  return {
    name: choice.name.trim(),
    role: relation ? `和用户是${relation}关系` : "",
    setting: describeSetting(choice),
    modelId,
  };
}

/** Chinese names work with the pinyin keyword spotter; anything else keeps the old wake word. */
export function wakeWordFor(name: string, fallback: string): string {
  const trimmed = name.trim();
  return /^[一-鿿]{2,8}$/u.test(trimmed) ? trimmed : fallback;
}

export interface AiServiceChoice {
  provider: AiProvider;
  baseUrl: string;
  model: string;
  apiKey: string;
}

export function aiServiceRequest(choice: AiServiceChoice): { baseUrl: string; model: string; apiKey: string } {
  const custom = choice.provider.id === "custom";
  return {
    baseUrl: (custom ? choice.baseUrl : choice.provider.baseUrl).trim(),
    model: (custom ? choice.model : choice.provider.model).trim(),
    apiKey: choice.apiKey.trim(),
  };
}

const TEXT_MODEL_ID = "text-default";

/**
 * Turns onboarding answers into a complete configuration: one AI service used
 * for both chat and reasoning, and the senses implied by the companion level.
 */
export function onboardingConfiguration(
  current: DesktopConfigurationView,
  answers: {
    service: AiServiceChoice; level: CompanionLevel; name: string; audioSupported: boolean;
    /** Optional decision model: a 百炼业务空间 ID (千问's key serves it) or a Jev key; without one the reasoning model judges. */
    jevApiKey?: string;
    bailianWorkspaceId?: string;
  },
): DesktopConfigurationInput {
  const { secrets: _secrets, ...base } = current;
  const service = aiServiceRequest(answers.service);
  const voiceModels = base.models.voice.length
    ? base.models.voice
    : [{ id: "voice-default", name: "默认语音服务", provider: "auto" as const }];
  const assigned = base.modelAssignments.speech;
  const speechId = assigned && voiceModels.some((model) => model.id === assigned) ? assigned : voiceModels[0]!.id;
  const speaking = answers.level !== "text";
  const watching = answers.level === "full";
  const workspace = watching && answers.service.provider.id === "qwen" ? answers.bailianWorkspaceId?.trim() : "";
  const jevKey = watching && !workspace ? answers.jevApiKey?.trim() : "";
  const companion = base.experimentalCompanion ?? { model: "jev-1.13.0", wakeSeconds: 5 };
  return {
    ...base,
    configured: true,
    ...workspace ? { experimentalCompanion: { ...companion, decisionProvider: "bailian" as const, bailianWorkspaceId: workspace } }
      : jevKey ? { experimentalCompanion: { ...companion, decisionProvider: "jev" as const } } : {},
    models: {
      text: [{
        id: TEXT_MODEL_ID,
        name: answers.service.provider.id === "custom" ? "AI 服务" : answers.service.provider.label,
        baseUrl: service.baseUrl,
        model: service.model,
      }],
      voice: voiceModels,
    },
    modelAssignments: {
      chat: TEXT_MODEL_ID,
      reasoning: TEXT_MODEL_ID,
      speech: speechId,
    },
    voiceEnabled: speaking,
    microphoneEnabled: speaking,
    playbackEnabled: speaking,
    proactiveCompanionEnabled: watching,
    perception: {
      ...base.perception,
      operationPerceptionEnabled: false,
      screenPerceptionEnabled: watching,
      audioPerceptionEnabled: watching && answers.audioSupported,
    },
    audio: { ...base.audio, wakeWord: wakeWordFor(answers.name, base.audio.wakeWord) },
    secrets: {
      // A 千问 (阿里云百炼) key also runs 通义 speech, so TA gets a cloud voice (小新) right away.
      modelApiKeys: {
        [TEXT_MODEL_ID]: service.apiKey,
        ...(answers.service.provider.id === "qwen" ? { [speechId]: service.apiKey } : {}),
      },
      ...(jevKey ? { jevApiKey: jevKey } : {}),
    },
  };
}

/** Drops Electron's "Error invoking remote method '…': Error: " wrapper from IPC failures. */
export function readableError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.replace(/^Error invoking remote method '[^']*': (?:\w*Error: )?/u, "");
}
