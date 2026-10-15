import type { DesktopPerceptionConfiguration, SystemPermissionKind, SystemPermissionState } from "./contracts";
import type { SettingsSection } from "../settings-next/model";

const SETTINGS_SECTIONS = new Set<string>(["ai", "jev", "companion", "usage", "extensions", "permissions", "general", "advanced"]);

/**
 * The pet's senses, named as in 设置 › 陪伴: 屏幕感知, 电脑音频感知, 麦克风收音
 * (perceiving the world) and 内心 (digesting it: 影音总结 and 内心思考).
 */
export type SenseKey = "screen" | "audio" | "microphone" | "inner";
export type SenseStatus = "on" | "off" | "noperm" | "waiting" | "partial";

export interface SenseView {
  key: SenseKey;
  title: string;
  status: SenseStatus;
  label: string;
  detail: string;
  on: boolean;
  /** The system permission that would bring this sense online. */
  fix: SystemPermissionKind | null;
}

export interface SensesSnapshot {
  senses: SenseView[];
  companionPaused: boolean;
  /** TA is asleep with 麦克风收音 on: only this word gets TA listening. */
  sleeping?: { wakeWord: string };
  /** Enough sentences went unrecognized that the owner could add them to the voiceprint. */
  ownerVoiceMissed?: boolean;
  /** Problems only the user can fix (a key, a quota), until that service works again. */
  problems?: SenseProblem[];
  /** Today's tokens, and what is still using the model in the background. */
  usage?: { today: string; background: string[] };
}

export interface SenseProblem { text: string; section: SettingsSection }

export type SensePanel = "closed" | "open" | "confirm-pause";

export type SenseAction =
  | { type: "panel"; panel: SensePanel }
  | { type: "toggle"; key: SenseKey }
  | { type: "pause"; closeSenses: boolean }
  | { type: "resume" }
  | { type: "fix"; kind: SystemPermissionKind }
  | { type: "settings"; section?: SettingsSection }
  | { type: "retry-models" }
  | { type: "owner-voice-add" };

export interface SenseInputs {
  perception: DesktopPerceptionConfiguration;
  microphoneEnabled: boolean;
  proactiveCompanionEnabled: boolean;
  audioSupported: boolean;
  permissions: Record<SystemPermissionKind, SystemPermissionState>;
  /** The permission the user was just sent to System Settings for. */
  waiting: SystemPermissionKind | null;
}

const LABELS: Record<SenseStatus, string> = {
  on: "在线", off: "已关闭", noperm: "没有权限", waiting: "等你打开开关", partial: "部分在线",
};

const usable = (state: SystemPermissionState): boolean => state === "granted" || state === "not-required";

export function senseViews(input: SenseInputs): SenseView[] {
  const { perception, permissions, waiting } = input;
  const inner = perception.innerLifeEnabled !== false;
  const missing = (kind: SystemPermissionKind): SenseStatus | null =>
    usable(permissions[kind]) ? null : waiting === kind ? "waiting" : "noperm";
  const view = (
    key: SenseKey, title: string, on: boolean, permission: SystemPermissionKind | null, detail: string,
  ): SenseView => {
    const blocked = on && permission ? missing(permission) : null;
    const status: SenseStatus = !on ? "off" : blocked ?? "on";
    return { key, title, status, label: LABELS[status], on, detail, fix: blocked ? permission : null };
  };

  const screen = view("screen", "屏幕感知", perception.screenPerceptionEnabled, "screen", "看得到你在屏幕上做什么");
  // Screen Recording covers computer sound too; its own switch matters only once the screen is allowed.
  const audioPermission: SystemPermissionKind = usable(permissions.screen) ? "systemAudio" : "screen";
  const audio = view("audio", "电脑音频感知", perception.audioPerceptionEnabled && input.audioSupported, audioPermission,
    input.audioSupported ? "听得到视频、音乐里的对白和声音" : "这台电脑的系统版本还不支持");
  // Computer sound only feeds 影音总结, so it waits for 内心.
  if (audio.status === "on" && !inner) {
    Object.assign(audio, { status: "partial", label: "要打开内心", detail: "听到的声音要靠内心来整理，内心关着时 TA 听了也记不住" });
  }
  const microphone = view("microphone", "麦克风收音", input.microphoneEnabled, "microphone", "喊名字就能和 TA 说话");
  const innerView = view("inner", "内心", inner, null, "整理看到听到的，闲下来自己想想");
  return [screen, audio, microphone, innerView];
}

export function sensesSnapshot(input: SenseInputs): SensesSnapshot {
  return { senses: senseViews(input), companionPaused: !input.proactiveCompanionEnabled };
}

export interface SenseChange {
  perception: DesktopPerceptionConfiguration;
  microphoneEnabled: boolean;
  proactiveCompanionEnabled: boolean;
}

/** What a capsule action writes back to the configuration. */
export function applySenseAction(
  current: SenseChange,
  action: Extract<SenseAction, { type: "toggle" | "pause" | "resume" }>,
  audioSupported: boolean,
): SenseChange {
  const perception = { ...current.perception };
  let { microphoneEnabled, proactiveCompanionEnabled } = current;
  if (action.type === "toggle") {
    if (action.key === "screen") perception.screenPerceptionEnabled = !perception.screenPerceptionEnabled;
    if (action.key === "audio") perception.audioPerceptionEnabled = !perception.audioPerceptionEnabled && audioSupported;
    if (action.key === "microphone") microphoneEnabled = !microphoneEnabled;
    if (action.key === "inner") perception.innerLifeEnabled = perception.innerLifeEnabled === false;
  } else if (action.type === "pause") {
    proactiveCompanionEnabled = false;
    if (action.closeSenses) {
      perception.screenPerceptionEnabled = false;
      perception.audioPerceptionEnabled = false;
      perception.innerLifeEnabled = false;
      microphoneEnabled = false;
    }
  } else {
    proactiveCompanionEnabled = true;
  }
  return { perception, microphoneEnabled, proactiveCompanionEnabled };
}

export function parseSenseAction(value: unknown): SenseAction | null {
  if (typeof value !== "object" || value === null) return null;
  const action = value as Record<string, unknown>;
  switch (action.type) {
    case "panel":
      return action.panel === "closed" || action.panel === "open" || action.panel === "confirm-pause"
        ? { type: "panel", panel: action.panel } : null;
    case "toggle":
      return action.key === "screen" || action.key === "audio" || action.key === "microphone" || action.key === "inner"
        ? { type: "toggle", key: action.key } : null;
    case "pause":
      return typeof action.closeSenses === "boolean" ? { type: "pause", closeSenses: action.closeSenses } : null;
    case "resume":
      return { type: "resume" };
    case "fix":
      return action.kind === "microphone" || action.kind === "screen" || action.kind === "systemAudio"
        ? { type: "fix", kind: action.kind } : null;
    case "settings":
      return typeof action.section === "string" && SETTINGS_SECTIONS.has(action.section)
        ? { type: "settings", section: action.section as SettingsSection } : { type: "settings" };
    case "retry-models":
      return { type: "retry-models" };
    case "owner-voice-add":
      return { type: "owner-voice-add" };
    default:
      return null;
  }
}
