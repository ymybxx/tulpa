import type { SenseKey, SensesSnapshot } from "../src/shared/pet-senses";
import type { DesktopUiLocale } from "../src/shared/contracts";

/**
 * The menu-bar icon's menu: the same names and grouping as the pet's
 * right-click quick panel, so 声音播放 or 屏幕感知 means one thing everywhere.
 */
export type TrayAction =
  | { type: "chat" }
  | { type: "main" }
  | { type: "pet" }
  | { type: "speaker" }
  | { type: "microphone" }
  | { type: "sense"; key: Exclude<SenseKey, "microphone"> }
  | { type: "companion" }
  | { type: "bubbles" }
  | { type: "look"; id: string }
  | { type: "passthrough" }
  | { type: "quit" };

export type TrayItem =
  | { separator: true }
  | { label: string; action?: TrayAction; checked?: boolean; radio?: boolean; enabled?: boolean; submenu?: TrayItem[] };

export interface TrayState {
  /** 鼠标穿透 is on: the pet ignores clicks, so the way back leads the menu. */
  locked: boolean;
  petVisible: boolean;
  playback: boolean;
  microphone: boolean;
  /** Older macOS can't record computer sound. */
  audioSupported: boolean;
  /** Null before setup is finished; the sense switches wait for it. */
  senses: SensesSnapshot | null;
  bubbles: boolean | null;
  looks: Array<{ id: string; name: string }>;
  lookId: string | null;
}

const SEPARATOR = { separator: true } as const;
const SENSES: Array<Exclude<SenseKey, "microphone">> = ["screen", "audio", "inner"];

export function trayMenu(state: TrayState, locale: DesktopUiLocale = "zh-CN"): TrayItem[] {
  const t = (zh: string, en: string): string => locale === "en" ? en : zh;
  const sense = (key: Exclude<SenseKey, "microphone">): TrayItem => {
    const view = state.senses?.senses.find((item) => item.key === key);
    return {
      label: { screen: t("屏幕感知", "Screen awareness"), audio: t("电脑音频感知", "Computer audio awareness"), inner: t("内心", "Inner life") }[key],
      action: { type: "sense", key },
      checked: Boolean(view?.on),
      enabled: Boolean(view) && (key !== "audio" || state.audioSupported),
    };
  };
  return [
    ...(state.locked ? [{ label: t("关闭鼠标穿透", "Disable click-through"), action: { type: "passthrough" } } as TrayItem, SEPARATOR] : []),
    { label: t("打开聊天", "Open chat"), action: { type: "chat" } },
    { label: t("打开主窗口", "Open main window"), action: { type: "main" } },
    { label: state.petVisible ? t("先藏起来", "Hide companion") : t("叫 TA 出来", "Show companion"), action: { type: "pet" } },
    SEPARATOR,
    { label: t("声音播放", "Voice playback"), action: { type: "speaker" }, checked: state.playback },
    { label: t("麦克风收音", "Microphone"), action: { type: "microphone" }, checked: state.microphone },
    ...SENSES.map(sense),
    { label: t("主动陪伴", "Proactive companion"), action: { type: "companion" }, checked: Boolean(state.senses && !state.senses.companionPaused), enabled: Boolean(state.senses) },
    { label: t("说话气泡", "Speech bubbles"), action: { type: "bubbles" }, checked: state.bubbles ?? true, enabled: state.bubbles !== null },
    SEPARATOR,
    {
      label: t("换形象", "Change appearance"),
      enabled: state.looks.length > 0,
      submenu: state.looks.map((look) => ({ label: look.name, action: { type: "look", id: look.id }, radio: true, checked: look.id === state.lookId })),
    },
    ...(state.locked ? [] : [{ label: t("鼠标穿透", "Click-through"), action: { type: "passthrough" }, checked: false } as TrayItem]),
    SEPARATOR,
    { label: t("退出 tulpa", "Quit tulpa"), action: { type: "quit" } },
  ];
}
