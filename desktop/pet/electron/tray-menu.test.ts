import { describe, expect, it } from "vitest";
import type { SensesSnapshot } from "../src/shared/pet-senses";
import { trayMenu, type TrayItem, type TrayState } from "./tray-menu";

const senses: SensesSnapshot = {
  companionPaused: true,
  senses: [
    { key: "screen", title: "屏幕感知", status: "on", label: "在线", detail: "", on: true, fix: null },
    { key: "audio", title: "电脑音频感知", status: "off", label: "已关闭", detail: "", on: false, fix: null },
    { key: "microphone", title: "麦克风收音", status: "on", label: "在线", detail: "", on: true, fix: null },
    { key: "inner", title: "内心", status: "off", label: "已关闭", detail: "", on: false, fix: null },
  ],
};
const state = (patch: Partial<TrayState> = {}): TrayState => ({
  locked: false, petVisible: true, playback: true, microphone: false, audioSupported: true,
  senses, bubbles: true, looks: [{ id: "mao", name: "Mao" }], lookId: "mao", ...patch,
});
const labels = (items: TrayItem[]) => items.map((item) => "separator" in item ? "—" : `${item.label}${item.checked ? " ✓" : ""}${item.enabled === false ? " (off)" : ""}`);

describe("menu-bar menu", () => {
  it("uses the quick panel's names, one switch per sense and 主动陪伴", () => {
    expect(labels(trayMenu(state()))).toEqual([
      "打开聊天", "打开主窗口", "先藏起来", "—",
      "声音播放 ✓", "麦克风收音", "屏幕感知 ✓", "电脑音频感知", "内心", "主动陪伴", "说话气泡 ✓", "—",
      "换形象", "鼠标穿透", "—", "退出 tulpa",
    ]);
  });

  it("puts the way out of 鼠标穿透 first, and waits for setup before the sense switches", () => {
    const items = labels(trayMenu(state({ locked: true, petVisible: false, senses: null, audioSupported: false })));
    expect(items.slice(0, 2)).toEqual(["关闭鼠标穿透", "—"]);
    expect(items).toContain("叫 TA 出来");
    expect(items).toContain("屏幕感知 (off)");
    expect(items).toContain("主动陪伴 (off)");
    expect(items.filter((item) => item.startsWith("鼠标穿透"))).toEqual([]);
  });

  it("switches the native menu to English while preserving user-created appearance names", () => {
    const items = trayMenu(state({ locked: true, looks: [{ id: "custom", name: "我的角色" }] }), "en");
    expect(labels(items)).toContain("Screen awareness ✓");
    expect(labels(items)).toContain("Disable click-through");
    expect(labels(items)).toContain("Quit tulpa");
    const looks = items.find((item) => "label" in item && item.label === "Change appearance");
    expect(looks && "submenu" in looks && looks.submenu?.[0]).toMatchObject({ label: "我的角色", action: { type: "look", id: "custom" } });
  });
});
