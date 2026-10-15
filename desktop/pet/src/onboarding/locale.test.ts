import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { changeLocale, initializeLocale } from "../i18n";
import { messageText, type UiMessage } from "../i18n/message";
import type { DesktopUiLocale } from "../shared/contracts";
import { characterSummary } from "../character/model";
import { dayLabel, originLabel, stateLabel } from "../chat/model";
import { inputReceiptText } from "../plugins/manual-inputs";
import { pendingCharacter, toggleTrait, traitBlock, traitNote } from "./flow";

let publish: ((locale: DesktopUiLocale) => void) | undefined;
beforeAll(async () => {
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    desktopPet: {
      getUiLocale: async () => "zh-CN",
      onUiLocale: (listener: (locale: DesktopUiLocale) => void) => { publish = listener; return () => {}; },
      setUiLocale: async (locale: DesktopUiLocale) => { publish?.(locale); },
    },
  });
  await initializeLocale();
});
afterAll(async () => {
  await changeLocale("zh-CN");
  vi.unstubAllGlobals();
});

describe("primary pages locale boundaries", () => {
  it("rerenders saved notices in the new language while preserving interpolation data", async () => {
    const stored: UiMessage = { key: "已导入「{{p0}}」。表情会在接上 AI 服务后自动匹配。", values: { p0: "我的角色English" } };
    await changeLocale("en");
    expect(messageText(stored)).toBe("Imported “我的角色English”. Expressions will be matched automatically once an AI service is connected.");
    expect(messageText("给 TA 起个名字吧")).toBe("Give your companion a name.");
    await changeLocale("zh-CN");
    expect(messageText(stored)).toBe("已导入「我的角色English」。表情会在接上 AI 服务后自动匹配。");
    expect(messageText("给 TA 起个名字吧")).toBe("给 TA 起个名字吧");
    expect(stored.values?.p0).toBe("我的角色English");
  });
  it("translates labels and dates while keeping character payloads stable", async () => {
    const choice = { name: "小星", relation: "朋友", traits: ["温柔", "元气"], customSetting: "" };
    const original = pendingCharacter(choice, "model-1");
    await changeLocale("en");
    expect(stateLabel("listening")).toBe("Listening to you");
    expect(originLabel({ kind: "self" }, "小星")).toBe("小星 wanted to share");
    expect(dayLabel(new Date("2026-09-26T10:12:00"), new Date("2026-09-26T11:00:00"))).toBe("Today 10:12");
    expect(characterSummary(original)).toBe("Friend · Gentle · Energetic");
    expect(pendingCharacter(choice, "model-1")).toEqual(original);
    expect(pendingCharacter({ ...choice, customSetting: "这是用户自己的描述" }, null).setting).toBe("这是用户自己的描述");
    await changeLocale("zh-CN");
    expect(stateLabel("listening")).toBe("在听你说");
  });

  it("keeps conflicting trait selection correct in both languages", async () => {
    await changeLocale("en");
    expect(traitBlock(["温柔"], "毒舌")).toContain("conflict");
    expect(traitNote(["温柔"], ["温柔", "毒舌"])).toContain("Sharp-tongued");
    expect(toggleTrait(["温柔"], "毒舌")).toEqual(["温柔"]);
    await changeLocale("zh-CN");
    expect(traitNote(["温柔"], ["温柔", "毒舌"])).toContain("不太搭");
  });

  it("translates plugin receipts without changing their disposition", async () => {
    const result = { disposition: "queued", reason: "no_playback_consumer" };
    await changeLocale("en");
    expect(inputReceiptText(result)).toBe("Queued until this session connects to voice playback.");
    expect(result).toEqual({ disposition: "queued", reason: "no_playback_consumer" });
    await changeLocale("zh-CN");
    expect(inputReceiptText(result)).toBe("已排队，等待当前会话连接语音伴侣。");
  });
});
