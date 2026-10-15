import { describe, expect, it } from "vitest";

import {
  changedFields,
  characterSummary,
  daysSince,
  profileInput,
  relationShown,
  relationStored,
  relationText,
  selectedRelation,
  selectedTraits,
  traitsText,
  wakeWordError,
  type CharacterDraft,
} from "./model";

const saved: CharacterDraft = {
  name: "阿澄",
  wakeWords: ["阿澄"],
  background: "住在电脑里",
  role: relationText("工作搭档"),
  setting: traitsText(["温柔", "可靠"]),
};

describe("character page model", () => {
  it("counts changed fields and only sends wake words when they changed", () => {
    const draft = { ...saved, background: "新的背景" };
    expect(changedFields(saved, draft)).toBe(1);
    expect(profileInput(draft, saved)).not.toHaveProperty("wakeWords");
    const withWord = { ...draft, wakeWords: ["阿澄", "小澄"] };
    expect(changedFields(saved, withWord)).toBe(2);
    expect(profileInput(withWord, saved).wakeWords).toEqual(["阿澄", "小澄"]);
  });

  it("keeps wake words short, Chinese, unique and at most four", () => {
    expect(wakeWordError("小澄", ["阿澄"])).toBeNull();
    expect(wakeWordError("Ace", [])).toBe("唤醒词要 2 到 8 个汉字");
    expect(wakeWordError("阿澄", ["阿澄"])).toBe("已经有这个唤醒词了");
    expect(wakeWordError("小澄", ["一一", "二二", "三三", "四四"])).toBe("最多 4 个唤醒词");
  });

  it("reads the chips back from the text they wrote", () => {
    expect(selectedRelation(saved.role)).toBe("工作搭档");
    expect(selectedTraits(saved.setting)).toEqual(["温柔", "可靠"]);
    expect(characterSummary(saved)).toBe("工作搭档 · 温柔 · 可靠");
    expect(selectedRelation("自己写的关系")).toBeNull();
  });

  it("counts the first day as day one", () => {
    const now = new Date(2026, 8, 25, 9, 0);
    expect(daysSince(new Date(2026, 8, 25, 1, 0).toISOString(), now)).toBe(1);
    expect(daysSince(new Date(2026, 8, 24, 23, 0).toISOString(), now)).toBe(2);
    expect(daysSince(new Date(2026, 8, 13, 1, 0).toISOString(), now)).toBe(13);
    expect(daysSince(null, now)).toBeNull();
  });
});

describe("relation wording", () => {
  it("shows the user as 你 and stores it back the model's way", () => {
    expect(relationShown(relationText("朋友"))).toBe("和你是朋友关系");
    expect(relationStored("和你是青梅竹马关系")).toBe("和用户是青梅竹马关系");
    expect(relationStored(relationShown("我们一起长大"))).toBe("我们一起长大");
  });
});
