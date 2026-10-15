import { describe, expect, it } from "vitest";

import { memoryText } from "./memory-wording";

describe("memory wording", () => {
  it("speaks to the user and names the character", () => {
    expect(memoryText("用户认为任何公司都骗不了那都", "小星")).toBe("你认为任何公司都骗不了那都");
    expect(memoryText("助手于2026年9月28日发现Claude界面弹出新弹窗，用户称该弹窗相关", "小星"))
      .toBe("小星于2026年9月28日发现Claude界面弹出新弹窗，你称该弹窗相关");
    expect(memoryText("该助手的名字为小星，该用户喜欢猫", "小星")).toBe("小星的名字为小星，你喜欢猫");
  });
});
