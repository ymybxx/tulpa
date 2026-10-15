import { describe, expect, it } from "vitest";

import { classifyPart, englishTokens } from "./part-classifier";

const part = (drawableId: string, ...chain: Array<[string, string]>) =>
  classifyPart({ drawableId, chain: chain.map(([id, name]) => ({ id, name })), side: "left" });

describe("classifyPart", () => {
  it("matches English IDs by whole words", () => {
    expect([...englishTokens("PartHairFront")]).toEqual(["part", "hair", "front"]);
    expect(part("ArtMesh1", ["PartHairFront", ""]).label).toBe("刘海与前发");
    expect(part("D_EYE_BALL_00", ["PartEye", ""]).label).toBe("眼珠与高光");
    // "Heart" must not read as "ear", "Pink" not as "ink".
    expect(part("ArtMesh2", ["PartHeart", ""]).group).toBe("unknown");
    expect(part("ArtMesh3", ["PartPinkBow", ""]).label).toBe("蝴蝶结与缎带");
  });

  it("reads Chinese and Japanese names and walks up the Part tree", () => {
    expect(part("ArtMesh9", ["Part24", "左"], ["Part130", "后脑勺"], ["Part3", "头发"]).label).toBe("后发");
    expect(part("ArtMesh5", ["Part8", "发卡"]).label).toBe("发饰");
    expect(part("ArtMesh6", ["Part9", "发光"]).label).toBe("特效");
    expect(part("ArtMesh7", ["Part10", "袖口"]).label).toBe("上衣与外套");
    expect(part("ArtMesh8", ["PartRobe", "ローブ"]).label).toBe("上衣与外套");
    expect(part("ArtMesh4", ["PartRabbit", "うさぎ"], ["PartHat", "帽子"]).label).toBe("帽子");
    expect(part("ArtMesh10", ["Part11", "星星眼"]).label).toBe("眼睛与睫毛");
    // A feathered eye mask is not a wing.
    expect(part("ArtMesh3", ["Part5", "羽翼眼罩"]).label).toBe("眼罩与面饰");
    expect(part("ArtMesh381", ["Part142", "眼部羽翼后"], ["Part145", "部件后"]).label).toBe("眼罩与面饰");
    expect(part("ArtMesh383", ["Part143", "翅膀"]).label).toBe("翅膀");
    // A broad 「身体」 folder far up the tree is not evidence for 身体.
    expect(part("ArtMesh11", ["Part12", "前"], ["Part13", "身体"]).group).toBe("unknown");
    expect(part("ArtMesh12", ["Part14", "身体"]).label).toBe("身体");
  });

  it("prefers the rigger's tag and falls back to the outermost named Part", () => {
    expect(classifyPart({ drawableId: "ArtMesh1", chain: [{ id: "Part1", name: "左" }], tag: "hair", side: "left" }).label).toBe("后发");
    expect(classifyPart({ drawableId: "ArtMesh1", chain: [], tag: "披肩", side: "left" }).label).toBe("上衣与外套");
    expect(classifyPart({ drawableId: "ArtMesh1", chain: [], tag: "小熊挂件", side: "left" }).label).toBe("小熊挂件");
    expect(part("ArtMesh1", ["Part1", "中2"], ["Part2", "水印.png"]).label).toBe("水印.png");
  });
});
