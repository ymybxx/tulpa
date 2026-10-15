import { describe, expect, it } from "vitest";

import {
  compileTextureVisualPrompt,
  splitTexturePromptClauses,
  validateTextureVisualEditSpec,
} from "./visual-edit-spec";

const originalPrompt = "服装改成参考图中的粉色小熊风格，袖口保留白色，上臂带子保持横向";

function input() {
  return {
    originalPrompt,
    clauses: splitTexturePromptClauses(originalPrompt),
    imageModel: "gpt-image-2" as const,
    positionReferenceAvailable: false,
    regions: [
      { key: "A", label: "左上臂袖子" },
      { key: "B", label: "右上臂袖子" },
    ],
    references: [
      {
        selectionId: "pink",
        id: "R1",
        source: "upload" as const,
        name: "粉色少女.png",
        description: "参考衣服的配色和小熊图案",
      },
    ],
  };
}

function validRawSpec() {
  return {
    intent: "style-transfer",
    styleProfiles: [
      {
        id: "S1",
        scope: "粉色小熊服装风格",
        attributes: {
          palette: {
            description: "低饱和浅粉主色，暖粉阴影和接近白色的柔和高光",
          },
          rendering: {
            description: "干净的二维赛璐璐渲染，保持柔和两级阴影",
          },
          motif: {
            description: "小尺寸、低密度的小熊与爱心重复图案",
          },
        },
      },
    ],
    regions: [
      {
        key: "A",
        role: "左上臂袖子",
        styleProfileIds: ["S1"],
        clauseIds: ["C1", "C3"],
        continuityGroup: "sleeves",
      },
      {
        key: "B",
        role: "右上臂袖子",
        styleProfileIds: ["S1"],
        clauseIds: ["C1", "C2", "C3"],
        continuityGroup: "sleeves",
      },
    ],
    warnings: [],
  };
}

describe("splitTexturePromptClauses", () => {
  it("preserves exact current-prompt clauses for structural requirements", () => {
    expect(splitTexturePromptClauses(originalPrompt)).toEqual([
      { id: "C1", text: "服装改成参考图中的粉色小熊风格" },
      { id: "C2", text: "袖口保留白色" },
      { id: "C3", text: "上臂带子保持横向" },
    ]);
  });
});

describe("validateTextureVisualEditSpec", () => {
  it("accepts source-bounded visual expansion and exact clause mapping", () => {
    const spec = validateTextureVisualEditSpec(validRawSpec(), input());
    expect(spec.regions[0]?.clauseIds).toEqual(["C1", "C3"]);
  });

  it("leaves a selected region the answer skipped as it is, instead of inventing a change for it", () => {
    const raw = validRawSpec();
    raw.regions = raw.regions.slice(0, 1);
    const spec = validateTextureVisualEditSpec(raw, input());
    expect(spec.regions[1]).toEqual({ key: "B", role: "右上臂袖子", styleProfileIds: [], clauseIds: [], keep: "AI 没给这块写要求，先保持原样" });
    expect(compileTextureVisualPrompt(spec, input())).toContain("只修改 A；");
  });

  it("uses the same spec shape with or without selected references", () => {
    const withoutReferences = validateTextureVisualEditSpec(validRawSpec(), {
      ...input(),
      references: [],
    });
    const withReferences = validateTextureVisualEditSpec(validRawSpec(), input());
    expect(withoutReferences.styleProfiles).toEqual(withReferences.styleProfiles);
    expect(withoutReferences.regions).toEqual(withReferences.regions);
  });

  it("accepts edit wording inside linework descriptions", () => {
    const raw = validRawSpec();
    (raw.styleProfiles[0]!.attributes as Record<string, { description: string }>).linework = {
      description: "将原有粗重线稿替换为纤细、均匀的深色轮廓线",
    };

    expect(validateTextureVisualEditSpec(raw, input()).styleProfiles[0]?.attributes.linework)
      .toEqual({ description: "将原有粗重线稿替换为纤细、均匀的深色轮廓线" });
  });
});

describe("compileTextureVisualPrompt", () => {
  it("produces a concise deterministic prompt with one global constraint block", () => {
    const compilerInput = input();
    const spec = validateTextureVisualEditSpec(validRawSpec(), compilerInput);
    const first = compileTextureVisualPrompt(spec, compilerInput);
    const second = compileTextureVisualPrompt(spec, compilerInput);

    expect(second).toBe(first);
    expect(first).toContain("视觉方向 S1（粉色小熊服装风格）");
    expect(first).toContain("配色：低饱和浅粉主色");
    expect(first).toContain("C3「上臂带子保持横向」");
    expect(first).toContain("A、B 属于同一视觉连续组");
    expect(first.match(/技术约束：/g)).toHaveLength(1);
    expect(first).not.toContain("当前固定结构");
    expect(first).not.toContain("允许修改");
    expect(first).not.toContain("禁止修改");
  });

  it("ignores unrelated free-form fields so historical prompts cannot enter output", () => {
    const raw = {
      ...validRawSpec(),
      overallInstruction: "去掉以前任务中的黑袜",
      historicalPrompt: "旧的粉色鞋任务",
    };
    const compilerInput = input();
    const prompt = compileTextureVisualPrompt(
      validateTextureVisualEditSpec(raw, compilerInput),
      compilerInput,
    );
    expect(prompt).not.toContain("黑袜");
    expect(prompt).not.toContain("旧的粉色鞋任务");
  });

  it("sends only the sentences that are drawn and keeps what the request doesn't touch", () => {
    const raw = {
      ...validRawSpec(),
      actions: [
        { clauseId: "C1", action: "repaint", part: "袖子", note: "重画袖子" },
        { clauseId: "c2", action: "toggle", part: "袖口", note: "去掉袖口要在开关里关" },
        { clauseId: "C3", action: "teleport", note: "未知动作按重画处理" },
        { clauseId: "C9", action: "unsupported", note: "不存在的句子被丢弃" },
      ],
    };
    const spec = validateTextureVisualEditSpec(raw, input());
    expect(spec.actions?.map((plan) => [plan.clauseId, plan.action])).toEqual([["C1", "repaint"], ["C2", "toggle"], ["C3", "repaint"]]);
    const prompt = compileTextureVisualPrompt(spec, input());
    expect(prompt).toContain("用户要求：服装改成参考图中的粉色小熊风格；上臂带子保持横向");
    expect(prompt).not.toContain("袖口保留白色");
    // palette, rendering and motif are set; everything else stays as drawn.
    expect(prompt).toContain("保持原样：光影、线稿、材质、整体气质与原纹理一致");
  });
});


describe("pieces the request leaves alone", () => {
  function withKept() {
    const raw = validRawSpec();
    raw.regions[1] = { key: "B", role: "右手皮肤", styleProfileIds: [], clauseIds: [], keep: "要求只说服装，这是手的皮肤" } as never;
    return raw;
  }

  it("accepts a kept piece without any change and keeps the reason", () => {
    const spec = validateTextureVisualEditSpec(withKept(), input());
    expect(spec.regions[1]).toEqual({ key: "B", role: "右手皮肤", styleProfileIds: [], clauseIds: [], keep: "要求只说服装，这是手的皮肤" });
  });

  it("treats a piece the answer gives nothing to do as left alone, instead of failing", () => {
    const raw = validRawSpec();
    raw.regions[1] = { key: "B", role: "右手皮肤", styleProfileIds: [], clauseIds: [] } as never;
    expect(validateTextureVisualEditSpec(raw, input()).regions[1]).toMatchObject({ key: "B", keep: "这次的要求没有涉及这块" });
  });

  it("tells the image model to leave a kept piece as it is and edit only the rest", () => {
    const spec = validateTextureVisualEditSpec(withKept(), input());
    const prompt = compileTextureVisualPrompt(spec, input());
    expect(prompt).toContain("- B（右手皮肤）：保持原样，不要改动。");
    expect(prompt).toContain("只修改 A；");
  });
});

describe("what a style change may redesign", () => {
  it("keeps the untouched categories without forbidding a redesign when the style changes", () => {
    const prompt = compileTextureVisualPrompt(validateTextureVisualEditSpec(validRawSpec(), input()), input());
    expect(prompt).toContain("与原纹理一致。");
    expect(prompt).not.toContain("不要重新设计");
  });

  it("still forbids a redesign for a plain recolor", () => {
    const raw = { ...validRawSpec(), intent: "recolor" };
    const prompt = compileTextureVisualPrompt(validateTextureVisualEditSpec(raw, input()), input());
    expect(prompt).toContain("与原纹理一致，不要重新设计。");
  });
});
