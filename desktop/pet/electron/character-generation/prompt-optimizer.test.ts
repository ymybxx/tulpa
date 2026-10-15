import { describe, expect, it, vi } from "vitest";

import {
  chatCompletionsUrl,
  DEFAULT_TEXTURE_PROMPT_TIMEOUT_MS,
  TexturePromptOptimizer,
} from "./prompt-optimizer";
import type { TextureReferenceSelection } from "../../src/shared/character-generation";

const visualContext = {
  canvas: { width: 1536, height: 1024 },
  referenceImageDataUrl: "data:image/png;base64,REFERENCE",
  regions: [
    {
      key: "A",
      label: "左袖 · 纹理 3",
      canvasBounds: { x: 16, y: 16, width: 178, height: 709 },
      modelBounds: { x: 0.3, y: 0.3, width: 0.1, height: 0.3 },
      imageDataUrl: "data:image/png;base64,LEFT_SLEEVE",
      modelLocatorImageDataUrl: "data:image/png;base64,LEFT_LOCATOR",
    },
    {
      key: "B",
      label: "右袖 · 纹理 1",
      canvasBounds: { x: 210, y: 16, width: 570, height: 1014 },
      modelBounds: { x: 0.6, y: 0.3, width: 0.1, height: 0.3 },
      imageDataUrl: "data:image/png;base64,RIGHT_SLEEVE",
      modelLocatorImageDataUrl: "data:image/png;base64,RIGHT_LOCATOR",
    },
  ],
} as const;

function optimizerResponse(content: unknown): Response {
  return new Response(JSON.stringify({ choices: [{ message: { content: JSON.stringify(content) } }] }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function request(references: TextureReferenceSelection[] = []) {
  return {
    templateId: "model-edit-mao",
    modelName: "Mao",
    textureUnitIds: ["left-sleeve", "right-sleeve"],
    prompt: "服装改成粉色小熊风格，袖口保留白色，上臂带子保持横向",
    planHash: "texture-batch-v4-test",
    imageModel: "gpt-image-2" as const,
    references,
  };
}

function rawSpec() {
  return {
    intent: "style-transfer",
    styleProfiles: [
      {
        id: "S1",
        scope: "粉色小熊服装风格",
        attributes: {
          palette: {
            description: "低饱和浅粉主色，暖粉阴影和柔和近白高光",
          },
          rendering: {
            description: "干净二维赛璐璐渲染，使用柔和两级阴影",
          },
        },
      },
    ],
    regions: [
      {
        key: "A",
        role: "左上臂袖子",
        styleProfileIds: ["S1"],
        clauseIds: ["C1", "C2", "C3"],
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

describe("TexturePromptOptimizer", () => {
  it("keeps the multimodal timeout and a bounded structured response budget", async () => {
    // A big selection with a thinking model can take well over a minute.
    expect(DEFAULT_TEXTURE_PROMPT_TIMEOUT_MS).toBe(300_000);
    const fetcher = vi.fn(async () => optimizerResponse(rawSpec()));
    await new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "vision-model",
      apiKey: "secret",
      fetcher,
    }).optimize(request(), visualContext);
    const body = JSON.parse(String(fetcher.mock.calls[0]?.[1]?.body)) as { max_tokens: number };
    // Bounded, with room for every region: 2 regions here.
    expect(body.max_tokens).toBe(2_500 + 80 * 2);
  });

  it("asks for source-bounded visual profiles instead of free-form region instructions", async () => {
    const fetcher = vi.fn(async () => optimizerResponse(rawSpec()));
    const result = await new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1/",
      model: "test-model",
      apiKey: "secret-key",
      fetcher,
    }).optimize(request(), visualContext);

    const [url, init] = fetcher.mock.calls[0]!;
    const body = String(init?.body);
    expect(url).toBe("https://example.test/v1/chat/completions");
    expect(new Headers(init?.headers).get("authorization")).toBe("Bearer secret-key");
    expect(body).not.toContain("secret-key");
    expect(body).not.toContain("texture-batch-v4-test");
    expect(body).toContain("styleProfiles");
    expect(body).not.toContain("\\\"sourceIds\\\"");
    expect(body).not.toContain("\\\"evidence\\\"");
    expect(body).not.toContain("不得出现‘改成");
    expect(body).toContain("clauseIds");
    expect(body).toContain("本次没有外部参考图");
    expect(body).not.toContain("\\\"id\\\":\\\"R1\\\"");
    expect(body).not.toContain("observedStructure");
    expect(body).not.toContain("allowedChanges");
    expect(result.spec.styleProfiles).toHaveLength(1);
  });

  it("does not send model locator images when no reference image was selected", async () => {
    const fetcher = vi.fn(async () => optimizerResponse(rawSpec()));
    await new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret-key",
      fetcher,
    }).optimize(request(), visualContext);

    const body = String(fetcher.mock.calls[0]?.[1]?.body);
    expect(body).not.toContain("REFERENCE");
    expect(body).not.toContain("LEFT_LOCATOR");
    expect(body).not.toContain("RIGHT_LOCATOR");
    expect(body).toContain("LEFT_SLEEVE");
    expect(body).toContain("RIGHT_SLEEVE");
    expect(body).not.toContain("\\\"sourceIds\\\"");
    expect(body).not.toContain("不是本次必须采用的来源");
  });

  it("sends one labeled sheet and the annotated body instead of an image per region, with the model's part names", async () => {
    const fetcher = vi.fn(async () => optimizerResponse(rawSpec()));
    await new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret-key",
      fetcher,
    }).optimize(
      { ...request(), parts: { selected: ["袖子"], others: ["靴子", "裙子"] } },
      { ...visualContext, regionSheetDataUrl: "data:image/png;base64,SHEET" },
    );
    const body = String(fetcher.mock.calls.find((call) => String(call[1]?.body).includes("视觉规格分析器"))?.[1]?.body);
    expect(body).toContain("SHEET");
    expect(body).toContain("REFERENCE");
    expect(body).not.toContain("LEFT_SLEEVE");
    expect(body).not.toContain("RIGHT_SLEEVE");
    expect(body.match(/"type":"image_url"/g)).toHaveLength(2);
    expect(body).toContain("模型上其他没选中的部位：靴子、裙子");
  });

  it("allows selected textures to reference each other without external references", async () => {
    const spec = rawSpec();
    const fetcher = vi.fn(async () => optimizerResponse(spec));
    const result = await new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret-key",
      fetcher,
    }).optimize(request(), visualContext);

    const body = String(fetcher.mock.calls[0]?.[1]?.body);
    expect(body).not.toContain("T1=区域 A");
    expect(body).not.toContain("T2=区域 B");
    expect(result.spec.styleProfiles[0]?.attributes.palette).toEqual({
      description: "低饱和浅粉主色，暖粉阴影和柔和近白高光",
    });
  });

  it("uses the same visual-spec shape when external references are present", async () => {
    const spec = rawSpec();
    const fetcher = vi.fn(async () => optimizerResponse(spec));
    const result = await new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret-key",
      fetcher,
    }).optimize(request([{ id: "upload", source: "upload", name: "style.png", description: "", path: "/tmp/style.png", textureUnitId: null, materialId: null, previewUrl: null }]), {
      ...visualContext,
      references: [{ selectionId: "upload", id: "R1", name: "style.png", source: "upload", description: "", imageDataUrl: "data:image/png;base64,STYLE" }],
    });

    expect(result.spec.styleProfiles[0]?.attributes.palette).toEqual({
      description: "低饱和浅粉主色，暖粉阴影和柔和近白高光",
    });
    expect(result.spec.regions[0]?.styleProfileIds).toEqual(["S1"]);
  });

  it("does not expose reference-selection decisions in the visual spec", async () => {
    const spec = rawSpec();
    const fetcher = vi.fn(async () => optimizerResponse(spec));
    const result = await new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret-key",
      fetcher,
    }).optimize(request(), visualContext);

    expect("referenceDecisions" in result).toBe(false);
  });

  it("compiles professional visual language while preserving exact current clauses", async () => {
    const result = await new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret",
      fetcher: async () => optimizerResponse(rawSpec()),
    }).optimize(request(), visualContext);

    expect(result.optimizedPrompt).toContain("视觉方向 S1（粉色小熊服装风格）");
    expect(result.optimizedPrompt).toContain("配色：低饱和浅粉主色");
    expect(result.optimizedPrompt).toContain("C3「上臂带子保持横向」");
    expect(result.optimizedPrompt.match(/技术约束：/g)).toHaveLength(1);
    expect(result.optimizedPrompt).not.toContain("当前固定结构");
    expect(result.optimizedPrompt).not.toContain("允许修改");
    expect(result.optimizedPrompt).not.toContain("禁止修改");
  });

  it("cannot leak free-form historical fields into the compiled prompt", async () => {
    const result = await new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret",
      fetcher: async () => optimizerResponse({
        ...rawSpec(),
        overallInstruction: "去除人物脚踝处的黑袜",
        previousTask: "旧的粉色鞋任务",
      }),
    }).optimize(request(), visualContext);
    expect(result.optimizedPrompt).not.toContain("脚踝");
    expect(result.optimizedPrompt).not.toContain("旧的粉色鞋任务");
  });

  it("keeps the same S specification shape when P0 and an external reference are present", async () => {
    const selections = [
      {
        id: "front", source: "model-preview" as const, name: "Mao 当前正面图", description: "",
        path: null, textureUnitId: null, materialId: null, previewUrl: "data:image/png;base64,FRONT",
      },
      {
        id: "pink", source: "upload" as const, name: "pink.png", description: "参考粉色少女服装",
        path: "/tmp/pink.png", textureUnitId: null, materialId: null, previewUrl: "data:image/png;base64,PINK",
      },
    ];
    const references = [
      { selectionId: "front", id: "P0", name: selections[0]!.name, source: selections[0]!.source, description: "", imageDataUrl: "data:image/png;base64,FRONT" },
      { selectionId: "pink", id: "R1", name: selections[1]!.name, source: selections[1]!.source, description: selections[1]!.description, imageDataUrl: "data:image/png;base64,PINK" },
    ];
    const spec = rawSpec();
    const fetcher = vi.fn(async () => optimizerResponse(spec));

    const result = await new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret",
      fetcher,
    }).optimize({ ...request(), references: selections }, { ...visualContext, references });

    const body = String(fetcher.mock.calls[0]?.[1]?.body);
    expect(body).not.toContain("\\\"sourceIds\\\"");
    expect(body).toContain("有无参考图必须返回完全相同的 JSON 结构");
    expect(result.optimizedPrompt).toContain("本次用户选择的参考图只用于补充视觉方向");
  });

  it("leaves a region missing from the answer unchanged instead of applying the request to it", async () => {
    const incomplete = rawSpec();
    incomplete.regions = incomplete.regions.slice(0, 1);
    const optimizer = new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret",
      fetcher: async () => optimizerResponse(incomplete),
    });
    const result = await optimizer.optimize(request(), visualContext);
    expect(result.spec.regions.find((region) => region.key === "B")?.keep).toBe("AI 没给这块写要求，先保持原样");
    expect(result.optimizedPrompt).toContain("只修改 A；");
  });

  it("reports malformed JSON and timeouts clearly", async () => {
    const invalid = new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret",
      fetcher: async () => new Response(JSON.stringify({ choices: [{ message: { content: "not-json" } }] })),
    });
    await expect(invalid.optimize(request(), visualContext)).rejects.toThrow(/JSON 无效/);

    const timeout = new TexturePromptOptimizer({
      baseUrl: "https://example.test/v1",
      model: "test-model",
      apiKey: "secret",
      timeoutMs: 1,
      fetcher: async (_url, init) => new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
      }),
    });
    await expect(timeout.optimize(request(), visualContext)).rejects.toThrow(/超时/);
  });
});

describe("chatCompletionsUrl", () => {
  it("normalizes compatible base URLs", () => {
    expect(chatCompletionsUrl("https://example.test/v1/"))
      .toBe("https://example.test/v1/chat/completions");
    expect(chatCompletionsUrl("https://example.test/v1/chat/completions"))
      .toBe("https://example.test/v1/chat/completions");
  });
});

describe("pieces the request leaves alone", () => {
  function answering(pieces: unknown) {
    return vi.fn(async (_url: string, init?: RequestInit) => String(init?.body).includes("确认一次 Live2D 模型纹理重画的范围")
      ? optimizerResponse({ pieces })
      : optimizerResponse(rawSpec()));
  }
  const sheetContext = { ...visualContext, regionSheetDataUrl: "data:image/png;base64,SHEET" };

  it("asks about every selected piece on its own and keeps the ones it says are not covered", async () => {
    const fetcher = answering([
      { key: "A", what: "左袖", change: true, reason: "" },
      { key: "B", what: "右手皮肤", change: false, reason: "要求只说服装，这是皮肤" },
    ]);
    const result = await new TexturePromptOptimizer({ baseUrl: "https://example.test/v1", model: "test-model", apiKey: "k", fetcher })
      .optimize(request(), sheetContext);
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(result.spec.regions.find((region) => region.key === "B")).toMatchObject({ keep: "要求只说服装，这是皮肤", styleProfileIds: [], clauseIds: [] });
    expect(result.spec.regions.find((region) => region.key === "A")?.keep).toBeUndefined();
    expect(result.optimizedPrompt).toContain("保持原样，不要改动");
    expect(result.optimizedPrompt).toContain("只修改 A；");
  });

  it("also reads an answer that is the bare list of pieces", async () => {
    const fetcher = vi.fn(async (_url: string, init?: RequestInit) => String(init?.body).includes("确认一次 Live2D 模型纹理重画的范围")
      ? optimizerResponse([{ key: "B", change: false, reason: "这次可以跳过" }])
      : optimizerResponse(rawSpec()));
    const result = await new TexturePromptOptimizer({ baseUrl: "https://example.test/v1", model: "test-model", apiKey: "k", fetcher })
      .optimize(request(), sheetContext);
    expect(result.spec.regions.find((region) => region.key === "B")?.keep).toBe("这次可以跳过");
  });

  it("reads the answer from the reasoning when a thinking model leaves the reply empty", async () => {
    const fetcher = vi.fn(async (_url: string, init?: RequestInit) => String(init?.body).includes("确认一次 Live2D 模型纹理重画的范围")
      ? new Response(JSON.stringify({ choices: [{ message: { content: "{}", reasoning_content: `想了一下。${JSON.stringify({ pieces: [{ key: "B", what: "皮肤", reason: "这次只换衣服", change: false }] })}` } }] }), { status: 200 })
      : optimizerResponse(rawSpec()));
    const result = await new TexturePromptOptimizer({ baseUrl: "https://example.test/v1", model: "test-model", apiKey: "k", fetcher })
      .optimize(request(), sheetContext);
    expect(result.spec.regions.find((region) => region.key === "B")).toMatchObject({ keep: "这次只换衣服", keepLabel: "皮肤" });
  });

  it("names every piece by what it shows in the image prompt", async () => {
    const fetcher = answering([
      { key: "A", what: "左袖上的金色刺绣", reason: "衣服，要换", change: true },
      { key: "B", what: "右手皮肤", reason: "这次只换衣服", change: false },
    ]);
    const result = await new TexturePromptOptimizer({ baseUrl: "https://example.test/v1", model: "test-model", apiKey: "k", fetcher })
      .optimize(request(), sheetContext);
    expect(result.optimizedPrompt).toContain("- A（左袖上的金色刺绣）：应用 S1");
    expect(result.optimizedPrompt).toContain("- B（右手皮肤）：保持原样，不要改动。");
  });

  it("shows the check each piece beside where it sits on the model, without the colour-coded body", async () => {
    const fetcher = answering([]);
    await new TexturePromptOptimizer({ baseUrl: "https://example.test/v1", model: "test-model", apiKey: "k", fetcher })
      .optimize(request(), { ...sheetContext, pieceSheetDataUrls: ["data:image/png;base64,PAIRED"] });
    const body = String(fetcher.mock.calls.find((call) => String(call[1]?.body).includes("确认一次 Live2D 模型纹理重画的范围"))?.[1]?.body);
    expect(body).toContain("PAIRED");
    expect(body).not.toContain("REFERENCE");
    expect(body).not.toContain("SHEET");
    expect(body).toContain("每格左边是这块纹理本身，右边是它在人物身上的位置");
  });

  it("reads an answer with a raw line break inside a reason", async () => {
    const fetcher = vi.fn(async (_url: string, init?: RequestInit) => String(init?.body).includes("确认一次 Live2D 模型纹理重画的范围")
      ? new Response(JSON.stringify({ choices: [{ message: { content: '{"pieces":[{"key":"B","what":"皮肤","reason":"这次\n只换衣服","change":false}]}' } }] }), { status: 200 })
      : optimizerResponse(rawSpec()));
    const result = await new TexturePromptOptimizer({ baseUrl: "https://example.test/v1", model: "test-model", apiKey: "k", fetcher })
      .optimize(request(), sheetContext);
    expect(result.spec.regions.find((region) => region.key === "B")?.keep).toBe("这次 只换衣服");
  });

  it("draws every piece when that question fails or answers nothing usable", async () => {
    const fetcher = vi.fn(async (_url: string, init?: RequestInit) => String(init?.body).includes("确认一次 Live2D 模型纹理重画的范围")
      ? new Response("busy", { status: 503 })
      : optimizerResponse(rawSpec()));
    const result = await new TexturePromptOptimizer({ baseUrl: "https://example.test/v1", model: "test-model", apiKey: "k", fetcher })
      .optimize(request(), sheetContext);
    expect(result.spec.regions.every((region) => !region.keep)).toBe(true);

    const unknown = await new TexturePromptOptimizer({ baseUrl: "https://example.test/v1", model: "test-model", apiKey: "k", fetcher: answering([{ key: "Z", change: false }]) })
      .optimize(request(), sheetContext);
    expect(unknown.spec.regions.every((region) => !region.keep)).toBe(true);
  });

  it("does not ask when the pieces are sent one image each", async () => {
    const fetcher = answering([]);
    await new TexturePromptOptimizer({ baseUrl: "https://example.test/v1", model: "test-model", apiKey: "k", fetcher })
      .optimize(request(), visualContext);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});
