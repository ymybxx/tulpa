import { describe, expect, it } from "vitest";

import {
  chooseTextureEditCanvasPreset,
  DEFAULT_CHARACTER_BUDGET_USD,
  estimateTextureEditUnitUsd,
  estimateTextureEditUsd,
  planTextureEditBatch,
  reconcileCharacterJobEstimate,
  reserveEstimatedCost,
  textureReferenceLabels,
  textureEditBatchPromptRegions,
  textureEditPlanRegions,
  TEXTURE_EDIT_ANCHOR_SIZE,
  transitionCharacterJob,
  validateMasterTemplateManifest,
  type CharacterGenerationJob,
  type MasterTemplateManifest,
} from "./character-generation";

describe("texture image model pricing and canvas selection", () => {
  it("uses the unified planning allowance for current and legacy model inputs", () => {
    expect(estimateTextureEditUsd("1024x1024")).toBe(0.24);
    expect(estimateTextureEditUsd("1536x1024")).toBe(0.2);
    expect(estimateTextureEditUsd("1024x1536")).toBe(0.2);
    expect(estimateTextureEditUsd("1024x1024", "gpt-image-1.5")).toBe(0.24);
    expect(estimateTextureEditUsd("1536x1024", "gpt-image-1.5")).toBe(0.2);
  });

  it("uses the same maximum integer-upscale canvas rule as artifact creation", () => {
    expect(chooseTextureEditCanvasPreset(700, 950)).toMatchObject({
      imageSize: "1024x1024",
      scale: 1,
    });
    expect(chooseTextureEditCanvasPreset(120, 80)).toMatchObject({
      imageSize: "1536x1024",
      scale: 12,
    });
    expect(estimateTextureEditUnitUsd(120, 80)).toBe(0.2);
    expect(estimateTextureEditUnitUsd(120, 80, "gpt-image-1.5")).toBe(0.2);
  });

  it("uses the full edit canvas for a long 1200px UV texture", () => {
    expect(chooseTextureEditCanvasPreset(1200, 206)).toMatchObject({
      imageSize: "1536x1024",
      scale: 1,
      editWidth: 1472,
    });
  });
});

describe("texture batch layout", () => {
  it("plans one deterministic canvas without exceeding any unit's solo scale", () => {
    const sources = [
      { id: "leg", label: "左腿", width: 178, height: 709 },
      { id: "shoe", label: "左鞋", width: 95, height: 169 },
    ];
    const first = planTextureEditBatch(sources);
    const second = planTextureEditBatch([...sources].reverse());
    expect(first.fits).toBe(true);
    expect(second).toEqual(first);
    if (!first.fits) return;
    expect(first.plan.imageModel).toBe("gpt-image-2.5-sunburst");
    expect(first.plan.items.find((item) => item.textureUnitId === "leg")?.scale)
      .toBe(chooseTextureEditCanvasPreset(178, 709).scale);
    // The shoe would get 8× alone; sharing the leg's canvas it is capped, but
    // never below 2× and never on a pricier canvas than the uncapped layout.
    const shoe = first.plan.items.find((item) => item.textureUnitId === "shoe")!.scale;
    expect(shoe).toBeGreaterThanOrEqual(2);
    expect(shoe).toBeLessThanOrEqual(chooseTextureEditCanvasPreset(95, 169).scale);
  });

  it("does not treat presentation-only label changes as a layout change", () => {
    const rendererPlan = planTextureEditBatch([
      { id: "sleeve", label: "右臂与右手 · 纹理 4", width: 146, height: 251 },
      { id: "cuff", label: "右臂与右手 · 纹理 5", width: 244, height: 318 },
    ]);
    const mainPlan = planTextureEditBatch([
      { id: "sleeve", label: "右袖上臂部分", width: 146, height: 251 },
      { id: "cuff", label: "右袖主体", width: 244, height: 318 },
    ]);
    expect(rendererPlan.fits).toBe(true);
    expect(mainPlan.fits).toBe(true);
    if (!rendererPlan.fits || !mainPlan.fits) return;
    expect(rendererPlan.plan.version).toBe(4);
    expect(rendererPlan.plan.hash).toBe(mainPlan.plan.hash);
    expect(rendererPlan.plan.items.map((item) => item.label))
      .not.toEqual(mainPlan.plan.items.map((item) => item.label));
  });

  it("uses the same canvas plan for current and legacy model selections", () => {
    const sources = [
      { id: "a", label: "A", width: 700, height: 950 },
      { id: "b", label: "B", width: 700, height: 950 },
    ];
    const image2 = planTextureEditBatch(sources);
    expect(image2.fits).toBe(true);
    if (image2.fits) {
      expect(image2.plan.imageModel).toBe("gpt-image-2.5-sunburst");
      expect(["1536x1024", "2048x1152", "2048x2048", "2160x3840"]).toContain(image2.plan.imageSize);
    }
    const image15 = planTextureEditBatch(sources, "gpt-image-1.5");
    expect(image15).toEqual(image2);
    expect(planTextureEditBatch(sources, "gpt-image-2")).toEqual(image2);
  });

  it("caps the upscale so a whole part of small pieces still fits one canvas", () => {
    const pieces = Array.from({ length: 9 }, (_, index) => ({ id: `iris-${index}`, label: `虹膜 ${index}`, width: 38, height: 38 }));
    const result = planTextureEditBatch(pieces);
    expect(result.fits).toBe(true);
    if (!result.fits) return;
    const scales = result.plan.items.map((item) => item.scale);
    expect(Math.min(...scales)).toBeGreaterThanOrEqual(1);
    // Small pieces no longer each claim a whole canvas at 25×.
    expect(Math.max(...scales)).toBeLessThan(25);
  });

  it("keeps texture rectangles separated by the configured gutter", () => {
    const result = planTextureEditBatch([
      { id: "a", label: "A", width: 50, height: 200 },
      { id: "b", label: "B", width: 50, height: 200 },
    ]);
    expect(result.fits).toBe(true);
    if (!result.fits) return;
    const [a, b] = result.plan.items;
    expect(a).toBeDefined();
    expect(b).toBeDefined();
    const separated =
      a!.x + a!.workWidth + result.plan.gutter <= b!.x ||
      b!.x + b!.workWidth + result.plan.gutter <= a!.x ||
      a!.y + a!.workHeight + result.plan.gutter <= b!.y ||
      b!.y + b!.workHeight + result.plan.gutter <= a!.y;
    expect(separated).toBe(true);
  });

  it("packs Mao's two hat textures instead of rejecting the 1200px strip", () => {
    const result = planTextureEditBatch([
      { id: "hat-main", label: "帽子 · 纹理 4", width: 554, height: 312 },
      { id: "hat-strip", label: "帽子 · 纹理 3", width: 1200, height: 206 },
    ]);
    expect(result.fits).toBe(true);
    if (!result.fits) return;
    expect(result.plan.imageSize).toBe("1536x1024");
    expect(result.plan.items).toHaveLength(2);
    expect(result.plan.items.find((item) => item.textureUnitId === "hat-main")?.scale).toBe(2);
    expect(result.plan.items.find((item) => item.textureUnitId === "hat-strip")?.scale).toBe(1);
    expect(Math.max(...result.plan.items.map((item) => item.x + item.workWidth)))
      .toBeLessThanOrEqual(result.plan.canvasWidth - result.plan.gutter);
  });

  it("splits a selection that no canvas can hold into several, anchored after the first", () => {
    const pieces = Array.from({ length: 10 }, (_, index) => ({ id: `hair-${index}`, label: `后发 ${index}`, width: 1000, height: 900 }));
    const result = planTextureEditBatch(pieces);
    expect(result.fits).toBe(true);
    if (!result.fits) return;
    expect(result.canvases.length).toBeGreaterThan(1);
    expect(result.plan).toBe(result.canvases[0]);
    expect(result.hash).not.toBe(result.canvases[0]!.hash);
    expect(result.estimatedUsd).toBeCloseTo(result.canvases.reduce((sum, plan) => sum + plan.estimatedUsd, 0));
    // Every piece is drawn exactly once.
    const drawn = result.canvases.flatMap((plan) => plan.items.map((item) => item.textureUnitId)).sort();
    expect(drawn).toEqual(pieces.map((piece) => piece.id).sort());
    // Letters run on across canvases.
    const keys = textureEditPlanRegions(result.canvases).map((region) => region.key);
    expect(new Set(keys).size).toBe(keys.length);
    expect(keys[0]).toBe("A");
    expect(result.canvases[0]!.anchor).toBeUndefined();
    for (const plan of result.canvases.slice(1)) {
      const anchor = plan.anchor!;
      expect(anchor.width).toBe(TEXTURE_EDIT_ANCHOR_SIZE);
      expect(anchor.x + anchor.width).toBeLessThanOrEqual(plan.canvasWidth - plan.gutter);
      expect(anchor.y + anchor.height).toBeLessThanOrEqual(plan.canvasHeight - plan.gutter);
      for (const item of plan.items) {
        const apart = item.x + item.workWidth <= anchor.x || anchor.x + anchor.width <= item.x ||
          item.y + item.workHeight <= anchor.y || anchor.y + anchor.height <= item.y;
        expect(apart).toBe(true);
      }
    }
  });

  it("keeps a selection that fits one canvas exactly as before", () => {
    const result = planTextureEditBatch([{ id: "a", label: "A", width: 200, height: 200 }]);
    expect(result.fits).toBe(true);
    if (!result.fits) return;
    expect(result.canvases).toEqual([result.plan]);
    expect(result.hash).toBe(result.plan.hash);
    expect(result.plan.anchor).toBeUndefined();
    expect(result.plan.regionKeyOffset).toBeUndefined();
  });

  it("refuses a selection that would take more canvases than one edit may use", () => {
    // Too small to be shrunk, too many to share four canvases.
    const pieces = Array.from({ length: 60 }, (_, index) => ({ id: `p-${index}`, label: `P ${index}`, width: 1000, height: 900 }));
    const result = planTextureEditBatch(pieces);
    expect(result.fits).toBe(false);
    if (result.fits) return;
    expect(result.reason).toContain("最多 4 张");
  });

  it("shrinks only the big pieces when their own size would take too many canvases", () => {
    const pieces = [
      ...Array.from({ length: 10 }, (_, index) => ({ id: `big-${index}`, label: `大 ${index}`, width: 2400, height: 1800 })),
      { id: "small", label: "小", width: 300, height: 200 },
    ];
    const result = planTextureEditBatch(pieces);
    expect(result.fits).toBe(true);
    if (!result.fits) return;
    expect(result.canvases.length).toBeLessThanOrEqual(4);
    const items = result.canvases.flatMap((plan) => plan.items);
    for (const item of items.filter((entry) => entry.textureUnitId.startsWith("big-"))) {
      expect(item.scale).toBeLessThan(1);
      expect(item.workWidth).toBe(Math.ceil(item.sourceWidth * item.scale - 1e-9));
    }
    expect(items.find((item) => item.textureUnitId === "small")!.scale).toBeGreaterThanOrEqual(1);
  });

  it("shrinks oversized textures where it can't be seen rather than paying for more canvases", () => {
    // A high-resolution model's skirt: three pieces, each filling a canvas at its own size.
    const result = planTextureEditBatch([
      { id: "front", label: "裙前", width: 3000, height: 2000 },
      { id: "back", label: "裙后", width: 2900, height: 1900 },
      { id: "lining", label: "内衬", width: 2600, height: 1800 },
    ]);
    expect(result.fits).toBe(true);
    if (!result.fits) return;
    expect(result.canvases).toHaveLength(1);
    for (const item of result.plan.items) {
      expect(item.scale).toBe(0.5);
      expect(Math.max(item.workWidth, item.workHeight)).toBeGreaterThanOrEqual(1024);
    }
  });

  it("draws a piece bigger than every canvas at a fraction of its size", () => {
    const result = planTextureEditBatch([{ id: "huge", label: "巨大", width: 5000, height: 4000 }]);
    expect(result.fits).toBe(true);
    if (!result.fits) return;
    expect(result.plan.items[0]!.scale).toBe(0.5);
    expect(result.plan.items[0]!.workWidth).toBe(2500);
    const hopeless = planTextureEditBatch([{ id: "x", label: "X", width: 20000, height: 20000 }]);
    expect(hopeless.fits).toBe(false);
  });

  it("assigns stable visual region keys by canvas position rather than internal id", () => {
    const result = planTextureEditBatch([
      { id: "z", label: "帽檐", width: 1200, height: 206 },
      { id: "a", label: "帽顶", width: 554, height: 312 },
    ]);
    expect(result.fits).toBe(true);
    if (!result.fits) return;
    const regions = textureEditBatchPromptRegions(result.plan);
    expect(regions.map((region) => region.key)).toEqual(["A", "B"]);
    expect(regions[0]!.y < regions[1]!.y || regions[0]!.x <= regions[1]!.x).toBe(true);
  });
});

describe("texture reference labels", () => {
  it("keeps the front view special and numbers only appearance references", () => {
    expect(textureReferenceLabels([
      { source: "model-preview" },
      { source: "texture-unit" },
      { source: "upload" },
    ])).toEqual(["定位图", "R1", "R2"]);
  });
});

const manifest: MasterTemplateManifest = {
  schemaVersion: 2,
  id: "girl-fullbody-v1",
  name: "全身少女母版",
  modelEntry: "girl.model3.json",
  alignmentGuide: "alignment-guide.png",
  coverage: "full-body",
  canvas: { width: 1200, height: 2000, originX: 600, originY: 1000, pixelsPerUnit: 1200 },
  parameters: { eyeBlink: ["ParamEyeLOpen", "ParamEyeROpen"], lipSync: ["ParamMouthOpenY"] },
  textureAtlases: [{ file: "girl.2048/texture_00.png", width: 2048, height: 2048 }],
  slots: [
    {
      id: "head",
      name: "头部",
      drawableId: "D_HEAD",
      partId: "PART_HEAD",
      group: "face",
      source: "completion",
      required: true,
      zIndex: 20,
      atlas: { texture: 0, x: 0, y: 0, width: 1024, height: 1024 },
      anchor: { x: 512, y: 560 },
      canvasBounds: { x: -0.5, y: 0.2, width: 1, height: 0.8 },
      mesh: {
        vertices: [-0.5, 1, 0.5, 1, 0.5, 0.2, -0.5, 0.2],
        uvs: [0, 1, 0.5, 1, 0.5, 0.5, 0, 0.5],
        indices: [0, 1, 2, 0, 2, 3],
      },
      silhouetteMask: "masks/head.png",
      completionPrompt: "完整补全头发后侧",
    },
    {
      id: "left-eye",
      name: "左眼",
      drawableId: "D_EYE_L",
      partId: "PART_EYE",
      group: "face",
      source: "visible",
      required: true,
      zIndex: 30,
      atlas: { texture: 0, x: 64, y: 64, width: 256, height: 128 },
      anchor: { x: 160, y: 112 },
      canvasBounds: { x: -0.2, y: 0.5, width: 0.1, height: 0.05 },
      mesh: {
        vertices: [-0.2, 0.55, -0.1, 0.55, -0.1, 0.5, -0.2, 0.5],
        uvs: [0.03, 0.97, 0.15, 0.97, 0.15, 0.9, 0.03, 0.9],
        indices: [0, 1, 2, 0, 2, 3],
      },
      silhouetteMask: "masks/left-eye.png",
      completionPrompt: "补全眼白和虹膜",
    },
  ],
};

function job(): CharacterGenerationJob {
  return {
    version: 2,
    id: "job-1",
    templateId: manifest.id,
    input: { description: "清纯可爱的黑发少女", referenceImage: null, sourceModelId: null },
    phase: "draft",
    paused: false,
    pauseReason: null,
    revision: 0,
    candidates: [],
    selectedCandidateId: null,
    falRequests: [],
    cache: {},
    artifacts: {
      normalizedImage: null,
      subjectImage: null,
      coarseLayers: [],
      completionSheets: {},
      slotTextures: {},
      repairSlotIds: [],
      requestedCompletions: [],
    },
    budget: {
      limitUsd: DEFAULT_CHARACTER_BUDGET_USD,
      estimatedUsd: 0,
      actualUsd: 0,
      pendingOperation: null,
    },
    issues: [],
    outputModelId: null,
    createdAt: "2026-08-17T00:00:00.000Z",
    updatedAt: "2026-08-17T00:00:00.000Z",
  };
}

describe("master template manifest", () => {
  it("accepts one valid full-body drawable manifest", () => {
    expect(validateMasterTemplateManifest(manifest)).toEqual(manifest);
  });

  it("rejects duplicate slots, unsafe paths, and atlas overflow", () => {
    const invalid = structuredClone(manifest);
    invalid.modelEntry = "../outside.model3.json";
    invalid.slots.push({ ...structuredClone(invalid.slots[0]!), zIndex: 99 });
    invalid.slots[0]!.atlas.x = 1900;

    expect(() => validateMasterTemplateManifest(invalid)).toThrow(/安全相对路径|重复|图集/);
  });

  it("rejects a template id that could escape userData", () => {
    expect(() => validateMasterTemplateManifest({ ...manifest, id: "../girl" })).toThrow(/母版 ID/);
  });
});

describe("character generation job", () => {
  it("only permits the documented phase order", () => {
    const decomposing = transitionCharacterJob(job(), "decomposing");
    const aligning = transitionCharacterJob(decomposing, "aligning");
    const validating = transitionCharacterJob(aligning, "validating");
    const preview = transitionCharacterJob(validating, "preview");
    const saved = transitionCharacterJob(preview, "saved");

    expect(saved.phase).toBe("saved");
    expect(saved.revision).toBe(5);
    expect(() => transitionCharacterJob(job(), "preview")).toThrow(/draft.*preview/);
  });

  it("supports a base preview checkpoint before targeted completion", () => {
    const decomposing = transitionCharacterJob(job(), "decomposing");
    const aligningBase = transitionCharacterJob(decomposing, "aligning");
    const basePreview = transitionCharacterJob(aligningBase, "base-preview");
    const completing = transitionCharacterJob(basePreview, "completing");
    const aligningCompleted = transitionCharacterJob(completing, "aligning");
    const validating = transitionCharacterJob(aligningCompleted, "validating");
    const preview = transitionCharacterJob(validating, "preview");

    expect(preview.phase).toBe("preview");
    expect(preview.revision).toBe(7);
  });

  it("pauses before an operation would exceed the default budget", () => {
    let value = reserveEstimatedCost(job(), "four-candidates", 0.7);
    value = reserveEstimatedCost(value, "layer-head", 0.15);

    expect(value.budget.estimatedUsd).toBe(0.7);
    expect(value.paused).toBe(true);
    expect(value.pauseReason).toBe("budget");
    expect(value.budget.pendingOperation).toEqual({ name: "layer-head", estimatedUsd: 0.15 });
  });

  it("repairs old flat-rate estimates for an eight-layer decomposition", () => {
    const old = job();
    old.budget.estimatedUsd = 0.28;
    old.falRequests = [
      { operation: "generate-candidates", endpoint: "fal-ai/qwen-image-2/text-to-image", requestId: "a", revision: 0 },
      { operation: "validate-candidates", endpoint: "openrouter/router/vision", requestId: "b", revision: 0 },
      { operation: "normalize", endpoint: "fal-ai/qwen-image-2/edit", requestId: "c", revision: 1 },
      { operation: "remove-background", endpoint: "fal-ai/birefnet/v2", requestId: "d", revision: 1 },
      { operation: "coarse-decomposition", endpoint: "fal-ai/qwen-image-layered", requestId: "e", revision: 1 },
      { operation: "material-face", endpoint: "fal-ai/qwen-image-2/edit", requestId: "f", revision: 1 },
      { operation: "in-flight", endpoint: "fal-ai/evf-sam", requestId: "g", revision: 1 },
    ];

    expect(reconcileCharacterJobEstimate(old).budget.estimatedUsd).toBe(0.625);
  });

  it("reconciles pose-controlled candidate batches at the conservative per-image estimate", () => {
    const old = job();
    old.falRequests = [{
      operation: "generate-candidates",
      endpoint: "fal-ai/z-image/turbo/controlnet",
      requestId: "pose-batch",
      revision: 0,
    }];

    expect(reconcileCharacterJobEstimate(old).budget.estimatedUsd).toBe(0.08);
  });

  it("reconciles targeted completion requests after reopening the app", () => {
    const old = job();
    old.budget.estimatedUsd = 0.04;
    old.falRequests = [
      { operation: "normalize", endpoint: "fal-ai/qwen-image-2/edit", requestId: "a", revision: 1 },
      { operation: "remove-background", endpoint: "fal-ai/birefnet/v2", requestId: "b", revision: 1 },
      { operation: "completion-mouth", endpoint: "fal-ai/qwen-image-2/edit", requestId: "c", revision: 2 },
      { operation: "completion-back-hair", endpoint: "fal-ai/qwen-image-2/edit", requestId: "d", revision: 2 },
    ];

    expect(reconcileCharacterJobEstimate(old).budget.estimatedUsd).toBe(0.11);
  });
});
