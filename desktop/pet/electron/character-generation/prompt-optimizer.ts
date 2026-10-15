import { loggedFetch } from "../llm-logs";
import type {
  TexturePromptOptimizationRequest,
  TexturePromptOptimizationResult,
  TextureVisualEditSpec,
  TextureReferencePromptId,
  TextureReferenceSource,
} from "../../src/shared/character-generation";
import {
  compileTextureVisualPrompt,
  splitTexturePromptClauses,
  validateTextureVisualEditSpec,
  type TextureVisualEditSpecInput,
} from "./visual-edit-spec";

type Fetcher = (input: string, init?: RequestInit) => Promise<Response>;

interface ChatCompletionResponse {
  choices?: Array<{ message?: { content?: unknown; reasoning_content?: unknown } }>;
}

export const DEFAULT_TEXTURE_PROMPT_TIMEOUT_MS = 300_000;
const SCREEN_GRACE_MS = 30_000;

export interface TexturePromptVisualRegion {
  key: string;
  label: string;
  canvasBounds: { x: number; y: number; width: number; height: number };
  modelBounds: { x: number; y: number; width: number; height: number };
  /** Only read when there is no region sheet. */
  imageDataUrl?: string;
  modelLocatorImageDataUrl?: string;
}

export interface TexturePromptVisualContext {
  canvas: { width: number; height: number };
  referenceImageDataUrl: string;
  /**
   * Every selected piece on one sheet, each labeled with its region letter.
   * When given it replaces the one-image-per-region list: one image reads in
   * seconds, dozens of them take over a minute.
   */
  regionSheetDataUrl?: string;
  /** The same, split over several sheets when there are many pieces. */
  regionSheetDataUrls?: readonly string[];
  /** Each piece beside where it sits on the model (the piece in colour, the rest dimmed), for the per-piece check. */
  pieceSheetDataUrls?: readonly string[];
  regions: readonly TexturePromptVisualRegion[];
  references?: ReadonlyArray<{
    selectionId?: string;
    id: string;
    name: string;
    source: TextureReferenceSource;
    description: string;
    imageDataUrl: string;
  }>;
}

export class TexturePromptOptimizer {
  constructor(
    private readonly configuration: {
      baseUrl: string;
      model: string;
      apiKey: string;
      fetcher?: Fetcher;
      timeoutMs?: number;
    },
  ) {}

  async optimize(
    request: TexturePromptOptimizationRequest,
    context: TexturePromptVisualContext,
  ): Promise<TexturePromptOptimizationResult> {
    const originalPrompt = request.prompt.trim();
    if (!originalPrompt) throw new Error("请输入需要优化的修改要求");
    if (context.regions.length === 0) throw new Error("没有可优化的纹理选择");
    if ((request.references?.length ?? 0) !== (context.references?.length ?? 0)) {
      throw new Error("参考图列表已经变化，请重新确认后再优化");
    }

    const compilerInput = visualSpecInput(request, context, originalPrompt);
    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(new DOMException("提示词优化超时", "TimeoutError")),
      this.configuration.timeoutMs ?? DEFAULT_TEXTURE_PROMPT_TIMEOUT_MS,
    );
    // Asked alongside, as its own small question: a whole group may be selected
    // while the request covers only some of its pieces. Without an answer, every piece is drawn.
    const screening = this.screenPieces(originalPrompt, context, controller.signal).catch(() => new Map<string, ScreenedPiece>());
    try {
      const response = await loggedFetch("desktop.texture", this.configuration.fetcher ?? fetch)(
        chatCompletionsUrl(this.configuration.baseUrl),
        {
          method: "POST",
          headers: {
            authorization: `Bearer ${this.configuration.apiKey}`,
            "content-type": "application/json",
          },
          body: JSON.stringify({
            model: this.configuration.model,
            stream: false,
            temperature: 0.1,
            // Every region and every sentence gets an entry; many pieces need room.
            max_tokens: 2_500 + 80 * context.regions.length,
            response_format: { type: "json_object" },
            messages: [
              { role: "system", content: optimizerSystemPrompt(compilerInput) },
              { role: "user", content: visualMessageContent(request, context, compilerInput) },
            ],
          }),
          signal: controller.signal,
        },
      );
      if (!response.ok) throw new Error(`提示词优化失败 (${response.status})`);
      const payload = await response.json() as ChatCompletionResponse;
      const content = payload.choices?.[0]?.message?.content;
      if (typeof content !== "string") throw new Error("LLM 没有返回可用的视觉规格");
      const rawSpec = parseVisualSpecJson(content);
      const validated = validateTextureVisualEditSpec(rawSpec, compilerInput);
      // The understanding is the answer; the screening may add to it but never holds it up for long.
      let grace: ReturnType<typeof setTimeout> | undefined;
      const kept = await Promise.race([
        screening,
        new Promise<Map<string, ScreenedPiece>>((resolve) => { grace = setTimeout(() => resolve(new Map()), SCREEN_GRACE_MS); }),
      ]);
      clearTimeout(grace);
      const spec = keepPieces(validated, kept);
      return {
        originalPrompt,
        optimizedPrompt: compileTextureVisualPrompt(spec, compilerInput),
        spec,
        regions: spec.regions,
        warnings: spec.warnings,
      };
    } catch (error) {
      if (controller.signal.aborted) throw new Error("视觉提示词优化超时，请重试");
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  /** Which selected pieces the request leaves as they are, with the reason for each. */
  private async screenPieces(
    prompt: string,
    context: TexturePromptVisualContext,
    signal: AbortSignal,
  ): Promise<Map<string, ScreenedPiece>> {
    const paired = context.pieceSheetDataUrls?.length ? context.pieceSheetDataUrls : null;
    const sheets = paired ?? regionSheets(context);
    if (sheets.length === 0 || context.regions.length < 2) return new Map();
    const keys = new Set(context.regions.map((region) => region.key.toUpperCase()));
    const response = await loggedFetch("desktop.texture", this.configuration.fetcher ?? fetch)(
      chatCompletionsUrl(this.configuration.baseUrl),
      {
        method: "POST",
        headers: {
          authorization: `Bearer ${this.configuration.apiKey}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: this.configuration.model,
          stream: false,
          temperature: 0.1,
          max_tokens: 1_500 + 90 * context.regions.length,
          response_format: { type: "json_object" },
          messages: [
            { role: "system", content: SCREEN_SYSTEM_PROMPT },
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: `用户要求：${prompt}\n选中的纹理块：${[...keys].join("、")}\n` + (paired
                    ? "下面把这些纹理块排在灰底上，每格下方是字母。每格左边是这块纹理本身，右边是它在人物身上的位置：这块保持原色，其余部分压暗。"
                    : `第一张是人物全身图，彩色框和字母标出每块在身上的位置；后面${sheets.length > 1 ? ` ${sheets.length} 张把这些纹理块分批` : "一张把这些纹理块"}排在灰底上，每块下方的字母对应。`),
                },
                ...(paired ? [] : [imageContent(context.referenceImageDataUrl)]),
                ...sheets.map(imageContent),
                { type: "text", text: "只返回 JSON。" },
              ],
            },
          ],
        }),
        signal,
      },
    );
    if (!response.ok) return new Map();
    const payload = await response.json() as ChatCompletionResponse;
    const message = payload.choices?.[0]?.message;
    // Some answers come back as the bare list instead of { pieces: [...] }, and a thinking
    // model sometimes leaves the reply empty with the whole answer in its reasoning.
    let pieces = typeof message?.content === "string" ? parseScreeningPieces(message.content) : [];
    if (!pieces.length && typeof message?.reasoning_content === "string") pieces = parseScreeningPieces(message.reasoning_content);
    const screened = new Map<string, ScreenedPiece>();
    for (const piece of pieces) {
      if (!piece || typeof piece !== "object") continue;
      const { key, what, change, reason } = piece as { key?: unknown; what?: unknown; change?: unknown; reason?: unknown };
      if (typeof key !== "string" || !keys.has(key.toUpperCase())) continue;
      screened.set(key.toUpperCase(), {
        what: typeof what === "string" ? what.trim().slice(0, 40) : "",
        skip: change === false,
        reason: typeof reason === "string" && reason.trim() ? reason.trim().slice(0, 200) : "不在这次要求的范围里",
      });
    }
    return screened;
  }
}

const SCREEN_SYSTEM_PROMPT = [
  "你在帮用户确认一次 Live2D 模型纹理重画的范围。下面的纹理块都是这个 Live2D 模型的图层贴图，重画后会贴回模型上的原位置。",
  "用户可能整组选了部位，其中有些图层这次其实可以跳过。逐块看它实际画的是什么、在 Live2D 模型里起什么作用（比如是独立的可见部件，还是叠在别的图层上的阴影、高光、光效），再结合用户这次的要求判断。",
  "每块先写 what（它实际画的是什么），再写 reason（结合用户这次的要求，用一句简短中文说明这块需不需要重画、为什么），最后才写 change：需要重画为 true，这次可以跳过为 false。拿不准时 change 为 true。",
  "每个选中的块恰好返回一项，不要遗漏或新增。",
  `返回 JSON：${JSON.stringify({ pieces: [{ key: "A", what: "这块实际画的是什么", reason: "结合要求，这块需不需要重画、为什么", change: true }] })}`,
].join("\n");

interface ScreenedPiece { what: string; skip: boolean; reason: string }

/**
 * Each piece is named by what it shows, so the image model knows a bow from a
 * lapel instead of seeing one part-group name for every piece; pieces the
 * request leaves alone are drawn unchanged and left off.
 */
function keepPieces(spec: TextureVisualEditSpec, screened: ReadonlyMap<string, ScreenedPiece>): TextureVisualEditSpec {
  if (screened.size === 0) return spec;
  return {
    ...spec,
    regions: spec.regions.map((region) => {
      const piece = screened.get(region.key);
      if (!piece) return region;
      const named = piece.what ? { ...region, role: piece.what } : region;
      if (!piece.skip) return named;
      const { continuityGroup: _group, ...rest } = named;
      return { ...rest, styleProfileIds: [], clauseIds: [], keep: piece.reason, ...(piece.what ? { keepLabel: piece.what.slice(0, 24) } : {}) };
    }),
  };
}

function parseScreeningPieces(value: string): unknown[] {
  // The JSON may sit inside other text (reasoning); take the outermost object or list.
  // A raw line break inside a string makes the whole answer unreadable; as plain spaces it parses.
  const text = value.trim().replace(/^```(?:json)?\s*/iu, "").replace(/\s*```$/u, "").trim().replace(/[\u0000-\u001f]+/gu, " ");
  const start = text.search(/[[{]/u);
  const end = Math.max(text.lastIndexOf("}"), text.lastIndexOf("]"));
  try {
    const parsed = JSON.parse(start >= 0 && end > start ? text.slice(start, end + 1) : text) as unknown;
    if (Array.isArray(parsed)) return parsed;
    const pieces = (parsed as { pieces?: unknown } | null)?.pieces;
    return Array.isArray(pieces) ? pieces : [];
  } catch {
    return [];
  }
}

function regionSheets(context: TexturePromptVisualContext): readonly string[] {
  return context.regionSheetDataUrls?.length
    ? context.regionSheetDataUrls
    : context.regionSheetDataUrl ? [context.regionSheetDataUrl] : [];
}

export function chatCompletionsUrl(baseUrl: string): string {
  const url = new URL(baseUrl.trim());
  url.pathname = url.pathname.replace(/\/+$/, "");
  if (!url.pathname.endsWith("/chat/completions")) url.pathname += "/chat/completions";
  url.search = "";
  url.hash = "";
  return url.toString();
}

function optimizerSystemPrompt(input: TextureVisualEditSpecInput): string {
  return [
    "你是 Live2D UV 纹理视觉规格分析器，不直接写最终生图提示词。",
    "你可以把用户不专业的外观描述扩展成准确的视觉语言，但只能使用 palette、rendering、shading、linework、material、motif、tone 七类属性。",
    "只为用户要求改变的类别写属性；用户没提到的类别不要写，它们会自动保持原纹理的样子。不要为了完整而补齐类别。",
    "intent 是整次修改的类型，只能取：recolor（只换颜色）、style-transfer（换成另一种风格的颜色和花纹）、material-transfer（换成另一种材质质感）、cleanup（清理瑕疵、不改风格）、alpha-remove（用户明确要求删掉某处、露出透明）。它和下面 actions 里的 action 是两组不同的取值。",
    "actions 为每个用户要求子句各返回一项，action 取值：repaint（重画选中纹理的颜色、花纹或材质）、recolor（只是换成另一种颜色，不改花纹材质）、toggle（整体去掉或显示某个部件，比如去掉靴子、摘掉帽子，这要用开关而不是重画）、unsupported（改形状、长短、增加新部件，只重画颜色做不到）、unselected（说的是没被选中的部位）。part 写这句说的部位名；note 用一句简短中文告诉用户会怎么处理或为什么做不到。",
    "视觉属性的 description 用简洁、准确的文字描述目标视觉结果；具体区域通过 styleProfileIds 和当前 clauseIds 关联。",
    "每个视觉属性只包含 description；不要附加来源、证据或参考元数据。",
    input.references.length > 0
      ? "本次有用户选择的参考图；可以直接比较参考图与选中纹理，归纳到 S 视觉方向中。参考图的发送由用户选择和发送层管理。"
      : "本次没有外部参考图；可以直接比较多个选中纹理的现有外观，归纳到同一个 S 视觉方向中。",
    "有无参考图必须返回完全相同的 JSON 结构；参考图只改变可观察的视觉输入，不改变输出协议。",
    "选中区域之间需要统一风格时，让它们共同引用同一个 S 视觉方向；不要为图片、纹理或来源创建编号。",
    "不得新增用户未要求的口袋、纽扣、腰带、蝴蝶结、徽章、发饰、鞋型、发型、服装结构、文字、Logo、删除或透明变化。",
    "当前用户要求已拆成 C1/C2。结构、方向和保留要求只能通过 clauseIds 引用原句，不得在自由文本中改写。",
    "所有已选 A/B/C 区域必须恰好返回一次，不得遗漏或新增区域。role 只写准确部位名；styleProfileIds 引用 S1/S2；continuityGroup 只用于确实需要一致的成组部件。",
    "styleProfiles 中的每个属性都必须能从当前用户要求或当前可见纹理、参考图获得，不能使用历史任务、常识补全或审美臆测。",
    "只返回 JSON，不要 Markdown，不要最终提示词，不要任务哈希、排版编号、当前固定结构、允许修改、禁止修改或全局技术约束。",
    `JSON 字段示例（只演示结构，不得照抄内容）：${JSON.stringify({
      intent: "recolor",
      styleProfiles: [{
        id: "S1",
        scope: "用户要求的目标风格",
        attributes: {
          palette: {
            description: "目标主色、阴影与高光关系",
          },
        },
      }],
      regions: [{
        key: "A",
        role: "图中实际部位",
        styleProfileIds: ["S1"],
        clauseIds: ["C1"],
      }],
      warnings: [],
      actions: [{ clauseId: "C1", action: "repaint", part: "部位名", note: "给用户的一句说明" }],
    })}。`,
  ].join("\n");
}

function visualSpecInput(
  request: TexturePromptOptimizationRequest,
  context: TexturePromptVisualContext,
  originalPrompt: string,
): TextureVisualEditSpecInput {
  return {
    originalPrompt,
    clauses: splitTexturePromptClauses(originalPrompt),
    imageModel: request.imageModel,
    positionReferenceAvailable: (context.references ?? [])
      .some((reference) => reference.source === "model-preview"),
    regions: context.regions.map((region) => ({ key: region.key, label: region.label })),
    references: (context.references ?? []).map((reference) => ({
        selectionId: reference.selectionId ?? reference.id,
        id: reference.source === "model-preview" ? "P0" : reference.id as TextureReferencePromptId,
        source: reference.source,
        name: reference.name,
        description: reference.description,
      })),
  };
}

function parseVisualSpecJson(value: string): unknown {
  const cleaned = value.trim()
    .replace(/^```(?:json)?\s*/iu, "")
    .replace(/\s*```$/u, "")
    .trim();
  try {
    const parsed = JSON.parse(cleaned) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("invalid response");
    return parsed;
  } catch {
    throw new Error("LLM 返回的视觉规格 JSON 无效，可以保留原文继续生成");
  }
}

function visualMessageContent(
  request: TexturePromptOptimizationRequest,
  context: TexturePromptVisualContext,
  input: TextureVisualEditSpecInput,
): Array<Record<string, unknown>> {
  const hasReferenceImages = (context.references?.length ?? 0) > 0;
  const content: Array<Record<string, unknown>> = [{
    type: "text",
    text: [
      `人物模型：${request.modelName}`,
      `当前用户要求：${input.originalPrompt}`,
      "当前用户要求子句：",
      ...input.clauses.map((clause) => `${clause.id}：${clause.text}`),
      ...(request.parts
        ? [
          `选中的部位：${request.parts.selected.join("、") || "无"}`,
          `模型上其他没选中的部位：${request.parts.others.join("、") || "无"}`,
        ]
        : []),
      `联合画布尺寸：${context.canvas.width}×${context.canvas.height}`,
      ...(hasReferenceImages
        ? [
          "下面第一张是 P0 当前模型定位图。默认只用于识别区域部位、左右侧和相邻关系，不提供外观参考。",
          input.positionReferenceAvailable
            ? "P0 已由用户加入最终参考图；只有当前要求明确授权时才可声明局部外观用途。"
            : "P0 不会发送给最终图片模型，因此绝对不能声明为外观来源。",
        ]
        : ["本次未选择参考图，不发送模型定位图；只依据选中的 UV 纹理和当前用户要求进行分析。"]),
    ].join("\n"),
  }];
  if (hasReferenceImages) content.push(imageContent(context.referenceImageDataUrl));

  const sheets = regionSheets(context);
  if (sheets.length) {
    content.push({
      type: "text",
      text: `选中的区域：${context.regions.map((region) => `${region.key}（${region.label}）`).join("、")}。` +
        `${hasReferenceImages ? "" : "下一张是人物全身图，彩色框和字母标出每个区域在身上的位置；"}` +
        `${sheets.length > 1 ? `再后面 ${sheets.length} 张把选中纹理分批排在灰底上` : "再下一张把所有选中纹理排在灰底上"}，每块下方的字母与区域对应。直接比较它们的视觉关系，统一风格归纳到 S 视觉方向。`,
    });
    if (!hasReferenceImages) content.push(imageContent(context.referenceImageDataUrl));
    for (const sheet of sheets) content.push(imageContent(sheet));
  } else {
    for (const region of context.regions) {
      content.push({
        type: "text",
        text: `区域 ${region.key} 已由用户选中，素材标签为“${region.label}”。${hasReferenceImages ? "下一张是人物定位高亮图，再下一张是透明 UV 纹理。" : "下面是透明 UV 纹理。"}直接比较当前选中纹理的视觉关系，并将统一风格归纳到 S 视觉方向。`,
      });
      if (hasReferenceImages && region.modelLocatorImageDataUrl) content.push(imageContent(region.modelLocatorImageDataUrl));
      if (region.imageDataUrl) content.push(imageContent(region.imageDataUrl));
    }
  }

  const references = context.references ?? [];
  if (references.length > 0) {
    content.push({ type: "text", text: "下面是本次独立外观参考图，编号与最终图片请求一致。" });
    for (const reference of references) {
      const note = reference.description.trim();
      const line = reference.source === "model-preview"
        ? "P0：用户选择的当前人物正面图，可作为外观参考；不要复制人物身份、姿态、比例或部件轮廓。"
        : reference.source === "texture-unit"
          ? `${reference.id}：模型素材“${reference.name}”。`
          : `${reference.id}：用户上传图片“${reference.name}”${note ? `；用户说明：${note}` : ""}。`;
      content.push({ type: "text", text: line }, imageContent(reference.imageDataUrl));
    }
  }
  content.push({
    type: "text",
    text: "只返回约定的视觉规格 JSON。每个当前区域恰好一项；专业扩展视觉语言，但不新增结构和用户未要求的设计内容。",
  });
  return content;
}

function imageContent(dataUrl: string): Record<string, unknown> {
  return { type: "image_url", image_url: { url: dataUrl, detail: "high" } };
}
