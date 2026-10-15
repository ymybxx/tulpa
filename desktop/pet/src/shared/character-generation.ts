export const CHARACTER_GENERATION_PHASES = [
  "draft",
  "decomposing",
  "aligning",
  "base-preview",
  "completing",
  "validating",
  "preview",
  "saved",
  "failed",
  "cancelled",
] as const;

export type CharacterGenerationPhase = (typeof CHARACTER_GENERATION_PHASES)[number];

export const CHARACTER_COMPLETION_KINDS = [
  "eyes",
  "mouth",
  "face-base",
  "back-hair",
  "body-hidden",
  "clothing-hidden",
] as const;

export type CharacterCompletionKind = (typeof CHARACTER_COMPLETION_KINDS)[number];

export const DEFAULT_CHARACTER_BUDGET_USD = 0.8;

export const TEXTURE_EDIT_IMAGE_MODELS = ["gpt-image-2.5-sunburst"] as const;
// Legacy names remain readable in saved jobs and older renderer requests.
// All new planning and provider submissions resolve them to the current model.
export type TextureEditImageModel = (typeof TEXTURE_EDIT_IMAGE_MODELS)[number] | "gpt-image-2" | "gpt-image-1.5";
export const DEFAULT_TEXTURE_EDIT_IMAGE_MODEL = TEXTURE_EDIT_IMAGE_MODELS[0];

export function resolveTextureEditImageModel(value?: unknown): typeof DEFAULT_TEXTURE_EDIT_IMAGE_MODEL {
  if (value === undefined || value === null || value === DEFAULT_TEXTURE_EDIT_IMAGE_MODEL ||
      value === "gpt-image-2" || value === "gpt-image-1.5") {
    return DEFAULT_TEXTURE_EDIT_IMAGE_MODEL;
  }
  throw new Error("不支持的纹理图片模型");
}

export function textureEditAlphaPolicy(spec?: TextureVisualEditSpec | null): "locked" | "shrink-only" {
  // Accepted appearance-only specifications can retain exact original alpha.
  // Plain-language requests may remove pixels, subject to prompt and UV limits.
  return spec && spec.intent !== "alpha-remove" ? "locked" : "shrink-only";
}

export type TextureEditImageSize =
  | "1024x1024"
  | "1536x1024"
  | "1024x1536"
  | "2048x1152"
  | "2048x2048"
  | "3840x2160"
  | "2160x3840";

interface TextureEditCanvasPreset {
  imageSize: TextureEditImageSize;
  canvasWidth: number;
  canvasHeight: number;
  editWidth: number;
  editHeight: number;
}

/**
 * Planning allowances retained across the Image 2.5 migration, not provider
 * quotes. Image 2.5 token usage varies; actual billing depends on the provider.
 */
export const TEXTURE_EDIT_ESTIMATED_USD: Readonly<Record<TextureEditImageSize, number>> = {
  "1024x1024": 0.24,
  "1536x1024": 0.20,
  "1024x1536": 0.20,
  "2048x1152": 0.29,
  "2048x2048": 0.50,
  "3840x2160": 0.95,
  "2160x3840": 0.95,
};

export const TEXTURE_EDIT_CANVAS_PRESETS = [
  {
    imageSize: "1024x1024",
    canvasWidth: 1024,
    canvasHeight: 1024,
    editWidth: 960,
    editHeight: 960,
  },
  {
    imageSize: "1536x1024",
    canvasWidth: 1536,
    canvasHeight: 1024,
    editWidth: 1472,
    editHeight: 960,
  },
  {
    imageSize: "1024x1536",
    canvasWidth: 1024,
    canvasHeight: 1536,
    editWidth: 960,
    editHeight: 1472,
  },
] as const;

const TEXTURE_EDIT_LARGE_CANVAS_PRESETS: readonly TextureEditCanvasPreset[] = [
  {
    imageSize: "2048x1152",
    canvasWidth: 2048,
    canvasHeight: 1152,
    editWidth: 1984,
    editHeight: 1088,
  },
  {
    imageSize: "2048x2048",
    canvasWidth: 2048,
    canvasHeight: 2048,
    editWidth: 1984,
    editHeight: 1984,
  },
  {
    imageSize: "3840x2160",
    canvasWidth: 3840,
    canvasHeight: 2160,
    editWidth: 3776,
    editHeight: 2096,
  },
  {
    imageSize: "2160x3840",
    canvasWidth: 2160,
    canvasHeight: 3840,
    editWidth: 2096,
    editHeight: 3776,
  },
];

// The gutter is also the protected visual legend area. A compact 5x7 badge is
// painted here without reducing the texture scale or breaking wide layouts.
export const TEXTURE_EDIT_LAYOUT_GUTTER = 16;

/** Stable colours shared by the batch preview, submitted canvas and model locator. */
export const TEXTURE_EDIT_REGION_COLORS = [
  [235, 58, 72],
  [0, 161, 214],
  [235, 159, 0],
  [139, 88, 217],
  [0, 166, 118],
] as const;

export interface TextureEditBatchSource {
  id: string;
  label: string;
  width: number;
  height: number;
}

export interface TextureEditBatchItemLayout {
  textureUnitId: string;
  label: string;
  x: number;
  y: number;
  scale: number;
  sourceWidth: number;
  sourceHeight: number;
  workWidth: number;
  workHeight: number;
}

export interface TextureEditBatchPlan {
  version: 4;
  hash: string;
  imageModel: TextureEditImageModel;
  imageSize: TextureEditImageSize;
  canvasWidth: number;
  canvasHeight: number;
  editArea: { x: number; y: number; width: number; height: number };
  gutter: number;
  estimatedUsd: number;
  unusedPixels: number;
  items: TextureEditBatchItemLayout[];
  /** Region letters run on across the canvases of one edit; the letter this canvas starts from. */
  regionKeyOffset?: number;
  /**
   * A read-only area on the second and later canvases showing what the first
   * canvas painted, so each separate image request matches its colors and patterns.
   */
  anchor?: { x: number; y: number; width: number; height: number };
}

/** Work pixels of the read-only area that carries the first canvas's result onto later ones. */
export const TEXTURE_EDIT_ANCHOR_SIZE = 448;
/** More canvases would mean more paid requests than one edit should quietly take. */
export const TEXTURE_EDIT_MAX_CANVASES = 4;

export interface TextureEditBatchPromptRegion extends TextureEditBatchItemLayout {
  /** Stable human-facing key for one confirmed layout, ordered top-to-bottom then left-to-right. */
  key: string;
}

export function textureEditBatchPromptRegions(
  plan: TextureEditBatchPlan,
): TextureEditBatchPromptRegion[] {
  const offset = plan.regionKeyOffset ?? 0;
  return [...plan.items]
    .sort((left, right) =>
      left.y - right.y || left.x - right.x || left.textureUnitId.localeCompare(right.textureUnitId)
    )
    .map((item, index) => ({ ...item, key: alphabeticalRegionKey(offset + index) }));
}

/** Every region of a multi-canvas edit, keyed A, B, C… across all its canvases. */
export function textureEditPlanRegions(
  canvases: readonly TextureEditBatchPlan[],
): Array<TextureEditBatchPromptRegion & { canvasIndex: number }> {
  return canvases.flatMap((plan, canvasIndex) =>
    textureEditBatchPromptRegions(plan).map((region) => ({ ...region, canvasIndex })));
}

function alphabeticalRegionKey(index: number): string {
  let value = index + 1;
  let output = "";
  while (value > 0) {
    value -= 1;
    output = String.fromCharCode(65 + value % 26) + output;
    value = Math.floor(value / 26);
  }
  return output;
}

export type TextureEditPlanned = Extract<TextureEditBatchPlanResult, { fits: true }>;

export type TextureEditBatchPlanResult =
  | {
    fits: true;
    /** The first canvas; the whole edit when it takes one. */
    plan: TextureEditBatchPlan;
    /** Canvases in the order they are drawn, one image request each. */
    canvases: TextureEditBatchPlan[];
    /** Identifies the whole layout, all canvases included. */
    hash: string;
    estimatedUsd: number;
  }
  | { fits: false; reason: string; textureUnitIds: string[] };

export type TexturePromptOptimizationState =
  | "not-requested"
  | "accepted"
  | "rejected";

export interface TexturePromptOptimization {
  originalPrompt: string;
  optimizedPrompt: string | null;
  state: TexturePromptOptimizationState;
}

export type TextureReferenceSource = "model-preview" | "upload" | "texture-unit";

/** Position guides are special and do not consume the user-facing R1/R2 sequence. */
export function textureReferenceLabels(
  references: readonly { source: TextureReferenceSource; promptId?: TextureReferencePromptId }[],
): string[] {
  let visualIndex = 0;
  return references.map((reference) => reference.promptId ?? (reference.source === "model-preview"
    ? "定位图"
    : `R${++visualIndex}`));
}

/** A renderer-side choice. Sources are resolved and validated by the main process. */
export interface TextureReferenceSelection {
  id: string;
  source: TextureReferenceSource;
  name: string;
  /** Optional user-authored note. Only upload references expose an editor for it. */
  description: string;
  path: string | null;
  textureUnitId: string | null;
  materialId: string | null;
  previewUrl: string | null;
  promptId?: TextureReferencePromptId;
  deliveryUse?: TextureReferenceDeliveryUse;
}

export type TextureReferenceDeliveryUse = "required" | "supplemental";

/** Immutable reference copied into a generation job's artifact directory. */
export interface TextureReferenceSnapshot {
  id: string;
  source: TextureReferenceSource;
  name: string;
  description: string;
  path: string;
  promptId?: TextureReferencePromptId;
  deliveryUse?: TextureReferenceDeliveryUse;
}

export const TEXTURE_VISUAL_ATTRIBUTE_CATEGORIES = [
  "palette",
  "rendering",
  "shading",
  "linework",
  "material",
  "motif",
  "tone",
] as const;

export type TextureVisualAttributeCategory =
  (typeof TEXTURE_VISUAL_ATTRIBUTE_CATEGORIES)[number];

export type TextureReferencePromptId = "P0" | `R${number}`;

export interface TextureAvailableReference {
  selectionId: string;
  id: TextureReferencePromptId;
  source: TextureReferenceSource;
  name: string;
  description: string;
}

export interface TexturePromptClause {
  id: string;
  text: string;
}

export interface TextureVisualAttribute {
  description: string;
}

export interface TextureStyleProfile {
  id: string;
  scope: string;
  attributes: Partial<Record<TextureVisualAttributeCategory, TextureVisualAttribute>>;
}

export interface TextureReferenceDeliveryDecision {
  selectionId: string;
  promptId: TextureReferencePromptId;
  required: boolean;
  appearanceScope: string | null;
  allowedAttributes: TextureVisualAttributeCategory[];
  regionKeys: string[];
}

export interface TextureRegionEdit {
  key: string;
  role: string;
  styleProfileIds: string[];
  clauseIds: string[];
  continuityGroup?: string;
  /** Why the request leaves this selected piece as it is; such a piece is drawn unchanged and not put on. */
  keep?: string;
  /** What the piece shows, in a few words, for the list of skipped pieces. */
  keepLabel?: string;
}

/** Selected pieces the understanding says the request leaves alone. */
export function keptRegionKeys(spec: Pick<TextureVisualEditSpec, "regions">): Set<string> {
  return new Set(spec.regions.filter((region) => region.keep).map((region) => region.key));
}

export type TextureVisualEditIntent =
  | "recolor"
  | "style-transfer"
  | "material-transfer"
  | "cleanup"
  | "alpha-remove";

/**
 * What happens to one sentence of the request:
 * - `repaint`: the image model redraws the selected pieces;
 * - `recolor`: only the color changes — 换色 does it at once, for free;
 * - `toggle`: a part is removed or shown whole — that is a switch, not a repaint;
 * - `unsupported`: a change of shape or a new element, which no repaint can make;
 * - `unselected`: it is about a part that isn't selected.
 */
export type TextureClauseAction = "repaint" | "recolor" | "toggle" | "unsupported" | "unselected";

export interface TextureClausePlan {
  clauseId: string;
  action: TextureClauseAction;
  /** The part the sentence is about, by the name the person knows it. */
  part?: string;
  /** One short sentence for the person. */
  note: string;
}

export interface TextureVisualEditSpec {
  intent: TextureVisualEditIntent;
  styleProfiles: TextureStyleProfile[];
  regions: TextureRegionEdit[];
  warnings: string[];
  /** One per sentence of the request; missing ones are repainted. */
  actions?: TextureClausePlan[];
}

/** Sentences the image model draws; the others go to 换色, the switches, or nowhere. */
export function drawnClauseIds(spec: Pick<TextureVisualEditSpec, "actions">, clauseIds: readonly string[]): string[] {
  const skipped = new Set((spec.actions ?? [])
    .filter((plan) => plan.action !== "repaint" && plan.action !== "recolor")
    .map((plan) => plan.clauseId));
  return clauseIds.filter((id) => !skipped.has(id));
}

export interface TexturePromptOptimizationRequest {
  templateId: string;
  modelName: string;
  textureUnitIds: string[];
  /** The model's parts by name, so a sentence about an unselected part can be told apart. */
  parts?: { selected: string[]; others: string[] };
  prompt: string;
  planHash: string;
  imageModel: TextureEditImageModel;
  references: TextureReferenceSelection[];
}

export interface TexturePromptOptimizationResult {
  originalPrompt: string;
  optimizedPrompt: string;
  spec: TextureVisualEditSpec;
  regions: TextureRegionEdit[];
  warnings: string[];
  /** Authoritative plan recomputed immediately before optimization. */
  plan?: TextureEditBatchPlan;
  /** The same, with every canvas when the edit takes several. */
  planned?: TextureEditPlanned;
  /** True when the renderer's cached plan was transparently refreshed. */
  planChanged?: boolean;
}

export function chooseTextureEditCanvasPreset(
  width: number,
  height: number,
  imageModel: TextureEditImageModel = DEFAULT_TEXTURE_EDIT_IMAGE_MODEL,
) {
  resolveTextureEditImageModel(imageModel);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error("纹理尺寸必须是正整数");
  }
  const choose = (presets: readonly TextureEditCanvasPreset[]) => presets.map((candidate) => ({
      ...candidate,
      scale: Math.floor(Math.min(candidate.editWidth / width, candidate.editHeight / height)),
    })).filter((candidate) => candidate.scale >= 1)
      .sort((left, right) =>
        right.scale - left.scale ||
        left.canvasWidth * left.canvasHeight - right.canvasWidth * right.canvasHeight
      )[0];
  const chosen = choose(TEXTURE_EDIT_CANVAS_PRESETS) ??
    TEXTURE_EDIT_LARGE_CANVAS_PRESETS.map((candidate) => ({
        ...candidate,
        scale: Math.floor(Math.min(candidate.editWidth / width, candidate.editHeight / height)),
      })).filter((candidate) => candidate.scale >= 1)
        .sort((left, right) =>
          left.canvasWidth * left.canvasHeight - right.canvasWidth * right.canvasHeight ||
          left.scale - right.scale
        )[0];
  if (!chosen) throw new Error(`纹理尺寸 ${width}×${height} 超过单次高保真编辑画布上限`);
  return chosen;
}

export function estimateTextureEditUsd(
  imageSize: TextureEditImageSize,
  imageModel: TextureEditImageModel = DEFAULT_TEXTURE_EDIT_IMAGE_MODEL,
): number {
  resolveTextureEditImageModel(imageModel);
  return TEXTURE_EDIT_ESTIMATED_USD[imageSize];
}

export function estimateTextureEditUnitUsd(
  width: number,
  height: number,
  imageModel: TextureEditImageModel = DEFAULT_TEXTURE_EDIT_IMAGE_MODEL,
): number {
  return estimateTextureEditUsd(
    chooseTextureEditCanvasPreset(width, height, imageModel).imageSize,
    imageModel,
  );
}

type RequestedTexture = TextureEditBatchSource & { scale: number; workWidth: number; workHeight: number };

const ANCHOR_ID = "\u0000anchor";

/** A piece bigger than every canvas may be drawn at 1/2, 1/3 or 1/4 of its size. */
const TEXTURE_EDIT_DOWNSCALE_DIVISORS = [2, 3, 4] as const;
/**
 * How far big pieces may shrink: never below `minSide` pixels on their long
 * side, and to at most 1/`divisor`. High-resolution models ship textures far
 * larger than they ever show on screen, so shrinking to 1024 costs no visible
 * detail; 512 is the last resort before refusing.
 */
interface ShrinkLevel { divisor: number; minSide: number }
const UNSEEN_SHRINK: ShrinkLevel = { divisor: 4, minSide: 1024 };
const LAST_RESORT_SHRINKS: readonly ShrinkLevel[] = [
  { divisor: 2, minSide: 512 },
  { divisor: 3, minSide: 512 },
  { divisor: 4, minSide: 512 },
];

/** Work pixels of `size` source pixels at `scale`; a shrunk piece rounds up so no pixel is lost. */
export function textureEditWorkSize(size: number, scale: number): number {
  return scale >= 1 ? size * scale : Math.ceil(size * scale - 1e-9);
}

/**
 * The scale a piece gets on its own: the integer upscale of its best canvas,
 * or 1/2–1/4 when it is bigger than every canvas. Null when even 1/4 is too big.
 */
export function soloTextureEditScale(
  width: number,
  height: number,
  imageModel: TextureEditImageModel = DEFAULT_TEXTURE_EDIT_IMAGE_MODEL,
): number | null {
  try {
    return chooseTextureEditCanvasPreset(width, height, imageModel).scale;
  } catch {
    // Bigger than any canvas at 1×.
  }
  for (const divisor of TEXTURE_EDIT_DOWNSCALE_DIVISORS) {
    const scale = 1 / divisor;
    if (ALL_CANVAS_PRESETS.some((preset) =>
      textureEditWorkSize(width, scale) <= preset.editWidth && textureEditWorkSize(height, scale) <= preset.editHeight
    )) return scale;
  }
  return null;
}

function withScale(item: RequestedTexture, scale: number): RequestedTexture {
  return { ...item, scale, workWidth: textureEditWorkSize(item.width, scale), workHeight: textureEditWorkSize(item.height, scale) };
}

/**
 * A piece's scale under `level`: a number caps the upscale; a shrink level
 * shrinks only big pieces, as far as it allows, and leaves small ones at
 * their own size.
 */
function levelScale(item: RequestedTexture, level: number | ShrinkLevel): number {
  if (typeof level === "number") return Math.min(item.scale, level);
  const longSide = Math.max(item.width, item.height);
  for (let divisor = level.divisor; divisor >= 2; divisor -= 1) {
    if (longSide / divisor >= level.minSide) return Math.min(item.scale, 1 / divisor);
  }
  return Math.min(item.scale, 1);
}

/**
 * Lays the selected texture units out on as few canvases as they need, each
 * canvas one paid request:
 * 1. One canvas at their own size or larger (see `planTextureCanvas`), else
 *    one canvas with oversized pieces shrunk where it can't be seen.
 * 2. Otherwise several canvases, biggest pieces first, choosing whichever of
 *    2×, own size or unseen shrinking takes the fewest (ties to the sharper).
 * 3. Only beyond TEXTURE_EDIT_MAX_CANVASES are big pieces shrunk visibly.
 * Every canvas after the first reserves a read-only anchor area for the
 * first canvas's result.
 */
export function planTextureEditBatch(
  sources: readonly TextureEditBatchSource[],
  imageModel: TextureEditImageModel = DEFAULT_TEXTURE_EDIT_IMAGE_MODEL,
): TextureEditBatchPlanResult {
  imageModel = resolveTextureEditImageModel(imageModel);
  if (sources.length === 0) {
    return { fits: false, reason: "请至少选择一个纹理", textureUnitIds: [] };
  }
  const ids = new Set<string>();
  const requested: RequestedTexture[] = [];
  for (const source of sources) {
    if (!source.id || ids.has(source.id) || source.id === ANCHOR_ID) throw new Error("批量纹理 ID 必须唯一");
    if (!Number.isInteger(source.width) || !Number.isInteger(source.height) || source.width <= 0 || source.height <= 0) {
      throw new Error("纹理尺寸必须是正整数");
    }
    ids.add(source.id);
    const scale = soloTextureEditScale(source.width, source.height, imageModel);
    if (scale === null) {
      return {
        fits: false,
        reason: `「${source.label}」太大（${source.width}×${source.height}），缩小到 1/4 也画不了，请去掉它`,
        textureUnitIds: [source.id],
      };
    }
    requested.push(withScale({ ...source, scale, workWidth: 0, workHeight: 0 }, scale));
  }
  requested.sort(byWorkArea);
  const single = planTextureCanvas(requested, imageModel, {}) ??
    planTextureCanvas(requested, imageModel, { shrink: UNSEEN_SHRINK });
  if (single) return plannedCanvases([single]);

  const binned = (level: number | ShrinkLevel) => {
    const bins: RequestedTexture[][] = [];
    for (const item of requested) {
      const bin = bins.find((members, index) => packsAt([...members, item], level, index > 0));
      if (bin) bin.push(item);
      else bins.push([item]);
    }
    return bins;
  };
  // Sharpest first, so a tie keeps the sharper layout.
  let chosen: { bins: RequestedTexture[][]; shrink?: ShrinkLevel } = { bins: binned(2) };
  for (const level of [1, UNSEEN_SHRINK]) {
    const bins = binned(level);
    if (bins.length < chosen.bins.length) chosen = typeof level === "number" ? { bins } : { bins, shrink: level };
  }
  for (const shrink of LAST_RESORT_SHRINKS) {
    if (chosen.bins.length <= TEXTURE_EDIT_MAX_CANVASES) break;
    chosen = { bins: binned(shrink), shrink };
  }
  if (chosen.bins.length > TEXTURE_EDIT_MAX_CANVASES) {
    return {
      fits: false,
      reason: `选得太多，要分 ${chosen.bins.length} 张画布画，一次最多 ${TEXTURE_EDIT_MAX_CANVASES} 张，请少选一些`,
      textureUnitIds: requested.map((item) => item.id),
    };
  }
  const canvases: TextureEditBatchPlan[] = [];
  for (const [index, members] of chosen.bins.entries()) {
    const options = {
      regionKeyOffset: canvases.reduce((sum, plan) => sum + plan.items.length, 0),
      shrink: chosen.shrink,
    };
    // A piece too big to share its canvas with the anchor still gets drawn, just without it.
    const plan = (index > 0 ? planTextureCanvas(members, imageModel, { ...options, anchor: true }) : null) ??
      planTextureCanvas(members, imageModel, options);
    if (!plan) {
      return {
        fits: false,
        reason: "所选纹理放不进画布，请减少选择",
        textureUnitIds: members.map((item) => item.id),
      };
    }
    canvases.push(plan);
  }
  return plannedCanvases(canvases);
}

function plannedCanvases(canvases: TextureEditBatchPlan[]): TextureEditBatchPlanResult {
  return {
    fits: true,
    plan: canvases[0]!,
    canvases,
    hash: textureEditCanvasesHash(canvases),
    estimatedUsd: Math.round(canvases.reduce((sum, plan) => sum + plan.estimatedUsd, 0) * 100) / 100,
  };
}

/** The hash of a whole edit: the plan's own hash when it takes one canvas. */
export function textureEditCanvasesHash(canvases: readonly TextureEditBatchPlan[]): string {
  if (canvases.length === 1) return canvases[0]!.hash;
  let hash = 0xcbf29ce484222325n;
  for (const character of canvases.map((plan) => plan.hash).join("|")) {
    hash ^= BigInt(character.charCodeAt(0));
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return `texture-canvases-v1-${canvases.length}-${hash.toString(16).padStart(16, "0")}`;
}

function byWorkArea(left: RequestedTexture, right: RequestedTexture): number {
  return right.workWidth * right.workHeight - left.workWidth * left.workHeight ||
    right.workHeight - left.workHeight ||
    left.id.localeCompare(right.id);
}

function anchorItem(): RequestedTexture {
  return {
    id: ANCHOR_ID,
    label: "",
    width: TEXTURE_EDIT_ANCHOR_SIZE,
    height: TEXTURE_EDIT_ANCHOR_SIZE,
    scale: 1,
    workWidth: TEXTURE_EDIT_ANCHOR_SIZE,
    workHeight: TEXTURE_EDIT_ANCHOR_SIZE,
  };
}

const ALL_CANVAS_PRESETS: readonly TextureEditCanvasPreset[] = [
  ...TEXTURE_EDIT_CANVAS_PRESETS,
  ...TEXTURE_EDIT_LARGE_CANVAS_PRESETS,
];

/** Whether these pieces share some canvas with each at its `levelScale`. */
function packsAt(members: readonly RequestedTexture[], level: number | ShrinkLevel, anchor: boolean): boolean {
  const items = [
    ...members.map((item) => withScale(item, levelScale(item, level))),
    ...(anchor ? [anchorItem()] : []),
  ].sort(byWorkArea);
  return ALL_CANVAS_PRESETS.some((preset) => packTextureRectangles(
    items,
    preset.editWidth + TEXTURE_EDIT_LAYOUT_GUTTER,
    preset.editHeight + TEXTURE_EDIT_LAYOUT_GUTTER,
  ) !== null);
}

/**
 * Packs texture units onto one canvas, each at the integer upscale it would
 * get when edited on its own when they all fit, otherwise at the highest
 * common cap (never below 1×). Rotation is intentionally forbidden because the
 * result must be mapped back to the original UVs. Null when they don't fit.
 */
function planTextureCanvas(
  requested: readonly RequestedTexture[],
  imageModel: TextureEditImageModel,
  options: { anchor?: boolean; regionKeyOffset?: number; shrink?: ShrinkLevel },
): TextureEditBatchPlan | null {
  const candidatesFor = (scaled: readonly RequestedTexture[]) => ALL_CANVAS_PRESETS.flatMap((preset) => {
    const packed = packTextureRectangles(
      options.anchor ? [...scaled, anchorItem()].sort(byWorkArea) : scaled,
      preset.editWidth + TEXTURE_EDIT_LAYOUT_GUTTER,
      preset.editHeight + TEXTURE_EDIT_LAYOUT_GUTTER,
    );
    if (!packed) return [];
    const anchor = packed.find((item) => item.id === ANCHOR_ID);
    const items = packed.filter((item) => item.id !== ANCHOR_ID).map((item) => ({
      textureUnitId: item.id,
      label: item.label,
      x: TEXTURE_EDIT_LAYOUT_GUTTER + item.x,
      y: TEXTURE_EDIT_LAYOUT_GUTTER + item.y,
      scale: item.scale,
      sourceWidth: item.width,
      sourceHeight: item.height,
      workWidth: item.workWidth,
      workHeight: item.workHeight,
    })).sort((left, right) => left.textureUnitId.localeCompare(right.textureUnitId));
    const occupied = packed.reduce((sum, item) => sum + item.workWidth * item.workHeight, 0);
    const unsigned: Omit<TextureEditBatchPlan, "hash"> = {
      version: 4 as const,
      imageModel,
      imageSize: preset.imageSize,
      canvasWidth: preset.canvasWidth,
      canvasHeight: preset.canvasHeight,
      editArea: {
        x: TEXTURE_EDIT_LAYOUT_GUTTER,
        y: TEXTURE_EDIT_LAYOUT_GUTTER,
        width: preset.editWidth,
        height: preset.editHeight,
      },
      gutter: TEXTURE_EDIT_LAYOUT_GUTTER,
      estimatedUsd: estimateTextureEditUsd(preset.imageSize, imageModel),
      unusedPixels: preset.editWidth * preset.editHeight - occupied,
      items,
      ...options.regionKeyOffset ? { regionKeyOffset: options.regionKeyOffset } : {},
      ...anchor ? {
        anchor: {
          x: TEXTURE_EDIT_LAYOUT_GUTTER + anchor.x,
          y: TEXTURE_EDIT_LAYOUT_GUTTER + anchor.y,
          width: anchor.workWidth,
          height: anchor.workHeight,
        },
      } : {},
    };
    return [{ ...unsigned, hash: textureBatchPlanHash(unsigned) }];
  }).sort((left, right) =>
    left.estimatedUsd - right.estimatedUsd ||
    left.unusedPixels - right.unusedPixels ||
    left.canvasWidth * left.canvasHeight - right.canvasWidth * right.canvasHeight
  );
  // Each cap limits every piece's upscale. The solo scales (no cap) give the
  // sharpest result but blow tiny pieces up into huge canvases, so the
  // cheapest canvas that still gives every piece at least 2× (or its own
  // maximum, when smaller) wins; 1× is the last resort when nothing else fits.
  const largest = Math.max(...requested.map((item) => item.scale));
  const scaledTo = (level: number | ShrinkLevel) =>
    requested.map((item) => withScale(item, levelScale(item, level))).sort(byWorkArea);
  let best: TextureEditBatchPlan | undefined;
  for (let cap = largest; cap >= 2; cap -= 1) {
    const candidate = candidatesFor(cap === largest ? requested : scaledTo(cap))[0];
    if (candidate && (!best || candidate.estimatedUsd < best.estimatedUsd)) best = candidate;
  }
  if (best) return best;
  const native = candidatesFor(scaledTo(1))[0];
  if (native || !options.shrink) return native ?? null;
  // Below 1× only as far as the caller allows, shrinking the least that fits.
  for (let divisor = 2; divisor <= options.shrink.divisor; divisor += 1) {
    const candidate = candidatesFor(scaledTo({ divisor, minSide: options.shrink.minSide }))[0];
    if (candidate) return candidate;
  }
  return null;
}

type PackedTexture = TextureEditBatchSource & {
  scale: number;
  workWidth: number;
  workHeight: number;
  x: number;
  y: number;
};

interface PackingRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

function packTextureRectangles(
  requested: ReadonlyArray<Omit<PackedTexture, "x" | "y">>,
  width: number,
  height: number,
): PackedTexture[] | null {
  let free: PackingRect[] = [{ x: 0, y: 0, width, height }];
  const packed: PackedTexture[] = [];
  for (const item of requested) {
    const footprintWidth = item.workWidth + TEXTURE_EDIT_LAYOUT_GUTTER;
    const footprintHeight = item.workHeight + TEXTURE_EDIT_LAYOUT_GUTTER;
    const choice = free.filter((rect) => footprintWidth <= rect.width && footprintHeight <= rect.height)
      .map((rect) => ({
        rect,
        shortSide: Math.min(rect.width - footprintWidth, rect.height - footprintHeight),
        longSide: Math.max(rect.width - footprintWidth, rect.height - footprintHeight),
      }))
      .sort((left, right) =>
        left.shortSide - right.shortSide || left.longSide - right.longSide ||
        left.rect.y - right.rect.y || left.rect.x - right.rect.x
      )[0];
    if (!choice) return null;
    const used = {
      x: choice.rect.x,
      y: choice.rect.y,
      width: footprintWidth,
      height: footprintHeight,
    };
    packed.push({ ...item, x: used.x, y: used.y });
    free = prunePackingRects(free.flatMap((rect) => splitPackingRect(rect, used)));
  }
  return packed;
}

function splitPackingRect(free: PackingRect, used: PackingRect): PackingRect[] {
  if (
    used.x >= free.x + free.width || used.x + used.width <= free.x ||
    used.y >= free.y + free.height || used.y + used.height <= free.y
  ) return [free];
  const values: PackingRect[] = [];
  if (used.x > free.x) values.push({ ...free, width: used.x - free.x });
  if (used.x + used.width < free.x + free.width) {
    values.push({
      ...free,
      x: used.x + used.width,
      width: free.x + free.width - used.x - used.width,
    });
  }
  if (used.y > free.y) values.push({ ...free, height: used.y - free.y });
  if (used.y + used.height < free.y + free.height) {
    values.push({
      ...free,
      y: used.y + used.height,
      height: free.y + free.height - used.y - used.height,
    });
  }
  return values.filter((rect) => rect.width > 0 && rect.height > 0);
}

function prunePackingRects(rects: PackingRect[]): PackingRect[] {
  return rects.filter((rect, index) => !rects.some((other, otherIndex) =>
    index !== otherIndex &&
    rect.x >= other.x && rect.y >= other.y &&
    rect.x + rect.width <= other.x + other.width &&
    rect.y + rect.height <= other.y + other.height
  ));
}

function textureBatchPlanHash(value: Omit<TextureEditBatchPlan, "hash">): string {
  // Labels are presentation-only metadata. The renderer gets them from the
  // capability manifest while the main process may reconstruct them from the
  // master template, so including them caused false "layout changed" errors
  // even though every pixel coordinate was identical.
  const canonical = JSON.stringify({
    ...value,
    items: value.items.map(({ label: _label, ...item }) => item),
  });
  let hash = 0xcbf29ce484222325n;
  for (let index = 0; index < canonical.length; index += 1) {
    hash ^= BigInt(canonical.charCodeAt(index));
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return `texture-batch-v4-${hash.toString(16).padStart(16, "0")}`;
}

export interface MasterTemplateAtlas {
  file: string;
  width: number;
  height: number;
}

export const MASTER_TEMPLATE_PART_GROUPS = [
  "face",
  "hair",
  "body",
  "arm-left",
  "arm-right",
  "leg-left",
  "leg-right",
  "clothing",
  "shoe-left",
  "shoe-right",
  "accessory",
  "effect",
  "template",
] as const;

export type MasterTemplatePartGroup = (typeof MASTER_TEMPLATE_PART_GROUPS)[number];
export type MasterTemplateSourceKind = "visible" | "completion" | "template";

export interface MasterTemplateCanvas {
  width: number;
  height: number;
  originX: number;
  originY: number;
  pixelsPerUnit: number;
}

export interface MasterTemplateMesh {
  /** Neutral-pose model-space vertices as x/y pairs. */
  vertices: number[];
  /** Cubism texture UVs as u/v pairs. */
  uvs: number[];
  /** Triangle vertex indices. */
  indices: number[];
}

export interface MasterTemplateSlot {
  id: string;
  name: string;
  drawableId: string;
  partId: string;
  group: MasterTemplatePartGroup;
  source: MasterTemplateSourceKind;
  required: boolean;
  zIndex: number;
  atlas: {
    texture: number;
    x: number;
    y: number;
    width: number;
    height: number;
  };
  anchor: { x: number; y: number };
  canvasBounds: { x: number; y: number; width: number; height: number };
  mesh: MasterTemplateMesh;
  silhouetteMask: string;
  completionPrompt: string;
  /** Optional semantic edit part produced by the model capability scanner. */
  editPartId?: string;
  /** Exact unique UV texture unit produced by the model capability scanner. */
  textureUnitId?: string;
}

export interface MasterTemplateManifest {
  schemaVersion: 2;
  id: string;
  name: string;
  modelEntry: string;
  /** Neutral template render used as the geometry reference during character alignment. */
  alignmentGuide: string;
  coverage: "full-body" | "half-body";
  canvas: MasterTemplateCanvas;
  parameters: {
    eyeBlink: string[];
    lipSync: string[];
  };
  textureAtlases: MasterTemplateAtlas[];
  slots: MasterTemplateSlot[];
}

export interface MasterTemplateSummary {
  id: string;
  name: string;
  slotCount: number;
  coverage: "full-body" | "half-body";
}

export interface CharacterCandidate {
  id: string;
  localPath: string;
  seed: number | null;
  previewUrl?: string;
  compatible: boolean;
  rejectionReason: string | null;
}

export interface CreateCharacterGenerationRequest {
  operation: "eye-restyle" | "part-restyle" | "texture-edit" | "character";
  templateId: string;
  description: string;
  referenceImage: string | null;
  sourceModelId: string | null;
  editPartId?: string | null;
  editMode?: import("./contracts").ModelEditMode | null;
  textureUnitIds?: string[];
  originalDescription?: string;
  optimizedDescription?: string | null;
  promptOptimization?: TexturePromptOptimizationState;
  promptVisualSpec?: TextureVisualEditSpec | null;
  textureBatchPlanHash?: string;
  /**
   * When the edit takes several canvases, the prompt for each, in canvas order,
   * naming only that canvas's regions. `description` still holds the whole prompt.
   */
  textureCanvasDescriptions?: string[];
  textureImageModel?: TextureEditImageModel;
  textureReferences?: TextureReferenceSelection[];
  budgetLimitUsd?: number;
}

export type TextureEditResultStatus =
  | "pending"
  | "running"
  | "review"
  | "accepted"
  | "failed"
  | "cancelled";

export interface TextureEditWorkLayout {
  canvasWidth: number;
  canvasHeight: number;
  offsetX: number;
  offsetY: number;
  scale: number;
  sourceWidth: number;
  sourceHeight: number;
}

export interface TextureEditSelection {
  textureUnitIds: string[];
  description: string;
  budgetLimitUsd: number;
  imageModel: TextureEditImageModel;
}

export interface TextureEditQualityReport {
  accepted: boolean;
  issues: string[];
  changedPixelRatio: number;
  alphaChangedPixelRatio: number;
  removedAlphaRatio: number;
  ignoredExpansionPixelRatio: number;
  sharpnessRatio: number;
}

export interface TextureEditResult {
  textureUnitId: string;
  status: TextureEditResultStatus;
  requestId: string | null;
  endpoint: string | null;
  estimatedUsd: number;
  sourcePath: string | null;
  sourcePreviewPath: string | null;
  workImagePath: string | null;
  workMaskPath: string | null;
  rawResultPath: string | null;
  editedPatchPath: string | null;
  editedPreviewPath: string | null;
  layout: TextureEditWorkLayout | null;
  quality: TextureEditQualityReport | null;
  error: string | null;
}

export interface TextureEditBatchArtifact {
  plan: TextureEditBatchPlan;
  requestId: string | null;
  endpoint: string | null;
  workImagePath: string | null;
  workMaskPath: string | null;
  rawResultPath: string | null;
  error: string | null;
}

export interface CharacterReferenceSelection {
  path: string;
  name: string;
  previewUrl: string;
}

export interface CharacterGenerationIssue {
  code: string;
  message: string;
  slotId: string | null;
  recoverable: boolean;
}

export interface CharacterGenerationJob {
  version: 2;
  id: string;
  templateId: string;
  input: {
    operation?: "eye-restyle" | "part-restyle" | "texture-edit" | "character";
    description: string;
    referenceImage: string | null;
    sourceModelId: string | null;
    editPartId?: string | null;
    editMode?: import("./contracts").ModelEditMode | null;
    textureUnitIds?: string[];
    textureAlphaPolicy?: "locked" | "shrink-only";
    textureBatchMode?: "joint-v1";
    textureBatchPlanHash?: string;
    textureCanvasDescriptions?: string[];
    textureImageModel?: TextureEditImageModel;
    originalDescription?: string;
    optimizedDescription?: string | null;
    promptOptimization?: TexturePromptOptimizationState;
    promptVisualSpec?: TextureVisualEditSpec | null;
    textureReferences?: TextureReferenceSnapshot[];
  };
  phase: CharacterGenerationPhase;
  paused: boolean;
  pauseReason: "user" | "budget" | "needs-review" | null;
  revision: number;
  candidates: CharacterCandidate[];
  selectedCandidateId: string | null;
  falRequests: Array<{
    operation: string;
    endpoint: string;
    requestId: string;
    revision: number;
  }>;
  cache: Record<string, string[]>;
  artifacts: {
    normalizedImage: string | null;
    subjectImage: string | null;
    coarseLayers: string[];
    completionSheets: Record<string, string>;
    slotTextures: Record<string, string>;
    repairSlotIds: string[];
    requestedCompletions: CharacterCompletionKind[];
    eyeMask?: string | null;
    eyeEditedImage?: string | null;
    eyeSlotIds?: string[];
    editMask?: string | null;
    editedImage?: string | null;
    editSlotIds?: string[];
    textureEdits?: Record<string, TextureEditResult>;
    /** The most recent canvas; kept for jobs written before edits could take several. */
    textureBatch?: TextureEditBatchArtifact | null;
    /** One per canvas, in drawing order. */
    textureBatches?: TextureEditBatchArtifact[];
  };
  budget: {
    limitUsd: number;
    estimatedUsd: number;
    actualUsd: number;
    pendingOperation: { name: string; estimatedUsd: number } | null;
  };
  issues: CharacterGenerationIssue[];
  outputModelId: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface CharacterGenerationProgress {
  jobId: string;
  phase: CharacterGenerationPhase;
  progress: number;
  detail: string;
  job: CharacterGenerationJob;
}

export interface CharacterProviderStatus {
  stage: "queued" | "running" | "completed";
  queuePosition: number | null;
  message: string | null;
  elapsedMs: number;
}

export interface CharacterGenerationProvider {
  generateCandidates(input: {
    prompt: string;
    referenceImagePath: string | null;
    templateGuidePath: string;
    imageSize: { width: number; height: number };
    count: 4;
    signal?: AbortSignal;
    onSubmitted?: (endpoint: string, requestId: string) => void | Promise<void>;
    onStatus?: (status: CharacterProviderStatus) => void;
  }): Promise<{ requestId: string; urls: string[]; seed: number | null }>;
  editImage(input: {
    prompt: string;
    imagePaths: string[];
    imageSize?: { width: number; height: number };
    signal?: AbortSignal;
    onSubmitted?: (endpoint: string, requestId: string) => void | Promise<void>;
    onStatus?: (status: CharacterProviderStatus) => void;
  }): Promise<{ requestId: string; urls: string[] }>;
  inpaintImage(input: {
    prompt: string;
    negativePrompt?: string;
    imagePath: string;
    maskPath: string;
    imageSize: { width: number; height: number } | "landscape_4_3";
    strength?: number;
    guidanceScale?: number;
    numInferenceSteps?: number;
    acceleration?: "none" | "regular" | "high";
    signal?: AbortSignal;
    onSubmitted?: (endpoint: string, requestId: string) => void | Promise<void>;
    onStatus?: (status: CharacterProviderStatus) => void;
  }): Promise<{ requestId: string; urls: string[] }>;
  editMaskedTexture(input: {
    prompt: string;
    imagePath: string;
    maskPath: string;
    imageSize: TextureEditImageSize;
    imageModel: TextureEditImageModel;
    referenceImagePaths?: string[];
    signal?: AbortSignal;
    onSubmitted?: (endpoint: string, requestId: string) => void | Promise<void>;
    onStatus?: (status: CharacterProviderStatus) => void;
  }): Promise<{ endpoint?: string; requestId: string; urls: string[] }>;
  decomposeImage(input: {
    imagePath: string;
    prompt: string;
    layerCount: number;
    signal?: AbortSignal;
    onSubmitted?: (endpoint: string, requestId: string) => void | Promise<void>;
    onStatus?: (status: CharacterProviderStatus) => void;
  }): Promise<{ requestId: string; urls: string[] }>;
  removeBackground(input: {
    imagePath: string;
    signal?: AbortSignal;
    onSubmitted?: (endpoint: string, requestId: string) => void | Promise<void>;
    onStatus?: (status: CharacterProviderStatus) => void;
  }): Promise<{ requestId: string; url: string; maskUrl: string | null }>;
  segmentImage(input: {
    imagePath: string;
    prompt: string;
    negativePrompt?: string;
    signal?: AbortSignal;
    onSubmitted?: (endpoint: string, requestId: string) => void | Promise<void>;
    onStatus?: (status: CharacterProviderStatus) => void;
  }): Promise<{ requestId: string; url: string }>;
  analyzeImages(input: {
    imagePaths: string[];
    prompt: string;
    signal?: AbortSignal;
    onSubmitted?: (endpoint: string, requestId: string) => void | Promise<void>;
    onStatus?: (status: CharacterProviderStatus) => void;
  }): Promise<{ requestId: string; content: string }>;
  cancel(endpoint: string, requestId: string): Promise<void>;
}

const ALLOWED_TRANSITIONS: Record<CharacterGenerationPhase, CharacterGenerationPhase[]> = {
  draft: ["decomposing", "failed", "cancelled"],
  decomposing: ["aligning", "failed", "cancelled"],
  aligning: ["base-preview", "validating", "failed", "cancelled"],
  "base-preview": ["completing", "saved", "failed", "cancelled"],
  completing: ["aligning", "failed", "cancelled"],
  validating: ["preview", "failed", "cancelled"],
  preview: ["decomposing", "saved", "failed", "cancelled"],
  saved: [],
  failed: ["draft", "decomposing", "cancelled"],
  cancelled: [],
};

export function transitionCharacterJob(
  job: CharacterGenerationJob,
  phase: CharacterGenerationPhase,
): CharacterGenerationJob {
  if (!ALLOWED_TRANSITIONS[job.phase].includes(phase)) {
    throw new Error(`不允许从 ${job.phase} 进入 ${phase}`);
  }
  return {
    ...job,
    phase,
    paused: false,
    pauseReason: null,
    revision: job.revision + 1,
    updatedAt: new Date().toISOString(),
  };
}

export function reserveEstimatedCost(
  job: CharacterGenerationJob,
  name: string,
  estimatedUsd: number,
): CharacterGenerationJob {
  if (!Number.isFinite(estimatedUsd) || estimatedUsd < 0) {
    throw new Error("预计费用必须是非负数");
  }
  const projected = Math.round((job.budget.estimatedUsd + estimatedUsd) * 1_000_000) / 1_000_000;
  if (projected > job.budget.limitUsd) {
    return {
      ...job,
      paused: true,
      pauseReason: "budget",
      budget: {
        ...job.budget,
        pendingOperation: { name, estimatedUsd },
      },
      updatedAt: new Date().toISOString(),
    };
  }
  return {
    ...job,
    budget: {
      ...job.budget,
      estimatedUsd: projected,
      pendingOperation: null,
    },
    updatedAt: new Date().toISOString(),
  };
}

/**
 * Repairs estimates written by older builds that treated Qwen Image Layered as
 * one flat-rate request. fal bills that endpoint per returned layer.
 */
export function reconcileCharacterJobEstimate(
  job: CharacterGenerationJob,
): CharacterGenerationJob {
  const requests = new Map<string, CharacterGenerationJob["falRequests"][number]>();
  for (const request of job.falRequests) {
    const key = `${request.endpoint}:${request.requestId}`;
    const previous = requests.get(key);
    if (!previous || previous.operation === "in-flight" || request.operation !== "in-flight") {
      requests.set(key, request);
    }
  }
  const requestTotal = [...requests.values()].reduce(
    (total, request) => total + estimatedRequestCost(request.operation, request.endpoint),
    0,
  );
  const reconciled = Math.round(Math.max(job.budget.estimatedUsd, requestTotal) * 1_000_000) / 1_000_000;
  if (reconciled === job.budget.estimatedUsd) return job;
  return {
    ...job,
    budget: { ...job.budget, estimatedUsd: reconciled },
  };
}

function estimatedRequestCost(operation: string, endpoint: string): number {
  if (operation === "generate-candidates") {
    return endpoint === "fal-ai/z-image/turbo/controlnet" ? 0.02 * 4 : 0.035 * 4;
  }
  if (
    operation === "validate-candidates" ||
    operation === "validate-alignment" ||
    operation === "validate-materials" ||
    operation === "final-validation"
  ) return 0.005;
  if (operation === "normalize") return 0.035;
  if (operation === "eye-restyle" || operation === "part-restyle") return 0.03;
  if (operation === "texture-edit") {
    // Texture requests reserve their exact model-and-canvas estimate before
    // submission. The endpoint alone cannot recover a custom Image 2 size, so
    // guessing here would overwrite the accurate plan with a different price.
    return 0;
  }
  if (operation === "remove-background") return 0.005;
  if (operation === "coarse-decomposition") return 0.05 * 8;
  if (operation.startsWith("completion-")) return 0.035;
  if (operation.startsWith("material-")) return 0.035;
  if (operation.startsWith("segment-")) return 0.005;
  if (operation === "in-flight") {
    if (endpoint === "fal-ai/qwen-image-2/text-to-image") return 0.035 * 4;
    if (endpoint === "fal-ai/qwen-image-2/edit") return 0.035;
    if (endpoint === "fal-ai/qwen-image-edit/inpaint") return 0.03;
    if (endpoint.includes("gpt-image-2") || endpoint.includes("gpt-image-1.5")) return 0;
    if (endpoint === "fal-ai/z-image/turbo/controlnet") return 0.02 * 4;
    if (endpoint === "fal-ai/qwen-image-layered") return 0.05 * 8;
    if (endpoint === "fal-ai/birefnet/v2") return 0.005;
    if (endpoint === "fal-ai/evf-sam" || endpoint === "openrouter/router/vision") return 0.005;
  }
  return 0;
}

export function validateMasterTemplateManifest(value: unknown): MasterTemplateManifest {
  if (!isRecord(value) || value.schemaVersion !== 2) {
    throw new Error("母版清单版本无效");
  }
  const id = requiredText(value.id, "母版 ID");
  if (!/^[a-z0-9][a-z0-9-]{0,79}$/.test(id)) {
    throw new Error("母版 ID 只能包含小写字母、数字和连字符");
  }
  const name = requiredText(value.name, "母版名称");
  const modelEntry = safeRelativePath(value.modelEntry, "模型入口");
  const alignmentGuide = safeRelativePath(value.alignmentGuide, "母版对齐参考图");
  if (value.coverage !== "full-body" && value.coverage !== "half-body") {
    throw new Error("母版必须明确声明 full-body 或 half-body 覆盖范围");
  }
  if (!isRecord(value.canvas)) throw new Error("母版缺少画布信息");
  const canvas = {
    width: positiveNumber(value.canvas.width, "画布宽度"),
    height: positiveNumber(value.canvas.height, "画布高度"),
    originX: finiteNumber(value.canvas.originX, "画布原点 x"),
    originY: finiteNumber(value.canvas.originY, "画布原点 y"),
    pixelsPerUnit: positiveNumber(value.canvas.pixelsPerUnit, "画布像素密度"),
  };
  if (!isRecord(value.parameters)) throw new Error("母版缺少参数映射");
  const parameters = {
    eyeBlink: stringArray(value.parameters.eyeBlink, "眨眼参数"),
    lipSync: stringArray(value.parameters.lipSync, "口型参数"),
  };
  if (!Array.isArray(value.textureAtlases) || value.textureAtlases.length === 0) {
    throw new Error("母版至少需要一个纹理图集");
  }
  const textureAtlases = value.textureAtlases.map((item, index) => {
    if (!isRecord(item)) throw new Error(`纹理图集 ${index} 无效`);
    return {
      file: safeRelativePath(item.file, `纹理图集 ${index}`),
      width: positiveInteger(item.width, `纹理图集 ${index} 宽度`),
      height: positiveInteger(item.height, `纹理图集 ${index} 高度`),
    };
  });
  if (!Array.isArray(value.slots) || value.slots.length === 0) {
    throw new Error("母版至少需要一个部件槽位");
  }
  const ids = new Set<string>();
  const drawableIds = new Set<string>();
  const parseSlot = (raw: unknown): MasterTemplateSlot => {
    if (!isRecord(raw)) throw new Error("母版部件槽位无效");
    const slotId = requiredText(raw.id, "槽位 ID");
    if (ids.has(slotId)) throw new Error(`槽位 ID 重复: ${slotId}`);
    ids.add(slotId);
    const drawableId = requiredText(raw.drawableId, `槽位 ${slotId} Drawable ID`);
    if (drawableIds.has(drawableId)) throw new Error(`Drawable ID 重复: ${drawableId}`);
    drawableIds.add(drawableId);
    if (!isRecord(raw.atlas) || !isRecord(raw.anchor)) {
      throw new Error(`槽位 ${slotId} 缺少图集或锚点`);
    }
    const texture = nonNegativeInteger(raw.atlas.texture, `槽位 ${slotId} 纹理索引`);
    const atlas = textureAtlases[texture];
    if (!atlas) throw new Error(`槽位 ${slotId} 指向不存在的纹理图集`);
    const region = {
      texture,
      x: nonNegativeInteger(raw.atlas.x, `槽位 ${slotId} x`),
      y: nonNegativeInteger(raw.atlas.y, `槽位 ${slotId} y`),
      width: positiveInteger(raw.atlas.width, `槽位 ${slotId} 宽度`),
      height: positiveInteger(raw.atlas.height, `槽位 ${slotId} 高度`),
    };
    if (region.x + region.width > atlas.width || region.y + region.height > atlas.height) {
      throw new Error(`槽位 ${slotId} 超出纹理图集范围`);
    }
    if (!isRecord(raw.canvasBounds) || !isRecord(raw.mesh)) {
      throw new Error(`槽位 ${slotId} 缺少画布范围或网格`);
    }
    const vertices = finiteNumberArray(raw.mesh.vertices, `槽位 ${slotId} 顶点`);
    const uvs = finiteNumberArray(raw.mesh.uvs, `槽位 ${slotId} UV`);
    const indices = nonNegativeIntegerArray(raw.mesh.indices, `槽位 ${slotId} 三角形索引`);
    if (vertices.length < 6 || vertices.length % 2 !== 0 || uvs.length !== vertices.length) {
      throw new Error(`槽位 ${slotId} 网格顶点或 UV 无效`);
    }
    if (indices.length < 3 || indices.length % 3 !== 0) {
      throw new Error(`槽位 ${slotId} 三角形索引无效`);
    }
    const vertexCount = vertices.length / 2;
    if (indices.some((index) => index >= vertexCount)) {
      throw new Error(`槽位 ${slotId} 三角形索引越界`);
    }
    const group = requiredText(raw.group, `槽位 ${slotId} 分组`);
    if (!MASTER_TEMPLATE_PART_GROUPS.includes(group as MasterTemplatePartGroup)) {
      throw new Error(`槽位 ${slotId} 分组无效`);
    }
    const source = requiredText(raw.source, `槽位 ${slotId} 素材来源`);
    if (source !== "visible" && source !== "completion" && source !== "template") {
      throw new Error(`槽位 ${slotId} 素材来源无效`);
    }
    return {
      id: slotId,
      name: requiredText(raw.name, `槽位 ${slotId} 名称`),
      drawableId,
      partId: requiredText(raw.partId, `槽位 ${slotId} Part ID`),
      group: group as MasterTemplatePartGroup,
      source,
      required: Boolean(raw.required),
      zIndex: finiteNumber(raw.zIndex, `槽位 ${slotId} 层级`),
      atlas: region,
      anchor: {
        x: finiteNumber(raw.anchor.x, `槽位 ${slotId} 锚点 x`),
        y: finiteNumber(raw.anchor.y, `槽位 ${slotId} 锚点 y`),
      },
      canvasBounds: {
        x: finiteNumber(raw.canvasBounds.x, `槽位 ${slotId} 画布 x`),
        y: finiteNumber(raw.canvasBounds.y, `槽位 ${slotId} 画布 y`),
        width: positiveNumber(raw.canvasBounds.width, `槽位 ${slotId} 画布宽度`),
        height: positiveNumber(raw.canvasBounds.height, `槽位 ${slotId} 画布高度`),
      },
      mesh: { vertices, uvs, indices },
      silhouetteMask: safeRelativePath(raw.silhouetteMask, `槽位 ${slotId} 轮廓蒙版`),
      completionPrompt: requiredText(raw.completionPrompt, `槽位 ${slotId} 补全提示`),
      editPartId: typeof raw.editPartId === "string" && raw.editPartId.trim()
        ? raw.editPartId.trim()
        : undefined,
      textureUnitId: typeof raw.textureUnitId === "string" && raw.textureUnitId.trim()
        ? raw.textureUnitId.trim()
        : undefined,
    };
  };
  return {
    schemaVersion: 2,
    id,
    name,
    modelEntry,
    alignmentGuide,
    coverage: value.coverage,
    canvas,
    parameters,
    textureAtlases,
    slots: value.slots.map(parseSlot),
  };
}

function safeRelativePath(value: unknown, label: string): string {
  const text = requiredText(value, label).replaceAll("\\", "/");
  if (text.startsWith("/") || text.split("/").some((part) => part === ".." || !part)) {
    throw new Error(`${label} 必须是安全相对路径`);
  }
  return text;
}

function requiredText(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} 不能为空`);
  return value.trim();
}

function positiveInteger(value: unknown, label: string): number {
  const parsed = nonNegativeInteger(value, label);
  if (parsed === 0) throw new Error(`${label} 必须大于 0`);
  return parsed;
}

function positiveNumber(value: unknown, label: string): number {
  const parsed = finiteNumber(value, label);
  if (parsed <= 0) throw new Error(`${label} 必须大于 0`);
  return parsed;
}

function stringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) throw new Error(`${label} 无效`);
  return value.map((item, index) => requiredText(item, `${label} ${index}`));
}

function finiteNumberArray(value: unknown, label: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${label} 无效`);
  return value.map((item, index) => finiteNumber(item, `${label} ${index}`));
}

function nonNegativeIntegerArray(value: unknown, label: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${label} 无效`);
  return value.map((item, index) => nonNegativeInteger(item, `${label} ${index}`));
}

function nonNegativeInteger(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new Error(`${label} 必须是非负整数`);
  }
  return value;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} 无效`);
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
