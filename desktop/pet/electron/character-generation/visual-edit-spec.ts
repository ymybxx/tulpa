import {
  drawnClauseIds,
  TEXTURE_VISUAL_ATTRIBUTE_CATEGORIES,
  type TextureClauseAction,
  type TextureClausePlan,
  type TextureEditImageModel,
  type TexturePromptClause,
  type TextureAvailableReference,
  type TextureRegionEdit,
  type TextureStyleProfile,
  type TextureVisualAttribute,
  type TextureVisualAttributeCategory,
  type TextureVisualEditIntent,
  type TextureVisualEditSpec,
} from "../../src/shared/character-generation";

const VISUAL_INTENTS = new Set<TextureVisualEditIntent>([
  "recolor",
  "style-transfer",
  "material-transfer",
  "cleanup",
  "alpha-remove",
]);
const CLAUSE_ACTIONS = new Set<TextureClauseAction>(["repaint", "recolor", "toggle", "unsupported", "unselected"]);
const CATEGORY_LABELS: Record<TextureVisualAttributeCategory, string> = {
  palette: "配色",
  rendering: "渲染",
  shading: "光影",
  linework: "线稿",
  material: "材质",
  motif: "图案",
  tone: "整体气质",
};

export interface TextureVisualEditSpecInput {
  originalPrompt: string;
  clauses: TexturePromptClause[];
  imageModel: TextureEditImageModel;
  positionReferenceAvailable: boolean;
  regions: ReadonlyArray<{ key: string; label: string }>;
  references: ReadonlyArray<TextureAvailableReference>;
}

export function splitTexturePromptClauses(prompt: string): TexturePromptClause[] {
  const parts = prompt.split(/[\n，,。；;！!？?]+/u)
    .map((part) => part.trim())
    .filter(Boolean);
  const clauses = parts.length > 0 ? parts : [prompt.trim()].filter(Boolean);
  return clauses.map((text, index) => ({ id: `C${index + 1}`, text }));
}

export function validateTextureVisualEditSpec(
  raw: unknown,
  input: TextureVisualEditSpecInput,
): TextureVisualEditSpec {
  const root = record(raw, "优化结果");
  const intent = requiredString(root.intent, "intent", 40) as TextureVisualEditIntent;
  if (!VISUAL_INTENTS.has(intent)) throw new Error(`未知编辑意图：${intent}`);
  if (intent === "alpha-remove" && !/(删除|去掉|去除|移除|透明|露出)/u.test(input.originalPrompt)) {
    throw new Error("当前用户要求未授权透明删除，不能启用 alpha-remove");
  }

  const styleProfiles = validateStyleProfiles(root.styleProfiles);
  const styleProfileIds = new Set(styleProfiles.map((profile) => profile.id));
  const regions = validateRegions(root.regions, input, styleProfileIds);

  const actions = validateActions(root.actions, input);
  return {
    intent,
    styleProfiles,
    regions,
    // Warnings are optional notes; an answer without them is still a whole answer.
    warnings: root.warnings === undefined || root.warnings === null ? [] : stringArray(root.warnings, "warnings", 8, 240),
    ...actions.length ? { actions } : {},
  };
}

/** One plan per known sentence; an unknown action falls back to repainting, which is what happened before. */
function validateActions(value: unknown, input: TextureVisualEditSpecInput): TextureClausePlan[] {
  if (!Array.isArray(value)) return [];
  const known = new Set(input.clauses.map((clause) => clause.id));
  const plans = new Map<string, TextureClausePlan>();
  for (const item of value) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const raw = item as Record<string, unknown>;
    const clauseId = optionalString(raw.clauseId, 16).toUpperCase();
    if (!known.has(clauseId) || plans.has(clauseId)) continue;
    const action = optionalString(raw.action, 20) as TextureClauseAction;
    const part = optionalString(raw.part, 40);
    plans.set(clauseId, {
      clauseId,
      action: CLAUSE_ACTIONS.has(action) ? action : "repaint",
      ...part ? { part } : {},
      note: optionalString(raw.note, 120),
    });
  }
  return input.clauses.flatMap((clause) => plans.get(clause.id) ?? []);
}

export function compileTextureVisualPrompt(
  spec: TextureVisualEditSpec,
  input: TextureVisualEditSpecInput,
): string {
  const clauses = new Map(input.clauses.map((clause) => [clause.id, clause.text]));
  // Sentences handled elsewhere (a switch, another part, a shape change) never reach the image model.
  const drawn = drawnClauseIds(spec, input.clauses.map((clause) => clause.id));
  const request = drawn.length === input.clauses.length
    ? input.originalPrompt
    : drawn.map((id) => clauses.get(id)).join("；");
  const lines = [
    "任务：",
    `编辑输入图 1 中标记的 Live2D UV 纹理。用户要求：${request}`,
    "",
    "参考：",
  ];

  lines.push(input.references.length > 0
    ? "- 本次用户选择的参考图只用于补充视觉方向；具体发送由用户选择和发送层管理，不改变区域结构。"
    : "- 本次没有额外参考图；视觉方向直接根据用户要求和选中纹理之间的现有视觉关系归纳。");

  for (const profile of spec.styleProfiles) {
    lines.push("", `视觉方向 ${profile.id}（${profile.scope}）：`);
    for (const category of TEXTURE_VISUAL_ATTRIBUTE_CATEGORIES) {
      const attribute = profile.attributes[category];
      if (attribute) lines.push(`- ${categoryLabel(category)}：${attribute.description}。`);
    }
  }

  // Whatever the request doesn't touch stays as the original draws it.
  const touched = new Set(spec.styleProfiles.flatMap((profile) => Object.keys(profile.attributes)));
  const kept = TEXTURE_VISUAL_ATTRIBUTE_CATEGORIES.filter((category) => !touched.has(category));
  // Changing style or material redesigns the surface; only a recolor or cleanup must not.
  const redesign = spec.intent === "style-transfer" || spec.intent === "material-transfer";
  if (kept.length) lines.push("", `保持原样：${kept.map(categoryLabel).join("、")}与原纹理一致${redesign ? "。" : "，不要重新设计。"}`);

  lines.push("", "区域：");
  for (const region of spec.regions) {
    if (region.keep) {
      lines.push(`- ${region.key}（${region.role}）：保持原样，不要改动。`);
      continue;
    }
    const parts: string[] = [];
    if (region.styleProfileIds.length > 0) parts.push(`应用 ${region.styleProfileIds.join("、")}`);
    const regionClauses = drawnClauseIds(spec, region.clauseIds);
    if (regionClauses.length > 0) {
      parts.push(regionClauses.map((id) => `${id}「${clauses.get(id) ?? id}」`).join("；执行 "));
    }
    lines.push(`- ${region.key}（${region.role}）：${parts.length ? parts.join("；") : "保持原样"}。`);
  }

  const continuityGroups = new Map<string, string[]>();
  for (const region of spec.regions) {
    if (!region.continuityGroup) continue;
    const members = continuityGroups.get(region.continuityGroup) ?? [];
    members.push(region.key);
    continuityGroups.set(region.continuityGroup, members);
  }
  const continuityLines = [...continuityGroups.values()].filter((members) => members.length > 1);
  if (continuityLines.length > 0) {
    lines.push("", "一致性：");
    for (const members of continuityLines) {
      lines.push(`- ${members.join("、")} 属于同一视觉连续组，配色、图案尺度、线稿和材质表现保持一致。`);
    }
  }

  lines.push(
    "",
    "技术约束：",
    `只修改 ${spec.regions.filter((region) => !region.keep).map((region) => region.key).join("、")}；保持原 UV 位置、比例和连接关系；除用户明确要求删除的部分外，完整覆盖原可见像素并保持外轮廓，不得偏移、跨区域串色或引入用户未要求的背景和物体。`,
    spec.intent === "alpha-remove"
      ? "仅按用户明确要求缩小 Alpha，不得扩大原透明轮廓。"
      : "保持原始 Alpha 和透明轮廓不变。",
  );
  return lines.join("\n").slice(0, 8_000);
}

function validateStyleProfiles(
  value: unknown,
): TextureStyleProfile[] {
  const rawProfiles = array(value, "styleProfiles");
  const seen = new Set<string>();
  return rawProfiles.map((value, index): TextureStyleProfile => {
    const raw = record(value, `styleProfiles[${index}]`);
    const id = requiredString(raw.id, `styleProfiles[${index}].id`, 24).toUpperCase();
    if (!/^S[1-9]\d*$/u.test(id)) throw new Error(`视觉方向编号无效：${id}`);
    if (seen.has(id)) throw new Error(`视觉方向 ${id} 重复`);
    seen.add(id);
    const attributesRaw = record(raw.attributes, `${id}.attributes`);
    const attributes: TextureStyleProfile["attributes"] = {};
    for (const category of TEXTURE_VISUAL_ATTRIBUTE_CATEGORIES) {
      const candidate = attributesRaw[category];
      if (candidate === undefined) continue;
      attributes[category] = validateVisualAttribute(candidate, category);
    }
    if (Object.keys(attributes).length === 0) throw new Error(`视觉方向 ${id} 没有可用视觉属性`);
    return { id, scope: requiredString(raw.scope, `${id}.scope`, 160), attributes };
  });
}

function validateVisualAttribute(
  value: unknown,
  category: TextureVisualAttributeCategory,
): TextureVisualAttribute {
  const raw = record(value, `${category} attribute`);
  const description = requiredString(raw.description, `${category}.description`, 360);
  return { description };
}

function validateRegions(
  value: unknown,
  input: TextureVisualEditSpecInput,
  styleProfileIds: ReadonlySet<string>,
): TextureRegionEdit[] {
  const rawRegions = array(value, "regions");
  const rawByKey = new Map<string, Record<string, unknown>>();
  for (const [index, value] of rawRegions.entries()) {
    const raw = record(value, `regions[${index}]`);
    const key = requiredString(raw.key, `regions[${index}].key`, 12).toUpperCase();
    if (rawByKey.has(key)) throw new Error(`区域 ${key} 重复`);
    rawByKey.set(key, raw);
  }
  // A region the model invented never reaches the prompt, so it is dropped
  // instead of failing the whole answer; a missing selected region still fails.
  const validClauses = new Set(input.clauses.map((clause) => clause.id));
  return input.regions.map((region): TextureRegionEdit => {
    const key = region.key.toUpperCase();
    const raw = rawByKey.get(key);
    // A piece the answer skipped is drawn as it is and left off, instead of failing the whole answer.
    if (!raw) return { key, role: region.label, styleProfileIds: [], clauseIds: [], keep: "AI 没给这块写要求，先保持原样" };
    const profileIds = stringArray(raw.styleProfileIds, `${key}.styleProfileIds`, 12, 24)
      .map((id) => id.toUpperCase());
    for (const id of profileIds) if (!styleProfileIds.has(id)) throw new Error(`区域 ${key} 引用了未知视觉方向 ${id}`);
    const clauseIds = stringArray(raw.clauseIds, `${key}.clauseIds`, 32, 16)
      .map((id) => id.toUpperCase());
    for (const id of clauseIds) if (!validClauses.has(id)) throw new Error(`区域 ${key} 引用了未知用户要求 ${id}`);
    // A piece the request doesn't cover stays as it is, with the reason shown to the user;
    // one the answer gives nothing to do is the same: the request leaves it alone.
    const keep = optionalString(raw.keep, 120) || (profileIds.length === 0 && clauseIds.length === 0 ? "这次的要求没有涉及这块" : "");
    const continuityGroup = optionalString(raw.continuityGroup, 80);
    return {
      key,
      role: requiredString(raw.role, `${key}.role`, 80),
      styleProfileIds: keep ? [] : unique(profileIds),
      clauseIds: keep ? [] : unique(clauseIds),
      ...(continuityGroup && !keep ? { continuityGroup } : {}),
      ...(keep ? { keep } : {}),
    };
  });
}

function categoryLabel(category: TextureVisualAttributeCategory): string {
  return CATEGORY_LABELS[category];
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${path} 必须是数组`);
  return value;
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    const actualType = value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
    const serialized = JSON.stringify(value) ?? String(value);
    throw new Error(
      `${path} 必须是对象（实际类型：${actualType}；实际值：${serialized.slice(0, 160)}）`,
    );
  }
  return value as Record<string, unknown>;
}

function requiredString(value: unknown, path: string, maxLength: number): string {
  const result = optionalString(value, maxLength);
  if (!result) throw new Error(`${path} 不能为空`);
  return result;
}

function optionalString(value: unknown, maxLength: number): string {
  return typeof value === "string"
    ? value.replace(/[\r\n]+/gu, " ").replace(/\s+/gu, " ").trim().slice(0, maxLength)
    : "";
}

function stringArray(
  value: unknown,
  path: string,
  maxItems: number,
  maxLength: number,
): string[] {
  if (!Array.isArray(value)) throw new Error(`${path} 必须是数组`);
  if (value.length > maxItems) throw new Error(`${path} 数量过多`);
  return value.map((item, index) => requiredString(item, `${path}[${index}]`, maxLength));
}

function unique<T>(values: readonly T[]): T[] {
  return [...new Set(values)];
}
