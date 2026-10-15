import catalog from "./facial-demo-parameters.json";
import type { NativeModelParameter } from "./contracts";

export interface ManualParameterBinding {
  parameterId: string;
  lowerValue: number;
  neutralValue: number;
  upperValue: number;
}
/**
 * A part's own list of model parameters, VTube Studio style: every entry maps
 * the part's weakest, usual and strongest step to that parameter's values.
 */
export type ManualParameterBindings = Record<string, ManualParameterBinding[]>;
/** Enough for a smile that also narrows the eyes and lifts the cheeks. */
export const MAX_PART_OUTPUTS = 6;
/**
 * How a part moves once Jev picks its step: stay there for the sentence, sway
 * between rest and the step (a waving arm), or reach the step and come back
 * (a nod, a shrug).
 */
export type FacialMotionStyle = "hold" | "swing" | "pulse";
export const FACIAL_MOTION_STYLES: readonly FacialMotionStyle[] = ["hold", "swing", "pulse"];
const motionStyle = (value: unknown): value is FacialMotionStyle => FACIAL_MOTION_STYLES.includes(value as FacialMotionStyle);

export interface FacialParameterSettings {
  enabled: boolean;
  multiplier: number;
  /** Deleted from the 说话表情 list; Jev leaves it alone until it is added back. */
  removed?: boolean;
  /** The person's own name for the part; display only, Jev reads the description. */
  label?: string;
  /** Absent means hold. */
  motion?: FacialMotionStyle;
}
export type FacialParameterSettingsMap = Record<string, FacialParameterSettings>;
export type FacialDescriptionOverrides = Record<string, string>;
export type FacialDecisionRule =
  | { mode: "score"; levels: string[] }
  | { mode: "binary"; falseDescription: string; trueDescription: string };
export type FacialCriteriaOverrides = Record<string, FacialDecisionRule>;
export interface CustomFacialParameter {
  /** An actual parameter ID from the selected model, used directly as the JEV question ID. */
  id: string;
  label: string;
  description: string;
  levels: string[];
  mode?: "score" | "binary";
  falseDescription?: string;
  trueDescription?: string;
  enabled: boolean;
  multiplier: number;
  /** Absent means hold. */
  motion?: FacialMotionStyle;
}
export interface FacialBindingConfiguration {
  manualBindings: ManualParameterBindings;
  parameterSettings: FacialParameterSettingsMap;
  descriptions: FacialDescriptionOverrides;
  criteria: FacialCriteriaOverrides;
  customParameters: CustomFacialParameter[];
}
export const DEFAULT_FACIAL_PARAMETER_SETTINGS: Readonly<FacialParameterSettings> = { enabled: true, multiplier: 1 };
const definitions = new Map(catalog.map(value => [value.id, value]));
export const AUTOMATIC_FACIAL_IDS = new Set(catalog.filter(value => value.owner !== "jev").map(value => value.id));

/** Local animation calibration, never included in model input. */
/** Parts that move with the voice as a speaker's do: nods and intonation, glances, raised brows. */
export type SpeechPart = "headPitch" | "eyeX" | "eyeY" | "brow";

export interface FacialMotionBindings {
  mouth: { id: string; closed: number; open: number } | null;
  eyes: Array<{ id: string; closed: number }>;
  transitions?: Record<string, {
    group: string; minimum: number; maximum: number; motion?: Exclude<FacialMotionStyle, "hold">;
    /**
     * The part of a speaker's face this parameter carries — head pitch (in
     * degrees), gaze or brow height (in the standard -1..1) — and how far it
     * moves per unit up and per unit down, for the motion that follows the voice.
     */
    follow?: { part: SpeechPart; up: number; down: number };
  }>;
}

export interface FacialOverlayBindings extends FacialMotionBindings {
  neutralValues: Record<string, number>;
  gazeParameterIds: string[];
}

export function assertManualStandardId(id: string): void {
  if (definitions.get(id)?.owner !== "jev") throw new Error("只能绑定由决策模型控制的标准部位");
}

function validPromptText(value: unknown, limit: number): value is string {
  return typeof value === "string" && !!value.trim() && value.length <= limit && !value.includes("\0");
}

export function validateDescriptionOverrides(value: unknown): FacialDescriptionOverrides {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("参数描述格式无效");
  return Object.fromEntries(Object.entries(value).map(([id, description]) => {
    assertManualStandardId(id);
    if (!validPromptText(description, 500)) throw new Error("参数描述须为 1–500 字");
    return [id, description.trim()];
  }));
}

export function validateCriteriaOverrides(value: unknown): FacialCriteriaOverrides {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("评分选项格式无效");
  return Object.fromEntries(Object.entries(value).map(([id, raw]) => {
    assertManualStandardId(id);
    const rule = Array.isArray(raw) ? { mode: "score", levels: raw } : raw;
    return [id, validateDecisionRule(rule)];
  }));
}

export function validateDecisionRule(value: unknown): FacialDecisionRule {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("参数判断模式无效");
  const rule = value as Record<string, unknown>;
  if (rule.mode === "score" && Object.keys(rule).sort().join(",") === "levels,mode"
    && Array.isArray(rule.levels) && rule.levels.length >= 2 && rule.levels.length <= 10
    && rule.levels.every(level => validPromptText(level, 100))
    && new Set(rule.levels.map(level => level.trim())).size === rule.levels.length) {
    return { mode: "score", levels: rule.levels.map(level => level.trim()) };
  }
  if (rule.mode === "binary" && Object.keys(rule).sort().join(",") === "falseDescription,mode,trueDescription"
    && validPromptText(rule.falseDescription, 100) && validPromptText(rule.trueDescription, 100)
    && rule.falseDescription.trim() !== rule.trueDescription.trim()) {
    return { mode: "binary", falseDescription: rule.falseDescription.trim(), trueDescription: rule.trueDescription.trim() };
  }
  throw new Error("渐进式需要 2–10 个不同描述；选择式需要不同的是/否描述");
}

export function validateCustomParameters(value: unknown): CustomFacialParameter[] {
  if (!Array.isArray(value) || value.length > 16) throw new Error("最多添加 16 个自定义参数");
  const seen = new Set<string>();
  return value.map(raw => {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("自定义参数格式无效");
    const entry = raw as CustomFacialParameter;
    const keys = Object.keys(entry);
    if (keys.some(key => !["description", "enabled", "id", "label", "levels", "multiplier", "mode", "falseDescription", "trueDescription", "motion"].includes(key))
      || !validPromptText(entry.id, 100) || /[\r\n]/u.test(entry.id)
      || ["__proto__", "constructor", "prototype"].includes(entry.id)
      || definitions.has(entry.id) || seen.has(entry.id)
      || !validPromptText(entry.label, 100)
      || !validPromptText(entry.description, 500)
      || !Array.isArray(entry.levels) || entry.levels.length < 2 || entry.levels.length > 10
      || !entry.levels.every(level => validPromptText(level, 100))
      || new Set(entry.levels.map(level => level.trim())).size !== entry.levels.length
      || (entry.mode !== undefined && entry.mode !== "score" && entry.mode !== "binary")
      || (entry.mode === "binary" && (!validPromptText(entry.falseDescription, 100)
        || !validPromptText(entry.trueDescription, 100)
        || entry.falseDescription.trim() === entry.trueDescription.trim()))
      || (entry.mode !== "binary" && (entry.falseDescription !== undefined || entry.trueDescription !== undefined))
      || typeof entry.enabled !== "boolean"
      || typeof entry.multiplier !== "number" || !Number.isFinite(entry.multiplier)
      || entry.multiplier < 0 || entry.multiplier > 100
      || (entry.motion !== undefined && !motionStyle(entry.motion))) {
      throw new Error("自定义参数须绑定非标准原生 ID，并提供名称、描述、判断模式及有效倍率");
    }
    seen.add(entry.id);
    return { id: entry.id, label: entry.label.trim(), description: entry.description.trim(),
      levels: entry.levels.map(level => level.trim()),
      mode: entry.mode ?? "score",
      ...(entry.mode === "binary" ? { falseDescription: entry.falseDescription!.trim(), trueDescription: entry.trueDescription!.trim() } : {}),
      enabled: entry.enabled, multiplier: entry.multiplier,
      ...entry.motion && entry.motion !== "hold" ? { motion: entry.motion } : {} };
  });
}

export function validateParameterSettings(value: unknown): FacialParameterSettingsMap {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("参数设置格式无效");
  return Object.fromEntries(Object.entries(value).map(([id, raw]) => {
    assertManualStandardId(id);
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("参数设置无效");
    const entry = raw as FacialParameterSettings;
    const keys = Object.keys(entry).filter((key) => key !== "label" && key !== "motion").sort().join(",");
    if ((keys !== "enabled,multiplier" && keys !== "enabled,multiplier,removed") || typeof entry.enabled !== "boolean"
      || typeof entry.multiplier !== "number" || !Number.isFinite(entry.multiplier) || entry.multiplier < 0
      || (entry.removed !== undefined && typeof entry.removed !== "boolean")
      || (entry.label !== undefined && typeof entry.label !== "string")
      || (entry.motion !== undefined && !motionStyle(entry.motion))) {
      throw new Error("自动生成须为开关，倍率须为大于等于 0 的有效数值");
    }
    const label = entry.label?.trim().slice(0, 30);
    // A removed part is never generated, whatever its switch said before.
    return [id, {
      ...(entry.removed
        ? { enabled: false, multiplier: entry.multiplier, removed: true }
        : { enabled: entry.enabled, multiplier: entry.multiplier }),
      ...label ? { label } : {},
      // Kept even when "hold": it may undo a part's own default, like a shrug's.
      ...entry.motion ? { motion: entry.motion } : {},
    }];
  }));
}

export function validateManualBindings(value: unknown): ManualParameterBindings {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("手动绑定格式无效");
  return Object.fromEntries(Object.entries(value).flatMap(([id, raw]) => {
    assertManualStandardId(id);
    // Saved before a part could drive several parameters: one entry, not a list.
    const outputs = (Array.isArray(raw) ? raw : [raw]).map(entry => validateOutput(id, entry));
    if (outputs.length > MAX_PART_OUTPUTS) throw new Error(`一个部位最多对应 ${MAX_PART_OUTPUTS} 个参数`);
    if (new Set(outputs.map(output => output.parameterId)).size !== outputs.length) throw new Error("同一个参数在一个部位里只能出现一次");
    return outputs.length ? [[id, outputs]] : [];
  }));
}

function validateOutput(id: string, raw: unknown): ManualParameterBinding {
  const definition = definitions.get(id)!;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("绑定参数无效");
  const entry = raw as ManualParameterBinding;
  if (Object.keys(entry).sort().join(",") !== "lowerValue,neutralValue,parameterId,upperValue"
    || typeof entry.parameterId !== "string" || !entry.parameterId.trim() || entry.parameterId.length > 200
    || ![entry.lowerValue, entry.neutralValue, entry.upperValue].every(value => typeof value === "number" && Number.isFinite(value))) {
    throw new Error("请选择原生参数并填写有效数值");
  }
  const low = Math.min(entry.lowerValue, entry.upperValue), high = Math.max(entry.lowerValue, entry.upperValue);
  if (low === high || entry.neutralValue < low || entry.neutralValue > high
    || definition.neutral === definition.min && entry.neutralValue !== entry.lowerValue
    || definition.neutral === definition.max && entry.neutralValue !== entry.upperValue) {
    throw new Error("端点必须不同，中性值须位于两个端点之间");
  }
  return { parameterId: entry.parameterId, lowerValue: entry.lowerValue,
    neutralValue: entry.neutralValue, upperValue: entry.upperValue };
}

export function initialManualBinding(id: string, parameter: NativeModelParameter): ManualParameterBinding {
  const definition = definitions.get(id)!;
  const neutral = definition.neutral === definition.min ? parameter.minimum
    : definition.neutral === definition.max ? parameter.maximum
    : Math.max(parameter.minimum, Math.min(parameter.maximum, parameter.defaultValue));
  return { parameterId: parameter.id,
    lowerValue: definition.neutral === definition.min ? neutral : parameter.minimum,
    neutralValue: neutral,
    upperValue: definition.neutral === definition.max ? neutral : parameter.maximum };
}

export function validateNativeBinding(id: string, binding: ManualParameterBinding, native: readonly NativeModelParameter[], blocked: Iterable<string>): void {
  validateManualBindings({ [id]: binding });
  const parameter = native.find(value => value.id === binding.parameterId);
  if (!parameter) throw new Error("绑定的原生参数已不存在，请重新选择");
  if (AUTOMATIC_FACIAL_IDS.has(parameter.id) || new Set(blocked).has(parameter.id)) throw new Error("此参数由呼吸或物理模拟控制，不能绑定");
  if (![parameter.minimum, parameter.maximum].every(Number.isFinite) || parameter.minimum >= parameter.maximum
    || [binding.lowerValue, binding.neutralValue, binding.upperValue].some(value => value < parameter.minimum || value > parameter.maximum)) {
    throw new Error(`数值须位于原生参数范围 ${parameter.minimum}～${parameter.maximum} 内`);
  }
}
