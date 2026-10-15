import catalog from "../shared/facial-demo-parameters.json";
import type { NativeModelParameter } from "../shared/contracts";
import type { FacialDecisionParameter } from "../shared/facial-test";
import type { ConflictPart } from "../shared/facial-conflicts";
import { vtsPartBindings, type VtsParameterSetting } from "../shared/vts-model-settings";
import { AUTOMATIC_FACIAL_IDS, DEFAULT_FACIAL_PARAMETER_SETTINGS, validateCriteriaOverrides, validateCustomParameters, validateDescriptionOverrides, validateNativeBinding, type CustomFacialParameter, type FacialCriteriaOverrides, type FacialDecisionRule, type FacialDescriptionOverrides, type FacialMotionStyle, type FacialParameterSettingsMap, type FacialMotionBindings, type FacialOverlayBindings, type ManualParameterBinding, type ManualParameterBindings, type SpeechPart } from "../shared/facial-bindings";

/** Who drives a standard part: JEV per sentence, the model's physics, automatic breathing, or the voice (mouth shapes). */
export type ParameterOwner = "jev" | "physics" | "breath" | "speech";
export interface StandardParameterDefinition {
  id: string;
  label: string;
  group: string;
  min: number;
  max: number;
  neutral: number;
  owner: ParameterOwner;
  description: string;
  levels: string[];
  anchors: number[];
  /**
   * Left and right mean nothing in dialogue: Jev is asked how much (0..max)
   * and the renderer picks the side, shared along one axis.
   */
  sided?: { axis: string; description: string; levels: string[]; anchors: number[] };
  /** How the part moves unless the person chose otherwise; absent means hold. */
  motion?: FacialMotionStyle;
}
/** One model parameter a part drives; without its own curve the part's value is used as is. */
export interface ParameterOutput {
  parameterId: string;
  manual: ManualParameterBinding | null;
}
export interface ParameterBinding extends StandardParameterDefinition {
  decisionMode: "score" | "binary";
  binaryCriteria?: { falseDescription: string; trueDescription: string };
  supported: boolean;
  actualId: string | null;
  defaultValue: number;
  controllable: boolean;
  reason: string;
  /** Every model parameter the part drives, the first one shown as its value. */
  outputs: ParameterOutput[];
  /** The parameters chosen by hand, as saved (some may no longer exist). */
  manualBindings?: ManualParameterBinding[];
  /** The first output's curve, for the part's single-value views. */
  manualBinding?: ManualParameterBinding;
  /** Wired as the model's own VTube Studio settings wire it, not chosen here. */
  fromVts?: boolean;
  enabled: boolean;
  multiplier: number;
  /** Deleted from the 说话表情 list by the user. */
  removed?: boolean;
  motion?: FacialMotionStyle;
}
export interface ParameterBindings {
  all: ParameterBinding[];
  active: ParameterBinding[];
}

/**
 * The standard parts: VTube Studio's face inputs (head, eyes, gaze, brows,
 * smile, cheek puff, tongue) plus brow angle and blush for emotion, named by
 * their usual Live2D IDs; the mouth's opening and VBridger's mouth shapes
 * belong to speech. IDs and working ranges follow the official list
 * (https://docs.live2d.com/en/cubism-editor-manual/standard-parameter-list/),
 * including its allowed 0..2 eye-opening extension; each model is limited to
 * the intersection with its real native range.
 */
export const STANDARD_PARAMETERS: readonly StandardParameterDefinition[] = catalog as StandardParameterDefinition[];
const clamp = (value: number, minimum: number, maximum: number) => Math.max(minimum, Math.min(maximum, value));
const spacedAnchors = (min: number, max: number, count: number) =>
  Array.from({ length: count }, (_, index) => index === 0 ? min : index === count - 1 ? max
    : min + (max - min) * index / (count - 1));

function applyDecision(binding: ParameterBinding, rule?: FacialDecisionRule): ParameterBinding {
  if (!rule) return binding;
  if (rule.mode === "binary") return { ...binding, decisionMode: "binary",
    binaryCriteria: { falseDescription: rule.falseDescription, trueDescription: rule.trueDescription } };
  return { ...binding, decisionMode: "score", levels: rule.levels,
    anchors: rule.levels.length === binding.anchors.length ? binding.anchors : spacedAnchors(binding.min, binding.max, rule.levels.length),
    binaryCriteria: undefined };
}

/** Real native IDs only; the caller supplies output-only IDs from the author's physics rig. */
export function buildBindings(
  native: readonly NativeModelParameter[],
  blushParameter: string | null = null,
  blockedParameterIds: Iterable<string> = [],
  manualBindings: ManualParameterBindings = {},
  parameterSettings: FacialParameterSettingsMap = {},
  descriptions: FacialDescriptionOverrides = {},
  customParameters: CustomFacialParameter[] = [],
  criteria: FacialCriteriaOverrides = {},
  vtsSettings: readonly VtsParameterSetting[] = [],
): ParameterBindings {
  const parameters = new Map(native.map(parameter => [parameter.id, parameter]));
  const blocked = new Set(blockedParameterIds);
  const overrides = validateDescriptionOverrides(descriptions);
  const criterionOverrides = validateCriteriaOverrides(criteria);
  // Without a choice made here, a part follows the model author's VTS wiring before its name.
  const authored = vtsPartBindings(vtsSettings, native, blocked);
  const all = STANDARD_PARAMETERS.map((standard): ParameterBinding => {
    const definition = { ...standard, description: overrides[standard.id] ?? standard.description, decisionMode: "score" as const };
    const controls = parameterSettings[definition.id] ?? DEFAULT_FACIAL_PARAMETER_SETTINGS;
    const manual = manualBindings[definition.id];
    if (manual?.length) {
      const outputs: ParameterOutput[] = [];
      let failure = "";
      for (const output of manual) {
        try { validateNativeBinding(definition.id, output, native, blocked); outputs.push({ parameterId: output.parameterId, manual: output }); }
        catch (error) { failure ||= error instanceof Error ? error.message : String(error); }
      }
      const supported = outputs.length > 0;
      return { ...definition, ...controls, manualBindings: manual, manualBinding: outputs[0]?.manual ?? undefined, outputs,
        actualId: outputs[0]?.parameterId ?? null, supported, controllable: supported, defaultValue: definition.neutral,
        reason: supported ? "手动绑定" : failure };
    }
    const fromVts = definition.owner === "jev" ? authored[definition.id] : undefined;
    if (fromVts) {
      const outputs = fromVts.flatMap((output): ParameterOutput[] => {
        try { validateNativeBinding(definition.id, output, native, blocked); return [{ parameterId: output.parameterId, manual: output }]; }
        catch { return []; }
      });
      if (outputs.length) {
        return { ...definition, ...controls, manualBinding: outputs[0]!.manual!, outputs, fromVts: true,
          actualId: outputs[0]!.parameterId, supported: true, controllable: true, defaultValue: definition.neutral,
          reason: "按模型自带的 VTube Studio 设置" };
      }
    }
    const parameter = parameters.get(definition.id)
      ?? (definition.id === "ParamCheek" && blushParameter ? parameters.get(blushParameter) : undefined);
    const owner = parameter && blocked.has(parameter.id) ? "physics" : definition.owner;
    const min = parameter ? Math.max(definition.min, parameter.minimum) : definition.min;
    const max = parameter ? Math.min(definition.max, parameter.maximum) : definition.max;
    const supported = !!parameter && Number.isFinite(min) && Number.isFinite(max) && min <= max;
    const defaultValue = parameter?.defaultValue ?? definition.neutral;
    const neutral = supported ? clamp(defaultValue, min, max) : definition.neutral;
    const controllable = supported && owner === "jev" && min < max;
    const reason = !parameter ? "模型没有此参数"
      : !supported ? "模型范围与标准范围不相交"
      : owner === "physics" ? "由作者物理模拟控制"
      : owner === "breath" ? "由自动呼吸控制"
      : owner === "speech" ? "由语音口型驱动"
      : min === max ? "模型参数为固定值"
      : parameter.id !== definition.id ? "使用作者脸红资产的明确绑定"
      : "原生标准参数";
    return { ...definition, ...controls, owner, min, max, neutral, defaultValue, supported, controllable, actualId: parameter?.id ?? null, reason,
      outputs: parameter ? [{ parameterId: parameter.id, manual: null }] : [] };
  });
  for (let index = 0; index < all.length; index++) all[index] = applyDecision(all[index]!, criterionOverrides[all[index]!.id]);
  for (const custom of validateCustomParameters(customParameters)) {
    const parameter = parameters.get(custom.id);
    const min = parameter?.minimum ?? 0, max = parameter?.maximum ?? 1;
    const validRange = Number.isFinite(min) && Number.isFinite(max) && min < max;
    const supported = !!parameter && validRange;
    const blockedByOwner = blocked.has(custom.id) || AUTOMATIC_FACIAL_IDS.has(custom.id);
    const claimedByStandard = all.some(binding => binding.controllable && binding.outputs.some(output => output.parameterId === custom.id));
    const neutral = supported && Number.isFinite(parameter.defaultValue)
      ? clamp(parameter.defaultValue, min, max) : (min + max) / 2;
    const binaryCriteria = custom.mode === "binary"
      ? { falseDescription: custom.falseDescription!, trueDescription: custom.trueDescription! } : undefined;
    all.push({ ...custom, group: "custom", owner: "jev", min, max, neutral,
      decisionMode: custom.mode ?? "score", binaryCriteria,
      defaultValue: neutral, anchors: spacedAnchors(min, max, custom.levels.length),
      supported, controllable: supported && !blockedByOwner && !claimedByStandard, actualId: supported ? custom.id : null,
      outputs: supported ? [{ parameterId: custom.id, manual: null }] : [],
      reason: !parameter ? "模型已不再提供这个原生参数" : !validRange ? "原生参数范围无效"
        : blockedByOwner ? "此参数由呼吸或物理模拟控制"
        : claimedByStandard ? "此参数已绑定到标准部位" : "自定义原生参数" });
  }
  return { all, active: all.filter(binding => binding.controllable && binding.enabled) };
}

export function neutralTargets(bindings: readonly ParameterBinding[]): Record<string, number> {
  return Object.fromEntries(bindings.filter(binding => binding.controllable).map(binding => [binding.id, binding.neutral]));
}

export function decisionParameter(binding: ParameterBinding): FacialDecisionParameter {
  return {
    id: binding.id, minimum: binding.min, maximum: binding.max,
    defaultValue: binding.defaultValue, description: binding.description,
    mode: binding.decisionMode,
    ...(binding.decisionMode === "score" ? { levels: [...binding.levels], anchors: [...binding.anchors] }
      : { falseDescription: binding.binaryCriteria!.falseDescription, trueDescription: binding.binaryCriteria!.trueDescription }),
  };
}

/** Convert canonical JEV targets to only verified, controllable native parameters. */
export function toNativeTargets(
  targets: Readonly<Record<string, number>>,
  bindings: readonly ParameterBinding[],
): Record<string, number> {
  return mapNativeTargets(targets, bindings, true);
}

function mapNativeTargets(targets: Readonly<Record<string, number>>, bindings: readonly ParameterBinding[], generated: boolean): Record<string, number> {
  const grouped = new Map<string, number[]>();
  for (const binding of bindings) {
    const value = targets[binding.id];
    if (!binding.controllable || !binding.outputs.length || !Number.isFinite(value) || generated && !binding.enabled) continue;
    const adjusted = generated && binding.multiplier !== 1
      ? clamp(binding.neutral + (clamp(value!, binding.min, binding.max) - binding.neutral) * binding.multiplier, binding.min, binding.max)
      : value!;
    for (const output of binding.outputs) {
      const values = grouped.get(output.parameterId) ?? [];
      values.push(outputValue(adjusted, binding, output));
      grouped.set(output.parameterId, values);
    }
  }
  // A model may share one native input for both sides; do not let the last side win.
  return Object.fromEntries([...grouped].map(([id, values]) => [id, values.reduce((sum, value) => sum + value, 0) / values.length]));
}

function interpolate(value: number, from: number, to: number, lower: number, upper: number): number {
  return from === to ? lower : lower + (upper - lower) * clamp((value - from) / (to - from), 0, 1);
}

/** The part's value on its first parameter. */
export function toNativeValue(value: number, binding: ParameterBinding): number {
  return outputValue(value, binding, { parameterId: binding.actualId ?? "", manual: binding.manualBinding ?? null });
}

/** The part's value on one of its parameters, through that parameter's curve. */
export function outputValue(value: number, binding: ParameterBinding, output: ParameterOutput): number {
  const manual = output.manual;
  if (!manual) return clamp(value, binding.min, binding.max);
  return value <= binding.neutral && binding.neutral > binding.min
    ? interpolate(value, binding.min, binding.neutral, manual.lowerValue, manual.neutralValue)
    : interpolate(value, binding.neutral, binding.max, manual.neutralValue, manual.upperValue);
}

/** Used only by the local parameter display, including reversed and saturated rigs. */
export function fromNativeTargets(values: Readonly<Record<string, number>>, bindings: readonly ParameterBinding[]): Record<string, number> {
  return Object.fromEntries(bindings.filter(binding => binding.controllable).map(binding => {
    const value = values[binding.actualId!];
    const manual = binding.manualBinding;
    if (!Number.isFinite(value)) return [binding.id, binding.neutral];
    if (!manual) return [binding.id, clamp(value!, binding.min, binding.max)];
    if (value === manual.neutralValue) return [binding.id, binding.neutral];
    const lowerSide = (value! - manual.neutralValue) * (manual.upperValue - manual.lowerValue) < 0;
    return [binding.id, lowerSide
      ? interpolate(value!, manual.lowerValue, manual.neutralValue, binding.min, binding.neutral)
      : interpolate(value!, manual.neutralValue, manual.upperValue, binding.neutral, binding.max)];
  }));
}

/** The standard parts that move with the voice, and how far (in their units) to measure each one's reach. */
const SPEECH_PARTS: Record<string, { part: SpeechPart; step: number }> = {
  ParamAngleY: { part: "headPitch", step: 4 },
  ParamEyeBallX: { part: "eyeX", step: 0.5 }, ParamEyeBallY: { part: "eyeY", step: 0.5 },
  ParamBrowLY: { part: "brow", step: 0.5 }, ParamBrowRY: { part: "brow", step: 0.5 },
};

export function motionBindings(bindings: readonly ParameterBinding[]): FacialMotionBindings {
  const mouth = bindings.find(binding => binding.id === "ParamMouthOpenY" && binding.controllable);
  const eyes = new Map<string, number[]>();
  for (const binding of bindings) {
    if (!binding.controllable || !["ParamEyeLOpen", "ParamEyeROpen"].includes(binding.id)) continue;
    for (const output of binding.outputs) {
      const values = eyes.get(output.parameterId) ?? [];
      values.push(outputValue(binding.min, binding, output)); eyes.set(output.parameterId, values);
    }
  }
  const transitions: NonNullable<FacialMotionBindings["transitions"]> = {};
  // Shared native parameters use the slower semantic role, regardless of binding order.
  const priority: Record<string, number> = { gaze: 0, eyes: 1, brows: 2, mouth: 2, hands: 3, head: 4, arms: 4, cheek: 5, body: 6, overall: 6 };
  for (const binding of bindings) {
    if (!binding.controllable) continue;
    for (const output of binding.outputs) {
    const lower = outputValue(binding.min, binding, output), upper = outputValue(binding.max, binding, output);
    const previous = transitions[output.parameterId];
    const motion = previous?.motion ?? (binding.motion && binding.motion !== "hold" ? binding.motion : undefined);
    // Each parameter a speaking part moves moves with it, by the model's own wiring either side of rest.
    const speech = SPEECH_PARTS[binding.id];
    const at = (offset: number) => outputValue(binding.neutral + offset, binding, output);
    const follow = speech
      ? { part: speech.part, up: (at(speech.step) - at(0)) / speech.step, down: (at(0) - at(-speech.step)) / speech.step }
      : previous?.follow;
    transitions[output.parameterId] = {
      group: previous && (priority[previous.group] ?? 2) >= (priority[binding.group] ?? 2) ? previous.group : binding.group,
      minimum: Math.min(lower, upper, previous?.minimum ?? Infinity),
      maximum: Math.max(lower, upper, previous?.maximum ?? -Infinity),
      ...motion ? { motion } : {},
      ...follow && (follow.up || follow.down) ? { follow } : {},
    };
    }
  }
  return { mouth: mouth ? { id: mouth.actualId!, closed: toNativeValue(mouth.min, mouth), open: toNativeValue(mouth.max, mouth) } : null,
    eyes: [...eyes].map(([id, values]) => ({ id, closed: values.reduce((sum, value) => sum + value, 0) / values.length })), transitions };
}

/**
 * The speaking layer's parts. A part switched off is left to the model, except
 * head pitch: the switch only stops the decision model choosing it, and the
 * head still lifts and lowers a little with the voice's intonation.
 */
export function overlayBindings(all: readonly ParameterBinding[]): FacialOverlayBindings {
  const bindings = all.filter(binding => binding.enabled || binding.id === "ParamAngleY");
  return { ...motionBindings(bindings), neutralValues: mapNativeTargets(neutralTargets(bindings), bindings, false),
    gazeParameterIds: bindings.filter(binding => binding.controllable && ["ParamEyeBallX", "ParamEyeBallY"].includes(binding.id))
      .flatMap(binding => binding.outputs.map(output => output.parameterId)) };
}

/** The conflict rules' view of bindings: which model parameter each part drives, and its range. */
export function conflictParts(bindings: readonly ParameterBinding[]): ConflictPart[] {
  return bindings.map(binding => ({ id: binding.id, nativeId: binding.actualId, group: binding.group,
    neutral: binding.neutral, min: binding.min, max: binding.max }));
}

/**
 * Parts drawn in mirrored pairs whose two sides normally move together. They
 * are asked as one question; limbs stay per side so one arm can wave alone.
 */
const SYMMETRIC_GROUPS = new Set(["eyes", "brows"]);
const STANDARD_BY_ID = new Map(STANDARD_PARAMETERS.map(parameter => [parameter.id, parameter]));

/** A standard part still worded as shipped; a person's own wording is asked as written. */
function asShipped(binding: ParameterBinding): StandardParameterDefinition | null {
  const standard = STANDARD_BY_ID.get(binding.id);
  if (!standard || binding.group === "custom" || binding.decisionMode !== "score") return null;
  if (binding.description !== standard.description || binding.levels.length !== standard.levels.length
    || binding.levels.some((level, index) => level !== standard.levels[index])) return null;
  return standard;
}

/**
 * The side each axis leans to. A gesture keeps its side while it lasts; each
 * new one after a rest picks a side at random, so the head does not always
 * tilt the same way.
 */
export class DecisionSides {
  private readonly sides = new Map<string, 1 | -1>();
  private readonly active = new Map<string, boolean>();
  private readonly lastAt = new Map<string, number>();
  constructor(
    private readonly random: () => number = () => Math.random(),
    private readonly now: () => number = () => Date.now(),
  ) {}
  sign(axis: string, strength: number): 1 | -1 {
    const moving = strength >= 0.15;
    const at = this.now();
    // Jev rarely answers a flat zero, so a quiet gap between replies is the
    // usual rest; without it the first side chosen would stick for good.
    const rested = !this.active.get(axis) || at - (this.lastAt.get(axis) ?? -Infinity) > DECISION_REST_MS;
    let side = this.sides.get(axis) ?? 1;
    if (moving && rested) side = this.random() < 0.5 ? -1 : 1;
    this.sides.set(axis, side);
    this.active.set(axis, moving);
    this.lastAt.set(axis, at);
    return side;
  }
}

/** A pause this long between sentences ends a gesture, so the next may lean the other way. */
export const DECISION_REST_MS = 4_000;

export interface DecisionPlan {
  /** What Jev answers, one entry per question. */
  parameters: FacialDecisionParameter[];
  /** Jev's answers, one per question, back to one value per binding ID. */
  expand(answers: Readonly<Record<string, number>>): Record<string, number> | null;
}

/**
 * The questions Jev answers for these bindings: mirrored face parts as one
 * question for both sides, left/right parts as how much, everything else as
 * is. Parts a person reworded are asked exactly as they wrote them.
 */
export function planDecisions(bindings: readonly ParameterBinding[], sides = new DecisionSides()): DecisionPlan {
  const byId = new Map(bindings.map(binding => [binding.id, binding]));
  const parameters: FacialDecisionParameter[] = [];
  const expanders: Array<(answers: Readonly<Record<string, number>>, result: Record<string, number>) => boolean> = [];
  const sidedAnswers: Array<{ binding: ParameterBinding; axis: string; maximum: number }> = [];
  const done = new Set<string>();
  for (const binding of bindings) {
    if (done.has(binding.id)) continue;
    const shipped = asShipped(binding);
    const partnerId = shipped && SYMMETRIC_GROUPS.has(binding.group) ? mirrorId(binding.id) : null;
    const partner = partnerId ? byId.get(partnerId) : undefined;
    // Jev reads the question ID, so the pair is named for both sides: ParamEyeOpen, ParamBrowY.
    const id = binding.id.replace(/(Eye|Brow)L(?=[A-Z]|$)/, "$1");
    if (partner && asShipped(partner) && !done.has(partner.id) && !byId.has(id)) {
      done.add(binding.id); done.add(partner.id);
      parameters.push({ id, minimum: binding.anchors[0]!, maximum: binding.anchors.at(-1)!,
        defaultValue: clamp(binding.neutral, binding.anchors[0]!, binding.anchors.at(-1)!),
        description: bothSides(binding.description), mode: "score",
        levels: binding.levels.map(bothSides), anchors: [...binding.anchors] });
      expanders.push((answers, result) => {
        const value = answers[id];
        if (!Number.isFinite(value)) return false;
        result[binding.id] = clamp(value!, binding.min, binding.max);
        result[partner.id] = clamp(value!, partner.min, partner.max);
        return true;
      });
      continue;
    }
    done.add(binding.id);
    if (shipped?.sided) {
      const { axis, description, levels, anchors } = shipped.sided;
      const maximum = anchors.at(-1)!;
      parameters.push({ id: binding.id, minimum: anchors[0]!, maximum, defaultValue: anchors[0]!,
        description, mode: "score", levels: [...levels], anchors: [...anchors] });
      sidedAnswers.push({ binding, axis, maximum });
      continue;
    }
    parameters.push(decisionParameter(binding));
    expanders.push((answers, result) => {
      const value = answers[binding.id];
      if (!Number.isFinite(value)) return false;
      result[binding.id] = value!;
      return true;
    });
  }
  return {
    parameters,
    expand(answers) {
      // An answer outside its question is broken, not something to clamp into shape.
      if (parameters.some(question => !Number.isFinite(answers[question.id])
        || answers[question.id]! < question.minimum || answers[question.id]! > question.maximum)) return null;
      const result: Record<string, number> = {};
      for (const expand of expanders) if (!expand(answers, result)) return null;
      // One side per axis for this sentence, chosen from how strongly the axis moves.
      const strengths = new Map<string, number>();
      for (const { binding, axis, maximum } of sidedAnswers) {
        const value = answers[binding.id];
        if (!Number.isFinite(value)) return null;
        strengths.set(axis, Math.max(strengths.get(axis) ?? 0, Math.abs(value!) / maximum));
      }
      const signs = new Map([...strengths].map(([axis, strength]) => [axis, sides.sign(axis, strength)]));
      for (const { binding, axis } of sidedAnswers) {
        result[binding.id] = clamp(binding.neutral + signs.get(axis)! * Math.abs(answers[binding.id]!), binding.min, binding.max);
      }
      return result;
    },
  };
}

function mirrorId(id: string): string | null {
  const mirrored = id.replace(/(Eye|Brow)L(?=[A-Z]|$)/, "$1R");
  return mirrored === id ? null : mirrored;
}

function bothSides(text: string): string {
  return text.replaceAll("左眼", "双眼").replaceAll("左眉", "双眉");
}

