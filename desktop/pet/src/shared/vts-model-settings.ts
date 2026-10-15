import type { NativeModelParameter } from "./contracts";
import { AUTOMATIC_FACIAL_IDS, MAX_PART_OUTPUTS, type ManualParameterBindings, type ManualParameterBinding } from "./facial-bindings";

/**
 * One row of a VTube Studio model's own settings (`<model>.vtube.json`): a
 * face input, the range of it the author uses, and the Live2D parameter and
 * range it lands on.
 */
export interface VtsParameterSetting {
  input: string;
  inputLower: number;
  inputUpper: number;
  outputLower: number;
  outputUpper: number;
  clampInput: boolean;
  clampOutput: boolean;
  parameterId: string;
}

/** Reads the settings rows from a `.vtube.json`, dropping anything malformed. */
export function readVtsParameterSettings(value: unknown): VtsParameterSetting[] {
  const rows = (value as { ParameterSettings?: unknown } | null)?.ParameterSettings;
  if (!Array.isArray(rows)) return [];
  return rows.slice(0, 500).flatMap((row): VtsParameterSetting[] => {
    if (!row || typeof row !== "object" || Array.isArray(row)) return [];
    const entry = row as Record<string, unknown>;
    const numbers = [entry.InputRangeLower, entry.InputRangeUpper, entry.OutputRangeLower, entry.OutputRangeUpper];
    if (typeof entry.Input !== "string" || !entry.Input || typeof entry.OutputLive2D !== "string" || !entry.OutputLive2D
      || entry.OutputLive2D.length > 200 || !numbers.every(number => typeof number === "number" && Number.isFinite(number))) return [];
    const [inputLower, inputUpper, outputLower, outputUpper] = numbers as number[];
    return [{ input: entry.Input, inputLower: inputLower!, inputUpper: inputUpper!, outputLower: outputLower!, outputUpper: outputUpper!,
      clampInput: entry.ClampInput === true, clampOutput: entry.ClampOutput === true, parameterId: entry.OutputLive2D }];
  });
}

/** The VTS input at the part's weakest, usual and strongest steps; a null usual step is the parameter's own default. */
type Points = (setting: VtsParameterSetting) => [number, number | null, number];
const same: Points = () => [-1, 0, 1];
const degrees: Points = () => [-30, 0, 30];
const fromZero: Points = () => [0, 0, 1];
// Eyes, brows and smile: the author's line from end to end, resting where the model itself rests.
const authorEnds: Points = setting => [setting.inputLower, null, setting.inputUpper];

/**
 * The standard part each VTS face input stands for, and where the part's
 * weakest, usual and strongest steps fall on that input. The mouth's opening and
 * the shapes lip sync drives (jaw, funnel, pucker) are speech's and not here;
 * VBridger's expressive mouth shapes (sideways, shrug, press/open) are.
 */
const VTS_INPUTS: Record<string, Array<{ part: string; points: Points }>> = {
  FaceAngleX: [{ part: "ParamAngleX", points: degrees }],
  FaceAngleY: [{ part: "ParamAngleY", points: degrees }],
  FaceAngleZ: [{ part: "ParamAngleZ", points: degrees }],
  EyeOpenLeft: [{ part: "ParamEyeLOpen", points: authorEnds }],
  EyeOpenRight: [{ part: "ParamEyeROpen", points: authorEnds }],
  EyeLeftX: [{ part: "ParamEyeBallX", points: same }],
  EyeRightX: [{ part: "ParamEyeBallX", points: same }],
  EyeLeftY: [{ part: "ParamEyeBallY", points: same }],
  EyeRightY: [{ part: "ParamEyeBallY", points: same }],
  BrowLeftY: [{ part: "ParamBrowLY", points: authorEnds }],
  BrowRightY: [{ part: "ParamBrowRY", points: authorEnds }],
  Brows: [{ part: "ParamBrowLY", points: authorEnds }, { part: "ParamBrowRY", points: authorEnds }],
  MouthSmile: [{ part: "ParamMouthForm", points: authorEnds }],
  MouthX: [{ part: "ParamMouthX", points: same }],
  MouthShrug: [{ part: "ParamMouthShrug", points: fromZero }],
  MouthPressLipOpen: [{ part: "ParamMouthPressLipOpen", points: authorEnds }],
  CheekPuff: [{ part: "ParamCheekPuffC", points: fromZero }],
  TongueOut: [{ part: "ParamTongueOut", points: fromZero }],
};

/**
 * The model author's own wiring, as the parts' parameter lists: a part's
 * three steps go through the VTS input and the author's range to the model
 * parameter, as VTube Studio would move it. Rows whose parameter the model
 * lacks, that physics or breathing drives, or that do not move are left out.
 */
export function vtsPartBindings(
  settings: readonly VtsParameterSetting[],
  native: readonly NativeModelParameter[],
  blocked: Iterable<string> = [],
): ManualParameterBindings {
  const parameters = new Map(native.map(parameter => [parameter.id, parameter]));
  const excluded = new Set(blocked);
  const result: ManualParameterBindings = {};
  for (const setting of settings) {
    const parameter = parameters.get(setting.parameterId);
    if (!parameter || AUTOMATIC_FACIAL_IDS.has(parameter.id) || excluded.has(parameter.id)
      || !(parameter.minimum < parameter.maximum) || setting.inputLower === setting.inputUpper) continue;
    for (const { part, points } of VTS_INPUTS[setting.input] ?? []) {
      const outputs = result[part] ??= [];
      if (outputs.length >= MAX_PART_OUTPUTS || outputs.some(output => output.parameterId === parameter.id)) continue;
      const [lowerInput, neutralInput, upperInput] = points(setting);
      const lowerValue = land(setting, parameter, lowerInput), upperValue = land(setting, parameter, upperInput);
      const neutralValue = neutralInput === null
        ? round(Math.max(Math.min(lowerValue, upperValue), Math.min(Math.max(lowerValue, upperValue), parameter.defaultValue)))
        : land(setting, parameter, neutralInput);
      if (lowerValue === upperValue || neutralValue < Math.min(lowerValue, upperValue) || neutralValue > Math.max(lowerValue, upperValue)) continue;
      const output: ManualParameterBinding = { parameterId: parameter.id, lowerValue, neutralValue, upperValue };
      outputs.push(output);
    }
  }
  return Object.fromEntries(Object.entries(result).filter(([, outputs]) => outputs.length));
}

/** Where VTube Studio puts the parameter for this input value, kept inside the model's range. */
function land(setting: VtsParameterSetting, parameter: NativeModelParameter, input: number): number {
  const low = Math.min(setting.inputLower, setting.inputUpper), high = Math.max(setting.inputLower, setting.inputUpper);
  const value = setting.clampInput ? Math.max(low, Math.min(high, input)) : input;
  let output = setting.outputLower + (value - setting.inputLower) / (setting.inputUpper - setting.inputLower)
    * (setting.outputUpper - setting.outputLower);
  if (setting.clampOutput) {
    output = Math.max(Math.min(setting.outputLower, setting.outputUpper), Math.min(Math.max(setting.outputLower, setting.outputUpper), output));
  }
  return round(Math.max(parameter.minimum, Math.min(parameter.maximum, output)));
}

function round(value: number): number { return Math.round(value * 1e4) / 1e4 || 0; }
