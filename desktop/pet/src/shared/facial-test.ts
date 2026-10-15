import type { DemoEvent } from "./facial-demo";
import type { ParameterEffects } from "./facial-conflicts";
import type { NativeModelParameter } from "./contracts";
import type { VtsParameterSetting } from "./vts-model-settings";
import type { ManualParameterBinding, ManualParameterBindings, FacialParameterSettings, FacialParameterSettingsMap, FacialDescriptionOverrides, FacialCriteriaOverrides, FacialDecisionRule, CustomFacialParameter } from "./facial-bindings";

export interface FacialDecisionParameter {
  id: string;
  minimum: number;
  maximum: number;
  defaultValue: number;
  description?: string;
  mode?: "score" | "binary";
  levels?: string[];
  anchors?: number[];
  falseDescription?: string;
  trueDescription?: string;
}

export interface FacialTestModelInfo {
  modelId: string;
  modelName: string;
  blushParameter: string | null;
  physicsOutputParameterIds: string[];
  /** The model's own VTube Studio settings, when it ships a `.vtube.json`. */
  vtsSettings?: VtsParameterSetting[];
  manualBindings?: ManualParameterBindings;
  parameterSettings?: FacialParameterSettingsMap;
  descriptions?: FacialDescriptionOverrides;
  criteria?: FacialCriteriaOverrides;
  customParameters?: CustomFacialParameter[];
  /** What each drivable parameter does to the model's drawables; feeds the conflict rules. */
  parameterEffects?: ParameterEffects;
}

export interface FacialTestStartRequest {
  id: string;
  modelId: string;
  text: string;
  parameters: FacialDecisionParameter[];
}

export interface FacialParameterSample {
  parameter: NativeModelParameter;
  frames: Array<{ kind: "default" | "target"; value: number; image: string }>;
}

export interface FacialParameterSuggestion {
  parameterId: string;
  confidence: "clear" | "uncertain";
  label: string;
  description: string;
  mode: "score" | "binary";
  levels?: string[];
  falseDescription?: string;
  trueDescription?: string;
  note: string;
}

/** What moving one parameter alone to each end does to the picture. */
export interface FacialParameterSurvey {
  id: string;
  parameter: NativeModelParameter;
  /** Share of the character's area that changes, 0..1. */
  changed: number;
  /** Where it changes, relative to the character's bounds (0..1); null when nothing visibly changes. */
  box: { x: number; y: number; width: number; height: number } | null;
}

/** 一键配置's findings, nothing saved until the person picks what to keep. */
export interface FacialAutoSetupProposal {
  /** Standard parts the model has under another name. */
  standard: Array<{ standardId: string; parameterId: string; parameterName: string; binding: ManualParameterBinding; note: string }>;
  /** Expressions and gestures worth adding as parts of their own. */
  custom: Array<{ parameterId: string; parameterName: string; kind: "expression" | "motion"; value: CustomFacialParameter; note: string }>;
  skipped: Array<{ parameterId: string; parameterName: string; reason: string }>;
  failed: Array<{ parameterId: string; parameterName: string; error: string }>;
}
export interface FacialAutoSetupProgress { text: string; done?: number; total?: number }
/** What the model already uses, from the panel's resolved parts. */
export interface FacialAutoSetupContext {
  /** Native parameters already driven by a part. */
  used: string[];
  /** Standard parts with nothing to drive yet (not deleted by the person). */
  open: string[];
  /** How many more custom parts fit. */
  customSlots: number;
}

export interface FacialTestApi {
  prepare(modelId: string): Promise<FacialTestModelInfo>;
  start(request: FacialTestStartRequest): Promise<void>;
  stop(id: string): Promise<void>;
  saveBinding(modelId: string, standardId: string, binding: ManualParameterBinding | ManualParameterBinding[] | null, settings?: FacialParameterSettings, description?: string, decision?: FacialDecisionRule): Promise<FacialTestModelInfo>;
  saveCustomParameter(modelId: string, value: CustomFacialParameter): Promise<FacialTestModelInfo>;
  deleteCustomParameter(modelId: string, id: string): Promise<FacialTestModelInfo>;
  suggestParameter(modelId: string, parameterId: string, mode: "score" | "binary", levelCount: number): Promise<FacialParameterSuggestion>;
  autoSetup(modelId: string, context: FacialAutoSetupContext): Promise<FacialAutoSetupProposal>;
  cancelAutoSetup(): Promise<void>;
  onAutoSetupProgress(callback: (progress: FacialAutoSetupProgress) => void): () => void;
  onEvent(callback: (event: DemoEvent) => void): () => void;
}
