import type { BehaviorCandidate, ModelRenderBundle, PetManifest } from "./contracts";
import type { BehaviorTarget } from "./behavior-target";
import type { FacialParameterSample, FacialParameterSurvey } from "./facial-test";

// Retain the IPC/storage names so existing emotion suggestions remain accessible.
export const EMOTION_ANALYSIS_VERSION = "behavior-v2";
export const EMOTION_SAMPLE_VERSION = "cubism-sample-v1";
export const EMOTION_MATCHING_CHANNEL = "pet:emotion-matching";
export type EmotionJobStatus = "queued" | "running" | "completed" | "partial" | "failed" | "cancelled" | "stale" | "interrupted";
export type EmotionSampleMode = "asset" | "runtime" | "speaking" | "state";

export interface EmotionAsset {
  id: string;
  kind: "motion" | "expression";
  name: string;
  contentHash: string;
  durationMs: number;
  timestampsMs: number[];
  parameters: string[];
  candidate: BehaviorCandidate;
  error?: string;
}

export interface EmotionFrame {
  id: string;
  timeMs: number;
  image: string;
}

export interface EmotionSample {
  asset: EmotionAsset;
  mode: EmotionSampleMode;
  frames: EmotionFrame[];
  faceExpressionIds: string[];
  elapsedMs: number;
}

export interface EmotionAssessment {
  assetId: string;
  classification: "behavior" | "utility" | "uncertain";
  matches: Array<{ target: BehaviorTarget; strength: "strong" | "possible" }>;
  evidence: Array<{ frameId: string; observation: string }>;
  reason: string;
  needsMoreFrames: boolean;
}

export interface EmotionSuggestion {
  id: string;
  target: BehaviorTarget;
  candidate: BehaviorCandidate;
  reason: string;
  strength: "strong" | "possible";
  sourceFingerprint: string;
  assetHashes: Record<string, string>;
  lipsync: PetManifest["lipsync"];
  speaking: "retained" | "transient";
  adopted: boolean;
}

export interface EmotionMatchingJob {
  id: string;
  modelId: string;
  modelName: string;
  analysisModelId: string;
  analysisModelName: string;
  analysisModelFingerprint: string;
  analysisVersion?: string;
  sourceFingerprint: string;
  revision: number;
  status: EmotionJobStatus;
  phase: "preparing" | "analyzing" | "validating" | "finished";
  totalAssets: number;
  processedAssets: number;
  cachedAssets: number;
  failures: Array<{ assetId: string; message: string }>;
  suggestions: EmotionSuggestion[];
  message: string;
  createdAt: string;
  updatedAt: string;
}

export interface EmotionAdoption {
  jobId: string;
  suggestionId: string;
  candidateId: string;
}

export interface EmotionMatchingApi {
  start(modelId: string, analysisModelId?: string): Promise<EmotionMatchingJob>;
  get(modelId: string): Promise<EmotionMatchingJob | null>;
  cancel(jobId: string): Promise<void>;
  validate(modelId: string, manifest: PetManifest, adoptions: EmotionAdoption[]): Promise<void>;
  onUpdate(callback: (job: EmotionMatchingJob) => void): () => void;
}

export type EmotionSamplerRequest = {
  id: string;
  operation: "load";
  bundle: ModelRenderBundle;
} | {
  id: string;
  operation: "sample";
  asset: EmotionAsset;
  mode: EmotionSampleMode;
} | {
  id: string;
  operation: "sample-parameter";
  parameterId: string;
  mode: "score" | "binary";
  levelCount: number;
} | {
  id: string;
  operation: "survey-parameters";
  /** null surveys every parameter of the model. */
  parameterIds: string[] | null;
} | { id: string; operation: "dispose" };

export interface EmotionSamplerResponse {
  id: string;
  error?: string;
  sample?: EmotionSample;
  parameterSample?: FacialParameterSample;
  survey?: FacialParameterSurvey[];
}

export function emotionJobActive(job: EmotionMatchingJob): boolean {
  return job.status === "queued" || job.status === "running";
}

export function candidateSignature(candidate: BehaviorCandidate): string {
  return JSON.stringify([candidate.motionAssetIds, candidate.expressionAssetIds]);
}
