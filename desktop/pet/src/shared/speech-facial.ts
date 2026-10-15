import type { FacialTestModelInfo } from "./facial-test";
import type { FacialDecisionParameter } from "./facial-test";

export interface SpeechFacialProfile {
  revision: string;
  modelId: string;
  modelName: string;
  parameters: FacialDecisionParameter[];
}

export interface SpeechFacialApi {
  prepare(modelId: string): Promise<FacialTestModelInfo>;
  publish(profile: SpeechFacialProfile | null): Promise<{ enabled: boolean }>;
}
