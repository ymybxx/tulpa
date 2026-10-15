export const LIP_VOWELS = ["a", "e", "i", "o", "u"] as const;

export type LipVowel = (typeof LIP_VOWELS)[number];
export type LipSyncAdapterMode = "open-form" | "viseme" | "unconfigured";
export type LipSyncMappingSource = "automatic" | "manual";
export type VowelParameterMap = Record<LipVowel, string | null>;

export interface ModelParameterDescriptor {
  id: string;
  name: string;
  groupId: string | null;
}

export interface LipSyncAdapterResolution {
  mode: LipSyncAdapterMode;
  source: LipSyncMappingSource;
  needsConfirmation: boolean;
  mouthOpenParameter: string | null;
  mouthFormParameter: string | null;
  /** VBridger rigs' jaw drop, pucker (−) to widen (+) and funnel, when the model has them. */
  jawParameter: string | null;
  puckerWidenParameter: string | null;
  funnelParameter: string | null;
  vowelParameters: VowelParameterMap;
}

const STANDARD_VOWEL_IDS: Record<LipVowel, string> = {
  a: "ParamA",
  e: "ParamE",
  i: "ParamI",
  o: "ParamO",
  u: "ParamU",
};

const EMPTY_VOWELS: VowelParameterMap = {
  a: null,
  e: null,
  i: null,
  o: null,
  u: null,
};

const NO_VBRIDGER_MOUTH = { jawParameter: null, puckerWidenParameter: null, funnelParameter: null };

export function resolveLipSyncAdapter(input: {
  lipSyncParameters?: string[];
  parameters?: ModelParameterDescriptor[];
}): LipSyncAdapterResolution {
  const declared = uniqueIds(input.lipSyncParameters ?? []);
  const described = new Set((input.parameters ?? []).map((item) => item.id));
  const available = new Set([...described, ...declared]);
  const vowelParameters = Object.fromEntries(
    LIP_VOWELS.map((vowel) => [
      vowel,
      available.has(STANDARD_VOWEL_IDS[vowel]) ? STANDARD_VOWEL_IDS[vowel] : null,
    ]),
  ) as VowelParameterMap;
  if (LIP_VOWELS.every((vowel) => vowelParameters[vowel])) {
    return {
      mode: "viseme",
      source: "automatic",
      needsConfirmation: false,
      mouthOpenParameter: null,
      mouthFormParameter: null,
      ...NO_VBRIDGER_MOUTH,
      vowelParameters,
    };
  }
  if (available.has("ParamMouthOpenY")) {
    return {
      mode: "open-form",
      source: "automatic",
      needsConfirmation: false,
      mouthOpenParameter: "ParamMouthOpenY",
      mouthFormParameter: available.has("ParamMouthForm") ? "ParamMouthForm" : null,
      jawParameter: available.has("ParamJawOpen") ? "ParamJawOpen" : null,
      puckerWidenParameter: available.has("ParamMouthPuckerWiden") ? "ParamMouthPuckerWiden" : null,
      funnelParameter: available.has("ParamMouthFunnel") ? "ParamMouthFunnel" : null,
      vowelParameters: { ...EMPTY_VOWELS },
    };
  }
  const customOpen = declared[0] ?? null;
  if (customOpen) {
    return {
      mode: "open-form",
      source: "automatic",
      needsConfirmation: true,
      mouthOpenParameter: customOpen,
      mouthFormParameter: null,
      ...NO_VBRIDGER_MOUTH,
      vowelParameters: { ...EMPTY_VOWELS },
    };
  }
  return {
    mode: "unconfigured",
    source: "automatic",
    needsConfirmation: true,
    mouthOpenParameter: null,
    mouthFormParameter: null,
    ...NO_VBRIDGER_MOUTH,
    vowelParameters: { ...EMPTY_VOWELS },
  };
}

function uniqueIds(values: string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))];
}
