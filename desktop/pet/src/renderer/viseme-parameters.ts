import type { LipVowel, VowelParameterMap } from "../shared/lipsync-adapter";

export type Vowel = LipVowel;

const DIRECT_VOWEL_PARAMETER_IDS: Record<Vowel, string> = {
  a: "ParamA",
  e: "ParamE",
  i: "ParamI",
  o: "ParamO",
  u: "ParamU",
};

export function detectDirectVowelParameters(core: {
  getParameterCount?: () => number;
  getParameterIndex?: (id: unknown) => number;
}): Partial<Record<Vowel, unknown>> {
  if (!core.getParameterCount || !core.getParameterIndex) return {};
  const count = core.getParameterCount();
  return Object.fromEntries((Object.entries(DIRECT_VOWEL_PARAMETER_IDS) as Array<[Vowel, string]>)
    .flatMap(([vowel, id]) => core.getParameterIndex!(id) < count ? [[vowel, id]] : []));
}

export function configuredVowelParameters(
  values: VowelParameterMap,
): Partial<Record<Vowel, unknown>> {
  return Object.fromEntries(
    (Object.entries(values) as Array<[Vowel, string | null]>)
      .filter((entry): entry is [Vowel, string] => Boolean(entry[1])),
  );
}
