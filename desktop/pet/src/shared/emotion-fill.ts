import type { PetManifest } from "./contracts";
import { candidateSignature, type EmotionSuggestion } from "./emotion-matching";
import { targetCandidates, targetId } from "./behavior-target";

/** Pending means absent from this exact target's current ordered layer combination. */
export function unappliedEmotionSuggestions(manifest: PetManifest, suggestions: EmotionSuggestion[]): EmotionSuggestion[] {
  const seen = new Set<string>();
  return suggestions.filter((suggestion) => {
    const signature = candidateSignature(suggestion.candidate);
    const key = `${targetId(suggestion.target)}:${signature}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return !targetCandidates(manifest, suggestion.target).some((candidate) => candidateSignature(candidate) === signature);
  });
}

export function mergeEmotionSuggestions(manifest: PetManifest, suggestions: EmotionSuggestion[]): PetManifest {
  const next = structuredClone(manifest);
  for (const suggestion of unappliedEmotionSuggestions(next, suggestions)) {
    const candidate = suggestion.candidate;
    if (candidate.motionAssetIds.some((id) => !next.motionAssets[id]) || candidate.expressionAssetIds.some((id) => !next.expressionAssets[id])) throw new Error("建议素材已改变，请重新分析");
    targetCandidates(next, suggestion.target).push(structuredClone(candidate));
  }
  return next;
}

/**
 * AI matching only fills states and emotions that have nothing set yet; what
 * the user already configured is never touched, so there is nothing to confirm.
 */
export function suggestionsForEmptyTargets(manifest: PetManifest, suggestions: EmotionSuggestion[]): EmotionSuggestion[] {
  return unappliedEmotionSuggestions(manifest, suggestions)
    .filter((suggestion) => targetCandidates(manifest, suggestion.target).length === 0);
}
