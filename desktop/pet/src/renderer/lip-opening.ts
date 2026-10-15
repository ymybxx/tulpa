import type { LipShape } from "./pet-renderer";

/** How far each vowel opens the mouth, relative to a full "a". */
const VOWEL_OPENING = { a: 1, e: 0.65, i: 0.45, o: 0.85, u: 0.5 } as const;

/**
 * Mouth opening for a rig without per-vowel shapes: the loudness, scaled by
 * the vowel being heard, so an "i" stays narrow however loud it is. Before a
 * vowel is heard (a consonant, the first frames) loudness alone opens it, and
 * a vowel on its own (the A–U previews) opens it by that vowel's amount.
 */
export function mouthOpening(shape: LipShape): number {
  const vowels = shape.a + shape.e + shape.i + shape.o + shape.u;
  if (vowels <= 1e-3) return clamp(shape.volume);
  const factor = (Object.keys(VOWEL_OPENING) as Array<keyof typeof VOWEL_OPENING>)
    .reduce((sum, vowel) => sum + shape[vowel] * VOWEL_OPENING[vowel], 0) / vowels;
  const heard = shape.volume > 1e-3 ? Math.min(1, vowels / shape.volume) : 1;
  return clamp(Math.max(shape.volume, vowels) * (1 - heard + heard * factor));
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/**
 * What each vowel does with a VBridger rig's extra mouth parts: the jaw drops
 * for "a" and "o", "o"/"u" pucker (−) and funnel, "e"/"i" widen (+). Each is
 * weighted by how strongly the vowel is heard, so it follows the loudness.
 */
const VOWEL_PARTS = {
  jaw: { a: 0.6, e: 0.25, i: 0.05, o: 0.35, u: 0.1 },
  puckerWiden: { a: 0, e: 0.2, i: 0.4, o: -0.5, u: -1 },
  funnel: { a: 0, e: 0, i: 0, o: 0.5, u: 0.8 },
} as const;

export function mouthParts(shape: LipShape): { jaw: number; puckerWiden: number; funnel: number } {
  const part = (weights: Record<keyof typeof VOWEL_OPENING, number>) =>
    (Object.keys(weights) as Array<keyof typeof VOWEL_OPENING>)
      .reduce((sum, vowel) => sum + shape[vowel] * weights[vowel], 0);
  return {
    jaw: clamp(part(VOWEL_PARTS.jaw)),
    puckerWiden: Math.max(-1, Math.min(1, part(VOWEL_PARTS.puckerWiden))),
    funnel: clamp(part(VOWEL_PARTS.funnel)),
  };
}
