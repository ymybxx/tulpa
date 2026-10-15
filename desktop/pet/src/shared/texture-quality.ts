export const PET_TEXTURE_DETAIL_LEVELS = [2048, 4096, 8192] as const;

export type PetTextureMaximumDimension = typeof PET_TEXTURE_DETAIL_LEVELS[number];

const GIB = 1024 ** 3;

/** Suggest a ceiling once for a new or legacy configuration; saved choices win. */
export function suggestedPetTextureMaximum(totalMemoryBytes: number): PetTextureMaximumDimension {
  if (totalMemoryBytes >= 24 * GIB) return 8192;
  if (totalMemoryBytes >= 12 * GIB) return 4096;
  return 2048;
}

export function petTextureMaximumOrDefault(
  value: unknown,
  fallback: PetTextureMaximumDimension,
): PetTextureMaximumDimension {
  return PET_TEXTURE_DETAIL_LEVELS.find((level) => level === value) ?? fallback;
}
