import type { LipShape } from "./pet-renderer";

export function standardMouthTarget(
  value: LipShape,
  maxOpen: number,
): { open: number; form: number } {
  const vowelOpen = Math.max(
    value.volume,
    value.a,
    value.e * 0.65,
    value.i * 0.45,
    value.o * 0.85,
    value.u * 0.5,
  );
  return {
    open: Math.min(1, vowelOpen) * Math.max(0, Math.min(1, maxOpen)),
    form: Math.max(
      -1,
      Math.min(1, (value.e + value.i) * 0.55 - (value.o + value.u) * 0.3),
    ),
  };
}
