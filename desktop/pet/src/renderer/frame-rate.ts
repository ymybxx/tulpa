import type { PetState } from "../shared/contracts";

// Keep every visible non-speaking animation fluid. Hidden windows pause the
// renderer entirely, while lip sync retains full display cadence.
export const IDLE_RENDER_FPS = 30;
export const ENGAGED_RENDER_FPS = 30;
export const ACTIVE_RENDER_FPS = 60;

export function live2DFrameRate(
  state: PetState,
  lipVolume: number,
  transientActive = false,
  proceduralActive = false,
): number {
  if (state === "speaking" || lipVolume > 0.01) return ACTIVE_RENDER_FPS;
  // Procedural idle shares the authored non-speaking animation budget; it must
  // not double the cost of the entire Cubism update/draw loop while silent.
  if (proceduralActive || transientActive || (state !== "standby" && state !== "unbound")) {
    return ENGAGED_RENDER_FPS;
  }
  return IDLE_RENDER_FPS;
}
