import type { TextureAtlasDelta } from "../shared/contracts";

export interface TextureStreamRevision {
  streamId: string;
  revision: number;
}

/** A new stream is an explicit snapshot switch; revisions only order within it. */
export function shouldApplyTextureSnapshot(
  currentStreamId: string | null,
  currentRevision: number,
  next: TextureStreamRevision,
): boolean {
  return next.streamId !== currentStreamId || next.revision > currentRevision;
}

/**
 * A cross-stream transition must carry atlas state. Otherwise the renderer
 * would record the new stream while leaving the previous stream's GPU overlay
 * intact, after which an equal revision could no longer repair it.
 */
export function assertTextureStreamTransitionHasState(
  currentStreamId: string | null,
  next: TextureAtlasDelta,
): void {
  if (
    currentStreamId !== null &&
    next.streamId !== currentStreamId &&
    next.atlases.length === 0 &&
    next.patchSets.length === 0
  ) {
    throw new Error("跨纹理数据流的快照必须包含明确的图集状态");
  }
}
