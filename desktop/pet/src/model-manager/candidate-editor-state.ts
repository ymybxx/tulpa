import type { BehaviorCandidate } from "../shared/contracts";
import type { AssetKind, BehaviorKind } from "./draft";

export interface CandidateEditorState {
  kind: BehaviorKind;
  key: string;
  candidateId: string;
  motionAssetIds: string[];
  expressionAssetIds: string[];
  previewing: boolean;
}

export function createCandidateEditorState(
  kind: BehaviorKind,
  key: string,
  candidateId: string,
  candidate?: BehaviorCandidate,
): CandidateEditorState {
  return {
    kind,
    key,
    candidateId,
    motionAssetIds: [...(candidate?.motionAssetIds ?? [])],
    expressionAssetIds: [...(candidate?.expressionAssetIds ?? [])],
    previewing: false,
  };
}

export function setCandidateAssetSelected(
  editor: CandidateEditorState,
  kind: AssetKind,
  assetId: string,
  selected: boolean,
): boolean {
  const ids = idsFor(editor, kind);
  const index = ids.indexOf(assetId);
  if (selected === (index >= 0)) return false;
  if (selected) ids.push(assetId);
  else ids.splice(index, 1);
  return true;
}

export function moveCandidateAsset(
  editor: CandidateEditorState,
  kind: AssetKind,
  assetId: string,
  delta: -1 | 1,
): boolean {
  const ids = idsFor(editor, kind);
  const index = ids.indexOf(assetId);
  const next = index + delta;
  if (index < 0 || next < 0 || next >= ids.length) return false;
  [ids[index], ids[next]] = [ids[next]!, ids[index]!];
  return true;
}

export function canSaveCandidate(editor: CandidateEditorState): boolean {
  return editor.motionAssetIds.length + editor.expressionAssetIds.length > 0;
}

export function candidateFromEditor(editor: CandidateEditorState): BehaviorCandidate {
  return {
    id: editor.candidateId,
    motionAssetIds: [...editor.motionAssetIds],
    expressionAssetIds: [...editor.expressionAssetIds],
  };
}

function idsFor(editor: CandidateEditorState, kind: AssetKind): string[] {
  return kind === "motion" ? editor.motionAssetIds : editor.expressionAssetIds;
}
