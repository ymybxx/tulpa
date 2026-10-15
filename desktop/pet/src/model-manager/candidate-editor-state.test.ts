import { describe, expect, it } from "vitest";

import {
  canSaveCandidate,
  candidateFromEditor,
  createCandidateEditorState,
  moveCandidateAsset,
  setCandidateAssetSelected,
} from "./candidate-editor-state";

describe("candidate editor", () => {
  it("can save as soon as something is picked, in any order", () => {
    const editor = createCandidateEditorState("state", "standby", "candidate-1");
    expect(canSaveCandidate(editor)).toBe(false);

    setCandidateAssetSelected(editor, "motion", "m1", true);
    setCandidateAssetSelected(editor, "motion", "m2", true);
    expect(canSaveCandidate(editor)).toBe(true);

    moveCandidateAsset(editor, "motion", "m2", -1);
    expect(editor.motionAssetIds).toEqual(["m2", "m1"]);
    expect(canSaveCandidate(editor)).toBe(true);

    setCandidateAssetSelected(editor, "motion", "m1", false);
    setCandidateAssetSelected(editor, "motion", "m2", false);
    expect(canSaveCandidate(editor)).toBe(false);
  });

  it("preserves stage-local order when creating the saved candidate", () => {
    const editor = createCandidateEditorState("emotion", "happy", "candidate-3");
    setCandidateAssetSelected(editor, "motion", "m1", true);
    setCandidateAssetSelected(editor, "motion", "m2", true);
    setCandidateAssetSelected(editor, "expression", "e1", true);
    expect(candidateFromEditor(editor)).toEqual({
      id: "candidate-3",
      motionAssetIds: ["m1", "m2"],
      expressionAssetIds: ["e1"],
    });
  });
});
