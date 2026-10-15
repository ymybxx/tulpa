import { describe, expect, it } from "vitest";

import {
  createDraft,
  deleteBehaviorCandidate,
  hasUnsavedChanges,
  isDraftValid,
  renameAsset,
  saveBehaviorCandidate,
  toManifest,
} from "./draft";
import { packageFixture } from "./test-fixture";

describe("model manager draft", () => {
  it("keeps default appearance edits isolated and drops references to missing expressions", () => {
    const model = packageFixture();
    const draft = createDraft(model);
    draft.manifest.defaultAppearance = { expressionAssetIds: ["smile", "missing", "smile"] };
    expect(hasUnsavedChanges(draft)).toBe(true);
    expect(model.manifest.defaultAppearance).toBeUndefined();
    expect(toManifest(draft).defaultAppearance).toEqual({ expressionAssetIds: ["smile"] });
    expect(createDraft(model).manifest.defaultAppearance).toBeUndefined();
  });

  it("adds and removes combined candidates without mutating the source package", () => {
    const model = packageFixture();
    const draft = createDraft(model);
    const candidate = { id: "happy-1", motionAssetIds: ["wave"], expressionAssetIds: ["smile"] };

    saveBehaviorCandidate(draft, "emotion", "happy", candidate);
    expect(draft.manifest.behaviors.emotions.happy).toEqual([candidate]);
    expect(model.manifest.behaviors.emotions.happy).toEqual([]);

    deleteBehaviorCandidate(draft, "emotion", "happy", candidate.id);
    expect(draft.manifest.behaviors.emotions.happy).toEqual([]);
  });

  it("rejects a candidate with no layers or a missing asset", () => {
    const draft = createDraft(packageFixture());
    expect(() => saveBehaviorCandidate(draft, "state", "standby", {
      id: "empty",
      motionAssetIds: [],
      expressionAssetIds: [],
    })).toThrow(/至少需要/);
    expect(() => saveBehaviorCandidate(draft, "state", "standby", {
      id: "missing",
      motionAssetIds: ["missing"],
      expressionAssetIds: [],
    })).toThrow(/不存在/);
  });

  it("tracks globally renamed assets and returns a detached manifest", () => {
    const draft = createDraft(packageFixture());

    expect(hasUnsavedChanges(draft)).toBe(false);
    renameAsset(draft, "expression", "smile", "  大   笑  ");
    expect(draft.manifest.expressionAssets.smile!.name).toBe("大 笑");
    expect(hasUnsavedChanges(draft)).toBe(true);

    const manifest = toManifest(draft);
    manifest.expressionAssets.smile!.name = "外部修改";
    expect(draft.manifest.expressionAssets.smile!.name).toBe("大 笑");
  });

  it("rejects an empty display name and caps normalized names at 80 characters", () => {
    const draft = createDraft(packageFixture());

    expect(() => renameAsset(draft, "motion", "wave", "  \t  ")).toThrow(/不能为空/);
    renameAsset(draft, "motion", "wave", `  ${"名".repeat(90)}  `);

    expect(draft.manifest.motionAssets.wave!.name).toHaveLength(80);
    expect(draft.manifest.motionAssets.wave!.file).toBe("wave.motion3.json");
    expect(draft.manifest.motionAssets.wave!.id).toBe("wave");
  });

  it("rejects drafts without a model name or mouth-open parameter", () => {
    const draft = createDraft(packageFixture());
    expect(isDraftValid(draft)).toBe(true);

    draft.manifest.name = "   ";
    expect(isDraftValid(draft)).toBe(false);
    draft.manifest.name = "Girl";
    draft.manifest.lipsync.mouthOpenParameter = "";
    expect(isDraftValid(draft)).toBe(false);
  });
});
