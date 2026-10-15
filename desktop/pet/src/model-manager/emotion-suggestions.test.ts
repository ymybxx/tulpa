import { describe, expect, it } from "vitest";
import { packageFixture } from "./test-fixture";
import { mergeEmotionSuggestions, suggestionsForEmptyTargets, unappliedEmotionSuggestions } from "./emotion-suggestions";
import type { EmotionSuggestion } from "../shared/emotion-matching";

const suggestion = { target: { kind: "emotion", key: "happy" }, candidate: { id: "ai-1", motionAssetIds: ["wave"], expressionAssetIds: ["smile"] } } as EmotionSuggestion;

describe("new behavior suggestions", () => {
  it("filters identical combinations even when candidate IDs differ, scoped to the same target", () => {
    const manifest = packageFixture().manifest;
    manifest.behaviors.emotions.happy = [{ ...suggestion.candidate, id: "manual" }];
    const state = { ...suggestion, target: { kind: "state", key: "listening" }, candidate: { ...suggestion.candidate, id: "state" } } as EmotionSuggestion;
    const duplicate = { ...state, candidate: { ...state.candidate, id: "duplicate" } };
    expect(unappliedEmotionSuggestions(manifest, [suggestion, state, duplicate])).toEqual([state]);
    const next = mergeEmotionSuggestions(manifest, [suggestion, state, duplicate]);
    expect(next.behaviors.emotions.happy[0]?.id).toBe("manual");
    expect(next.behaviors.states.listening).toEqual([state.candidate]);
    expect(mergeEmotionSuggestions(next, [suggestion, state])).toEqual(next);
  });

  it("adds missing complete combinations while preserving all existing configuration", () => {
    const manifest = packageFixture().manifest;
    manifest.name = "手动名称";
    manifest.behaviors.emotions.happy = [{ id: "manual", motionAssetIds: [], expressionAssetIds: ["smile"] }];
    manifest.behaviors.states.thinking = [{ id: "manual-state", motionAssetIds: ["wave"], expressionAssetIds: [] }];
    expect(unappliedEmotionSuggestions(manifest, [suggestion])).toEqual([suggestion]);
    const next = mergeEmotionSuggestions(manifest, [suggestion]);
    expect(next.behaviors.emotions.happy.map((c) => c.id)).toEqual(["manual", "ai-1"]);
    expect(next.behaviors.states).toEqual(manifest.behaviors.states);
    expect(next.name).toBe("手动名称");
    expect(next.lipsync).toEqual(manifest.lipsync);
    expect(manifest.behaviors.emotions.happy).toHaveLength(1);
  });

  it("keeps different layer orders because they can produce different expressions", () => {
    const manifest = packageFixture().manifest;
    manifest.expressionAssets.second = { ...manifest.expressionAssets.smile!, id: "second", name: "Second" };
    const first = { ...suggestion, candidate: { id: "first", motionAssetIds: [], expressionAssetIds: ["smile", "second"] } };
    const reversed = { ...first, candidate: { ...first.candidate, id: "reversed", expressionAssetIds: ["second", "smile"] } };
    const next = mergeEmotionSuggestions(manifest, [first, reversed]);
    expect(next.behaviors.emotions.happy).toEqual([first.candidate, reversed.candidate]);
  });

  it("shows a suggestion again after its candidate is removed, regardless of old adoption metadata", () => {
    const manifest = packageFixture().manifest;
    const adopted = { ...suggestion, adopted: true };
    manifest.behaviors.emotions.happy = [suggestion.candidate];
    expect(unappliedEmotionSuggestions(manifest, [adopted])).toEqual([]);
    manifest.behaviors.emotions.happy = [];
    expect(unappliedEmotionSuggestions(manifest, [adopted])).toEqual([adopted]);
  });

  it("rejects missing assets without modifying the current configuration", () => {
    const manifest = packageFixture().manifest;
    const before = structuredClone(manifest);
    const missing = { ...suggestion, candidate: { ...suggestion.candidate, motionAssetIds: ["missing"] } };
    expect(() => mergeEmotionSuggestions(manifest, [missing])).toThrow("建议素材已改变");
    expect(manifest).toEqual(before);
  });
});

describe("filling empty states and emotions", () => {
  it("never touches a state or emotion the user already set", () => {
    const manifest = packageFixture().manifest;
    manifest.behaviors.emotions.happy = [{ id: "manual", motionAssetIds: [], expressionAssetIds: ["smile"] }];
    manifest.behaviors.states.listening = [];
    const listening = { ...suggestion, id: "s2", target: { kind: "state", key: "listening" }, candidate: { ...suggestion.candidate, id: "state" } } as EmotionSuggestion;
    expect(suggestionsForEmptyTargets(manifest, [suggestion, listening])).toEqual([listening]);
    const next = mergeEmotionSuggestions(manifest, suggestionsForEmptyTargets(manifest, [suggestion, listening]));
    expect(next.behaviors.emotions.happy.map((candidate) => candidate.id)).toEqual(["manual"]);
    expect(next.behaviors.states.listening).toEqual([listening.candidate]);
    expect(suggestionsForEmptyTargets(next, [suggestion, listening])).toEqual([]);
  });
});
