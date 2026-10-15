import { mkdtemp } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { OnboardingDraftStore, parseOnboardingDraft } from "./onboarding-draft";
import { assertPrivateFile } from "../test-support/private-file";

const draft = {
  step: "permissions", modelId: "mao-1", name: "小鱼", relation: "朋友", relations: ["朋友", "青梅竹马"],
  traits: ["温柔"], traitOptions: ["温柔", "爱吐槽"], customSetting: "", provider: "doubao",
  baseUrl: "", model: "", apiKey: "sk-1", level: "full", jevApiKey: "jev-1", bailianWorkspaceId: "ws-1234",
} as const;

describe("OnboardingDraftStore", () => {
  it("restores answers after a relaunch and forgets them when cleared", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "draft-"));
    const store = new OnboardingDraftStore(directory);
    await store.save(draft);
    expect(await new OnboardingDraftStore(directory).load()).toEqual(draft);
    await assertPrivateFile(path.join(directory, "onboarding-draft.json"));
    await store.clear();
    expect(await store.load()).toBeNull();
  });
});

describe("parseOnboardingDraft", () => {
  it("rejects an unknown step and falls back on unknown choices", () => {
    expect(parseOnboardingDraft({ ...draft, step: "done" })).toBeNull();
    expect(parseOnboardingDraft({ ...draft, provider: "evil", level: "max" })).toMatchObject({ provider: "qwen", level: "full" });
  });

  it("drops values of the wrong type", () => {
    expect(parseOnboardingDraft({ step: "soul", traits: ["温柔", 3], apiKey: 42 })).toMatchObject({ traits: ["温柔"], apiKey: "" });
  });
});
