import { describe, expect, it } from "vitest";

import {
  modelManagerTabKeepsPreview,
  modelManagerTabUsesAppearance,
} from "./preview-resource-policy";

describe("model manager preview resource policy", () => {
  it("keeps one selected-model preview under every model-management tab", () => {
    expect(modelManagerTabKeepsPreview("pools")).toBe(true);
    expect(modelManagerTabKeepsPreview("assets")).toBe(true);
    expect(modelManagerTabKeepsPreview("defaults")).toBe(true);
    expect(modelManagerTabKeepsPreview("appearance")).toBe(true);
    expect(modelManagerTabKeepsPreview("display")).toBe(true);
    expect(modelManagerTabKeepsPreview("facial")).toBe(true);
    expect(modelManagerTabKeepsPreview("edit")).toBe(true);
  });

  it("rejects unknown non-model views", () => {
    expect(modelManagerTabKeepsPreview("chat")).toBe(false);
  });
});

describe("modelManagerTabUsesAppearance", () => {
  it("loads appearance data for Wardrobe and AI editable parts", () => {
    expect(modelManagerTabUsesAppearance("appearance")).toBe(true);
    expect(modelManagerTabUsesAppearance("edit")).toBe(true);
  });

  it("does not load appearance data for unrelated tabs", () => {
    expect(modelManagerTabUsesAppearance("pools")).toBe(false);
    expect(modelManagerTabUsesAppearance("assets")).toBe(false);
    expect(modelManagerTabUsesAppearance("defaults")).toBe(false);
    expect(modelManagerTabUsesAppearance("display")).toBe(false);
    expect(modelManagerTabUsesAppearance("facial")).toBe(false);
  });
});
