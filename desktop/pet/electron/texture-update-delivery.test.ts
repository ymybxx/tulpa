import { describe, expect, it } from "vitest";

import { textureUpdateRecipients } from "./texture-update-delivery";

describe("textureUpdateRecipients", () => {
  it("returns to the caller and broadcasts only to other renderers", () => {
    expect(textureUpdateRecipients({
      senderId: 10,
      mainId: 10,
      petId: 20,
      modelIsSelectedPet: true,
    })).toEqual([20]);
  });

  it("excludes the pet when the pet itself invoked the mutation", () => {
    expect(textureUpdateRecipients({
      senderId: 20,
      mainId: 10,
      petId: 20,
      modelIsSelectedPet: true,
    })).toEqual([10]);
  });

  it("delivers a service-originated save to both active consumers", () => {
    expect(textureUpdateRecipients({
      senderId: null,
      mainId: 10,
      petId: 20,
      modelIsSelectedPet: true,
    })).toEqual([10, 20]);
  });

  it("does not wake the pet renderer for another model", () => {
    expect(textureUpdateRecipients({
      senderId: 30,
      mainId: 10,
      petId: 20,
      modelIsSelectedPet: false,
    })).toEqual([10]);
  });

  it("deduplicates aliases of the same renderer", () => {
    expect(textureUpdateRecipients({
      senderId: null,
      mainId: 10,
      petId: 10,
      modelIsSelectedPet: true,
    })).toEqual([10]);
  });

  it("omits unavailable renderer targets", () => {
    expect(textureUpdateRecipients({
      senderId: 10,
      mainId: null,
      petId: null,
      modelIsSelectedPet: true,
    })).toEqual([]);
  });
});
