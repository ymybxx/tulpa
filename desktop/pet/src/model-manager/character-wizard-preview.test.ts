import { describe, expect, it } from "vitest";

import type { CharacterGenerationJob } from "../shared/character-generation";
import type { TextureAtlasDelta } from "../shared/contracts";
import {
  acceptedTextureResultCount,
  deliverAcceptedTexturePreviewChange,
  textureBatchPreviewMaxWidth,
} from "./character-wizard";

describe("texture batch preview sizing", () => {
  it("limits a portrait canvas width so its height stays within the preview cap", () => {
    expect(textureBatchPreviewMaxWidth(1024, 1536, 760, 620)).toBeCloseTo(413.333, 2);
  });

  it("keeps a landscape canvas within the normal preview width", () => {
    expect(textureBatchPreviewMaxWidth(1536, 1024, 760, 620)).toBe(760);
  });
});

describe("accepted texture preview attachment", () => {
  it("detects when cancelling one result removes the final accepted overlay", () => {
    const artifacts = {
      textureEdits: {
        shoe: { status: "review" },
        sock: { status: "failed" },
      },
    } as unknown as CharacterGenerationJob["artifacts"];

    expect(acceptedTextureResultCount({ artifacts } as CharacterGenerationJob)).toBe(0);
    artifacts.textureEdits!.shoe = {
      ...artifacts.textureEdits!.shoe!,
      status: "accepted",
    };
    expect(acceptedTextureResultCount({ artifacts } as CharacterGenerationJob)).toBe(1);
  });

  it("applies the final revert before returning no pending attachment", async () => {
    const job = {
      id: "job-1",
      artifacts: { textureEdits: { shoe: { status: "review" } } },
    } as unknown as CharacterGenerationJob;
    const delta = { modelId: "mao" } as TextureAtlasDelta;
    const applied: TextureAtlasDelta[] = [];

    await expect(deliverAcceptedTexturePreviewChange(
      job,
      delta,
      async (value) => { applied.push(value); },
    )).resolves.toBeNull();
    expect(applied).toEqual([delta]);
  });

  it("defers a snapshot while at least one accepted result needs a new host", async () => {
    const job = {
      id: "job-1",
      artifacts: { textureEdits: { shoe: { status: "accepted" } } },
    } as unknown as CharacterGenerationJob;
    const delta = { modelId: "mao" } as TextureAtlasDelta;
    let applied = false;

    await expect(deliverAcceptedTexturePreviewChange(
      job,
      delta,
      async () => { applied = true; },
    )).resolves.toEqual({ jobId: "job-1", delta });
    expect(applied).toBe(false);
  });
});
