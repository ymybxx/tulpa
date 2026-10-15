import { describe, expect, it } from "vitest";

import { resolveLipSyncAdapter } from "./lipsync-adapter";

describe("resolveLipSyncAdapter", () => {
  it("prefers a complete AEIOU parameter set", () => {
    expect(resolveLipSyncAdapter({
      lipSyncParameters: ["ParamA"],
      parameters: ["ParamA", "ParamI", "ParamU", "ParamE", "ParamO"].map((id) => ({
        id,
        name: id,
        groupId: "ParamGroupMouth",
      })),
    })).toEqual({
      mode: "viseme",
      source: "automatic",
      needsConfirmation: false,
      mouthOpenParameter: null,
      mouthFormParameter: null,
      jawParameter: null,
      puckerWidenParameter: null,
      funnelParameter: null,
      vowelParameters: {
        a: "ParamA",
        e: "ParamE",
        i: "ParamI",
        o: "ParamO",
        u: "ParamU",
      },
    });
  });

  it("recognizes the standard open and form parameters", () => {
    expect(resolveLipSyncAdapter({
      lipSyncParameters: ["ParamMouthOpenY"],
      parameters: ["ParamMouthOpenY", "ParamMouthForm"].map((id) => ({
        id,
        name: id,
        groupId: "ParamGroupMouth",
      })),
    })).toMatchObject({
      mode: "open-form",
      source: "automatic",
      needsConfirmation: false,
      mouthOpenParameter: "ParamMouthOpenY",
      mouthFormParameter: "ParamMouthForm",
    });
  });

  it("also drives a VBridger rig's jaw, pucker/widen and funnel when present", () => {
    const ids = ["ParamMouthOpenY", "ParamMouthForm", "ParamJawOpen", "ParamMouthPuckerWiden", "ParamMouthFunnel"];
    expect(resolveLipSyncAdapter({
      parameters: ids.map((id) => ({ id, name: id, groupId: "ParamGroupMouth" })),
    })).toMatchObject({
      mode: "open-form",
      jawParameter: "ParamJawOpen",
      puckerWidenParameter: "ParamMouthPuckerWiden",
      funnelParameter: "ParamMouthFunnel",
    });
    expect(resolveLipSyncAdapter({
      parameters: [{ id: "ParamMouthOpenY", name: "open", groupId: null }],
    })).toMatchObject({ jawParameter: null, puckerWidenParameter: null, funnelParameter: null });
  });

  it("uses a declared custom LipSync parameter but asks for one confirmation", () => {
    expect(resolveLipSyncAdapter({
      lipSyncParameters: ["MyCustomMouth"],
      parameters: [{ id: "MyCustomMouth", name: "嘴巴", groupId: "Mouth" }],
    })).toMatchObject({
      mode: "open-form",
      source: "automatic",
      needsConfirmation: true,
      mouthOpenParameter: "MyCustomMouth",
      mouthFormParameter: null,
    });
  });

  it("reports an unconfigured model when no mouth parameter is discoverable", () => {
    expect(resolveLipSyncAdapter({ lipSyncParameters: [], parameters: [] }))
      .toMatchObject({
        mode: "unconfigured",
        source: "automatic",
        needsConfirmation: true,
        mouthOpenParameter: null,
      });
  });
});
