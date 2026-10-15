import { describe, expect, it } from "vitest";

import {
  configuredVowelParameters,
  detectDirectVowelParameters,
} from "./viseme-parameters";

describe("detectDirectVowelParameters", () => {
  it("detects Mao-style independent AEIOU parameters", () => {
    const ids = ["ParamA", "ParamI", "ParamU", "ParamE", "ParamO"];
    expect(detectDirectVowelParameters({
      getParameterCount: () => ids.length,
      getParameterIndex: (id) => {
        const index = ids.indexOf(String(id));
        return index >= 0 ? index : ids.length;
      },
    })).toEqual({ a: "ParamA", e: "ParamE", i: "ParamI", o: "ParamO", u: "ParamU" });
  });

  it("does not treat one generic open parameter as a direct vowel rig", () => {
    const ids = ["ParamA"];
    expect(Object.keys(detectDirectVowelParameters({
      getParameterCount: () => ids.length,
      getParameterIndex: (id) => ids.indexOf(String(id)) >= 0 ? 0 : ids.length,
    }))).toEqual(["a"]);
  });

  it("uses per-model custom vowel IDs without assuming standard names", () => {
    expect(configuredVowelParameters({
      a: "MouthAh",
      e: "MouthEh",
      i: "MouthEe",
      o: "MouthOh",
      u: "MouthOo",
    })).toEqual({
      a: "MouthAh",
      e: "MouthEh",
      i: "MouthEe",
      o: "MouthOh",
      u: "MouthOo",
    });
  });
});
