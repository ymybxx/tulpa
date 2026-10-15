import { describe, expect, it } from "vitest";

import type { NativeModelParameter } from "./contracts";
import { readVtsParameterSettings, vtsPartBindings } from "./vts-model-settings";

const parameter = (id: string, minimum = -1, maximum = 1, defaultValue = 0): NativeModelParameter => ({ id, name: id, minimum, maximum, defaultValue });
const row = (Input: string, OutputLive2D: string, inLow: number, inHigh: number, outLow: number, outHigh: number) => ({
  Input, OutputLive2D, InputRangeLower: inLow, InputRangeUpper: inHigh, OutputRangeLower: outLow, OutputRangeUpper: outHigh,
  ClampInput: false, ClampOutput: false, Smoothing: 50,
});
// Rows from 空.vtube.json, an author's VBridger model.
const kong = readVtsParameterSettings({ ParameterSettings: [
  row("MouthSmile", "ParamMouthForm", 0.1, 1, -1, 1),
  row("MouthSmile", "ParamCheek", 0, 1, 0.5, 1),
  row("MouthOpen", "ParamMouthOpenY", 0, 0.85, 0, 1),
  row("FaceAngleX", "ParamBodyAngleX", -30, 30, -10, 10),
  row("FaceAngleX", "ParamAngleX", -30, 30, -30, 30),
  row("EyeRightX", "ParamEyeBallX", -1, 1, 1, -1),
  row("BrowLeftY", "ParamBrowLYY", 0.2, 0.8, -1, 1),
  row("TongueOut", "ParamMouthX5", 0, 1, 0, 1),
  row("CheekPuff", "ParamMouthPuff", 0, 1, 0, 1),
  row("FacePositionZ", "Param22", 0, 6.5, 0, 0),
  { Input: "", OutputLive2D: "ParamBreath", InputRangeLower: 0, InputRangeUpper: 1, OutputRangeLower: 0, OutputRangeUpper: 1 },
] });
const native = [
  parameter("ParamMouthForm"), parameter("ParamCheek", 0, 1), parameter("ParamMouthOpenY", 0, 1),
  parameter("ParamBodyAngleX", -10, 10), parameter("ParamAngleX", -30, 30), parameter("ParamEyeBallX"),
  parameter("ParamBrowLYY"), parameter("ParamMouthX5", 0, 1), parameter("ParamMouthPuff", 0, 1), parameter("ParamBreath", 0, 1),
];

describe("a model's own VTube Studio settings", () => {
  it("reads only complete rows", () => {
    expect(kong).toHaveLength(10);
    expect(readVtsParameterSettings({ ParameterSettings: [{ Input: "MouthSmile" }, "x", null] })).toEqual([]);
    expect(readVtsParameterSettings("broken")).toEqual([]);
  });

  it("wires each face input's part to the author's parameters and ranges, speech and position left out", () => {
    const parts = vtsPartBindings(kong, native);
    expect(Object.keys(parts).sort()).toEqual(["ParamAngleX", "ParamBrowLY", "ParamCheekPuffC", "ParamEyeBallX", "ParamMouthForm", "ParamTongueOut"]);
    // The smile also warms the cheeks, resting where the model rests; the head also turns the body.
    expect(parts.ParamMouthForm).toEqual([
      { parameterId: "ParamMouthForm", lowerValue: -1, neutralValue: 0, upperValue: 1 },
      { parameterId: "ParamCheek", lowerValue: 0.5, neutralValue: 0.5, upperValue: 1 },
    ]);
    expect(parts.ParamAngleX).toEqual([
      { parameterId: "ParamBodyAngleX", lowerValue: -10, neutralValue: 0, upperValue: 10 },
      { parameterId: "ParamAngleX", lowerValue: -30, neutralValue: 0, upperValue: 30 },
    ]);
    // The author's own reversed gaze and brow driver, kept inside the model's range.
    expect(parts.ParamEyeBallX).toEqual([{ parameterId: "ParamEyeBallX", lowerValue: 1, neutralValue: 0, upperValue: -1 }]);
    expect(parts.ParamBrowLY).toEqual([{ parameterId: "ParamBrowLYY", lowerValue: -1, neutralValue: 0, upperValue: 1 }]);
    expect(parts.ParamTongueOut).toEqual([{ parameterId: "ParamMouthX5", lowerValue: 0, neutralValue: 0, upperValue: 1 }]);
    expect(parts.ParamCheekPuffC).toEqual([{ parameterId: "ParamMouthPuff", lowerValue: 0, neutralValue: 0, upperValue: 1 }]);
  });

  it("wires VBridger's expressive mouth shapes, not the ones lip sync drives", () => {
    const parts = vtsPartBindings(readVtsParameterSettings({ ParameterSettings: [
      row("MouthX", "ParamMouthX", -1, 1, -1, 1),
      row("MouthShrug", "ParamMouthShrug", 0, 1, 0, 1),
      row("MouthPressLipOpen", "ParamMouthPressLipOpen", -1, 1.2, -1, 1),
      row("JawOpen", "ParamJawOpen", 0, 1, 0, 1),
    ] }), [parameter("ParamMouthX"), parameter("ParamMouthShrug", 0, 1), parameter("ParamMouthPressLipOpen"), parameter("ParamJawOpen", 0, 1)]);
    expect(Object.keys(parts).sort()).toEqual(["ParamMouthPressLipOpen", "ParamMouthShrug", "ParamMouthX"]);
    expect(parts.ParamMouthX).toEqual([{ parameterId: "ParamMouthX", lowerValue: -1, neutralValue: 0, upperValue: 1 }]);
    expect(parts.ParamMouthShrug).toEqual([{ parameterId: "ParamMouthShrug", lowerValue: 0, neutralValue: 0, upperValue: 1 }]);
    expect(parts.ParamMouthPressLipOpen).toEqual([{ parameterId: "ParamMouthPressLipOpen", lowerValue: -1, neutralValue: 0, upperValue: 1 }]);
  });

  it("leaves out parameters the model lacks or its physics drives", () => {
    const parts = vtsPartBindings(kong, native.filter(item => item.id !== "ParamCheek"), ["ParamBodyAngleX"]);
    expect(parts.ParamMouthForm!.map(output => output.parameterId)).toEqual(["ParamMouthForm"]);
    expect(parts.ParamAngleX!.map(output => output.parameterId)).toEqual(["ParamAngleX"]);
  });

  it("keeps the model's own resting eye as the usual step, with room to widen up to the author's end", () => {
    const eyes = vtsPartBindings(readVtsParameterSettings({ ParameterSettings: [row("EyeOpenLeft", "ParamEyeLOpen", 0, 0.85, 0, 1.9)] }),
      [parameter("ParamEyeLOpen", 0, 1.9, 1)]);
    expect(eyes.ParamEyeLOpen).toEqual([{ parameterId: "ParamEyeLOpen", lowerValue: 0, neutralValue: 1, upperValue: 1.9 }]);
  });
});
