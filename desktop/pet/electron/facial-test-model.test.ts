import { describe, expect, it } from "vitest";
import type { ModelPackage } from "../src/shared/contracts";
import { facialTestModelInfo } from "./facial-test-model";

const model = { id: "another-model", name: "其他模型", manifest: { entry: "nested/model.model3.json" } } as ModelPackage;

describe("facial test model metadata", () => {
  it("works without a blush asset or physics, including models other than the original demo", async () => {
    expect(await facialTestModelInfo(model, async () => JSON.stringify({ FileReferences: {} }))).toEqual({
      modelId: model.id, modelName: model.name, blushParameter: null, physicsOutputParameterIds: [],
    });
  });

  it("resolves descriptor-relative assets and excludes output-only physics controls", async () => {
    const files: Record<string, unknown> = {
      "nested/model.model3.json": { FileReferences: { Physics: "rig.json", Expressions: [{ Name: "脸红", File: "expressions/blush.json" }] } },
      "nested/rig.json": { PhysicsSettings: [
        { Input: [{ Source: { Target: "Parameter", Id: "ParamAngleX" } }], Output: [
          { Destination: { Target: "Parameter", Id: "ParamAngleX" } },
          { Destination: { Target: "Parameter", Id: "OutputOnly" } },
        ] },
      ] },
      "nested/expressions/blush.json": { Parameters: [{ Id: "AuthorBlush", Value: 1, Blend: "Add" }] },
    };
    const result = await facialTestModelInfo(model, async file => {
      if (!(file in files)) throw new Error(`unexpected path: ${file}`);
      return JSON.stringify(files[file]);
    });
    expect(result.blushParameter).toBe("AuthorBlush");
    expect(result.physicsOutputParameterIds).toEqual(["OutputOnly"]);
  });

  it("reads the VTube Studio settings saved beside the model", async () => {
    const files: Record<string, unknown> = {
      "nested/model.model3.json": { FileReferences: {} },
      "nested/model.vtube.json": { ParameterSettings: [{ Input: "MouthSmile", OutputLive2D: "ParamMouthForm",
        InputRangeLower: 0, InputRangeUpper: 1, OutputRangeLower: -1, OutputRangeUpper: 1, ClampInput: false, ClampOutput: true }] },
    };
    const info = await facialTestModelInfo(model, async file => {
      if (!(file in files)) throw new Error("missing");
      return JSON.stringify(files[file]);
    });
    expect(info.vtsSettings).toEqual([{ input: "MouthSmile", inputLower: 0, inputUpper: 1, outputLower: -1, outputUpper: 1,
      clampInput: false, clampOutput: true, parameterId: "ParamMouthForm" }]);
  });

  it("does not guess one parameter from a multi-parameter or missing author expression", async () => {
    const read = async (file: string) => JSON.stringify(file.endsWith("model3.json")
      ? { FileReferences: { Expressions: [{ Name: "Blush", File: "blush.json" }] } }
      : { Parameters: [{ Id: "A", Value: 1 }, { Id: "B", Value: 1 }] });
    expect((await facialTestModelInfo(model, read)).blushParameter).toBeNull();
  });
});
