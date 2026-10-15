import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { inspectCubismModel } from "./cubism-core-inspector";

// Mimics the Emscripten loader in live2dcubismcore.min.js: it detects Node
// through process.versions and registers process-wide abort handlers.
const FAKE_CORE = `
if (typeof process.versions.node !== "string") throw new Error("not node");
process.on("uncaughtException", (error) => { throw error; });
process.on("unhandledRejection", (reason) => { throw reason; });
const model = {
  canvasinfo: { CanvasWidth: 2, CanvasHeight: 3, CanvasOriginX: 1, CanvasOriginY: 1, PixelsPerUnit: 1 },
  parameters: { ids: ["ParamAngleX"] },
  parts: { ids: ["Part"], parentIndices: [-1] },
  drawables: {
    ids: ["Mesh"], textureIndices: [0], drawOrders: [500], blendModes: [0],
    opacities: [1], parentPartIndices: [0], vertexPositions: [[0, 1]],
    vertexUvs: [[0, 1]], indices: [[0]],
  },
  getRenderOrders: () => [0], update() {}, release() {},
};
globalThis.Live2DCubismCore = {
  Moc: { fromArrayBuffer: () => ({ _release() {} }) },
  Model: { fromMoc: () => model },
  Version: { csmGetMocVersion: () => 5 },
};
`;

const directories: string[] = [];

afterEach(async () => {
  await Promise.all(directories.splice(0).map((directory) =>
    rm(directory, { recursive: true, force: true })));
});

describe("inspectCubismModel", () => {
  it("keeps the core's process handlers off the app process", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-cubism-core-"));
    directories.push(root);
    const core = path.join(root, "live2dcubismcore.min.js");
    await writeFile(core, FAKE_CORE);
    const before = {
      rejection: process.listenerCount("unhandledRejection"),
      exception: process.listenerCount("uncaughtException"),
    };

    for (const name of ["a.moc3", "b.moc3"]) {
      const moc = path.join(root, name);
      await writeFile(moc, new Uint8Array([1, 2, 3]));
      const inspected = await inspectCubismModel(core, moc);
      expect(inspected.parameters).toEqual(["ParamAngleX"]);
      expect(inspected.drawables[0]).toMatchObject({ id: "Mesh", vertices: [0, 1] });
    }

    expect(process.listenerCount("unhandledRejection")).toBe(before.rejection);
    expect(process.listenerCount("uncaughtException")).toBe(before.exception);
  });
});
