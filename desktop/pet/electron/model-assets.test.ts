import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { scanModelAssets } from "./model-assets";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { recursive: true, force: true }),
    ),
  );
});

describe("scanModelAssets", () => {
  it("catalogs registered motions and expressions with stable ids", async () => {
    const root = await temporaryDirectory();
    await mkdir(path.join(root, "motions"));
    await mkdir(path.join(root, "expressions"));
    await writeFile(path.join(root, "motions", "idle.motion3.json"), "idle");
    await writeFile(path.join(root, "motions", "wave.motion3.json"), "wave");
    await writeFile(path.join(root, "expressions", "happy.exp3.json"), "happy");
    const model = {
      FileReferences: {
        Motions: {
          Idle: [{ File: "motions/idle.motion3.json" }],
          Wave: [{ File: "motions/wave.motion3.json" }],
        },
        Expressions: [
          { Name: "happy", File: "expressions/happy.exp3.json" },
        ],
      },
    };

    const first = await scanModelAssets(root, "girl.model3.json", model);
    const second = await scanModelAssets(root, "girl.model3.json", model);

    expect(Object.values(first.motionAssets).map((asset) => asset.name)).toEqual([
      "idle",
      "wave",
    ]);
    expect(Object.values(first.expressionAssets)[0]).toMatchObject({
      expression: "happy",
      file: "expressions/happy.exp3.json",
    });
    expect(Object.keys(second.motionAssets)).toEqual(Object.keys(first.motionAssets));
    expect(Object.keys(second.expressionAssets)).toEqual(
      Object.keys(first.expressionAssets),
    );
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(os.tmpdir(), "fish-memory-assets-test-"));
  temporaryDirectories.push(directory);
  return directory;
}
