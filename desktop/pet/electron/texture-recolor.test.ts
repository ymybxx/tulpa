import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { PNG } from "pngjs";

import type { TextureEditUnit } from "../src/shared/contracts";
import { writeRecoloredPatches } from "./texture-recolor";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))); });

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), "recolor-"));
  roots.push(root);
  const image = new PNG({ width: 2, height: 1 });
  image.data.set([60, 70, 140, 255, 60, 70, 140, 255]);
  const mask = new PNG({ width: 2, height: 1 });
  mask.data.set([255, 255, 255, 255, 255, 255, 255, 0]);
  await writeFile(path.join(root, "current.png"), PNG.sync.write(image));
  await writeFile(path.join(root, "mask.png"), PNG.sync.write(mask));
  const unit = (id: string) => ({
    id, label: id, atlasIndex: 0, atlasFile: "a.png", bounds: { x: 0, y: 0, width: 2, height: 1 }, maskPath: "mask.png",
  }) as unknown as TextureEditUnit;
  return {
    root,
    units: [unit("a"), unit("b")],
    currentImage: async () => path.join(root, "current.png"),
    maskFile: () => path.join(root, "mask.png"),
  };
}

describe("writeRecoloredPatches", () => {
  it("writes each piece recolored inside its mask, with the mask beside it", async () => {
    const f = await fixture();
    const patches = await writeRecoloredPatches({ ...f, hex: "#C8323C", directory: path.join(f.root, "out"), name: "1" });
    expect(patches).toHaveLength(2);
    const out = PNG.sync.read(await readFile(patches![0]!.editedPatchPath));
    expect(out.data[0]).toBeGreaterThan(out.data[2]!);
    expect([...out.data.subarray(4, 8)]).toEqual([60, 70, 140, 255]);
    expect(await readFile(patches![0]!.maskPath)).toEqual(await readFile(path.join(f.root, "mask.png")));
  });

  it("stops when a newer color replaced the request", async () => {
    const f = await fixture();
    let calls = 0;
    const patches = await writeRecoloredPatches({ ...f, hex: "#C8323C", directory: path.join(f.root, "out"), name: "1", superseded: () => ++calls > 1 });
    expect(patches).toBeNull();
  });
});
