import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { PNG } from "pngjs";
import { afterEach, describe, expect, it } from "vitest";

import { MasterTemplateStore } from "./master-template-store";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("MasterTemplateStore", () => {
  it("installs a complete template into userData and never needs a bundled model", async () => {
    const source = await fixture();
    const userData = await mkdtemp(path.join(os.tmpdir(), "master-target-"));
    roots.push(userData);
    const store = new MasterTemplateStore(userData);

    const installed = await store.installFolder(source);

    expect(installed.id).toBe("girl-v1");
    expect(installed.name).toBe("Girl V1");
    expect(installed.root).toBe(path.join(userData, "master-templates", "girl-v1"));
    expect((await store.list()).map((item) => item.id)).toEqual(["girl-v1"]);
  });

  it("rejects templates with missing runtime or mask files", async () => {
    const source = await fixture(false);
    const userData = await mkdtemp(path.join(os.tmpdir(), "master-target-"));
    roots.push(userData);
    const store = new MasterTemplateStore(userData);

    await expect(store.installFolder(source)).rejects.toThrow(/masks\/body.png/);
  });

  it("rejects a mother template without fixed moc, physics, motions, and expressions", async () => {
    const source = await fixture();
    await writeFile(path.join(source, "girl.model3.json"), "{}");
    const userData = await mkdtemp(path.join(os.tmpdir(), "master-target-"));
    roots.push(userData);

    await expect(new MasterTemplateStore(userData).installFolder(source)).rejects.toThrow(/Moc|Physics|Motions|Expressions/);
  });

  it("rejects a required resource symlink", async () => {
    const source = await fixture();
    const outside = path.join(await mkdtemp(path.join(os.tmpdir(), "master-outside-")), "outside.moc3");
    roots.push(path.dirname(outside));
    // Directory junctions exercise the same lstat rejection on Windows without
    // requiring Developer Mode or the privilege to create file symlinks.
    if (process.platform === "win32") await mkdir(outside);
    else await writeFile(outside, "moc");
    await rm(path.join(source, "girl.moc3"));
    await symlink(outside, path.join(source, "girl.moc3"), process.platform === "win32" ? "junction" : "file");
    const userData = await mkdtemp(path.join(os.tmpdir(), "master-target-"));
    roots.push(userData);

    await expect(new MasterTemplateStore(userData).installFolder(source)).rejects.toThrow(/符号链接/);
  });

  it("rejects a silhouette mask whose size differs from its atlas slot", async () => {
    const source = await fixture();
    await writePng(path.join(source, "masks", "body.png"), 2, 2);
    const userData = await mkdtemp(path.join(os.tmpdir(), "master-target-"));
    roots.push(userData);

    await expect(new MasterTemplateStore(userData).installFolder(source)).rejects.toThrow(/蒙版尺寸/);
  });
});

async function fixture(includeMask = true): Promise<string> {
  const source = await mkdtemp(path.join(os.tmpdir(), "master-source-"));
  roots.push(source);
  await mkdir(path.join(source, "textures"), { recursive: true });
  await mkdir(path.join(source, "masks"), { recursive: true });
  await writeFile(path.join(source, "girl.moc3"), "moc");
  await writeFile(path.join(source, "girl.physics3.json"), "{}");
  await writeFile(path.join(source, "idle.motion3.json"), "{}");
  await writeFile(path.join(source, "happy.exp3.json"), "{}");
  await writeFile(path.join(source, "girl.model3.json"), JSON.stringify({
    FileReferences: {
      Moc: "girl.moc3",
      Textures: ["textures/atlas.png"],
      Physics: "girl.physics3.json",
      Motions: { Idle: [{ File: "idle.motion3.json" }] },
      Expressions: [{ Name: "happy", File: "happy.exp3.json" }],
    },
  }));
  await writePng(path.join(source, "textures", "atlas.png"), 4, 4);
  await writePng(path.join(source, "alignment-guide.png"), 512, 512);
  await writeFile(path.join(source, "pet.manifest.json"), "{}");
  if (includeMask) await writePng(path.join(source, "masks", "body.png"), 4, 4);
  await writeFile(path.join(source, "master.template.json"), JSON.stringify({
    schemaVersion: 2,
    id: "girl-v1",
    name: "Girl V1",
    modelEntry: "girl.model3.json",
    alignmentGuide: "alignment-guide.png",
    coverage: "full-body",
    canvas: { width: 4, height: 4, originX: 2, originY: 2, pixelsPerUnit: 2 },
    parameters: { eyeBlink: ["ParamEyeOpen"], lipSync: ["ParamMouthOpenY"] },
    textureAtlases: [{ file: "textures/atlas.png", width: 4, height: 4 }],
    slots: [{
      id: "body",
      name: "Body",
      drawableId: "D_BODY",
      partId: "PART_BODY",
      group: "body",
      source: "completion",
      required: true,
      zIndex: 1,
      atlas: { texture: 0, x: 0, y: 0, width: 4, height: 4 },
      anchor: { x: 2, y: 2 },
      canvasBounds: { x: -1, y: -1, width: 2, height: 2 },
      mesh: {
        vertices: [-1, 1, 1, 1, 1, -1, -1, -1],
        uvs: [0, 1, 1, 1, 1, 0, 0, 0],
        indices: [0, 1, 2, 0, 2, 3],
      },
      silhouetteMask: "masks/body.png",
      completionPrompt: "complete body",
    }],
  }));
  return source;
}

async function writePng(file: string, width: number, height: number): Promise<void> {
  const png = new PNG({ width, height });
  png.data.fill(255);
  await writeFile(file, PNG.sync.write(png));
}
