import { app } from "electron";
import { createRequire } from "node:module";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { PNG } from "pngjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);

app.once("ready", () => void run());

async function run() {
  let temporaryRoot;
  let exitCode = 0;
  try {
    temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "atlas-worker-qa-"));
    const source = path.join(temporaryRoot, "textures", "atlas.png");
    const patch = path.join(temporaryRoot, "patch.png");
    const mask = path.join(temporaryRoot, "mask.png");
    const contentKey = "a".repeat(64);
    const target = path.join(
      temporaryRoot,
      ".appearance",
      "compiled",
      contentKey,
      "textures",
      "atlas.png",
    );
    await mkdir(path.dirname(source), { recursive: true });
    await Promise.all([
      writePng(source, 2, 2, [10, 20, 30, 255]),
      writePng(patch, 1, 1, [200, 100, 50, 128]),
      writePng(mask, 1, 1, [255, 255, 255, 255]),
    ]);
    const { UtilityProcessAtlasCompactionRunner } = require(
      "../dist-electron/electron/atlas-compaction-utility-runner.js"
    );
    const runner = new UtilityProcessAtlasCompactionRunner(
      path.join(root, "dist-electron", "electron", "atlas-compaction-worker.js"),
      { timeoutMs: 15_000 },
    );
    await runner.compact({
      modelId: "qa-model",
      revision: 1,
      allowedRoots: [temporaryRoot],
      atlases: [{
        contentKey,
        atlasFile: "textures/atlas.png",
        sourcePath: source,
        targetPath: target,
        edits: [{
          textureUnitId: "qa-unit",
          atlasIndex: 0,
          bounds: { x: 0, y: 0, width: 1, height: 1 },
          maskPath: mask,
          imagePath: patch,
        }],
      }],
    });
    const output = PNG.sync.read(await readFile(target));
    const firstPixel = [...output.data.slice(0, 4)];
    const ok = firstPixel.join(",") === "200,100,50,128";
    process.stdout.write(`${JSON.stringify({ ok, firstPixel, width: output.width, height: output.height })}\n`);
    if (!ok) exitCode = 1;
  } catch (error) {
    console.error(error instanceof Error ? error.stack : String(error));
    exitCode = 1;
  } finally {
    if (temporaryRoot && path.basename(temporaryRoot).startsWith("atlas-worker-qa-")) {
      await rm(temporaryRoot, { recursive: true, force: true });
    }
    app.exit(exitCode);
  }
}

async function writePng(file, width, height, rgba) {
  const image = new PNG({ width, height });
  for (let offset = 0; offset < image.data.length; offset += 4) image.data.set(rgba, offset);
  await writeFile(file, PNG.sync.write(image));
}
