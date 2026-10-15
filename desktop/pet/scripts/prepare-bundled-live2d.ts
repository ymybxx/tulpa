/**
 * Puts the Live2D Cubism Core and the default Mao look into the app bundle, so
 * the first launch shows TA at once instead of downloading from overseas.
 *
 * Usage: vite-node scripts/prepare-bundled-live2d.ts <output dir> [--seed <live2d userData dir>]
 * The output is <bundled models>/live2d; a seed is an app data folder's `live2d`.
 */
import { createHash } from "node:crypto";
import { access, copyFile, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

import JSZip from "jszip";

import {
  BUNDLED_CORE, BUNDLED_MODEL_DIR, CORE_ARCHIVE_ENTRY, CUBISM_SDK_ARCHIVE_SHA256, CUBISM_SDK_ARCHIVE_URL,
  MODEL_FILES, SAMPLE_COMMIT, SAMPLE_ROOT, isCompatibleR5CoreSource,
} from "../electron/live2d-sources";

// jsDelivr's fastly edge is reachable from mainland China and ~3x faster than raw GitHub there.
const SAMPLE_MIRROR = `https://fastly.jsdelivr.net/gh/Live2D/CubismWebSamples@${SAMPLE_COMMIT}/Samples/Resources/Mao`;

const log = (message: string) => console.log(`[bundled-live2d] ${message}`);
const exists = (file: string) => access(file).then(() => true, () => false);

async function fetchBytes(url: string): Promise<Buffer> {
  const response = await fetch(url, { signal: AbortSignal.timeout(120_000) });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
  return Buffer.from(await response.arrayBuffer());
}

async function place(target: string, bytes: Buffer): Promise<void> {
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(`${target}.part`, bytes);
  await rename(`${target}.part`, target);
}

async function prepareCore(output: string, seed: string | null): Promise<void> {
  const target = path.join(output, BUNDLED_CORE);
  if (await exists(target) && isCompatibleR5CoreSource(await readFile(target, "utf8"))) return;
  const seeded = seed ? path.join(seed, "runtime", BUNDLED_CORE) : null;
  if (seeded && await exists(seeded) && isCompatibleR5CoreSource(await readFile(seeded, "utf8"))) {
    await mkdir(output, { recursive: true });
    await copyFile(seeded, target);
    log("copied Cubism Core from seed");
    return;
  }
  log("downloading Cubism SDK for its Core");
  const archive = await fetchBytes(CUBISM_SDK_ARCHIVE_URL);
  if (createHash("sha256").update(archive).digest("hex") !== CUBISM_SDK_ARCHIVE_SHA256) {
    throw new Error("Cubism SDK checksum mismatch");
  }
  const entry = (await JSZip.loadAsync(archive)).file(CORE_ARCHIVE_ENTRY);
  if (!entry) throw new Error("Cubism SDK has no Web Core");
  const core = await entry.async("nodebuffer");
  if (!isCompatibleR5CoreSource(core.toString("utf8"))) throw new Error("Cubism Core is not R5");
  await place(target, core);
}

async function prepareModel(output: string, seed: string | null): Promise<void> {
  for (const relative of MODEL_FILES) {
    const target = path.join(output, BUNDLED_MODEL_DIR, ...relative.split("/"));
    if (await exists(target)) continue;
    const seeded = seed ? path.join(seed, "default-model", "Mao", ...relative.split("/")) : null;
    if (seeded && await exists(seeded)) {
      await mkdir(path.dirname(target), { recursive: true });
      await copyFile(seeded, target);
      continue;
    }
    log(`downloading ${relative}`);
    const bytes = await fetchBytes(`${SAMPLE_MIRROR}/${relative}`).catch(() => fetchBytes(`${SAMPLE_ROOT}/${relative}`));
    await place(target, bytes);
  }
  const model = JSON.parse(await readFile(path.join(output, BUNDLED_MODEL_DIR, "Mao.model3.json"), "utf8"));
  if (model.FileReferences?.Moc !== "Mao.moc3") throw new Error("Mao.model3.json does not describe Mao");
}

const args = process.argv.slice(2);
const seedIndex = args.indexOf("--seed");
const seed = seedIndex >= 0 && args[seedIndex + 1] ? path.resolve(args[seedIndex + 1]!) : null;
const positional = args.filter((value, index) => !value.startsWith("--") && (seedIndex < 0 || index !== seedIndex + 1));
if (positional.length !== 1) throw new Error("usage: prepare-bundled-live2d.ts <output> [--seed <dir>]");
const output = path.resolve(positional[0]!);
await prepareCore(output, seed);
await prepareModel(output, seed);
log(`ready: ${output}`);
