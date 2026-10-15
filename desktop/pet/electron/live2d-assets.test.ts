import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it, vi } from "vitest";

const netFetch = vi.hoisted(() => vi.fn(async () => { throw new Error("offline"); }));
vi.mock("electron", () => ({
  net: { fetch: netFetch },
  session: { defaultSession: { setProxy: vi.fn(async () => undefined) } },
}));

import { Live2dAssetInstaller, isCompatibleR5CoreSource } from "./live2d-assets";
import { MODEL_FILES } from "./live2d-sources";

const R5_CORE = "Live2D Cubism Core MocVersion_53=6 ColorBlendType_Overlay csmGetLatestMocVersion" + "x".repeat(100_000);

describe("Cubism R5 Core validation", () => {
  it("requires moc3 version 6 and advanced blend support", () => {
    expect(isCompatibleR5CoreSource(R5_CORE)).toBe(true);
    expect(isCompatibleR5CoreSource(
      "Live2D Cubism Core MocVersion_50=5 csmGetLatestMocVersion" + "x".repeat(100_000),
    )).toBe(false);
  });
});

describe("Live2dAssetInstaller", () => {
  it("takes the Core and Mao from the app bundle without going online", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "live2d-bundle-"));
    try {
      const bundle = path.join(root, "bundle");
      await mkdir(bundle, { recursive: true });
      await writeFile(path.join(bundle, "live2dcubismcore.min.js"), R5_CORE);
      for (const relative of MODEL_FILES) {
        const file = path.join(bundle, "Mao", ...relative.split("/"));
        await mkdir(path.dirname(file), { recursive: true });
        await writeFile(file, relative === "Mao.model3.json"
          ? JSON.stringify({ FileReferences: { Moc: "Mao.moc3", Textures: ["Mao.2048/texture_00.png"] } })
          : "x");
      }
      const installer = new Live2dAssetInstaller(path.join(root, "user"), bundle);
      expect(await installer.ready()).toBe(false);
      await installer.install(() => undefined);
      expect(await installer.ready()).toBe(true);
      expect(await readFile(installer.corePath, "utf8")).toBe(R5_CORE);
      expect(netFetch).not.toHaveBeenCalled();
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
