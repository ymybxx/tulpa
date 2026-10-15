import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { LatestAtomicFileWriter } from "./latest-atomic-file-writer";

describe("LatestAtomicFileWriter", () => {
  it("serializes concurrent replacements and persists the newest complete value", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "atomic-writer-"));
    try {
      const target = path.join(root, "state.json");
      const writer = new LatestAtomicFileWriter(target);
      await Promise.all(Array.from({ length: 100 }, (_, index) =>
        writer.write(JSON.stringify({ index }))
      ));
      await writer.flush();
      expect(JSON.parse(await readFile(target, "utf8"))).toEqual({ index: 99 });
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
