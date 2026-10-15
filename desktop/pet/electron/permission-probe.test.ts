import { describe, expect, it } from "vitest";

import { probePermission } from "./permission-probe";

const answer = async () => "{\"screenCapture\":true,\"microphone\":\"denied\"}\n";

describe("probePermission", () => {
  it("reads each permission from the helper's JSON answer", async () => {
    let seen: string[] = [];
    await expect(probePermission("/helper", "screen", async (_file, args) => { seen = args; return answer(); })).resolves.toBe(true);
    expect(seen).toEqual(["--permission-probe"]);
    await expect(probePermission("/helper", "microphone", answer)).resolves.toBe(false);
  });

  it("returns null when the helper is missing or prints something else", async () => {
    await expect(probePermission("/helper", "screen", async () => { throw new Error("ENOENT"); })).resolves.toBeNull();
    await expect(probePermission("/helper", "microphone", async () => "hello")).resolves.toBeNull();
  });
});
