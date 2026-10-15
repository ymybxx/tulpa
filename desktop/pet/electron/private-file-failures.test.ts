import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppConfigStore } from "./app-config";
import { OnboardingDraftStore } from "./onboarding-draft";
import { GatewayServer } from "./gateway-server";

const execute = vi.hoisted(() => vi.fn((...args: unknown[]) => {
  (args.at(-1) as (error: Error) => void)(new Error("ACL update denied"));
}));
vi.mock("node:child_process", () => ({ execFile: execute }));
const roots: string[] = [];
afterEach(async () => {
  vi.unstubAllGlobals(); vi.clearAllMocks();
  await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true })));
});

describe("credential ACL failures", () => {
  it.each(["secrets.json", "onboarding-draft.json", "gateway-key"])("keeps %s intact and retries failed permission preparation", async name => {
    vi.stubGlobal("process", { ...process, platform: "win32" });
    const root = await mkdtemp(path.join(os.tmpdir(), "private-failure-"));
    roots.push(root);
    const file = path.join(root, name);
    const original = name === "secrets.json" ? JSON.stringify({ version: 2, values: { internalApiKey: "existing-fake-key" } }) : "existing-fake-key";
    await writeFile(file, original);
    const store = name === "secrets.json" ? new AppConfigStore(root) : name === "onboarding-draft.json" ? new OnboardingDraftStore(root) : new GatewayServer({ keyFile: file, forward: async () => ({ status: 202, body: {} }) });
    const load = () => store instanceof AppConfigStore ? store.runtimeSecrets() : store instanceof OnboardingDraftStore ? store.load() : store.status();
    await expect(load()).rejects.toThrow("ACL update denied");
    await expect(load()).rejects.toThrow("ACL update denied");
    expect(execute).toHaveBeenCalledTimes(2);
    expect(await readFile(file, "utf8")).toBe(original);
    expect(await readdir(root)).toEqual([name]);
    expect(JSON.stringify(execute.mock.calls)).not.toContain("existing-fake-key");
  });
});
