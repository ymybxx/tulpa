import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  CharacterAppearanceStore,
  announceBirth,
  createCharacter,
  PendingCharacterStore,
  validateCharacterProfile,
  validatePendingCharacter,
} from "./character-profile";

const profile = { name: "小鱼", role: "你的朋友", setting: "温柔、元气，说话简短" };

async function temporaryDirectory(): Promise<string> {
  return mkdtemp(path.join(os.tmpdir(), "character-"));
}

describe("validateCharacterProfile", () => {
  it("trims values and requires a name", () => {
    expect(validateCharacterProfile({ name: " 小鱼 ", role: " 朋友 ", setting: "" })).toEqual({
      name: "小鱼", role: "朋友", setting: "",
    });
    expect(() => validateCharacterProfile({ name: "  " })).toThrow("请给角色起个名字");
  });

  it("rejects text longer than the soul limits", () => {
    expect(() => validateCharacterProfile({ name: "小鱼", setting: "长".repeat(201) })).toThrow("200");
  });

  it("keeps the chosen appearance with a pending character", () => {
    expect(validatePendingCharacter({ ...profile, modelId: "mao" })).toEqual({ ...profile, modelId: "mao" });
    expect(validatePendingCharacter({ ...profile, modelId: "" })).toEqual({ ...profile, modelId: null });
  });
});

describe("PendingCharacterStore", () => {
  it("round-trips and clears the pending character", async () => {
    const store = new PendingCharacterStore(await temporaryDirectory());
    expect(await store.pending()).toBeNull();
    await store.save({ ...profile, modelId: "mao" });
    expect(await store.pending()).toEqual({ ...profile, modelId: "mao" });
    await store.clear();
    expect(await store.pending()).toBeNull();
  });

  it("ignores a corrupt file instead of blocking startup", async () => {
    const directory = await temporaryDirectory();
    await writeFile(path.join(directory, "pending-character.json"), "{");
    expect(await new PendingCharacterStore(directory).pending()).toBeNull();
  });
});

describe("CharacterAppearanceStore", () => {
  it("binds and removes a model per character", async () => {
    const store = new CharacterAppearanceStore(await temporaryDirectory());
    await store.bind("fish-1", "mao");
    await store.bind("fish-2", "hiyori");
    expect(await store.modelFor("fish-1")).toBe("mao");
    await store.remove("fish-1");
    expect(await store.modelFor("fish-1")).toBeNull();
    expect(await store.modelFor("fish-2")).toBe("hiyori");
  });
});

describe("createCharacter", () => {
  it("creates a new fish, makes it the assistant and keeps its name as wake word", async () => {
    const requests: Array<{ url: string; method: string; body?: string }> = [];
    const fetcher = async (input: string | URL | Request, init?: RequestInit) => {
      requests.push({ url: String(input), method: init?.method ?? "GET", body: init?.body as string | undefined });
      return init?.method === "POST"
        ? Response.json({ fishId: "fish-new" }, { status: 201 })
        : Response.json({ fishId: "fish-new" });
    };

    await expect(createCharacter("http://api", "key", profile, fetcher)).resolves.toBe("fish-new");
    expect(requests.map((request) => `${request.method} ${request.url}`)).toEqual([
      "POST http://api/v1/fishes",
      "PUT http://api/v1/fishes/fish-new/assistant",
      "PUT http://api/v1/fishes/fish-new/wake-words",
    ]);
    expect(JSON.parse(requests[0]!.body!)).toEqual(profile);
    expect(JSON.parse(requests[2]!.body!)).toEqual({ words: [profile.name] });
  });

  it("surfaces backend failures", async () => {
    const fetcher = async () => new Response("nope", { status: 500 });
    await expect(createCharacter("http://api", "key", profile, fetcher)).rejects.toThrow("500");
  });
});

describe("announceBirth", () => {
  it("asks the API to let the new character speak first", async () => {
    const requests: string[] = [];
    const fetcher = async (input: string | URL | Request, init?: RequestInit) => {
      requests.push(`${init?.method} ${String(input)} ${new Headers(init?.headers).get("Authorization")}`);
      return new Response(null, { status: 202 });
    };
    await announceBirth("http://api", "key", "fish new", fetcher);
    expect(requests).toEqual(["POST http://api/v1/fishes/fish%20new/birth Bearer key"]);
    await expect(announceBirth("http://api", "key", "x", async () => new Response(null, { status: 409 }))).rejects.toThrow("409");
  });
});
