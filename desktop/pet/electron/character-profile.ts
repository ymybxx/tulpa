import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import type { CharacterProfile, PendingCharacter } from "../src/shared/contracts";

const MAX_NAME_LENGTH = 64;
const MAX_TEXT_LENGTH = 200;

export function validateCharacterProfile(value: unknown): CharacterProfile {
  if (typeof value !== "object" || value === null) throw new Error("角色设定无效");
  const record = value as Record<string, unknown>;
  const name = text(record.name, "名字", MAX_NAME_LENGTH);
  if (!name) throw new Error("请给角色起个名字");
  return {
    name,
    role: text(record.role, "身份与关系", MAX_TEXT_LENGTH),
    setting: text(record.setting, "性格与表达", MAX_TEXT_LENGTH),
  };
}

export function validatePendingCharacter(value: unknown): PendingCharacter {
  const profile = validateCharacterProfile(value);
  const modelId = (value as Record<string, unknown>).modelId;
  if (modelId !== null && modelId !== undefined && typeof modelId !== "string") {
    throw new Error("角色形象无效");
  }
  return { ...profile, modelId: modelId || null };
}

/**
 * Holds the character created during onboarding until the memory API is
 * running. The API only starts after an AI service is configured, so the
 * onboarding window cannot create the character directly.
 */
export class PendingCharacterStore {
  private readonly filePath: string;

  constructor(userDataPath: string) {
    this.filePath = path.join(userDataPath, "pending-character.json");
  }

  async pending(): Promise<PendingCharacter | null> {
    const value = await readJsonFile(this.filePath);
    if (value === undefined) return null;
    try {
      return validatePendingCharacter(value);
    } catch (error) {
      console.error("Ignoring invalid pending character:", error);
      return null;
    }
  }

  async save(character: PendingCharacter): Promise<void> {
    await writeJsonFile(this.filePath, validatePendingCharacter(character));
  }

  async clear(): Promise<void> {
    await rm(this.filePath, { force: true });
  }
}

/** Which 形象 (model) each character wears. Characters live in the memory API. */
export class CharacterAppearanceStore {
  private readonly filePath: string;

  constructor(userDataPath: string) {
    this.filePath = path.join(userDataPath, "character-appearances.json");
  }

  async modelFor(characterId: string): Promise<string | null> {
    return (await this.all())[characterId] ?? null;
  }

  async bind(characterId: string, modelId: string | null): Promise<void> {
    const current = await this.all();
    if (modelId) current[characterId] = modelId;
    else delete current[characterId];
    await writeJsonFile(this.filePath, current);
  }

  async remove(characterId: string): Promise<void> {
    await this.bind(characterId, null);
  }

  private async all(): Promise<Record<string, string>> {
    const value = await readJsonFile(this.filePath);
    if (typeof value !== "object" || value === null || Array.isArray(value)) return {};
    return Object.fromEntries(
      Object.entries(value).filter((entry): entry is [string, string] => typeof entry[1] === "string"),
    );
  }
}

type Fetcher = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

/** Lets a newly created character say its first words; the API answers each character once. */
export async function announceBirth(
  apiUrl: string,
  apiKey: string,
  characterId: string,
  fetcher: Fetcher = fetch,
): Promise<void> {
  const response = await fetcher(`${apiUrl}/v1/fishes/${encodeURIComponent(characterId)}/birth`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) throw new Error(`出生事件没送到（${response.status}）`);
}

/** Creates a new character and makes it the one the desktop pet plays. */
export async function createCharacter(
  apiUrl: string,
  apiKey: string,
  profile: CharacterProfile,
  fetcher: Fetcher = fetch,
): Promise<string> {
  const headers = { Authorization: `Bearer ${apiKey}` };
  const created = await fetcher(`${apiUrl}/v1/fishes`, {
    method: "POST",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify(validateCharacterProfile(profile)),
  });
  if (!created.ok) throw new Error(`创建角色失败（${created.status}）`);
  const { fishId } = await created.json() as { fishId?: unknown };
  if (typeof fishId !== "string" || !fishId) throw new Error("创建角色失败");
  const bound = await fetcher(`${apiUrl}/v1/fishes/${encodeURIComponent(fishId)}/assistant`, {
    method: "PUT",
    headers,
  });
  if (!bound.ok) throw new Error(`切换到新角色失败（${bound.status}）`);
  // The name is the character's first wake word; the keyword spotter needs Chinese.
  const name = validateCharacterProfile(profile).name;
  if (/^[一-鿿]{2,8}$/u.test(name)) {
    const words = await fetcher(`${apiUrl}/v1/fishes/${encodeURIComponent(fishId)}/wake-words`, {
      method: "PUT",
      headers: { ...headers, "Content-Type": "application/json" },
      body: JSON.stringify({ words: [name] }),
    });
    if (!words.ok) console.error(`Saving the wake word failed (${words.status})`);
  }
  return fishId;
}

async function readJsonFile(filePath: string): Promise<unknown> {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      console.error(`Ignoring unreadable ${path.basename(filePath)}:`, error);
    }
    return undefined;
  }
}

async function writeJsonFile(filePath: string, value: unknown): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.tmp`;
  await writeFile(temporary, JSON.stringify(value), { mode: 0o600 });
  await rename(temporary, filePath);
}

function text(value: unknown, label: string, max: number): string {
  if (value === undefined || value === null) return "";
  if (typeof value !== "string") throw new Error(`${label}无效`);
  const trimmed = value.trim();
  if (trimmed.length > max) throw new Error(`${label}不能超过 ${max} 个字`);
  return trimmed;
}
