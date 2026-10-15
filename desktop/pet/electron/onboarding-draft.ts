import { rm } from "node:fs/promises";
import path from "node:path";

import type { OnboardingDraft } from "../src/shared/contracts";
import { readPrivateText, writePrivateText } from "./private-file";

const STEPS = ["look", "soul", "brain", "companion", "permissions"] as const;
const PROVIDERS = ["qwen", "doubao", "custom"] as const;
const LEVELS = ["text", "voice", "full"] as const;

/**
 * Onboarding answers survive a relaunch, e.g. when macOS offers to
 * "quit and reopen" after Screen Recording is switched on. The file sits next
 * to secrets.json with the same owner-only mode because it may hold the API
 * key, and it is deleted once setup completes.
 */
export class OnboardingDraftStore {
  private readonly filePath: string;

  constructor(userDataPath: string) {
    this.filePath = path.join(userDataPath, "onboarding-draft.json");
  }

  async load(): Promise<OnboardingDraft | null> {
    try {
      return parseOnboardingDraft(JSON.parse(await readPrivateText(this.filePath)));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT" || error instanceof SyntaxError) return null;
      throw error;
    }
  }

  async save(value: unknown): Promise<void> {
    const draft = parseOnboardingDraft(value);
    if (!draft) throw new Error("引导进度无效");
    await writePrivateText(this.filePath, JSON.stringify(draft));
  }

  async clear(): Promise<void> {
    await rm(this.filePath, { force: true });
  }
}

export function parseOnboardingDraft(value: unknown): OnboardingDraft | null {
  if (typeof value !== "object" || value === null) return null;
  const record = value as Record<string, unknown>;
  const step = oneOf(record.step, STEPS);
  if (!step) return null;
  return {
    step,
    modelId: text(record.modelId, 200) || null,
    name: text(record.name, 64),
    relation: text(record.relation, 40),
    relations: list(record.relations),
    traits: list(record.traits),
    traitOptions: list(record.traitOptions),
    customSetting: text(record.customSetting, 200),
    provider: oneOf(record.provider, PROVIDERS) ?? "qwen",
    baseUrl: text(record.baseUrl, 500),
    model: text(record.model, 200),
    apiKey: text(record.apiKey, 500),
    level: oneOf(record.level, LEVELS) ?? "full",
    jevApiKey: text(record.jevApiKey, 500),
    bailianWorkspaceId: text(record.bailianWorkspaceId, 100),
  };
}

function oneOf<T extends string>(value: unknown, options: readonly T[]): T | null {
  return typeof value === "string" && (options as readonly string[]).includes(value) ? value as T : null;
}

function text(value: unknown, max: number): string {
  return typeof value === "string" ? value.slice(0, max) : "";
}

function list(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string").slice(0, 30).map((item) => item.slice(0, 20)) : [];
}
