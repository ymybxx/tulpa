import { t } from "../i18n";
import type { CharacterDetails, CharacterProfileInput } from "../shared/contracts";
import { MAX_TRAITS, RELATIONS, TRAITS, toggleTrait, traitBlock, traitNote } from "../onboarding/flow";

export type CharacterTab = "profile" | "voice" | "memory";

export const CHARACTER_TABS: ReadonlyArray<{ id: CharacterTab; label: string; latin: string; tip: string }> = [
  { id: "profile", label: "性格", latin: "CHARACTER", tip: "叫我哪个名字都行，喊哪个我都会应～" },
  { id: "voice", label: "声音", latin: "VOICE", tip: "点 ▶ 听听看，挑一个你喜欢的。" },
  { id: "memory", label: "记忆", latin: "MEMORY", tip: "这些都是我记得的关于你的事，只存在这台电脑上。" },
];

export const MAX_WAKE_WORDS = 4;
export { MAX_TRAITS, RELATIONS, TRAITS, toggleTrait, traitBlock, traitNote };

/** The editable part of a character, compared field by field to find unsaved changes. */
export interface CharacterDraft {
  name: string;
  wakeWords: string[];
  background: string;
  role: string;
  setting: string;
}

export function draftFrom(profile: CharacterDetails): CharacterDraft {
  return {
    name: profile.name,
    wakeWords: [...profile.wakeWords],
    background: profile.background,
    role: profile.role,
    setting: profile.setting,
  };
}

export function changedFields(saved: CharacterDraft, draft: CharacterDraft): number {
  return (Object.keys(saved) as Array<keyof CharacterDraft>)
    .filter((key) => JSON.stringify(saved[key]) !== JSON.stringify(draft[key])).length;
}

export function profileInput(draft: CharacterDraft, saved: CharacterDraft): CharacterProfileInput {
  return {
    name: draft.name.trim(),
    role: draft.role.trim(),
    setting: draft.setting.trim(),
    background: draft.background.trim(),
    ...(JSON.stringify(draft.wakeWords) !== JSON.stringify(saved.wakeWords) ? { wakeWords: draft.wakeWords } : {}),
  };
}

/** The keyword spotter reads pinyin, so wake words are short Chinese phrases. */
export function wakeWordError(word: string, existing: readonly string[]): string | null {
  const value = word.trim();
  if (!/^[一-鿿]{2,8}$/u.test(value)) return t("唤醒词要 2 到 8 个汉字");
  if (existing.includes(value)) return t("已经有这个唤醒词了");
  if (existing.length >= MAX_WAKE_WORDS) return t("最多 {{max}} 个唤醒词", { max: MAX_WAKE_WORDS });
  return null;
}

export function relationText(relation: string): string {
  return `和用户是${relation}关系`;
}

/**
 * The model reads the relation as "和用户是…"; the person editing it is that
 * user, so the field shows "和你是…" and stores it back the model's way.
 */
export function relationShown(role: string): string {
  return role.replace(/^和用户是/, "和你是");
}

export function relationStored(shown: string): string {
  return shown.replace(/^和你是/, "和用户是");
}

/** Which relation chip the free text still matches, if any. */
export function selectedRelation(role: string): string | null {
  return RELATIONS.find((relation) => role.includes(relation)) ?? null;
}

export function selectedTraits(setting: string): string[] {
  return TRAITS.filter((trait) => setting.includes(trait));
}

export function traitsText(traits: readonly string[]): string {
  return `性格${traits.length ? traits.join("、") : "平和"}，说话${traits.includes("话痨") ? "自然" : "自然、简短"}。`;
}

export function daysSince(iso: string | null, now = new Date()): number | null {
  if (!iso) return null;
  const start = new Date(iso);
  if (Number.isNaN(start.getTime())) return null;
  // Calendar days: meeting yesterday evening is already the second day.
  const day = (value: Date) => Date.UTC(value.getFullYear(), value.getMonth(), value.getDate());
  return Math.max(1, Math.round((day(now) - day(start)) / 86_400_000) + 1);
}

/** Summary line under the character's name on the stage. */
export function characterSummary(draft: Pick<CharacterDraft, "role" | "setting">): string {
  const parts = [selectedRelation(draft.role), ...selectedTraits(draft.setting)].filter(Boolean);
  return parts.map((part) => t(part!)).join(" · ");
}
