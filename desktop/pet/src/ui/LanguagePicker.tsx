import type { DesktopUiLocale } from "../shared/contracts";
import { Icon } from "./icons";

const LANGUAGES = [
  { id: "zh-CN", name: "简体中文" },
  { id: "en", name: "English" },
] as const;

/** Language names and the bilingual label remain recognizable in either locale. */
export function LanguagePicker({ locale, busy = false, onChange, showLabel = true }: {
  locale: DesktopUiLocale;
  busy?: boolean;
  onChange: (locale: DesktopUiLocale) => void;
  showLabel?: boolean;
}) {
  return (
    <div class="ui-language" aria-busy={busy}>
      {showLabel && <span class="ui-language__label">
        <Icon name="language" size={16} />
        <span lang="zh-CN">语言</span><span aria-hidden="true">/</span><span lang="en">Language</span>
      </span>}
      <div class="ui-segmented ui-language__choices" role="radiogroup" aria-label="语言 / Language"
        onKeyDown={(event) => {
          if (busy || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === "Home" ? "zh-CN" : event.key === "End" ? "en" : locale === "en" ? "zh-CN" : "en";
          event.currentTarget.querySelector<HTMLButtonElement>(`[data-locale="${next}"]`)?.focus();
          if (next !== locale) onChange(next);
        }}>
        {LANGUAGES.map((language) => <button type="button" role="radio" lang={language.id}
          data-locale={language.id} aria-checked={language.id === locale} disabled={busy}
          tabIndex={language.id === locale ? 0 : -1}
          onClick={() => { if (language.id !== locale) onChange(language.id); }}>
          <span class="ui-language__check" aria-hidden="true"><Icon name="check" size={12} weight={2.7} /></span>
          {language.name}
        </button>)}
      </div>
    </div>
  );
}
