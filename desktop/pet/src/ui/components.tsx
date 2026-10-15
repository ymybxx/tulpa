import { t } from "../i18n";
import type { ComponentChildren, JSX } from "preact";
import { useState } from "preact/hooks";

import { Icon, type IconName } from "./icons";

type Tone = "ok" | "warn" | "info" | "muted";
export type Hue = "slate" | "blue" | "amber" | "rose" | "green";

export function PillButton({
  variant = "primary", icon, children, size = "md", ...rest
}: {
  variant?: "primary" | "secondary";
  icon?: IconName;
  size?: "md" | "sm";
  children: ComponentChildren;
} & Omit<JSX.ButtonHTMLAttributes<HTMLButtonElement>, "icon" | "size">) {
  return (
    <button type="button" class={`ui-pill ui-pill--${variant} ui-pill--${size}${icon ? "" : " ui-pill--plain"}`} {...rest}>
      {icon && <span class="ui-pill__icon"><Icon name={icon} size={size === "sm" ? 11 : 14} weight={3} /></span>}
      <span>{children}</span>
    </button>
  );
}

/** Going back a level: a pill with an arrow, above the page's title, saying where to. */
export function BackButton({ children, onClick }: { children: ComponentChildren; onClick: () => void }) {
  return (
    <button type="button" class="ui-back" onClick={onClick}>
      <span class="ui-back__icon"><Icon name="back" size={14} weight={2.6} /></span>
      {children}
    </button>
  );
}

export function TextButton({ children, ...rest }: { children: ComponentChildren } & JSX.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button type="button" class="ui-text-button" {...rest}>{children}</button>;
}

/** Centered section heading with the thin gold ornament lines. */
export function OrnamentTitle({ children }: { children: ComponentChildren }) {
  return (
    <div class="ui-ornament">
      <svg width="44" height="10" viewBox="0 0 44 10" aria-hidden="true"><path d="M0 5 H26 M30 5 c2 -4 6 -4 8 0 c2 4 6 4 6 0" /></svg>
      <span>{children}</span>
      <svg width="44" height="10" viewBox="0 0 44 10" aria-hidden="true"><path d="M44 5 H18 M14 5 c-2 -4 -6 -4 -8 0 c-2 4 -6 4 -6 0" /></svg>
    </div>
  );
}

export function Panel({ children, footer, class: className = "" }: {
  children: ComponentChildren;
  footer?: ComponentChildren;
  class?: string;
}) {
  return (
    <section class={`ui-panel ${className}`}>
      <div class="ui-panel__body">{children}</div>
      {footer && <footer class="ui-panel__footer">{footer}</footer>}
    </section>
  );
}

export function Card({ title, aside, children }: { title?: string; aside?: ComponentChildren; children: ComponentChildren }) {
  return (
    <section class="ui-card">
      {(title || aside) && (
        <header class="ui-card__header">
          {title && <h2>{title}</h2>}
          {aside && <span class="ui-card__aside">{aside}</span>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Badge({ tone, children }: { tone: Tone; children: ComponentChildren }) {
  return <span class={`ui-badge ui-badge--${tone}`}>{children}</span>;
}

export function Chip({ selected, children, ...rest }: {
  selected: boolean;
  children: ComponentChildren;
} & JSX.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button type="button" class="ui-chip" aria-pressed={selected} {...rest}>
      {selected && <span class="ui-chip__check"><Icon name="check" size={10} weight={3.2} /></span>}
      {children}
    </button>
  );
}

/** A chip that turns into a small input so users can add their own tag. */
export function AddChip({ placeholder, label, onAdd }: {
  placeholder: string;
  label: string;
  onAdd: (value: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const commit = () => {
    const value = draft.trim();
    if (value) onAdd(value);
    setDraft("");
    setEditing(false);
  };
  if (!editing) {
    return (
      <button type="button" class="ui-chip ui-chip--add" onClick={() => setEditing(true)}>
        <Icon name="plus" size={12} weight={2.6} />{t("添加")}</button>
    );
  }
  return (
    <span class="ui-chip-input">
      <input
        aria-label={label}
        value={draft}
        placeholder={placeholder}
        maxLength={12}
        ref={(element) => element?.focus()}
        onInput={(event) => setDraft(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") commit();
          if (event.key === "Escape") setEditing(false);
        }}
        onBlur={commit}
      />
      <button type="button" aria-label={t("添加")} onMouseDown={(event) => event.preventDefault()} onClick={commit}>
        <Icon name="check" size={12} weight={3} />
      </button>
    </span>
  );
}

export function Segmented<T extends string>({ options, value, onChange, label }: {
  options: ReadonlyArray<{ id: T; label: string }>;
  value: T;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div class="ui-segmented" role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button
          type="button"
          role="radio"
          aria-checked={option.id === value}
          onClick={() => onChange(option.id)}
        >{t(option.label)}</button>
      ))}
    </div>
  );
}

/** Large selectable card used for single-choice questions. */
export function OptionCard({ selected, icon, hue, title, description, tags, recommended, onSelect }: {
  selected: boolean;
  icon: IconName;
  hue: Hue;
  title: string;
  description: string;
  tags?: readonly string[];
  recommended?: boolean;
  onSelect: () => void;
}) {
  return (
    <button type="button" role="radio" aria-checked={selected} class="ui-option" onClick={onSelect}>
      <span class={`ui-option__icon ui-hue-${hue}`}>
        <Icon name={icon} size={20} weight={1.9} />
      </span>
      <span class="ui-option__text">
        <span class="ui-option__title">{title}</span>
        <span class="ui-option__description">{description}</span>
        {tags && tags.length > 0 && (
          <span class="ui-option__tags">{tags.map((tag) => <span class="ui-tag">{tag}</span>)}</span>
        )}
      </span>
      {recommended && <span class="ui-option__flag">{t("推荐")}</span>}
    </button>
  );
}

export function Switch({ checked, label, onChange, disabled = false }: {
  checked: boolean; label: string; onChange: (value: boolean) => void; disabled?: boolean;
}) {
  return (
    <button type="button" role="switch" aria-checked={checked} aria-label={label} class="ui-switch" disabled={disabled}
      onClick={() => onChange(!checked)}>
      <span class="ui-switch__knob" />
    </button>
  );
}

export function Row({ icon, hue, title, subtitle, status, action, highlighted }: {
  icon: IconName;
  hue: Hue;
  title: string;
  subtitle?: string;
  status?: ComponentChildren;
  action?: ComponentChildren;
  highlighted?: boolean;
}) {
  return (
    <div class={`ui-row${highlighted ? " ui-row--highlighted" : ""}`}>
      <span class={`ui-row__icon ui-hue-${hue}`}><Icon name={icon} size={16} /></span>
      <span class="ui-row__text">
        <span class="ui-row__title">{title}{status}</span>
        {subtitle && <span class="ui-row__subtitle">{subtitle}</span>}
      </span>
      {action !== undefined && <span class="ui-row__action">{action}</span>}
    </div>
  );
}

/** A line spoken by the character, placed where the user needs to act. */
export function CharacterTip({ name, avatar, children }: { name: string; avatar?: ComponentChildren; children: ComponentChildren }) {
  return (
    <div class="ui-tip">
      {avatar && <span class="ui-tip__avatar">{avatar}</span>}
      <span><b>{name}</b>　{children}</span>
    </div>
  );
}

/** Modal confirmation in the parchment style; the primary action carries the yellow check. */
export function Dialog({ title, children, confirmLabel = t("确定"), onConfirm, onCancel, footer }: {
  title: string;
  children: ComponentChildren;
  confirmLabel?: string;
  onConfirm?: () => void;
  onCancel: () => void;
  /** Buttons of its own, in place of 取消 and the confirm button. */
  footer?: ComponentChildren;
}) {
  return (
    <div class="ui-dialog-backdrop" onClick={(event) => { if (event.target === event.currentTarget) onCancel(); }}>
      <section class="ui-dialog" role="dialog" aria-modal="true" aria-labelledby="ui-dialog-title"
        onKeyDown={(event) => { if (event.key === "Escape") onCancel(); }}>
        <div class="ui-dialog__body">
          <h2 id="ui-dialog-title">{title}</h2>
          {children}
        </div>
        <footer class="ui-dialog__footer">
          {footer ?? (
            <>
              <PillButton variant="secondary" icon="close" onClick={onCancel}>{t("取消")}</PillButton>
              <PillButton icon="check" onClick={onConfirm} autoFocus>{confirmLabel}</PillButton>
            </>
          )}
        </footer>
      </section>
    </div>
  );
}

/**
 * A message about what just happened: an error, a warning, a success or a
 * plain note. Always a soft tinted box with its icon, never bare red text.
 */
export function Notice({ tone = "error", children, action }: {
  tone?: "error" | "warn" | "ok" | "info";
  children: ComponentChildren;
  action?: ComponentChildren;
}) {
  const icon: IconName = tone === "ok" ? "check" : tone === "info" ? "spark" : "warn";
  return (
    <div class={`ui-notice ui-notice--${tone}`} role={tone === "error" ? "alert" : "status"}>
      <span class="ui-notice__icon"><Icon name={icon} size={12} weight={2.6} /></span>
      <span class="ui-notice__text">{children}</span>
      {action && <span class="ui-notice__action">{action}</span>}
    </div>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: ComponentChildren; children: ComponentChildren }) {
  return (
    <label class="ui-field">
      <span class="ui-field__label">{label}</span>
      {children}
      {hint && <span class="ui-field__hint">{hint}</span>}
    </label>
  );
}

export function StepDots({ total, current }: { total: number; current: number }) {
  return (
    <span class="ui-step-dots" aria-hidden="true">
      {Array.from({ length: total }, (_, index) => (
        <span class={index < current ? "done" : index === current ? "current" : ""} />
      ))}
    </span>
  );
}
