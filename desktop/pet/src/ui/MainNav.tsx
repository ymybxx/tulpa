import { useLocale } from "../i18n/preact";
import { t } from "../i18n";
import { Icon, type IconName } from "./icons";
import { openLook } from "./remembered";

export type MainNavPage = "chat" | "character" | "look" | "settings";

const PAGES: ReadonlyArray<{ id: MainNavPage; label: string; icon: IconName }> = [
  { id: "chat", label: "聊天", icon: "chat" },
  { id: "character", label: "角色", icon: "person" },
  { id: "look", label: "形象", icon: "face" },
  { id: "settings", label: "设置", icon: "gear" },
];

/** The main window's top navigation, drawn on the sky. */
export function MainNav({ current }: { current: MainNavPage }) {
  useLocale();
  return (
    <nav class="sn-nav" aria-label={t("主导航")}>
      {PAGES.map((page) => (
        <a href="#" aria-current={page.id === current ? "page" : undefined}
          onClick={(event) => { event.preventDefault(); if (page.id !== current) page.id === "look" ? openLook() : window.desktopPet.showMainView(page.id); }}>
          <span><Icon name={page.icon} size={14} weight={2.2} /></span>{t(page.label)}
        </a>
      ))}
    </nav>
  );
}
