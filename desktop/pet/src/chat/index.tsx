import { render } from "preact";

import "../ui/theme.css";
import "../ui/ui.css";
import "../ui/sky.css";
import "../settings-next/settings.css";
import "../onboarding/onboarding.css";
import "../character/character.css";
import "./chat.css";
import { followScene } from "../ui/scene";
import { ChatApp } from "./ChatApp";

let mounted: { root: HTMLElement; stopScene: () => void } | null = null;

/** Shows the 聊天 page: the conversation with the current character. */
export function mountChat(root: HTMLElement): void {
  if (mounted?.root === root) return;
  unmountChat();
  root.classList.add("ui-theme");
  mounted = { root, stopScene: followScene(root) };
  render(<ChatApp />, root);
}

export function unmountChat(): void {
  if (!mounted) return;
  render(null, mounted.root);
  mounted.stopScene();
  mounted = null;
}
