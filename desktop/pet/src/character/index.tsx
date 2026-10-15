import { render } from "preact";

import "../ui/theme.css";
import "../ui/ui.css";
import "../ui/sky.css";
import "../settings-next/settings.css";
import "../onboarding/onboarding.css";
import "./character.css";
import { followScene } from "../ui/scene";
import { CharacterApp } from "./CharacterApp";

let mounted: { root: HTMLElement; stopScene: () => void } | null = null;

/** Shows the 角色 page: who the character is, how TA looks, sounds and what TA remembers. */
export function mountCharacter(root: HTMLElement): void {
  if (mounted?.root === root) return;
  unmountCharacter();
  root.classList.add("ui-theme");
  mounted = { root, stopScene: followScene(root) };
  render(<CharacterApp />, root);
}

export function unmountCharacter(): void {
  if (!mounted) return;
  render(null, mounted.root);
  mounted.stopScene();
  mounted = null;
}
