import { render } from "preact";

import "../ui/theme.css";
import "../ui/ui.css";
import "../ui/sky.css";
import "../settings-next/settings.css";
import "./looks.css";
import { followScene } from "../ui/scene";
import { LooksApp } from "./LooksApp";

let mounted: { root: HTMLElement; stopScene: () => void } | null = null;

/** Shows 形象: the looks the current character can wear. */
export function mountLooks(root: HTMLElement): void {
  if (mounted?.root === root) return;
  unmountLooks();
  root.classList.add("ui-theme");
  mounted = { root, stopScene: followScene(root) };
  render(<LooksApp />, root);
}

export function unmountLooks(): void {
  if (!mounted) return;
  render(null, mounted.root);
  mounted.stopScene();
  mounted = null;
}
