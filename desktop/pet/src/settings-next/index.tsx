import { render } from "preact";

import "../ui/theme.css";
import "../ui/ui.css";
import "../ui/sky.css";
import "./settings.css";
import { followScene } from "../ui/scene";
import { SettingsApp } from "./SettingsApp";

let mounted: { root: HTMLElement; stopScene: () => void } | null = null;

/** Shows the settings page. */
export function mountSettings(root: HTMLElement): void {
  if (mounted?.root === root) return;
  unmountSettings();
  root.classList.add("ui-theme");
  mounted = { root, stopScene: followScene(root) };
  render(<SettingsApp />, root);
}

export function unmountSettings(): void {
  if (!mounted) return;
  render(null, mounted.root);
  mounted.stopScene();
  mounted = null;
}
