import { render } from "preact";
import { useLocale } from "../i18n/preact";

import "../ui/theme.css";
import "../ui/ui.css";
import "../ui/sky.css";
import "../settings-next/settings.css";
import "./look-editor.css";
import { followScene } from "../ui/scene";
import { MainNav } from "../ui/MainNav";
import { SkyStage } from "../ui/SkyStage";

let mounted: { root: HTMLElement; chrome: HTMLElement; stopScene: () => void } | null = null;

/**
 * 形象编辑 keeps the model manager's own DOM; this adds the sky, the main
 * navigation and the parchment theme around it.
 */
export function mountLookEditor(root: HTMLElement, chrome: HTMLElement): void {
  if (mounted?.root === root) return;
  unmountLookEditor();
  root.classList.add("ui-theme");
  mounted = { root, chrome, stopScene: followScene(root) };
  render(<LookEditorChrome />, chrome);
}

export function unmountLookEditor(): void {
  if (!mounted) return;
  render(null, mounted.chrome);
  mounted.stopScene();
  mounted = null;
}

function LookEditorChrome() {
  useLocale();
  return <><SkyStage /><div class="sn"><MainNav current="look" /></div></>;
}
