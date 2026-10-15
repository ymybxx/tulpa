import type { DesktopMainWindowState } from "../shared/contracts";

interface VisibilityTarget {
  hidden: boolean;
}

interface SelectionTarget {
  classList: { toggle(name: string, force: boolean): void };
  setAttribute(name: string, value: string): void;
}

interface FrameTarget {
  getAttribute(name: string): string | null;
  setAttribute(name: string, value: string): void;
}

export interface MainViewElements {
  chat: VisibilityTarget;
  models: VisibilityTarget;
  settings: VisibilityTarget;
  loading: VisibilityTarget;
  chatButton: SelectionTarget;
  modelsButton: SelectionTarget;
  settingsButton: SelectionTarget;
  frame: FrameTarget;
  plugins?: VisibilityTarget;
  pluginsButton?: SelectionTarget;
  onboarding?: VisibilityTarget;
}

export function applyMainView(
  elements: MainViewElements,
  state: DesktopMainWindowState,
): void {
  const chatSelected = state.view === "chat";
  const modelsSelected = state.view === "models";
  const settingsSelected = state.view === "settings";
  const pluginsSelected = state.view === "plugins";
  if (elements.plugins) elements.plugins.hidden = !pluginsSelected;
  if (elements.onboarding) elements.onboarding.hidden = state.view !== "onboarding";
  if (elements.pluginsButton) select(elements.pluginsButton, pluginsSelected);
  elements.chat.hidden = !chatSelected;
  elements.models.hidden = !modelsSelected;
  elements.settings.hidden = !settingsSelected;
  select(elements.chatButton, chatSelected);
  select(elements.modelsButton, modelsSelected);
  select(elements.settingsButton, settingsSelected);
  elements.loading.hidden = !chatSelected || state.chatUrl !== null;
  if (state.chatUrl && elements.frame.getAttribute("src") !== state.chatUrl) {
    elements.frame.setAttribute("src", state.chatUrl);
  }
}

function select(target: SelectionTarget, selected: boolean): void {
  target.classList.toggle("active", selected);
  target.setAttribute("aria-selected", String(selected));
}
