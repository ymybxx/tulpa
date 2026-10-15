import { localizeStatic } from "../i18n/dom";
import "./base.css";
import "./styles.css";
import { PluginPanel } from "../plugins/panel";

import { ModelManager } from "../model-manager";
import { applyMainView, type MainViewElements } from "./view";
import { shouldActivateModelManager } from "./model-manager-visibility";
import { remember } from "../ui/remembered";

const chatFrame = required<HTMLIFrameElement>("chat-frame");
const elements: MainViewElements = {
  chat: required("chat-view"),
  models: required("models-view"),
  settings: required("settings-next-view"),
  loading: required("chat-loading"),
  chatButton: required("nav-chat"),
  modelsButton: required("nav-models"),
  settingsButton: required("nav-settings"),
  frame: chatFrame,
  plugins: required("plugins-view"),
  pluginsButton: required("nav-plugins"),
  onboarding: required("onboarding-view"),
};
const pluginPanel = new PluginPanel(required("plugins-view"));
const modelManager = new ModelManager();
let mainView: Parameters<typeof applyMainView>[1]["view"] = "chat";
let mainWindowVisible = false;
let mainWindowState: Parameters<typeof applyMainView>[1] | null = null;

// The character page opens the look editor on a specific look (worn or not).
window.addEventListener("look-editor:edit", (event) => {
  const modelId = (event as CustomEvent<unknown>).detail;
  if (typeof modelId === "string") modelManager.edit(modelId);
});

required("nav-chat").addEventListener("click", () => {
  window.desktopPet.showMainView("chat");
});
required("nav-models").addEventListener("click", () => {
  window.desktopPet.showMainView("character");
});
required("nav-settings").addEventListener("click", () => {
  window.desktopPet.showMainView("settings");
});
required("nav-plugins").addEventListener("click", () => window.desktopPet.showMainView("plugins"));
required("look-editor-back").addEventListener("click", (event) => {
  event.preventDefault();
  window.desktopPet.showMainView("looks");
});

window.desktopPet.onMainWindowState(showView);
window.desktopPet.onMainWindowVisibility((visible) => {
  mainWindowVisible = visible;
  if (mainWindowState) showView(mainWindowState);
  syncModelManager();
});
void window.desktopPet.getMainWindowState()
  .then(showView)
  .catch((error) => {
    required("chat-loading").textContent = error instanceof Error
      ? error.message
      : String(error);
  });
void window.desktopPet.getMainWindowVisible()
  .then((visible) => {
    mainWindowVisible = visible;
    if (mainWindowState) showView(mainWindowState);
    syncModelManager();
  })
  .catch(() => undefined);
window.addEventListener("beforeunload", () => modelManager.destroy());
document.addEventListener("visibilitychange", syncModelManager);

function showView(state: Parameters<typeof applyMainView>[1]): void {
  mainWindowState = state;
  mainView = state.view;
  // The chat page is the desktop's own now; the old web chat is never loaded.
  applyMainView(elements, { ...state, chatUrl: null });
  pluginPanel.setVisible(state.view === "plugins" && mainWindowVisible);
  syncOnboarding(state.view === "onboarding");
  syncSettingsNext(state.view === "settings");
  syncCharacter(state.view === "character");
  syncLooks(state.view === "looks");
  // 形象 reopens on the list or the editor, whichever was last shown.
  if (state.view === "looks" || state.view === "models") remember("look-page", state.view);
  syncChat(state.view === "chat");
  syncLookEditor(state.view === "models");
  // A hidden frame can contain an unsaved management form or a recording.
  // Keep that document mounted across visibility/navigation changes; closing
  // the BrowserWindow still releases it. Only initial loading is visibility-gated.
  syncModelManager();
}

function syncOnboarding(active: boolean): void {
  document.body.classList.toggle("onboarding-active", active);
  void import("../onboarding").then(({ mountOnboarding, unmountOnboarding }) => {
    if (active) mountOnboarding(required("onboarding-view"));
    else unmountOnboarding();
  });
}

function syncSettingsNext(active: boolean): void {
  document.body.classList.toggle("settings-next-active", active);
  required("settings-next-view").hidden = !active;
  void import("../settings-next").then(({ mountSettings, unmountSettings }) => {
    if (active) mountSettings(required("settings-next-view"));
    else unmountSettings();
  });
}

function syncChat(active: boolean): void {
  document.body.classList.toggle("chat-active", active);
  required("chat-next-view").hidden = !active;
  void import("../chat").then(({ mountChat, unmountChat }) => {
    if (active) mountChat(required("chat-next-view"));
    else unmountChat();
  });
}

function syncCharacter(active: boolean): void {
  document.body.classList.toggle("character-active", active);
  required("character-view").hidden = !active;
  void import("../character").then(({ mountCharacter, unmountCharacter }) => {
    if (active) mountCharacter(required("character-view"));
    else unmountCharacter();
  });
}

function syncLooks(active: boolean): void {
  document.body.classList.toggle("looks-active", active);
  required("looks-view").hidden = !active;
  void import("../looks").then(({ mountLooks, unmountLooks }) => {
    if (active) mountLooks(required("looks-view"));
    else unmountLooks();
  });
}

function syncLookEditor(active: boolean): void {
  document.body.classList.toggle("look-editor-active", active);
  void import("../look-editor").then(({ mountLookEditor, unmountLookEditor }) => {
    if (active) mountLookEditor(required("models-view"), required("look-editor-chrome"));
    else unmountLookEditor();
  });
}

function syncModelManager(): void {
  if (shouldActivateModelManager(mainView, document.hidden, mainWindowVisible)) {
    void modelManager.enter();
  } else {
    modelManager.leave();
  }
}

function required<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`缺少主窗口元素 #${id}`);
  return element as T;
}

localizeStatic(document);
