/** Isolated native QA only; the production entry never imports this module. */
import type { Live2DRenderer } from "../renderer/live2d-renderer";
import type { ModelEditCapabilityManifest, ModelPackage } from "../shared/contracts";
import { CharacterWizard } from "./character-wizard";

export function textureSelectionHarness(renderer: Live2DRenderer, viewport: HTMLElement) {
  let wizard: CharacterWizard | null = null;
  let pick: ((ids: string[]) => void) | undefined;
  let requestedAI = 0;
  const canvas = viewport.querySelector("canvas")!;
  const park = () => { pick = undefined; renderer.highlightLayers([]); document.body.prepend(viewport); };
  return {
    open: async (model: ModelPackage, capability: ModelEditCapabilityManifest) => {
      if (!wizard) {
        document.body.style.cssText = "margin:0;background:#0b1712;color:#eaf8f3;font:14px system-ui";
        const shell = document.createElement("main"); shell.className = "models-shell";
        shell.innerHTML = '<div id="character-wizard" class="character-wizard" hidden><section class="character-wizard-panel"><header><h2 id="character-wizard-title"></h2><button id="character-wizard-close">关闭</button></header><div id="character-wizard-body"></div></section></div>';
        document.body.append(shell);
        window.desktopPet = new Proxy({
          onCharacterProgress: () => () => {}, listCharacterJobs: async () => [],
        }, { get: (value, key: string) => key in value ? value[key as keyof typeof value] : () => { requestedAI++; throw new Error(`Unexpected desktop call: ${key}`); } }) as unknown as typeof window.desktopPet;
        wizard = new CharacterWizard({
          models: () => [model],
          alignmentGuideUrl: async () => `pet-model://${model.id}/.edit-capabilities/alignment-guide.png`,
          texturePreviewUrl: unit => `pet-model://${model.id}/${unit.previewPath}`,
          textureMaterials: () => [],
          previewModel: async () => {}, previewTextureDelta: async () => {},
          prepareModelReference: async () => {}, refreshModels: async () => {},
          attachPreview: async (host, _id, handler) => {
            host.append(viewport); viewport.className = "models-preview-stage";
            viewport.style.width = "100%"; viewport.style.height = "100%";
            renderer.setEditingPreview(true); renderer.setPaused(true); renderer.resize(); pick = handler;
          },
          highlightLayers: ids => renderer.highlightLayers(ids),
          parkPreview: park, restorePreview: async () => park(),
          setPreviewPaused: value => renderer.setPaused(value),
        });
      }
      await wizard.open(capability);
      await new Promise(resolve => requestAnimationFrame(resolve));
      renderer.resize();
    },
    pick: (x: number, y: number) => {
      const ids = renderer.pickLayers(x, y); pick?.(ids); return ids;
    },
    hover: (id: string, entered: boolean) => {
      const card = [...document.querySelectorAll<HTMLElement>("[data-texture-unit-id]")].find(card => card.dataset.textureUnitId === id);
      if (!card) throw new Error(`Missing card ${id}`);
      card.dispatchEvent(new MouseEvent(entered ? "mouseenter" : "mouseleave"));
    },
    control: (name: string, value = "") => {
      const root = document.querySelector("#character-wizard-body")!;
      if (name === "description") {
        const field = root.querySelector("textarea")!;
        field.value = value; field.dispatchEvent(new Event("input", { bubbles: true })); return;
      }
      const button = [...root.querySelectorAll("button")].find(item => item.textContent === name);
      if (!button) throw new Error(`Missing control ${name}`);
      button.click();
    },
    snapshot: () => {
      const selection = (wizard as unknown as { textureSelection: { ids: string[]; prompt: string } | null }).textureSelection;
      return { selected: selection?.ids ?? [], prompt: selection?.prompt, requestedAI,
        cards: document.querySelectorAll(".texture-unit-card").length,
        canvasCount: document.querySelectorAll("canvas").length,
        picking: !!pick, image: canvas.toDataURL("image/png"),
        hint: document.querySelector(".texture-person-locator-status small")?.textContent };
    },
    close: async () => { await wizard?.close(); renderer.setPaused(true); },
  };
}
