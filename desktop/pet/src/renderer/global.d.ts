import type { DesktopPetApi } from "../shared/contracts";

declare global {
  interface Window {
    desktopPet: DesktopPetApi;
    Live2DCubismCore?: unknown;
  }
}

export {};
