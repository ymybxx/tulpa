import { app } from "electron";
import path from "node:path";

export type PetCursor = "default" | "grabbing" | "pointer";

export interface PetApplicationActivation {
  isActive(): boolean;
  makeKeyWindow(handle: Buffer): boolean;
  acquire(): object | null;
  restore(token: object): boolean;
  resignKeyWindow(handle: Buffer): boolean;
  setCursor(kind: PetCursor): boolean;
}

export interface PetMouseMonitor {
  startMouseMonitor(onMovement: () => void): boolean;
  stopMouseMonitor(): void;
  /** Presses in other apps' windows or on the desktop. */
  startClickMonitor(onPress: () => void): boolean;
  stopClickMonitor(): void;
}

export interface WindowsNativeWindowSupport {
  initializeHostJob?(): boolean;
  setNonActivating?(handle: Buffer): boolean;
  attachProcessToJob?(pid: number): boolean;
  terminateProcessJob?(pid: number): boolean;
}

export function loadPetApplicationActivation(): PetApplicationActivation & PetMouseMonitor & WindowsNativeWindowSupport {
  const root = app.isPackaged
    ? path.join(process.resourcesPath, "native")
    : path.join(__dirname, process.platform === "win32" ? "../../runtime/win32-x64" : "../../runtime/mac-arm64");
  return require(path.join(root, "pet-application-activation.node")) as PetApplicationActivation & PetMouseMonitor & WindowsNativeWindowSupport;
}
