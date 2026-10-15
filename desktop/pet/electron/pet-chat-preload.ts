import { contextBridge, ipcRenderer } from "electron";
import type { PetChatApi, PetChatState } from "../src/shared/pet-chat";
import type { DesktopUiLocale } from "../src/shared/contracts";

const api: PetChatApi = {
  getUiLocale: () => ipcRenderer.invoke("desktop:ui-locale:get"),
  onUiLocale: (callback) => {
    const listener = (_event: Electron.IpcRendererEvent, locale: DesktopUiLocale): void => callback(locale);
    ipcRenderer.on("desktop:ui-locale", listener);
    return () => ipcRenderer.removeListener("desktop:ui-locale", listener);
  },
  ready: () => ipcRenderer.send("pet:chat:ready"),
  open: () => ipcRenderer.send("pet:chat:open"),
  close: () => ipcRenderer.send("pet:chat:close"),
  composing: (active) => ipcRenderer.send("pet:chat:composing", active === true),
  openHistory: () => ipcRenderer.send("pet:chat:history"),
  send: (id, content, attachmentIds) => ipcRenderer.invoke("pet:chat:send", id, content, attachmentIds ?? []),
  attach: (upload) => ipcRenderer.invoke("pet:chat:attach", upload),
  newConversation: () => ipcRenderer.invoke("pet:chat:new-conversation"),
  imageInput: () => ipcRenderer.invoke("desktop:model-image-input:get"),
  senseAction: (action) => ipcRenderer.send("pet:senses:action", action),
  quickAction: (action) => ipcRenderer.send("pet:quick:action", action),
  openMain: () => ipcRenderer.send("pet:main:open"),
  onState: (callback) => {
    const listener = (_event: Electron.IpcRendererEvent, state: PetChatState): void => callback(state);
    ipcRenderer.on("pet:chat:state", listener);
    return () => ipcRenderer.removeListener("pet:chat:state", listener);
  },
  onPointer: (callback) => {
    const listener = (_event: Electron.IpcRendererEvent, point: { x: number; y: number } | null): void => callback(point);
    ipcRenderer.on("pet:chat:pointer", listener);
    return () => ipcRenderer.removeListener("pet:chat:pointer", listener);
  },
};
contextBridge.exposeInMainWorld("petChat", api);
