/**
 * Pure routing input for a texture-material mutation.
 *
 * The invoking renderer applies the value returned by `ipcRenderer.invoke`, so
 * it must not receive the same mutation again through `pet:command`. A
 * service-originated mutation has no sender and is broadcast to every active
 * consumer instead.
 */
export interface TextureUpdateDeliveryRoute {
  senderId?: number | null;
  mainId?: number | null;
  petId?: number | null;
  modelIsSelectedPet: boolean;
}

/**
 * Resolve a stable, duplicate-free list of WebContents IDs that still need the
 * update. The main renderer is ordered first, followed by the desktop pet.
 */
export function textureUpdateRecipients(route: TextureUpdateDeliveryRoute): number[] {
  const recipients: number[] = [];
  const seen = new Set<number>();

  addRecipient(route.mainId);
  if (route.modelIsSelectedPet) addRecipient(route.petId);
  return recipients;

  function addRecipient(id: number | null | undefined): void {
    if (id == null || id === route.senderId || seen.has(id)) return;
    seen.add(id);
    recipients.push(id);
  }
}
