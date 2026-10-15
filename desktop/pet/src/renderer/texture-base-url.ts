/**
 * Marks renderer atlas requests as immutable base reads. The main-process
 * protocol keeps legacy compiled-atlas substitution for explicit file
 * consumers, but a renderer must always rebuild from base + its snapshot.
 */
export function immutableTextureBaseUrl(value: string): string {
  const url = new URL(value);
  if (url.protocol === "pet-model:") url.searchParams.set("textureBase", "1");
  return url.toString();
}
