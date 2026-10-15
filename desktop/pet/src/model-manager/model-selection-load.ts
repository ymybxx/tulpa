/**
 * Loads one selected-model preview without allowing a late bundle response to
 * enter the renderer after a newer model has already become the selection.
 */
export async function loadCurrentModelSelection<TBundle, TResult>(input: {
  loadBundle(): Promise<TBundle>;
  loadPreview(bundle: TBundle): Promise<TResult>;
  isCurrent(): boolean;
}): Promise<{ current: true; result: TResult } | { current: false }> {
  const bundle = await input.loadBundle();
  if (!input.isCurrent()) return { current: false };
  const result = await input.loadPreview(bundle);
  if (!input.isCurrent()) return { current: false };
  return { current: true, result };
}
