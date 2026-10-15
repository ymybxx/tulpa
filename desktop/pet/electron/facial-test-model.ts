import path from "node:path";
import type { ModelPackage } from "../src/shared/contracts";
import type { FacialTestModelInfo } from "../src/shared/facial-test";
import { readVtsParameterSettings, type VtsParameterSetting } from "../src/shared/vts-model-settings";

type Json = Record<string, any>;

/** Only authored assets provide aliases; an absent optional control is normal. */
export async function facialTestModelInfo(
  model: ModelPackage,
  readAsset: (relative: string) => Promise<string>,
): Promise<FacialTestModelInfo> {
  const descriptor: Json = JSON.parse(await readAsset(model.manifest.entry));
  const references = descriptor.FileReferences ?? {};
  const relativeToEntry = (file: string) => path.posix.join(path.posix.dirname(model.manifest.entry), file);
  const inputs = new Set<string>(), outputs = new Set<string>();
  if (typeof references.Physics === "string" && references.Physics) {
    const physics: Json = JSON.parse(await readAsset(relativeToEntry(references.Physics)));
    for (const setting of physics.PhysicsSettings ?? []) {
      for (const { Source: source } of setting.Input ?? []) {
        if (source?.Target === "Parameter" && typeof source.Id === "string") inputs.add(source.Id);
      }
      for (const { Destination: destination } of setting.Output ?? []) {
        if (destination?.Target === "Parameter" && typeof destination.Id === "string") outputs.add(destination.Id);
      }
    }
  }
  let blushParameter: string | null = null;
  const expressions: Json[] = Array.isArray(references.Expressions) ? references.Expressions : [];
  const blush = expressions.find(item => /^(脸红|blush)$/i.test(item.Name ?? "") && typeof item.File === "string");
  if (blush) {
    try {
      const expression: Json = JSON.parse(await readAsset(relativeToEntry(blush.File)));
      const items = expression.Parameters;
      if (Array.isArray(items) && items.length === 1 && typeof items[0]?.Id === "string"
        && items[0].Id && Number.isFinite(items[0].Value) && items[0].Value > 0
        && [undefined, "Add", "Overwrite"].includes(items[0].Blend)) {
        blushParameter = items[0].Id;
      }
    } catch { /* Optional author alias is not required for standard parameters. */ }
  }
  // VTube Studio saves a model's settings beside it, named after the model: 空.model3.json → 空.vtube.json.
  let vtsSettings: VtsParameterSetting[] = [];
  const vtsFile = model.manifest.entry.replace(/\.model3\.json$/iu, ".vtube.json");
  if (vtsFile !== model.manifest.entry) {
    try { vtsSettings = readVtsParameterSettings(JSON.parse(await readAsset(vtsFile))); }
    catch { /* Most models are not from VTube Studio; names then decide. */ }
  }
  return { modelId: model.id, modelName: model.name, blushParameter,
    physicsOutputParameterIds: [...outputs].filter(id => !inputs.has(id)),
    ...vtsSettings.length ? { vtsSettings } : {} };
}
