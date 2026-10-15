import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { LatestAtomicFileWriter } from "./latest-atomic-file-writer";
import standardParameters from "../src/shared/facial-demo-parameters.json";
import { assertManualStandardId, validateCriteriaOverrides, validateCustomParameters, validateDecisionRule, validateDescriptionOverrides, validateManualBindings, validateParameterSettings, type CustomFacialParameter, type FacialBindingConfiguration, type FacialCriteriaOverrides, type FacialDecisionRule, type FacialDescriptionOverrides, type FacialParameterSettings, type FacialParameterSettingsMap, type ManualParameterBinding, type ManualParameterBindings } from "../src/shared/facial-bindings";

interface BindingFile {
  models: Record<string, ManualParameterBindings>;
  parameterSettings: Record<string, FacialParameterSettingsMap>;
  descriptions: Record<string, FacialDescriptionOverrides>;
  criteria: Record<string, FacialCriteriaOverrides>;
  customParameters: Record<string, CustomFacialParameter[]>;
}

/** Local test bindings are stored separately from authored model assets. */
export class FacialBindingStore {
  private readonly file: string;
  private readonly writer: LatestAtomicFileWriter;
  private pending: Promise<unknown> = Promise.resolve();
  constructor(private readonly directory: string) {
    this.file = path.join(directory, "facial-test-bindings.json");
    this.writer = new LatestAtomicFileWriter(this.file);
  }
  private async read(): Promise<BindingFile> {
    try {
      const value = JSON.parse(await readFile(this.file, "utf8"));
      if (value.version !== 1 || !value.models || typeof value.models !== "object" || Array.isArray(value.models)) throw new Error("绑定配置格式无效");
      if (value.parameterSettings !== undefined && (!value.parameterSettings || typeof value.parameterSettings !== "object" || Array.isArray(value.parameterSettings))) throw new Error("参数设置格式无效");
      if (value.descriptions !== undefined && (!value.descriptions || typeof value.descriptions !== "object" || Array.isArray(value.descriptions))) throw new Error("参数描述格式无效");
      if (value.criteria !== undefined && (!value.criteria || typeof value.criteria !== "object" || Array.isArray(value.criteria))) throw new Error("评分描述格式无效");
      if (value.customParameters !== undefined && (!value.customParameters || typeof value.customParameters !== "object" || Array.isArray(value.customParameters))) throw new Error("自定义参数格式无效");
      return withoutRetiredParts({ models: value.models, parameterSettings: value.parameterSettings ?? {},
        descriptions: value.descriptions ?? {}, criteria: value.criteria ?? {}, customParameters: value.customParameters ?? {} });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return { models: {}, parameterSettings: {}, descriptions: {}, criteria: {}, customParameters: {} };
      throw error;
    }
  }
  async get(modelId: string): Promise<ManualParameterBindings> {
    return (await this.getConfiguration(modelId)).manualBindings;
  }
  async getConfiguration(modelId: string): Promise<FacialBindingConfiguration> {
    await this.pending;
    const { models, parameterSettings, descriptions, criteria, customParameters } = await this.read();
    return {
      manualBindings: validateManualBindings(Object.hasOwn(models, modelId) ? models[modelId] : {}),
      parameterSettings: validateParameterSettings(Object.hasOwn(parameterSettings, modelId) ? parameterSettings[modelId] : {}),
      descriptions: validateDescriptionOverrides(Object.hasOwn(descriptions, modelId) ? descriptions[modelId] : {}),
      criteria: validateCriteriaOverrides(Object.hasOwn(criteria, modelId) ? criteria[modelId] : {}),
      customParameters: validateCustomParameters(Object.hasOwn(customParameters, modelId) ? customParameters[modelId] : []),
    };
  }
  /** `binding` is the part's parameters: a list, one (as 一键配置 finds it), or null for the automatic one. */
  async save(modelId: string, standardId: string, binding: ManualParameterBinding | ManualParameterBinding[] | null, settings?: FacialParameterSettings, description?: string, decision?: FacialDecisionRule): Promise<ManualParameterBindings> {
    validateModelId(modelId);
    assertManualStandardId(standardId);
    const clean = binding === null ? null : validateManualBindings({ [standardId]: binding })[standardId] ?? null;
    const cleanSettings = settings === undefined ? undefined : validateParameterSettings({ [standardId]: settings })[standardId]!;
    const cleanDescription = description === undefined ? undefined : description.trim()
      ? validateDescriptionOverrides({ [standardId]: description })[standardId]! : "";
    const cleanDecision = decision === undefined ? undefined : validateDecisionRule(decision);
    const operation = this.pending.then(async () => {
      const { models, parameterSettings, descriptions, criteria, customParameters } = await this.read();
      const next = validateManualBindings(Object.hasOwn(models, modelId) ? models[modelId] : {});
      if (clean === null) delete next[standardId];
      else next[standardId] = clean;
      models[modelId] = next;
      if (cleanSettings) {
        const controls = validateParameterSettings(Object.hasOwn(parameterSettings, modelId) ? parameterSettings[modelId] : {});
        if (cleanSettings.enabled && cleanSettings.multiplier === 1 && !cleanSettings.label && !cleanSettings.motion) delete controls[standardId];
        else controls[standardId] = cleanSettings;
        parameterSettings[modelId] = controls;
      }
      if (cleanDescription !== undefined) {
        const current = validateDescriptionOverrides(Object.hasOwn(descriptions, modelId) ? descriptions[modelId] : {});
        if (cleanDescription) current[standardId] = cleanDescription;
        else delete current[standardId];
        descriptions[modelId] = current;
      }
      if (cleanDecision !== undefined) {
        const current = validateCriteriaOverrides(Object.hasOwn(criteria, modelId) ? criteria[modelId] : {});
        const standard = standardParameters.find(parameter => parameter.id === standardId)!;
        if (cleanDecision.mode === "score" && cleanDecision.levels.length === standard.levels.length
          && cleanDecision.levels.every((level, index) => level === standard.levels[index])) delete current[standardId];
        else current[standardId] = cleanDecision;
        criteria[modelId] = current;
      }
      await mkdir(this.directory, { recursive: true });
      await this.writer.write(JSON.stringify({ version: 1, models, parameterSettings, descriptions, criteria, customParameters }, null, 2));
      return next;
    });
    this.pending = operation.catch(() => {});
    return operation;
  }

  async saveCustom(modelId: string, value: CustomFacialParameter | null, id = value?.id): Promise<CustomFacialParameter[]> {
    validateModelId(modelId);
    if (!id || typeof id !== "string") throw new Error("自定义参数 ID 无效");
    const clean = value ? validateCustomParameters([value])[0]! : null;
    const operation = this.pending.then(async () => {
      const file = await this.read();
      const current = validateCustomParameters(Object.hasOwn(file.customParameters, modelId) ? file.customParameters[modelId] : []);
      const next = validateCustomParameters([
        ...current.filter(item => item.id !== id),
        ...(clean ? [clean] : []),
      ]);
      if (!clean && next.length === current.length) throw new Error("自定义参数不存在");
      file.customParameters[modelId] = next;
      await mkdir(this.directory, { recursive: true });
      await this.writer.write(JSON.stringify({ version: 1, ...file }, null, 2));
      return next;
    });
    this.pending = operation.catch(() => {});
    return operation;
  }
}

/**
 * Parts that left the standard set keep no settings, and a model's own part
 * that became standard is now configured as the standard one.
 */
function withoutRetiredParts(file: BindingFile): BindingFile {
  const standard = new Set(standardParameters.filter(parameter => parameter.owner === "jev").map(parameter => parameter.id));
  const known = new Set(standardParameters.map(parameter => parameter.id));
  const perModel = <T>(value: Record<string, Record<string, T>>) => Object.fromEntries(Object.entries(value).map(([model, entries]) => [model,
    entries && typeof entries === "object" && !Array.isArray(entries)
      ? Object.fromEntries(Object.entries(entries).filter(([id]) => standard.has(id))) : entries]));
  return {
    models: perModel(file.models), parameterSettings: perModel(file.parameterSettings),
    descriptions: perModel(file.descriptions), criteria: perModel(file.criteria),
    customParameters: Object.fromEntries(Object.entries(file.customParameters).map(([model, entries]) => [model,
      Array.isArray(entries) ? entries.filter(entry => !known.has((entry as { id?: string })?.id ?? "")) : entries])),
  };
}

function validateModelId(value: string): void {
  if (typeof value !== "string" || !value || value.length > 200 || ["__proto__", "constructor", "prototype"].includes(value)) throw new Error("模型 ID 无效");
}
