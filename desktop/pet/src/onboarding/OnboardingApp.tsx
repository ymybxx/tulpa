import { t, changeLocale } from "../i18n";
import { useLocale } from "../i18n/preact";
import { messageText, type UiMessage } from "../i18n/message";
import { useCallback, useEffect, useRef, useState } from "preact/hooks";

import { permissionPlace, systemSettingsNames, type SystemSettingsNames } from "../shared/system-settings-names";
import type {
  DesktopConfigurationView,
  ModelPackage,
  ModelRenderBundle,
  OnboardingPreparation,
  PetEmotion,
  SystemPermissionKind,
  SystemPermissionState,
} from "../shared/contracts";
import {
  AddChip, Badge, Chip, Dialog, Field, Notice, OptionCard, OrnamentTitle, Panel, PillButton, Row, StepDots, TextButton, type Hue,
} from "../ui/components";
import { Icon, type IconName } from "../ui/icons";
import { SkyStage } from "../ui/SkyStage";
import { LanguagePicker } from "../ui/LanguagePicker";
import { CharacterStage } from "./CharacterStage";
import {
  AI_PROVIDERS, COMPANION_OPTIONS, MAX_TRAITS, PERMISSION_LABELS, RELATIONS, TRAITS,
  aiServiceRequest, describeSetting, onboardingConfiguration, onboardingSteps, pendingCharacter, readableError, toggleTrait, traitBlock, traitNote,
  type AiProvider, type CompanionLevel, type OnboardingStep,
} from "./flow";
import { BAILIAN_WORKSPACE_PATTERN } from "../shared/decision-model";

const STEP_TITLES: Record<OnboardingStep, { eyebrow: string; title: string; subtitle: string }> = {
  look: { eyebrow: "形 象", title: "选一个 TA 的样子", subtitle: "随时可以换。换形象不会改变 TA 的性格和记忆。" },
  soul: { eyebrow: "性 格", title: "TA 叫什么，什么性子？", subtitle: "点选标签，或者自己写。" },
  brain: { eyebrow: "A I 服 务", title: "给 TA 接上大脑", subtitle: "选一个服务，粘贴 Key 就好，其余的自动配好。" },
  companion: { eyebrow: "陪 伴 方 式", title: "TA 该怎么陪着你？", subtitle: "需要的权限会自动列出来，以后随时能改。" },
  permissions: { eyebrow: "权 限", title: "开启权限", subtitle: "一直点下面的按钮就行。系统设置里打开开关后，这里会自动打勾。" },
};

const TRAIT_LINES: Record<string, string> = {
  温柔: "今天也辛苦啦，要不要歇一会儿？", 元气: "嘿！今天也一起加油吧！", 毒舌: "又熬夜？你的黑眼圈比我还显眼。",
  傲娇: "才、才不是特意在等你呢。", 安静: "嗯，我在。", 话痨: "你知道吗你知道吗，我刚想到一件超好玩的事！",
  可靠: "交给我吧，我帮你记着。", 天然呆: "诶？我刚才要说什么来着……",
};
const TRAIT_EMOTIONS: Record<string, PetEmotion> = { 元气: "happy", 话痨: "happy", 温柔: "happy", 傲娇: "angry", 毒舌: "angry" };
const COMPANION_ICONS: Record<CompanionLevel, { icon: IconName; hue: Hue }> = {
  text: { icon: "chat", hue: "slate" }, voice: { icon: "mic", hue: "blue" }, full: { icon: "eye", hue: "amber" },
};
const PERMISSION_ICONS: Record<SystemPermissionKind, IconName> = { microphone: "mic", screen: "screen", systemAudio: "sound" };

type KeyState = { phase: "idle" } | { phase: "checking" } | { phase: "ok" } | { phase: "error"; message: string };

export function OnboardingApp({ initialStep = "look" }: { initialStep?: OnboardingStep } = {}) {
  const locale = useLocale();
  const api = window.desktopPet;
  const isWindows = api.platform === "win32";
  const [configuration, setConfiguration] = useState<DesktopConfigurationView | null>(null);
  const [preparation, setPreparation] = useState<OnboardingPreparation>({ phase: "preparing", percent: 0, detail: t("正在准备运行环境…") });
  const [preparationAttempt, setPreparationAttempt] = useState(0);
  const [models, setModels] = useState<ModelPackage[]>([]);
  const [modelId, setModelId] = useState<string | null>(null);
  const [bundle, setBundle] = useState<ModelRenderBundle | null>(null);
  const [importNote, setImportNote] = useState<UiMessage | null>(null);
  const [importedIds, setImportedIds] = useState<string[]>([]);
  const [previewFailed, setPreviewFailed] = useState<string[]>([]);

  const [name, setName] = useState("小星");
  const [relation, setRelation] = useState<string>("朋友");
  const [relations, setRelations] = useState<string[]>([...RELATIONS]);
  const [traits, setTraits] = useState<string[]>(["温柔", "元气"]);
  const [traitOptions, setTraitOptions] = useState<string[]>([...TRAITS]);
  const [customSetting, setCustomSetting] = useState("");
  const [showCustom, setShowCustom] = useState(false);

  const [provider, setProvider] = useState<AiProvider>(AI_PROVIDERS[0]!);
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [keyState, setKeyState] = useState<KeyState>({ phase: "idle" });
  const [keyAttempt, setKeyAttempt] = useState(0);
  const [showKey, setShowKey] = useState(false);
  const [keyHelp, setKeyHelp] = useState(false);

  const [level, setLevel] = useState<CompanionLevel>("full");
  const [jevApiKey, setJevApiKey] = useState("");
  const [workspace, setWorkspace] = useState("");
  const [jevState, setJevState] = useState<KeyState>({ phase: "idle" });
  const [jevAttempt, setJevAttempt] = useState(0);
  const [permissions, setPermissions] = useState<Record<SystemPermissionKind, SystemPermissionState> | null>(null);
  const [skipped, setSkipped] = useState<SystemPermissionKind[]>([]);
  const [audioSupported, setAudioSupported] = useState(false);
  const [settingsNames, setSettingsNames] = useState<SystemSettingsNames>(() => systemSettingsNames(24));

  const [step, setStep] = useState<OnboardingStep>(initialStep);
  const [draftReady, setDraftReady] = useState(false);
  const [finishing, setFinishing] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [changingLanguage, setChangingLanguage] = useState(false);
  const selectLanguage = async (next: "zh-CN" | "en") => {
    setChangingLanguage(true);
    try { await changeLocale(next); }
    catch (error) { setFailure(readableError(error)); }
    finally { setChangingLanguage(false); }
  };

  // ---------- draft: survive "quit and reopen" ----------
  useEffect(() => {
    void api.onboarding.loadDraft().then((draft) => {
      if (!draft) return;
      // A draft past the AI step without a key (an older or hand-edited file) resumes at that step.
      const resumeAt: OnboardingStep = draft.apiKey || !["companion", "permissions"].includes(draft.step) ? draft.step : "brain";
      if (initialStep === "look") setStep(resumeAt);
      if (draft.modelId) setModelId(draft.modelId);
      if (draft.name) setName(draft.name);
      if (draft.relation) setRelation(draft.relation);
      if (draft.relations.length) setRelations(draft.relations);
      setTraits(draft.traits);
      if (draft.traitOptions.length) setTraitOptions(draft.traitOptions);
      setCustomSetting(draft.customSetting);
      if (draft.customSetting) setShowCustom(true);
      setProvider(AI_PROVIDERS.find((item) => item.id === draft.provider) ?? AI_PROVIDERS[0]!);
      setBaseUrl(draft.baseUrl);
      setModel(draft.model);
      setApiKey(draft.apiKey);
      setLevel(draft.level);
      setJevApiKey(draft.jevApiKey);
      setWorkspace(draft.bailianWorkspaceId ?? "");
    }).catch(() => undefined).finally(() => setDraftReady(true));
  }, []);

  useEffect(() => {
    if (!draftReady || finishing) return;
    const timer = window.setTimeout(() => {
      void api.onboarding.saveDraft({
        step, modelId, name, relation, relations, traits, traitOptions, customSetting,
        provider: provider.id, baseUrl, model, apiKey, level, jevApiKey, bailianWorkspaceId: workspace,
      }).catch(() => undefined);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [draftReady, finishing, step, modelId, name, relation, relations, traits, traitOptions, customSetting, provider, baseUrl, model, apiKey, level, jevApiKey, workspace]);

  // ---------- loading ----------
  useEffect(() => {
    void api.getConfiguration().then(setConfiguration).catch((error) => setFailure(String(error)));
    void api.getPerceptionStatus().then((status) => {
      setAudioSupported(Boolean(status.audioSupported));
      if (status.systemSettings) setSettingsNames(status.systemSettings);
    }).catch(() => undefined);
    void api.onboarding.permissionStatus().then(setPermissions).catch(() => undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const update = (value: OnboardingPreparation) => { if (!cancelled) setPreparation(value); };
    const stop = api.onboarding.onPreparation(update);
    void api.onboarding.prepare().then(update).catch((error) => {
      update({ phase: "failed", percent: null, detail: readableError(error) });
    });
    return () => { cancelled = true; stop(); };
  }, [preparationAttempt]);

  const modelRefresh = useRef(0);
  const refreshModels = useCallback(async (prefer?: string) => {
    const attempt = ++modelRefresh.current;
    const list = await api.listModels();
    const selected = await api.getSelectedModel().catch(() => null);
    // A preparation retry must not replace a more recent manual import.
    if (attempt !== modelRefresh.current) return;
    setModels(list);
    setModelId((current) => {
      const wanted = prefer ?? current;
      if (wanted && list.some((item) => item.id === wanted)) return wanted;
      return selected?.id ?? list[0]?.id ?? null;
    });
  }, []);

  useEffect(() => {
    if (preparation.phase === "ready") void refreshModels();
  }, [preparation.phase]);

  useEffect(() => {
    if (!modelId) return;
    let cancelled = false;
    void api.getModelRenderBundle(modelId).then((value) => { if (!cancelled) setBundle(value); });
    return () => { cancelled = true; };
  }, [modelId]);

  // ---------- AI service check ----------
  const request = aiServiceRequest({ provider, baseUrl, model, apiKey });
  const checkable = request.apiKey.length >= 8 && Boolean(request.baseUrl) && Boolean(request.model);
  useEffect(() => {
    if (!checkable) { setKeyState({ phase: "idle" }); return; }
    setKeyState({ phase: "checking" });
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void api.onboarding.checkAiService(request).then((result) => {
        if (cancelled) return;
        setKeyState(result.ok ? { phase: "ok" } : { phase: "error", message: result.message });
        if (result.ok) setFailure(null);
      }).catch((error) => { if (!cancelled) setKeyState({ phase: "error", message: readableError(error) }); });
    }, 700);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [request.baseUrl, request.model, request.apiKey, keyAttempt]);

  // ---------- decision model (optional, judges 主动陪伴 better) ----------
  // 千问's key is a 百炼 key, so only the 业务空间 ID is asked; other services offer Jev.
  const bailianDecision = provider.id === "qwen";
  const decisionValue = bailianDecision ? workspace.trim() : jevApiKey.trim();
  const jevCheckable = level === "full" && (bailianDecision
    ? BAILIAN_WORKSPACE_PATTERN.test(decisionValue) && apiKey.trim().length > 0 : decisionValue.length >= 8);
  useEffect(() => {
    if (!jevCheckable) { setJevState({ phase: "idle" }); return; }
    setJevState({ phase: "checking" });
    let cancelled = false;
    const timer = window.setTimeout(() => {
      const check = bailianDecision ? api.onboarding.checkBailianDecision(decisionValue, apiKey.trim()) : api.onboarding.checkJev(decisionValue);
      void check.then((result) => {
        if (!cancelled) setJevState(result.ok ? { phase: "ok" } : { phase: "error", message: result.message });
      }).catch((error) => { if (!cancelled) setJevState({ phase: "error", message: readableError(error) }); });
    }, 700);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [jevCheckable, bailianDecision, decisionValue, apiKey, jevAttempt]);

  // ---------- permissions ----------
  const needed = COMPANION_OPTIONS.find((option) => option.id === level)!.permissions
    .filter((kind) => permissions?.[kind] !== "not-required");
  const missing = needed.filter((kind) => permissions?.[kind] !== "granted" && !skipped.includes(kind));
  const [waiting, setWaiting] = useState<SystemPermissionKind | null>(null);
  const [recheckNote, setRecheckNote] = useState<UiMessage | null>(null);
  const [confirming, setConfirming] = useState<SystemPermissionKind | null>(null);
  // After Screen Recording is on, macOS still asks once more on the first real capture.
  const [warmUp, setWarmUp] = useState<"idle" | "asking" | "running" | "done">("idle");
  const [settingsName, setSettingsName] = useState("tulpa");
  useEffect(() => { void api.onboarding.settingsAppName().then(setSettingsName).catch(() => undefined); }, []);
  // Poll while the user is (or may be) in System Settings; the heads-up closes itself once granted.
  const watched = waiting ?? confirming;
  useEffect(() => {
    if (!watched) return;
    const poll = () => void api.onboarding.permissionStatus().then(setPermissions);
    const timer = window.setInterval(poll, 1500);
    window.addEventListener("focus", poll);
    return () => { window.clearInterval(timer); window.removeEventListener("focus", poll); };
  }, [watched]);
  useEffect(() => {
    if (waiting && permissions?.[waiting] === "granted") setWaiting(null);
    if (confirming && permissions?.[confirming] === "granted") setConfirming(null);
  }, [permissions, waiting, confirming]);

  const grant = async (kind: SystemPermissionKind) => {
    setRecheckNote(null);
    const state = await api.onboarding.requestPermission(kind);
    setPermissions((current) => current && { ...current, [kind]: state });
    setWaiting(state === "granted" ? null : kind);
  };

  /**
   * Permissions that must be switched on in System Settings get a heads-up first:
   * macOS then offers "quit and reopen", which the user should decline. The app
   * re-checks from a fresh helper process, so no restart is needed.
   */
  const needsSettings = (kind: SystemPermissionKind) => kind === "screen" || permissions?.[kind] === "denied";
  const startGrant = (kind: SystemPermissionKind) => {
    if (needsSettings(kind) && !(isWindows && kind !== "microphone")) setConfirming(kind);
    else void grant(kind);
  };

  /** The user says they switched it on: check right away instead of waiting for the poll. */
  const recheck = async () => {
    if (!waiting) return;
    const value = await api.onboarding.permissionStatus();
    setPermissions(value);
    if (value[waiting] === "granted") {
      setWaiting(null);
      setRecheckNote(null);
    } else {
      setRecheckNote(isWindows ? { key: "请确认已允许桌面应用访问麦克风，然后回到这里重试。" } : { key: "还没检测到开关打开。确认打开的是「{{p0}}」那一项；列表里没有的话，按下面的步骤添加。", values: { p0: settingsName } });
    }
  };

  // ---------- flow ----------
  const steps = onboardingSteps(level, needed.length > 0);
  const index = steps.indexOf(step);

  const finish = async () => {
    if (!configuration) return;
    setFailure(null);
    if (!apiKey.trim()) {
      setStep("brain");
      setFailure("还差一步：填好 API Key 就能完成");
      return;
    }
    setFinishing(true);
    try {
      await api.onboarding.saveCharacter(pendingCharacter({ name, relation, traits, customSetting }, modelId));
      if (modelId) await api.selectModel(modelId);
      await api.saveConfiguration(onboardingConfiguration(configuration, {
        service: { provider, baseUrl, model, apiKey }, level, name, audioSupported, jevApiKey, bailianWorkspaceId: workspace,
      }));
    } catch (error) {
      setFailure(readableError(error));
      setFinishing(false);
    }
  };

  const next = () => {
    if (step === "permissions") {
      if (waiting) { void recheck(); return; }
      const pending = missing[0];
      if (pending) startGrant(pending);
      else if (!isWindows && needed.includes("screen") && permissions?.screen === "granted" && warmUp === "idle") setWarmUp("asking");
      else void finish();
      return;
    }
    const following = steps[index + 1];
    if (following) setStep(following);
    else void finish();
  };

  const canContinue = (() => {
    if (finishing || warmUp === "running") return false;
    if (step === "look") return models.some((item) => item.id === modelId);
    if (step === "soul") return name.trim().length > 0;
    if (step === "brain") return keyState.phase === "ok";
    // The decision model is optional, but what was typed in must check out before it is saved.
    if (step === "companion") return level !== "full" || !decisionValue || jevState.phase === "ok";
    if (step === "permissions") return true;
    return true;
  })();

  const nextLabel = (() => {
    if (warmUp === "running") return t("等你点「允许」…");
    if (finishing) return t("正在准备…");
    if (step === "look") return t("就选 TA");
    if (step === "permissions") {
      if (waiting) return t("我已经打开了");
      const pending = missing[0];
      return pending ? t("开启{{p0}}", { p0: t(PERMISSION_LABELS[pending].title) }) : t("完成");
    }
    return steps[index + 1] ? t("下一步") : t("完成");
  })();

  // ---------- character voice ----------
  const lastTrait = traits[traits.length - 1];
  const displayName = name.trim() || "TA";
  const line = (() => {
    if (step === "look") return models.length > 1 ? t("嗨，是我！先帮我挑个样子吧。") : t("嗨，是我！这就是我现在的样子～");
    if (step === "soul") return (lastTrait && TRAIT_LINES[lastTrait]) || (lastTrait ? t("好，我就照「{{p0}}」的样子跟你说话。", { p0: t(lastTrait) }) : t("很高兴认识你。"));
    if (step === "brain") return keyState.phase === "ok" ? t("脑子转起来了！") : t("接上大脑，我才能好好陪你聊天。");
    if (step === "companion") {
      return level === "full" ? t("我能看懂你在做什么，也猜得到你的心情～")
        : level === "voice" ? t("叫一声「{{p0}}」，我就来。", { p0: displayName }) : t("你想聊的时候，点我就好。");
    }
    if (waiting) return t("打开系统设置里的那个开关就好，我等你～");
    if (warmUp === "running") return t("系统在问你呢，点「允许」就好～");
    return missing.length === 0 ? t("谢谢你！这下我能看见、也能听见了。点「完成」，我们就开始吧！") : t("授权以后，我就能看见、听见你身边的事了。");
  })();
  const emotion: PetEmotion = step === "soul" && lastTrait ? TRAIT_EMOTIONS[lastTrait] ?? "neutral"
    : step === "brain" && keyState.phase === "ok" ? "happy"
    : step === "permissions" && missing.length === 0 && !waiting ? "happy" : "neutral";

  const title = STEP_TITLES[step];
  const traitHint = traitNote(traits, traitOptions);
  const prepText = preparation.phase === "ready" ? ""
    : preparation.phase === "failed" ? t("没准备好：{{p0}}", { p0: preparation.detail })
    : t("正在做准备{{p0}}", { p0: preparation.percent === null ? "…" : ` ${preparation.percent}%` });

  if (!draftReady) return <div class="onb"><SkyStage /></div>;

  return (
    <div class="onb">
      <SkyStage />
      <div class="onb-stage">
        <header class="onb-stage__header">
          <div class="ui-scene-title">
            <span class="ui-scene-title__badge"><Icon name="sun" size={18} weight={1.8} /></span>
            <span class="ui-scene-title__text"><strong>{t("初次相遇")}</strong><small>FIRST MEETING</small></span>
          </div>
          <div class="onb-language"><LanguagePicker locale={locale} busy={changingLanguage} onChange={(next) => void selectLanguage(next)} /></div>
          <div class="ui-scene-meta">
            <StepDots total={steps.length} current={index} />
            <span>{index + 1} / {steps.length} · {locale === "zh-CN" ? title.eyebrow.replace(/ /gu, "") : t(title.eyebrow)}</span>
          </div>
        </header>
        <CharacterStage
          bundle={bundle}
          emotion={emotion}
          ready={preparation.phase === "ready"}
          fallback={preparation.phase === "ready" ? t("正在请 TA 出来…") : t("正在准备形象…")}
          onSnapshot={(id, png) => {
            void api.saveModelPreview(id, png).then((saved) => {
              setModels((list) => list.map((item) => item.id === saved.id ? saved : item));
            }).catch(() => setPreviewFailed((ids) => [...ids, id]));
          }}
        />
        <div class="ui-caption onb-stage__caption" aria-live="polite">
          <span class="ui-caption__name">{displayName}</span>
          <span class="ui-caption__rule" aria-hidden="true" />
          <span class="ui-caption__line">{t(line)}</span>
        </div>
      </div>

      <Panel
        class="onb-panel"
        footer={(
          <>
            <span class={`onb-panel__status${failure || preparation.phase === "failed" ? " onb-panel__status--warn" : ""}`} role={failure || preparation.phase === "failed" ? "alert" : undefined}>
              {failure ? t(failure) : prepText}
            </span>
            {preparation.phase === "failed" && (
              <TextButton disabled={finishing} onClick={() => {
                setPreparation({ phase: "preparing", percent: 0, detail: t("正在准备运行环境…") });
                setPreparationAttempt((attempt) => attempt + 1);
              }}>{t("重新准备")}</TextButton>
            )}
            {step === "permissions" && missing.length > 0 && (
              <TextButton onClick={() => { setSkipped(missing); void finish(); }}>{t("跳过剩下的")}</TextButton>
            )}
            {index > 0 && (
              <PillButton variant="secondary" icon="back" onClick={() => setStep(steps[index - 1]!)} disabled={finishing}>{t("返回")}</PillButton>
            )}
            <PillButton icon="check" onClick={next} disabled={!canContinue}>{nextLabel}</PillButton>
          </>
        )}
      >
        <OrnamentTitle>{t(title.eyebrow)}</OrnamentTitle>
        <h1 class="onb-panel__title">{step === "look" && models.length === 1 ? t("这是 TA 的样子")
          : step !== "permissions" ? t(title.title) : missing.length === 0 && !waiting ? t("全部就绪") : t("开启 {{p0}} 项权限", { p0: needed.length })}</h1>
        <p class="onb-panel__subtitle">{step === "look" && models.length === 1 ? t("先用这个，以后随时能换，换了也不会改变 TA 的性格和记忆。")
          : step === "permissions" && missing.length === 0 && !waiting ? t("需要的权限都开好了，点「完成」就行。") : t(title.subtitle)}</p>

        {step === "look" && (
          <div class="onb-section">
            <div class={`onb-models${models.length === 1 ? " onb-models--single" : ""}`} role="radiogroup" aria-label={t("形象")}>
              {models.map((item) => (
                <button type="button" role="radio" aria-checked={item.id === modelId} class="onb-model" onClick={() => setModelId(item.id)}>
                  <span class="onb-model__art">
                    {item.previewUrl ? <img src={item.previewUrl} alt="" />
                      : previewFailed.includes(item.id) ? <Icon name="person" size={40} weight={1.4} />
                      : <span class="onb-model__loading">{t("生成预览中…")}</span>}
                  </span>
                  <span class="onb-model__name">{item.name}</span>
                  <span class="onb-model__meta">{importedIds.includes(item.id) ? t("你导入的形象") : t("内置形象")}</span>
                </button>
              ))}
              {models.length === 0 && <p class="onb-empty">{preparation.phase === "failed" ? t("默认形象没能准备好，可以先导入自己的 Live2D 模型。") : t("正在准备默认形象…")}</p>}
            </div>
            <div class="onb-import-box">
              <span class="onb-import-box__title">{t("想用自己的形象？")}<small>{t("能导入网上下载的 Live2D 角色（一种会动的二次元形象），没有的话跳过就好")}</small></span>
              <span class="onb-import-box__actions">
                {(["folder", "zip"] as const).map((kind) => (
                  <PillButton variant="secondary" size="sm" icon="upload" onClick={async () => {
                    const result = await api.importModel(kind);
                    if (!result) return;
                    if (!result.model) {
                      setImportNote(result.report.errors[0]?.message ?? "这个模型没法导入");
                      return;
                    }
                    setImportedIds((ids) => [...ids, result.model!.id]);
                    setImportNote({ key: "已导入「{{p0}}」。表情会在接上 AI 服务后自动匹配。", values: { p0: result.model.name } });
                    await refreshModels(result.model.id);
                  }}>{kind === "folder" ? t("选择文件夹") : t("选择压缩包")}</PillButton>
                ))}
              </span>
              {importNote && <span class="onb-import-box__note">{messageText(importNote)}</span>}
            </div>
          </div>
        )}

        {step === "soul" && (
          <div class="onb-section">
            <Field label={t("名字")} hint={t("也是唤醒词：喊「{{p0}}」就能叫醒 TA", { p0: displayName })}>
              <input class="ui-input" value={name} maxLength={16} onInput={(event) => setName(event.currentTarget.value)} />
            </Field>
            <div class="onb-group">
              <span class="ui-field__label">{t("你们的关系")}</span>
              <div class="ui-chip-row">
                {relations.map((value) => <Chip selected={value === relation} onClick={() => setRelation(value)}>{RELATIONS.some((item) => item === value) ? t(value) : value}</Chip>)}
                <AddChip label={t("自定义关系")} placeholder={t("比如：青梅竹马")} onAdd={(value) => {
                  setRelations((list) => list.includes(value) ? list : [...list, value]);
                  setRelation(value);
                }} />
              </div>
            </div>
            <div class="onb-group">
              <span class="ui-field__label">{t("性格")}<small>{traitHint}</small></span>
              <div class="ui-chip-row">
                {traitOptions.map((value) => {
                  const blocked = traitBlock(traits, value);
                  return (
                    <Chip selected={traits.includes(value)} disabled={Boolean(blocked)} title={blocked ?? undefined}
                      onClick={() => setTraits((list) => toggleTrait(list, value))}>{TRAITS.some((item) => item === value) ? t(value) : value}</Chip>
                  );
                })}
                <AddChip label={t("自定义性格")} placeholder={t("比如：爱吐槽")} onAdd={(value) => {
                  setTraitOptions((list) => list.includes(value) ? list : [...list, value]);
                  setTraits((list) => list.includes(value) ? list : toggleTrait(list, value));
                }} />
              </div>
            </div>
            <TextButton aria-expanded={showCustom} onClick={() => {
              if (!showCustom && !customSetting) setCustomSetting(describeSetting({ traits, customSetting: "" }));
              setShowCustom(!showCustom);
            }}>{showCustom ? t("收起设定") : t("想更细？写一段 TA 的设定")}</TextButton>
            {showCustom && (
              <textarea class="ui-input" aria-label={t("角色设定")} rows={3} maxLength={200} value={customSetting}
                onInput={(event) => setCustomSetting(event.currentTarget.value)} />
            )}
          </div>
        )}

        {step === "brain" && (
          <div class="onb-section">
            <div class="onb-providers" role="radiogroup" aria-label={t("AI 服务")}>
              {AI_PROVIDERS.map((item) => (
                <button type="button" role="radio" aria-checked={item.id === provider.id} class="onb-provider" onClick={() => setProvider(item)}>
                  <strong>{t(item.label)}</strong><small>{t(item.hint)}</small>
                  {item.recommended && <span class="ui-option__flag">{t("推荐")}</span>}
                </button>
              ))}
            </div>
            {provider.id === "custom" && (
              <div class="onb-two">
                <Field label={t("接口地址")}><input class="ui-input" placeholder="https://…/v1" value={baseUrl} onInput={(event) => setBaseUrl(event.currentTarget.value)} /></Field>
                <Field label={t("模型名称")}><input class="ui-input" placeholder={t("例如 gpt-4o-mini")} value={model} onInput={(event) => setModel(event.currentTarget.value)} /></Field>
              </div>
            )}
            <Field
              label="API Key"
              hint={keyState.phase === "error"
                ? <Notice action={<TextButton disabled={!checkable} onClick={() => {
                  setKeyState({ phase: "checking" });
                  setKeyAttempt((attempt) => attempt + 1);
                }}>{t("重新验证")}</TextButton>}>{t(keyState.message)}。{keyAdvice(provider, keyState.message)}</Notice>
                : keyState.phase === "checking" ? t("正在验证…")
                : keyState.phase === "ok" ? <span class="onb-ok"><Icon name="check" size={12} weight={2.6} />{t("连上了，TA 可以开始思考了")}</span>
                : undefined}
            >
              <span class="onb-key">
                <input class={`ui-input${keyState.phase === "ok" ? " ui-input--ok" : ""}`} type={showKey ? "text" : "password"} autocomplete="off"
                  spellcheck={false} placeholder={t("粘贴到这里")} value={apiKey} onInput={(event) => setApiKey(event.currentTarget.value)} />
                <button type="button" class="onb-key__toggle" onClick={() => setShowKey(!showKey)}>{showKey ? t("隐藏") : t("显示")}</button>
              </span>
            </Field>
            <div class="onb-key-meta">
              <TextButton aria-expanded={keyHelp} onClick={() => setKeyHelp(!keyHelp)}>{t("什么是 API Key，怎么拿到？")}</TextButton>
              <span><Icon name="lock" size={12} /> {t("只保存在这台电脑上")}</span>
            </div>
            {keyHelp && <KeyHelp provider={provider} />}
          </div>
        )}

        {step === "companion" && (
          <div class="onb-section onb-options" role="radiogroup" aria-label={t("陪伴方式")}>
            {COMPANION_OPTIONS.map((option) => (
              <OptionCard
                selected={option.id === level}
                icon={COMPANION_ICONS[option.id].icon}
                hue={COMPANION_ICONS[option.id].hue}
                title={t(option.title)}
                description={t(option.description)}
                tags={option.permissions.length ? option.permissions.map((kind) => t(PERMISSION_LABELS[kind].title)) : [t("无需权限")]}
                recommended={option.recommended}
                onSelect={() => setLevel(option.id)}
              />
            ))}
            {level === "full" && (
              <p class="onb-note"><Icon name="lock" size={12} /> {t("看屏幕只是为了懂你在做什么：画面交给你接的 AI 服务去理解，TA 记住的是文字小结。随时能在 设置 › 陪伴 里关掉。")}</p>
            )}
            {level === "full" && (
              <p class="onb-note"><Icon name="bolt" size={12} /> {t("主动陪伴会一直看屏幕、在后台想事情，对话和推理模型大约每小时消耗 100 万 token 以上。想省一点可以选「语音聊天」或「打字聊天」，随时能在 设置 › 陪伴 里改。")}</p>
            )}
            {level === "full" && (
              <div class="onb-jev">
                <Field
                  label={bailianDecision ? t("百炼业务空间 ID（可选，没有就跳过）") : t("决策模型 Jev 的 API Key（可选，没有就跳过）")}
                  hint={<>
                    {jevState.phase === "error" ? <Notice action={<TextButton disabled={!jevCheckable} onClick={() => {
                      setJevState({ phase: "checking" });
                      setJevAttempt((attempt) => attempt + 1);
                    }}>{t("重新验证")}</TextButton>}>{t(jevState.message)}</Notice>
                      : jevState.phase === "ok" ? <span class="onb-ok"><Icon name="check" size={12} weight={2.6} />{t("连上了")}</span>
                      : jevState.phase === "checking" ? t("正在验证…") : null}
                    {bailianDecision ? (
                      <span class="onb-jev__about">{t("决策模型专门帮 TA 拿主意：什么时候该开口、说话配什么表情。它用上一步的百炼 Key，只要再填业务空间 ID，不过目前判断不如 Jev 准。不填也能主动陪伴，由上一步的 AI 服务来判断；以后也能在 设置 › 决策模型 里加。业务空间 ID 在")}<a href="https://bailian.console.aliyun.com/cn-beijing/settings/workspace" target="_blank" rel="noopener">{t("百炼的业务空间页面")}</a>{t("能看到。")}</span>
                    ) : (
                      <span class="onb-jev__about">{t("决策模型专门帮 TA 拿主意：什么时候该开口、说话配什么表情。不填也能主动陪伴，由上一步的 AI 服务来判断；以后也能在 设置 › 决策模型 里加，也可以用百炼的。想用 Jev 的话：在")}<a href="https://typesafe.ai/" target="_blank" rel="noopener">typesafe.ai</a> {t("注册登录，到")}<a href="https://console.typesafe.ai/keys" target="_blank" rel="noopener">{t("控制台的 Keys 页面")}</a> {t("创建一个。")}</span>
                    )}
                  </>}
                >
                  {bailianDecision ? (
                    <input class={`ui-input${jevState.phase === "ok" ? " ui-input--ok" : ""}`} type="text" autocomplete="off"
                      spellcheck={false} placeholder={t("例如 ws-xxxx 或 llm-xxxx")} value={workspace} onInput={(event) => setWorkspace(event.currentTarget.value)} />
                  ) : (
                    <span class="onb-key">
                      <input class={`ui-input${jevState.phase === "ok" ? " ui-input--ok" : ""}`} type={showKey ? "text" : "password"} autocomplete="off"
                        spellcheck={false} placeholder={t("有的话粘贴到这里")} value={jevApiKey} onInput={(event) => setJevApiKey(event.currentTarget.value)} />
                      <button type="button" class="onb-key__toggle" onClick={() => setShowKey(!showKey)}>{showKey ? t("隐藏") : t("显示")}</button>
                    </span>
                  )}
                </Field>
              </div>
            )}
          </div>
        )}

        {step === "permissions" && (
          <div class="onb-section">
            <div class="ui-card">
              {needed.map((kind) => {
                const state = permissions?.[kind];
                const granted = state === "granted";
                return (
                  <Row
                    icon={PERMISSION_ICONS[kind]}
                    hue={granted ? "green" : waiting === kind ? "blue" : "amber"}
                    title={t(PERMISSION_LABELS[kind].title)}
                    subtitle={waiting === kind
                      ? isWindows ? t("请确认已允许桌面应用访问麦克风，然后回到这里重试。") : t("已打开系统设置，找到「{{p0}}」打开开关", { p0: settingsName })
                      : t(PERMISSION_LABELS[kind].why)}
                    highlighted={waiting === kind}
                    status={granted ? <Badge tone="ok">{t("已开启")}</Badge>
                      : waiting === kind ? <Badge tone="info">{t("等你打开开关")}</Badge>
                      : skipped.includes(kind) ? <Badge tone="muted">{t("已跳过")}</Badge>
                      : <Badge tone="warn">{t("未开启")}</Badge>}
                    action={waiting === kind
                      ? <PillButton size="sm" variant="secondary" icon="external" onClick={() => void grant(kind)}>{t("重新打开设置")}</PillButton>
                      : null}
                  />
                );
              })}
            </div>
            {recheckNote && <p class="onb-note onb-note--warn" role="status">{messageText(recheckNote)}</p>}
            {!isWindows && waiting === "screen" && (
              <div class="onb-help">
                <span class="onb-help__lead">{t("打开开关后，系统会问要不要「退出并重新打开」：")}<b>{t("选「以后」就行，不用重启")}</b>{t("。万一选了重启，重新打开后也会回到这一步。")}</span>
                <b>{t("系统设置里找不到「{{name}}」？", { name: settingsName })}</b>
                <ol>
                  <li>{t("点列表下方的")}<b>+</b>{t("（可能要先输入电脑密码）")}</li>
                  <li>{t("在「应用程序」里选中")}<b>{settingsName}</b>{t("，点「打开」")}</li>
                  <li>{t("打开它旁边的开关，回到这里会自动打勾")}</li>
                </ol>
                <span>{t("开关打开了还是没反应？先选中「{{name}}」点 − 删掉，再重新添加。", { name: settingsName })}</span>
              </div>
            )}
            {missing.length > 0 && <p class="onb-note">{t("没开的权限随时能补：在 设置 › 权限 里再开。")}</p>}
          </div>
        )}
      </Panel>

      {!isWindows && warmUp === "asking" && (
        <Dialog
          title={t("系统还会再确认一下")}
          confirmLabel={t("好的")}
          onCancel={() => setWarmUp("idle")}
          onConfirm={() => {
            setWarmUp("running");
            void api.onboarding.warmUpCapture()
              .catch(() => null)
              .finally(() => { setWarmUp("done"); void finish(); });
          }}
        >
          <span>{t("{{name}} 第一次看屏幕时，macOS 会弹出确认框，问能不能直接访问屏幕。", { name: settingsName })}</span>
          <span><strong>{t("都点「允许」就行")}</strong>{t("，这是正常的一步，不是出错了。")}</span>
        </Dialog>
      )}

      {confirming && (
        <Dialog
          title={t("马上去系统设置")}
          confirmLabel={t("知道了，去打开")}
          onCancel={() => setConfirming(null)}
          onConfirm={() => { const kind = confirming; setConfirming(null); void grant(kind); }}
        >
          <span>{isWindows ? t("在 Windows 设置的「隐私和安全性 › 麦克风」中，打开「麦克风访问权限」和「允许桌面应用访问麦克风」。") : t("在「{{place}}」里打开「{{name}}」的开关。", { place: permissionPlace(confirming, settingsNames, locale), name: settingsName })}</span>
          {!isWindows && confirming === "screen" && <span>{t("系统接着会问要不要「退出并重新打开」——")}<strong>{t("选「以后」就行，不用重启")}</strong>{t("，回到这里会自动打勾。")}</span>}
        </Dialog>
      )}
    </div>
  );
}

/** Where the key comes from, in words a first-time user can follow. */
function KeyHelp({ provider }: { provider: (typeof AI_PROVIDERS)[number] }) {
  if (!provider.keyUrl) {
    return (
      <div class="onb-help">
        <span class="onb-help__lead">{t("API Key 是 AI 服务发给你的一串密码，TA 靠它去调用那边的模型。填上服务给你的接口地址、模型名称和 Key 就行。")}</span>
      </div>
    );
  }
  return (
    <div class="onb-help">
      <span class="onb-help__lead">{t("API Key 是 AI 服务发给你的一串密码，TA 靠它去「{{provider}}」那里思考。按用量付费，新账号一般有免费额度，日常聊天花不了多少。", { provider: t(provider.label) })}</span>
      <ol>
        <li>{t("打开")}<a href={provider.keyUrl} target="_blank" rel="noopener">{provider.keySite ? t(provider.keySite) : ""}</a>{t("，用手机号注册登录（可能要实名认证）")}</li>
        <li>{t("找到「API Key」页面，点「创建 API Key」")}</li>
        {provider.activate && <li>{t(provider.activate)}</li>}
        <li>{t("复制那一串字符，回来粘贴到上面的框里")}</li>
      </ol>
    </div>
  );
}

/** The next thing to try after a failed check. */
function keyAdvice(provider: (typeof AI_PROVIDERS)[number], message: string): string {
  const site = provider.keySite ? t(provider.keySite) : null;
  if (!site) return t("看看地址、模型名称和 Key 是不是都填对了");
  if (message.startsWith("Key 不对")) return t("看看是不是没复制全，或者多了空格；也可以回{{p0}}重新创建一个", { p0: site });
  if (message.startsWith("找不到")) return t("这个模型可能还没开通，到{{p0}}里开通后再试", { p0: site });
  if (message.includes("额度")) return t("到{{p0}}看看余额，或者稍等一会儿", { p0: site });
  return t("稍后再试一次");
}
