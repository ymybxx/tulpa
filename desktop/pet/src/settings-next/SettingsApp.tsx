import { changeLocale, getLocale, t } from "../i18n";
import { useLocale } from "../i18n/preact";
import type { DesktopUiLocale } from "../shared/contracts";
import type { ComponentChildren } from "preact";
import { permissionPlace, systemSettingsNames, type SystemSettingsNames } from "../shared/system-settings-names";
import { openCharacter, recall, remember } from "../ui/remembered";
import { useEffect, useRef, useState } from "preact/hooks";

import type {
  DesktopConfiguration,
  DesktopConfigurationInput,
  DesktopConfigurationView,
  DesktopRuntimeStatus,
  DesktopTextModel,
  GatewayStatus,
  ModelImageInput,
  ModelImageInputStatus,
  ModelReadiness,
  OwnerVoiceEnrollmentState,
  OwnerVoiceStatus,
  SystemPermissionKind,
  SystemPermissionState,
} from "../shared/contracts";
import { AI_PROVIDERS, aiServiceRequest, readableError, type AiProvider, type AiServiceChoice } from "../onboarding/flow";
import { BackButton, Badge, Dialog, Field, Notice, PillButton, Segmented, Switch, TextButton } from "../ui/components";
import { Icon, type IconName } from "../ui/icons";
import { MainNav } from "../ui/MainNav";
import { LanguagePicker } from "../ui/LanguagePicker";
import { SkyStage } from "../ui/SkyStage";
import { scenePreference, setScenePreference, type ScenePreference } from "../ui/scene";
import { bailianKeyModelId, bailianWorkspaceId, chatPickEnabled, decisionProvider, decisionReady, type DecisionProvider } from "../shared/decision-model";
import { LlmLogPanel } from "./log-panel";
import { ownerVoiceControls } from "./owner-voice-state";
import {
  ADVANCED_DEFAULTS,
  SETTINGS_SECTIONS,
  TEXTURE_OPTIONS,
  blockedFeatures,
  configuredModels,
  conversationService,
  modelLabel,
  withAssignedModel,
  withChanges,
  withRemovedModel,
  withSavedModel,
  type SettingsSection,
} from "./model";
import { webSearchSupported, WEB_SEARCH_SERVICES } from "../shared/web-search";
import { tokenText, USAGE_GROUPS, type UsageGroup, type UsageSummary, type UsageTotals } from "../shared/usage";

type Permissions = Record<SystemPermissionKind, SystemPermissionState>;
type Check = { phase: "idle" | "checking" | "ok" } | { phase: "error"; message: string };

const SECTION_ICONS: Record<SettingsSection, { icon: IconName; hue: string }> = {
  ai: { icon: "spark", hue: "blue" },
  jev: { icon: "bolt", hue: "amber" },
  companion: { icon: "heart", hue: "rose" },
  usage: { icon: "chart", hue: "amber" },
  extensions: { icon: "check", hue: "green" },
  permissions: { icon: "shield", hue: "slate" },
  general: { icon: "gear", hue: "slate" },
  advanced: { icon: "sliders", hue: "slate" },
};

const PERMISSION_INFO: Record<SystemPermissionKind, { title: string; usedBy: string; icon: IconName }> = {
  screen: { title: "屏幕与系统录音", usedBy: "屏幕感知、电脑音频感知", icon: "screen" },
  systemAudio: { title: "电脑声音", usedBy: "电脑音频感知", icon: "sound" },
  microphone: { title: "麦克风", usedBy: "麦克风收音、录入声纹", icon: "mic" },
};

const PERMISSION_ORDER: readonly SystemPermissionKind[] = ["screen", "systemAudio", "microphone"];

const usable = (state: SystemPermissionState | undefined): boolean => state === "granted" || state === "not-required";

const SECTION_IDS: readonly SettingsSection[] = ["ai", "jev", "companion", "usage", "extensions", "permissions", "general", "advanced"];

export function SettingsApp() {
  const locale = useLocale();
  const api = window.desktopPet;
  const isWindows = api.platform === "win32";
  const [view, setView] = useState<DesktopConfigurationView | null>(null);
  const [section, showSection] = useState<SettingsSection>(() => recall("settings-section", SECTION_IDS, "companion"));
  // Closing the window drops the page; reopening it returns to the logs if that is where the user was.
  const [logsOpen, showLogs] = useState(() => recall("settings-logs", ["open", "closed"], "closed") === "open");
  const setLogsOpen = (open: boolean) => { remember("settings-logs", open ? "open" : "closed"); showLogs(open); };
  const setSection = (next: SettingsSection) => { remember("settings-section", next); showSection(next); setLogsOpen(false); };
  useEffect(() => {
    // Opened to fix something (a key, a quota): go straight to that section.
    const take = () => void api.takeSettingsSection().then((next) => { if (next) setSection(next); });
    take();
    return api.onSettingsSection(take);
  }, []);
  const [permissions, setPermissions] = useState<Permissions | null>(null);
  const [audioSupported, setAudioSupported] = useState(true);
  const [settingsNames, setSettingsNames] = useState<SystemSettingsNames>(() => systemSettingsNames(24));
  const [character, setCharacter] = useState("TA");
  const [appName, setAppName] = useState("tulpa");
  const [notice, setNotice] = useState<string | null>(null);
  const [waiting, setWaiting] = useState<SystemPermissionKind | null>(null);
  const [confirming, setConfirming] = useState<SystemPermissionKind | null>(null);
  const [readiness, setReadiness] = useState<ModelReadiness | null>(null);
  useEffect(() => {
    void api.getModelReadiness().then(setReadiness).catch(() => undefined);
    return api.onModelReadiness(setReadiness);
  }, []);
  const [imageInput, setImageInput] = useState<ModelImageInputStatus | null>(null);
  const models = view ? JSON.stringify([view.models.text, view.modelAssignments]) : "";
  useEffect(() => {
    if (!models) return;
    let current = true;
    setImageInput(null);
    // TA restarts its models after a change; ask once they are back.
    const timer = window.setTimeout(() => {
      void api.getModelImageInput().then((value) => { if (current) setImageInput(value); }).catch(() => undefined);
    }, 1500);
    return () => { current = false; window.clearTimeout(timer); };
  }, [models]);

  const refresh = () => {
    void api.getConfiguration().then(setView).catch((error) => setNotice(readableError(error)));
    void api.onboarding.permissionStatus().then(setPermissions).catch(() => undefined);
  };
  useEffect(() => {
    refresh();
    void api.getPerceptionStatus().then((status) => {
      setAudioSupported(status.audioSupported !== false);
      if (status.systemSettings) setSettingsNames(status.systemSettings);
    }).catch(() => undefined);
    void api.onboarding.settingsAppName().then(setAppName).catch(() => undefined);
    window.addEventListener("focus", refresh);
    return () => window.removeEventListener("focus", refresh);
  }, []);

  const runtimeReady = useRuntimeReady();
  useEffect(() => {
    void api.getCurrentCharacter().then((value) => { if (value) setCharacter(value.name); }).catch(() => undefined);
  }, [runtimeReady]);

  // Permissions change in System Settings; keep checking while the user is there.
  // The heads-up dialog also counts: the user may have gone there on their own.
  const watched = waiting ?? confirming;
  useEffect(() => {
    if (!watched) return;
    const timer = window.setInterval(() => {
      void api.onboarding.permissionStatus().then(setPermissions);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [watched]);
  useEffect(() => {
    if (waiting && usable(permissions?.[waiting])) setWaiting(null);
    if (confirming && usable(permissions?.[confirming])) setConfirming(null);
  }, [permissions, waiting, confirming]);

  const save = async (input: DesktopConfigurationInput): Promise<boolean> => {
    setNotice(null);
    try {
      setView(await api.saveConfiguration(input));
      return true;
    } catch (error) {
      setNotice(readableError(error));
      return false;
    }
  };
  const change = (partial: Partial<DesktopConfiguration>) => { if (view) void save(withChanges(view, partial)); };

  const grant = async (kind: SystemPermissionKind) => {
    const state = await api.onboarding.requestPermission(kind);
    setPermissions((current) => current && { ...current, [kind]: state });
    setWaiting(usable(state) ? null : kind);
  };
  const startGrant = (kind: SystemPermissionKind) => {
    if (isWindows && kind !== "microphone") { void grant(kind); return; }
    if (kind !== "microphone" || permissions?.[kind] === "denied") setConfirming(kind);
    else void grant(kind);
  };

  if (!view) return <div class="sn"><SkyStage /></div>;

  const decisionConnected = decisionOf(view).ready;
  const blocked = blockedFeatures(view, permissions);
  const permissionMissing = permissions !== null && PERMISSION_ORDER.some((kind) => !usable(permissions[kind]));
  const current = SETTINGS_SECTIONS.find((item) => item.id === section)!;
  const tips: Record<SettingsSection, string> = {
    ai: decisionConnected ? t("两个大脑都接好了，我现在状态满满！") : t("有对话大脑就能陪你了；接上决策模型，我找你聊天的时机会更准。"),
    jev: decisionConnected ? t("决策模型接上了，该不该开口、怎么回你，我判断得又快又准～") : t("没接决策模型我也能做，只是慢一点；说话表情要等接上才有。"),
    companion: blocked.screen.length ? t("还差一个权限，点「去开启」就好，不用重启～")
      : t("想安静的时候，在我头顶点「暂停陪伴」就好。"),
    usage: t("不找你聊的时候，我也可能在后台想事情、看屏幕，这里都记着。"),
    extensions: t("想查点新消息、开个灯，或者有事交给别人办，跟我说就行～"),
    permissions: permissionMissing ? t("有权限还没开，开了我才能看、才能听。") : t("权限都开好了，谢谢你信任我。"),
    general: t("外观默认跟着时间走，晚上会自动变暗。"),
    advanced: t("这里是给懂行的人调的，一般不用动～"),
  };
  const props = {
    view, change, save, setSection, permissions, startGrant, waiting, audioSupported, appName, character, readiness, settingsNames,
    imageInput, locale, reload: refresh, openLogs: () => setLogsOpen(true),
  };
  const heading = logsOpen
    ? { label: t("运行日志"), subtitle: t("TA 和 AI 服务之间每一次请求和返回"), latin: "LOGS", footer: current.footer }
    : current;

  return (
    <div class="sn">
      <SkyStage />
      <MainNav current="settings" />

      <aside class="sn-aside" aria-label={t("设置分类")}>
        <div class="sn-menu">
          <div class="sn-ornament"><span />{t("设 置")}<span /></div>
          {SETTINGS_SECTIONS.map((item, index) => {
            const dot = (item.id === "permissions" && permissionMissing)
              || (item.id === "companion" && readiness?.state === "failed");
            const starts = index > 0 && SETTINGS_SECTIONS[index - 1]!.group !== item.group;
            return (
              <button type="button" class={`sn-menu__item${starts ? " sn-menu__item--group" : ""}`} aria-current={item.id === section ? "page" : undefined} onClick={() => setSection(item.id)}>
                <span class={`sn-menu__icon ui-hue-${SECTION_ICONS[item.id].hue}`}><Icon name={SECTION_ICONS[item.id].icon} size={15} /></span>
                <span class="sn-menu__label">{t(item.label)}</span>
                {dot && <span class="sn-dot" aria-label={t("有待处理的事项")} />}
              </button>
            );
          })}
        </div>
        <div class="sn-tip">
          <span class="sn-tip__name">{character}</span>
          <span>{tips[section]}</span>
        </div>
      </aside>

      <section class="sn-panel" aria-labelledby="sn-title">
        <header class="sn-panel__header">
          <div>
            {logsOpen && <div class="sn-back"><BackButton onClick={() => setLogsOpen(false)}>{t("通用设置")}</BackButton></div>}
            <h1 id="sn-title">{t(heading.label)}</h1>
            <p>{t(heading.subtitle)}</p>
          </div>
          <span class="sn-latin">{heading.latin}</span>
        </header>
        <div class="sn-panel__body">
          {notice && <Notice tone="warn">{t(notice)}</Notice>}
          {logsOpen && <LogsView retentionDays={view.llmLogRetentionDays ?? 7} />}
          {!logsOpen && section === "ai" && <AiSection {...props} />}
          {!logsOpen && section === "jev" && <JevSection {...props} />}
          {!logsOpen && section === "companion" && <CompanionSection {...props} />}
          {!logsOpen && section === "usage" && <UsageSection {...props} />}
          {!logsOpen && section === "extensions" && <ExtensionsSection {...props} />}
          {!logsOpen && section === "permissions" && <PermissionsSection {...props} />}
          {!logsOpen && section === "general" && <GeneralSection {...props} />}
          {!logsOpen && section === "advanced" && <AdvancedSection {...props} />}
        </div>
        <footer class="sn-panel__footer"><Icon name="lock" size={12} /><span>{t(current.footer)}</span></footer>
      </section>

      {confirming && (
        <Dialog
          title={confirming === "systemAudio" && permissions?.systemAudio !== "denied" ? t("系统会问一下") : t("马上去系统设置")}
          confirmLabel={confirming === "systemAudio" && permissions?.systemAudio !== "denied" ? t("好的") : t("知道了，去打开")}
          onCancel={() => setConfirming(null)}
          onConfirm={() => { const kind = confirming; setConfirming(null); void grant(kind); }}
        >
          {isWindows ? <p>{t("在 Windows 设置的「隐私和安全性 › 麦克风」中，打开「麦克风访问权限」和「允许桌面应用访问麦克风」。")}</p> : confirming === "systemAudio" && permissions?.systemAudio !== "denied" ? (
            <p>{t("macOS 会询问是否允许「{{app}}」录制电脑声音。点「允许」即可；没有弹框时，会打开系统设置。", { app: appName })}</p>
          ) : (
            <>
              <p>{t("在「{{place}}」里打开「{{app}}」的开关。", { place: permissionPlace(confirming, settingsNames, getLocale()), app: appName })}</p>
              {confirming === "screen" && <p>{t("系统问要不要「退出并重新打开」时，选「以后」即可，不用重启；回到这里会自动打勾。")}</p>}
              {confirming === "systemAudio" && settingsNames.audioOnly && <p>{t("「{{permission}}」在同一页的下方，要往下翻一翻。打开后回到这里会自动打勾。", { permission: t(settingsNames.audioOnly) })}</p>}
            </>
          )}
        </Dialog>
      )}
    </div>
  );
}

interface SectionProps {
  view: DesktopConfigurationView;
  change: (partial: Partial<DesktopConfiguration>) => void;
  save: (input: DesktopConfigurationInput) => Promise<boolean>;
  setSection: (section: SettingsSection) => void;
  permissions: Permissions | null;
  startGrant: (kind: SystemPermissionKind) => void;
  waiting: SystemPermissionKind | null;
  audioSupported: boolean;
  settingsNames: SystemSettingsNames;
  appName: string;
  character: string;
  readiness: ModelReadiness | null;
  imageInput: ModelImageInputStatus | null;
  reload: () => void;
  openLogs: () => void;
  locale: DesktopUiLocale;
}

function ImageInputBadge({ value }: { value: ModelImageInput | undefined }) {
  if (value === "supported") return <Badge tone="ok">{t("能看图")}</Badge>;
  if (value === "unsupported") return <Badge tone="warn">{t("不能看图")}</Badge>;
  return null;
}

/** Whether the reasoning model, which does the looking up, can search the web. */
function searchCapable(view: DesktopConfigurationView): boolean {
  const model = view.models.text.find((item) => item.id === view.modelAssignments.reasoning);
  return model !== undefined && webSearchSupported(model.baseUrl);
}

function SearchBadge({ view }: { view: DesktopConfigurationView }) {
  return searchCapable(view) ? <Badge tone="ok">{t("能联网")}</Badge> : <Badge tone="warn">{t("不能联网")}</Badge>;
}

/** What the screen switch cannot do with the models as configured, in one sentence. */
function screenImageHint(input: ModelImageInputStatus | null, shared: boolean): string | null {
  if (!input) return null;
  const chat = input.chat === "unsupported";
  const reasoning = input.reasoning === "unsupported";
  if (chat && (reasoning || shared)) {
    return t("{{brains}}不能看图，TA 看不到你的屏幕。换一个能看图的模型就好", { brains: shared ? t("对话大脑") : t("对话大脑和推理大脑都") });
  }
  if (chat) return t("对话大脑不能看图，TA 回你话时看不到屏幕，只能参考推理大脑记下的屏幕内容");
  if (reasoning) return t("推理大脑不能看图，屏幕上的事不会被记下来；TA 回你话时还能看到当下的屏幕");
  return null;
}

function ServiceCard({ icon, hue, title, tag, subtitle, status, highlighted, children }: {
  icon: IconName; hue: string; title: string; tag?: string; subtitle: string; status: ComponentChildren; highlighted?: boolean; children?: ComponentChildren;
}) {
  return (
    <div class={`sn-card sn-card--service${highlighted ? " sn-card--attention" : ""}`}>
      <div class="sn-card__head">
        <span class={`sn-card__icon ui-hue-${hue}`}><Icon name={icon} size={19} weight={1.9} /></span>
        <span class="sn-card__text">
          <span class="sn-card__title">{title}{tag && <span class="ui-tag">{tag}</span>}</span>
          <span class="sn-card__subtitle">{subtitle}</span>
        </span>
        {status}
      </div>
      {children}
    </div>
  );
}

function KeyField({ id, label, placeholder, onVerify, actionLabel = t("验证并保存"), plain = false }: {
  id: string; label: string; placeholder: string; actionLabel?: string;
  onVerify: (key: string) => Promise<Check>;
  /** Not a secret (an ID): shown as typed. */
  plain?: boolean;
}) {
  const [value, setValue] = useState("");
  const [check, setCheck] = useState<Check>({ phase: "idle" });
  const submit = async () => {
    if (!plain && value.trim().length < 8) { setCheck({ phase: "error", message: "Key 看起来不完整" }); return; }
    setCheck({ phase: "checking" });
    setCheck(await onVerify(value.trim()));
  };
  return (
    <div class="sn-key">
      <label for={id}>{label}</label>
      <input id={id} class="ui-input" type={plain ? "text" : "password"} autocomplete="off" spellcheck={false} placeholder={placeholder} value={value}
        onInput={(event) => { setValue(event.currentTarget.value); setCheck({ phase: "idle" }); }}
        onKeyDown={(event) => { if (event.key === "Enter") void submit(); }} />
      <PillButton size="sm" icon="check" disabled={check.phase === "checking" || !value.trim()} onClick={() => void submit()}>
        {check.phase === "checking" ? t("正在验证…") : actionLabel}
      </PillButton>
      {check.phase === "error" && <div class="sn-key__notice"><Notice>{t(check.message)}</Notice></div>}
    </div>
  );
}

const OPENAI_IMAGE_BASE_URL = "https://api.openai.com/v1";

/**
 * 图片模型: only AI 换装 uses it, through an OpenAI-compatible image endpoint
 * (OpenAI itself when the address is left empty). An older fal key keeps working.
 */
function ImageModelCard({ view, save }: Pick<SectionProps, "view" | "save">) {
  const compatible = Boolean(view.imageEditing.openaiCompatibleBaseUrl);
  const ready = compatible ? view.secrets.openaiCompatibleImageApiKey : view.secrets.falApiKey;
  const [editing, setEditing] = useState(false);
  const [baseUrl, setBaseUrl] = useState(view.imageEditing.openaiCompatibleBaseUrl ?? "");
  const store = async (apiKey: string): Promise<Check> => {
    const address = baseUrl.trim() || OPENAI_IMAGE_BASE_URL;
    if (!/^https?:\/\//.test(address)) return { phase: "error", message: "接口地址要以 https:// 开头，或者留空" };
    const saved = await save(withChanges(view, { imageEditing: { openaiCompatibleBaseUrl: address } }, { openaiCompatibleImageApiKey: apiKey }));
    if (saved) setEditing(false);
    return saved ? { phase: "ok" } : { phase: "error", message: "保存没有成功，请再试一次" };
  };
  return (
    <ServiceCard icon="spark" hue="green" title={t("图片模型")} tag={t("AI 换装用")}
      subtitle={t("重画衣服、头发的颜色和花纹。按次计费，每次开始前会告诉你大约多少钱")}
      status={null}>
      {ready && !editing ? (
        <div class="sn-key">
          <span class="sn-key__saved">{!compatible ? "fal" : view.imageEditing.openaiCompatibleBaseUrl === OPENAI_IMAGE_BASE_URL ? t("OpenAI 官方") : t("OpenAI 兼容 · {{url}}", { url: view.imageEditing.openaiCompatibleBaseUrl ?? "" })} {t("· Key 已保存")}</span>
          <PillButton size="sm" variant="secondary" onClick={() => setEditing(true)}>{t("更换")}</PillButton>
        </div>
      ) : (
        <>
          <p class="sn-hint">{t("需要能调用 gpt-image-2.5 的接口。用 OpenAI 官方 Key 时，接口地址留空就行")}</p>
          <Field label={t("接口地址")}><input class="ui-input" placeholder={t("留空即 OpenAI 官方（{{url}}）", { url: OPENAI_IMAGE_BASE_URL })} value={baseUrl} onInput={(event) => setBaseUrl(event.currentTarget.value)} /></Field>
          <KeyField id="sn-image-key" label="API Key" placeholder={t("粘贴图片接口的 Key")} actionLabel={t("保存")} onVerify={store} />
          {ready && <div class="sn-inline"><TextButton onClick={() => setEditing(false)}>{t("取消")}</TextButton></div>}
        </>
      )}
    </ServiceCard>
  );
}

const JEV_KEYS_URL = "https://console.typesafe.ai/keys";
const BAILIAN_WORKSPACE_URL = "https://bailian.console.aliyun.com/cn-beijing/settings/workspace";

/** Which decision model judges and whether it can, from the saved-key flags. */
function decisionOf(view: DesktopConfigurationView) {
  const hasKey = (id: string) => view.secrets.modelApiKeys?.[id] === true;
  const jevKey = view.secrets.jevApiKey === true;
  return {
    provider: decisionProvider(view, hasKey, jevKey),
    ready: decisionReady(view, hasKey, jevKey),
    bailianKey: bailianKeyModelId(view, hasKey) !== null,
    workspace: bailianWorkspaceId(view),
    jevKey,
  };
}

/** Picks an AI service and saves it once its key checks out. */
function ServiceChooser({ id, current, onSave, onCancel }: {
  id: string;
  current: { provider: AiProvider; model: DesktopTextModel | undefined };
  onSave: (choice: AiServiceChoice) => Promise<boolean>;
  onCancel?: () => void;
}) {
  const api = window.desktopPet;
  const custom = current.provider.id === "custom";
  const [provider, setProvider] = useState<AiProvider>(current.provider);
  const [baseUrl, setBaseUrl] = useState(custom ? current.model?.baseUrl ?? "" : "");
  const [model, setModel] = useState(custom ? current.model?.model ?? "" : "");
  const verify = async (apiKey: string): Promise<Check> => {
    const result = await api.onboarding.checkAiService(aiServiceRequest({ provider, baseUrl, model, apiKey }));
    if (!result.ok) return { phase: "error", message: result.message };
    return await onSave({ provider, baseUrl, model, apiKey })
      ? { phase: "ok" } : { phase: "error", message: "保存没有成功，请再试一次" };
  };
  return (
    <>
      <Segmented label={t("AI 服务")} value={provider.id}
        options={AI_PROVIDERS.map((item) => ({ id: item.id, label: item.label }))}
        onChange={(next) => setProvider(AI_PROVIDERS.find((item) => item.id === next)!)} />
      {provider.id === "custom" && (
        <div class="sn-two">
          <Field label={t("接口地址")}><input class="ui-input" placeholder="https://…/v1" value={baseUrl} onInput={(event) => setBaseUrl(event.currentTarget.value)} /></Field>
          <Field label={t("模型名称")}><input class="ui-input" placeholder={t("例如 gpt-4o-mini")} value={model} onInput={(event) => setModel(event.currentTarget.value)} /></Field>
        </div>
      )}
      <KeyField id={id} label="API Key" placeholder={t("粘贴到这里")} onVerify={verify} />
      <div class="sn-inline">
        {provider.keyUrl && <a class="ui-text-button" href={provider.keyUrl} target="_blank" rel="noopener">{t("去{{site}}获取 Key", { site: t(provider.keySite ?? "") })}</a>}
        {onCancel && <TextButton onClick={onCancel}>{t("取消")}</TextButton>}
      </div>
    </>
  );
}

/** Every saved model: edit its service or key, remove it, or add another. */
function SavedModelsCard({ view, save, imageInput }: Pick<SectionProps, "view" | "save" | "imageInput">) {
  const models = configuredModels(view);
  const [editing, setEditing] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);
  const target = models.find((item) => item.model.id === removing);
  const removal = removing ? withRemovedModel(view, removing) : null;
  const fallback = removal ? models.find((item) => item.model.id === removal.modelAssignments.chat) : undefined;
  const store = (id?: string) => async (choice: AiServiceChoice) => {
    const saved = await save(withSavedModel(view, choice, id ? { id } : {}));
    if (saved) { setEditing(null); setAdding(false); }
    return saved;
  };
  const imageOf = (id: string) =>
    id === view.modelAssignments.chat ? imageInput?.chat : id === view.modelAssignments.reasoning ? imageInput?.reasoning : undefined;
  return (
    <ServiceCard icon="sliders" hue="slate" title={t("已配置的模型")} subtitle={t("配过的模型都在这里，两个大脑直接从中选")}
      status={null}>
      {models.map(({ model, provider }) => editing === model.id ? (
        <ServiceChooser key={model.id} id={`sn-model-${model.id}`} current={{ provider, model }}
          onSave={store(model.id)} onCancel={() => setEditing(null)} />
      ) : (
        <div class="sn-key" key={model.id}>
          <span class="sn-key__saved">{modelLabel(model)}</span>
          <ImageInputBadge value={imageOf(model.id)} />
          <PillButton size="sm" variant="secondary" onClick={() => setEditing(model.id)}>{t("修改")}</PillButton>
          <PillButton size="sm" variant="secondary" onClick={() => setRemoving(model.id)}>{t("删除")}</PillButton>
        </div>
      ))}
      {adding ? (
        <ServiceChooser id="sn-model-new" current={{ provider: AI_PROVIDERS[0]!, model: undefined }}
          onSave={store()} onCancel={() => setAdding(false)} />
      ) : (
        <div class="sn-inline"><TextButton onClick={() => setAdding(true)}>{t("+ 新增模型")}</TextButton></div>
      )}
      {target && (
        <Dialog title={removal ? t("删除「{{model}}」？", { model: modelLabel(target.model) }) : t("这个模型不能删")} confirmLabel={removal ? t("删除") : t("知道了")}
          onCancel={() => setRemoving(null)}
          onConfirm={() => { if (removal) void save(removal); setRemoving(null); }}>
          <p>{!removal ? t("这是唯一的模型，对话大脑离不开它。先新增一个模型，再删这个。")
            : target.usedBy.length ? t("{{brains}}正在用它，删掉后改用「{{model}}」。它的 Key 也会从这台电脑上删掉。", { brains: target.usedBy.map((label) => t(label)).join(t("和")), model: modelLabel(fallback!.model) })
            : t("它的 Key 也会从这台电脑上删掉。")}</p>
        </Dialog>
      )}
    </ServiceCard>
  );
}

const ADD_MODEL = "__add__";
const FOLLOW_CHAT = "__follow__";

/** A brain's model, chosen from the saved ones; the last option adds one. */
function ModelSelect({ label, value, options, onChange }: {
  label: string; value: string; options: Array<{ id: string; label: string }>; onChange: (id: string) => void;
}) {
  return (
    <div class="sn-key">
      <label>{label}</label>
      <select class="ui-input sn-select" value={value} onChange={(event) => onChange(event.currentTarget.value)}>
        {options.map((option) => <option value={option.id}>{option.label}</option>)}
        <option value={ADD_MODEL}>{t("新增一个模型…")}</option>
      </select>
    </div>
  );
}

function AiSection({ view, save, setSection, imageInput }: SectionProps) {
  const models = configuredModels(view);
  const chatId = view.modelAssignments.chat;
  const reasoningId = view.modelAssignments.reasoning;
  const chat = models.find((item) => item.model.id === chatId);
  const reasoning = models.find((item) => item.model.id === reasoningId);
  const follows = !reasoning || reasoningId === chatId;
  // Which brain is filling in a new model right away; with none saved, 对话大脑 is.
  const [adding, setAdding] = useState<"chat" | "reasoning" | null>(null);
  const decisionConnected = decisionOf(view).ready;
  const options = models.map((item) => ({ id: item.model.id, label: modelLabel(item.model) }));
  const blank = { provider: AI_PROVIDERS[0]!, model: undefined };
  const addFor = (use: "chat" | "reasoning") => async (choice: AiServiceChoice) => {
    const saved = await save(withSavedModel(view, choice, { use }));
    if (saved) setAdding(null);
    return saved;
  };
  const pick = (module: "chat" | "reasoning") => (id: string) => {
    if (id === ADD_MODEL) setAdding(module);
    else void save(withAssignedModel(view, module, id === FOLLOW_CHAT ? chatId : id));
  };
  return (
    <div class="sn-stack">
      <ServiceCard icon="chat" hue="blue" title={t("对话大脑")} tag={t("必需")}
        subtitle={follows ? t("聊天和推理都用它") : t("和你聊天、主动说话用它")}
        status={chat ? <span class="sn-status-row">
          <ImageInputBadge value={imageInput?.chat} />
          {follows && <SearchBadge view={view} />}
        </span> : null}
        highlighted={!chat}>
        {adding === "chat" || !models.length ? (
          <ServiceChooser id="sn-chat-key" current={blank} onSave={addFor("chat")}
            onCancel={models.length ? () => setAdding(null) : undefined} />
        ) : (
          <ModelSelect label={t("模型")} value={chat ? chatId : ""} options={options} onChange={pick("chat")} />
        )}
      </ServiceCard>

      <ServiceCard icon="spark" hue="blue" title={t("推理大脑")} tag={t("可选")}
        subtitle={t("整理记忆、总结屏幕、在心里想事情用它")}
        status={follows ? null : <span class="sn-status-row">
          <ImageInputBadge value={imageInput?.reasoning} />
          <SearchBadge view={view} />
        </span>}>
        {adding === "reasoning" ? (
          <ServiceChooser id="sn-reasoning-key" current={blank} onSave={addFor("reasoning")} onCancel={() => setAdding(null)} />
        ) : models.length > 0 && (
          <ModelSelect label={t("模型")} value={follows ? FOLLOW_CHAT : reasoningId}
            options={[{ id: FOLLOW_CHAT, label: t("和对话大脑一样") }, ...options]} onChange={pick("reasoning")} />
        )}
      </ServiceCard>

      {models.length > 0 && <SavedModelsCard view={view} save={save} imageInput={imageInput} />}

      <ServiceCard icon="bolt" hue="amber" title={t("决策模型")} tag={t("可选 · 效果更好")}
        subtitle={t("百炼或 Jev，怎么接、用在哪，都在「决策模型」里")}
        status={<PillButton size="sm" variant="secondary" onClick={() => setSection("jev")}>{decisionConnected ? t("去 决策模型") : t("接上决策模型")}</PillButton>} />

      <ImageModelCard view={view} save={save} />

      <ServiceCard icon="sound" hue="rose" title={t("说话的声音")} subtitle={t("每个角色自己选用通义语音还是本机语音，Key 也在那里填")}
        status={<PillButton size="sm" variant="secondary" onClick={() => openCharacter("voice")}>{t("去 角色 › 声音")}</PillButton>} />
    </div>
  );
}

/** 决策模型: 百炼 or Jev and what it needs, then every judgement it takes over, each with its switch. */
function JevSection({ view, save, change, setSection }: SectionProps) {
  const decision = decisionOf(view);
  const tuning = view.experimentalCompanion ?? { model: "jev-1.13.0", wakeSeconds: 5 };
  const inner = view.perception.innerLifeEnabled !== false;
  const choose = (provider: DecisionProvider) => change({ experimentalCompanion: { ...tuning, decisionProvider: provider } });

  return (
    <div class="sn-stack">
      <ServiceCard icon="bolt" hue="amber" title={t("决策模型")} tag={t("可选 · 效果更好")}
        subtitle={t("专门做判断的模型：比让对话大脑来判断更准、更快、更省")}
        status={null}>
        <Segmented label={t("决策模型")} value={decision.provider}
          options={[{ id: "jev", label: "Jev" }, { id: "bailian", label: t("百炼") }]}
          onChange={(value) => choose(value as DecisionProvider)} />
        {decision.provider === "bailian" && <p class="sn-hint">{t("百炼的决策模型目前不如 Jev 准：有时该回你时不回，该交给 Agent 的事也可能自己答了。Jev 用不了时可以先用它")}</p>}
        {decision.provider === "bailian"
          ? <BailianDecision view={view} save={save} setSection={setSection} decision={decision} />
          : <JevDecision view={view} save={save} />}
      </ServiceCard>

      {!decision.ready && (
        <div class="sn-strip sn-strip--quiet">
          <Icon name="bolt" size={14} />
          <span>{t("还没接上决策模型：前三项先由对话大脑来判断，慢一些、没那么准；说话表情要接上才有")}</span>
        </div>
      )}

      <div class="sn-card">
        <div class="sn-group"><span>{t("决策模型帮 TA 做这些判断")}</span><small>{t("「主动找你聊」「内心」和 陪伴 里是同一个开关")}</small></div>
        <FeatureRow icon="heart" hue="rose" title={t("主动找你聊")} description={t("判断此刻该开口，还是安静陪着；开着时会一直在后台看屏幕、想事情，每小时大约消耗 100 万 token 以上")}
          checked={view.proactiveCompanionEnabled} onChange={(value) => change({ proactiveCompanionEnabled: value })} />
        <FeatureRow icon="chat" hue="green" title={t("聊天时挑怎么回")}
          description={t("你说完话，它来决定是直接回、先想想以前的事、先查资料，还是交给「Agent 任务」去办；关掉就按固定顺序来")}
          checked={chatPickEnabled(view, decision.ready)}
          onChange={(value) => change({ experimentalCompanion: { ...tuning, chatPickEnabled: value } })} />
        <FeatureRow icon="spark" hue="violet" title={t("内心")} description={t("闲下来挑最值得想的念头，也判断看到听到的东西值不值得记下")}
          checked={inner} onChange={(value) => change({ perception: { ...view.perception, innerLifeEnabled: value } })} />
        <FeatureRow icon="face" hue="amber" title={t("说话时的表情和动作")}
          description={decision.ready ? t("每说一句，配上合适的眉眼、嘴角和手臂动作；部位在 形象编辑 里调") : t("每说一句，配上合适的眉眼、嘴角和手臂动作 · 接上决策模型才能用")}
          checked={decision.ready && tuning.speechFacialEnabled !== false} disabled={!decision.ready}
          onChange={(value) => change({ experimentalCompanion: { ...tuning, speechFacialEnabled: value } })} />
      </div>
    </div>
  );
}

/** 百炼: uses the Bailian key already saved in AI 服务; only the 业务空间 ID is asked. */
function BailianDecision({ view, save, setSection, decision }: Pick<SectionProps, "view" | "save" | "setSection"> & { decision: ReturnType<typeof decisionOf> }) {
  const api = window.desktopPet;
  const [editing, setEditing] = useState(false);
  const tuning = view.experimentalCompanion ?? { model: "jev-1.13.0", wakeSeconds: 5 };
  if (!decision.bailianKey) {
    return (
      <div class="sn-inline">
        <span class="sn-hint">{t("百炼决策模型用你在 AI 服务 里填的百炼 Key，现在还没有")}</span>
        <PillButton size="sm" variant="secondary" onClick={() => setSection("ai")}>{t("去添加百炼")}</PillButton>
      </div>
    );
  }
  const verify = async (workspaceId: string): Promise<Check> => {
    const result = await api.onboarding.checkBailianDecision(workspaceId);
    if (!result.ok) return { phase: "error", message: result.message };
    const saved = await save(withChanges(view, { experimentalCompanion: { ...tuning, decisionProvider: "bailian", bailianWorkspaceId: workspaceId } }));
    if (saved) setEditing(false);
    return saved ? { phase: "ok" } : { phase: "error", message: "保存没有成功，请再试一次" };
  };
  return decision.workspace && !editing ? (
    <div class="sn-key">
      <span class="sn-key__saved">{t("业务空间 {{id}} · 用 AI 服务 里的百炼 Key", { id: decision.workspace })}</span>
      <PillButton size="sm" variant="secondary" onClick={() => setEditing(true)}>{t("更换")}</PillButton>
    </div>
  ) : (
    <>
      <KeyField id="sn-bailian-workspace" label={t("业务空间 ID")} placeholder={t("例如 ws-xxxx 或 llm-xxxx")} onVerify={verify} plain />
      <p class="sn-hint">{t("模型 decision-model-preview（北京）· Key 用 AI 服务 里的百炼 Key")}{editing && <TextButton onClick={() => setEditing(false)}>{t("取消")}</TextButton>}</p>
      <div class="sn-inline"><a class="ui-text-button" href={BAILIAN_WORKSPACE_URL} target="_blank" rel="noopener">{t("去百炼查看业务空间")}</a><span class="sn-hint">{t("登录百炼后，在「业务空间」页面能看到 ID")}</span></div>
    </>
  );
}

/** Jev: its own key. */
function JevDecision({ view, save }: Pick<SectionProps, "view" | "save">) {
  const api = window.desktopPet;
  const [editing, setEditing] = useState(false);
  const tuning = view.experimentalCompanion ?? { model: "jev-1.13.0", wakeSeconds: 5 };
  const verify = async (apiKey: string): Promise<Check> => {
    const result = await api.onboarding.checkJev(apiKey);
    if (!result.ok) return { phase: "error", message: result.message };
    const saved = await save(withChanges(view, { experimentalCompanion: { ...tuning, decisionProvider: "jev" } }, { jevApiKey: apiKey }));
    if (saved) setEditing(false);
    return saved ? { phase: "ok" } : { phase: "error", message: "保存没有成功，请再试一次" };
  };
  return view.secrets.jevApiKey === true && !editing ? (
    <div class="sn-key">
      <span class="sn-key__saved">{t("模型 {{model}} · Key 已保存", { model: tuning.model })}</span>
      <PillButton size="sm" variant="secondary" onClick={() => setEditing(true)}>{t("更换")}</PillButton>
      <TextButton onClick={() => void save(withChanges(view, {}, { jevApiKey: null }))}>{t("移除")}</TextButton>
    </div>
  ) : (
    <>
      <KeyField id="sn-jev-key" label="API Key" placeholder={t("粘贴 Jev API Key")} onVerify={verify} />
      <p class="sn-hint">{t("模型 {{model}} · 填好后下面这些马上换成 Jev 判断", { model: tuning.model })}{editing && <TextButton onClick={() => setEditing(false)}>{t("取消")}</TextButton>}</p>
      <div class="sn-inline"><a class="ui-text-button" href={JEV_KEYS_URL} target="_blank" rel="noopener">{t("去 Jev 获取 Key")}</a><span class="sn-hint">{t("在 typesafe.ai 注册登录，到控制台的 Keys 页面创建")}</span></div>
    </>
  );
}

function FeatureRow({ icon, hue, title, needs, description, checked, disabled, onChange, action }: {
  icon: IconName; hue: string; title: string; needs?: string; description: string;
  checked?: boolean; disabled?: boolean; onChange?: (value: boolean) => void; action?: ComponentChildren;
}) {
  return (
    <div class="sn-feature">
      <span class={`sn-feature__icon ui-hue-${checked === false ? "slate" : hue}`}><Icon name={icon} size={16} /></span>
      <span class="sn-feature__text">
        <span class="sn-feature__title">{title}{needs && <span class="ui-tag">{needs}</span>}</span>
        <span class="sn-feature__description">{description}</span>
      </span>
      {action ?? (onChange && <Switch checked={Boolean(checked)} label={title} disabled={disabled} onChange={onChange} />)}
    </div>
  );
}

function PermissionStrip({ features, kind, startGrant, waiting, setSection }: {
  features: string[]; kind: SystemPermissionKind; startGrant: (kind: SystemPermissionKind) => void;
  waiting: SystemPermissionKind | null; setSection: (section: SettingsSection) => void;
}) {
  if (!features.length) return null;
  const title = PERMISSION_INFO[kind].title;
  return (
    <div class={`sn-strip${waiting === kind ? " sn-strip--waiting" : ""}`}>
      <Icon name="lock" size={14} />
      <span>{waiting === kind ? t("等你在系统设置里打开「{{permission}}」…", { permission: t(title) }) : t("{{features}}需要「{{permission}}」权限，现在还没开", { features: features.map((label) => t(label)).join(t("和")), permission: t(title) })}</span>
      <TextButton onClick={() => setSection("permissions")}>{t("查看权限")}</TextButton>
      <PillButton size="sm" icon="check" onClick={() => startGrant(kind)}>{waiting === kind ? t("重新打开") : t("去开启")}</PillButton>
    </div>
  );
}

/** Speech models still downloading or failed: said plainly so voice does not look broken. */
function ReadinessStrip({ readiness }: { readiness: ModelReadiness | null }) {
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState<string | null>(null);
  const retry = async () => {
    if (retrying) return;
    setRetrying(true);
    setRetryError(null);
    try { await window.desktopPet.retryModelPreparation(); }
    catch (error) { setRetryError(readableError(error)); }
    finally { setRetrying(false); }
  };
  const [exporting, setExporting] = useState(false);
  const [exported, setExported] = useState<"saved" | "failed" | null>(null);
  const exportDiagnostics = async () => {
    if (exporting) return;
    setExporting(true);
    setExported(null);
    try { await window.desktopPet.exportDiagnostics(); setExported("saved"); }
    catch { setExported("failed"); }
    finally { setExporting(false); }
  };
  if (!readiness || readiness.state === "ready") return null;
  const failed = readiness.state === "failed";
  return (
    <div class={`sn-strip${failed ? "" : " sn-strip--waiting"}`} role="status">
      <Icon name={failed ? "warn" : "sound"} size={14} />
      <span><strong>{readiness.text}</strong> · {readiness.detail}
        {retryError && <span role="alert"> · {t("重试未完成，请再试一次")} {retryError}</span>}
        {exported === "saved" && <span> · {t("诊断信息已保存到桌面，发给开发者就好")}</span>}
        {exported === "failed" && <span role="alert"> · {t("导出没完成，请再试一次")}</span>}
      </span>
      {failed && (
        <PillButton size="sm" variant="secondary" disabled={exporting}
          onClick={() => void exportDiagnostics()}>
          {exporting ? t("正在导出…") : t("导出诊断信息")}
        </PillButton>
      )}
      {failed && (
        <PillButton size="sm" icon="check" disabled={retrying}
          onClick={() => void retry()}>
          {retrying ? t("正在重试…") : t("重试")}
        </PillButton>
      )}
    </div>
  );
}

function CompanionSection({ view, change, setSection, permissions, startGrant, waiting, audioSupported, readiness, imageInput, reload }: SectionProps) {
  const api = window.desktopPet;
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const runtimeReady = useRuntimeReady();
  useEffect(() => { void api.getUsage().then(setUsage).catch(() => undefined); }, [runtimeReady]);
  /** What a switch cost in the last hour, beside the switch itself. */
  const lastHour = (group: UsageGroup): string => {
    const found = usage?.groups.find((item) => item.group === group);
    return found && found.hour.calls > 0 ? t(" · 最近 1 小时用了 {{tokens}} token", { tokens: tokenText(found.hour.tokens, getLocale()) }) : "";
  };
  const decision = decisionOf(view);
  const blocked = blockedFeatures(view, permissions);
  const inner = view.perception.innerLifeEnabled !== false;
  const microphone = view.microphoneEnabled ?? view.voiceEnabled;
  const playback = view.playbackEnabled ?? view.voiceEnabled;
  const perception = (patch: Partial<DesktopConfiguration["perception"]>) => change({ perception: { ...view.perception, ...patch } });
  const voice = (next: { microphone?: boolean; playback?: boolean }) => {
    const mic = next.microphone ?? microphone;
    const play = next.playback ?? playback;
    change({ microphoneEnabled: mic, playbackEnabled: play, voiceEnabled: mic || play });
  };

  return (
    <div class="sn-stack">
      <ReadinessStrip readiness={readiness} />
      <div class="sn-card">
        <FeatureRow icon="heart" hue="amber" title={t("主动陪伴")}
          description={t("根据 TA 感知到的内容，自然地找你聊。关掉后只在你找 TA 时回应") + lastHour("proactive")}
          checked={view.proactiveCompanionEnabled}
          onChange={(value) => change({ proactiveCompanionEnabled: value })} />
        {view.proactiveCompanionEnabled && (
          <p class="sn-hint">
            {t("什么时候开口由 {{brain}} 判断", { brain: !decision.ready ? t("对话大脑（接上决策模型会更准）") : decision.provider === "bailian" ? t("百炼决策模型") : "Jev" })} · <TextButton onClick={() => setSection("jev")}>{t("在 决策模型 里看")}</TextButton>
          </p>
        )}
        <p class="sn-hint">
          {t("开着时 TA 会一直看屏幕、在后台想事情，对话和推理模型大约每小时消耗 100 万 token 以上，屏幕变化越多用得越多")} · <TextButton onClick={() => setSection("usage")}>{t("看用量")}</TextButton>
        </p>
      </div>

      <div class="sn-card">
        <div class="sn-group"><span>{t("TA 能感知什么")}</span><small>{t("主动陪伴靠这些了解你")}</small></div>
        <PermissionStrip features={blocked.screen} kind="screen" startGrant={startGrant} waiting={waiting} setSection={setSection} />
        <PermissionStrip features={blocked.systemAudio} kind="systemAudio" startGrant={startGrant} waiting={waiting} setSection={setSection} />
        <FeatureRow icon="screen" hue="amber" title={t("屏幕感知")} needs={t("屏幕与系统录音")} description={t("看得到你在屏幕上做什么") + lastHour("senses")}
          checked={view.perception.screenPerceptionEnabled} onChange={(value) => perception({ screenPerceptionEnabled: value })} />
        {view.perception.screenPerceptionEnabled && screenImageHint(imageInput, !conversationService(view).split) && (
          <p class="sn-hint">
            {screenImageHint(imageInput, !conversationService(view).split)} · <TextButton onClick={() => setSection("ai")}>{t("去 AI 服务 换模型")}</TextButton>
          </p>
        )}
        <FeatureRow icon="sound" hue="blue" title={t("电脑音频感知")} needs={t("屏幕与系统录音")}
          description={!audioSupported ? t("这台电脑的系统版本还不支持") : t("听得到视频、音乐里的对白和声音")}
          checked={view.perception.audioPerceptionEnabled && audioSupported} disabled={!audioSupported}
          onChange={(value) => perception({ audioPerceptionEnabled: value })} />
        {audioSupported && view.perception.audioPerceptionEnabled && !inner && (
          <div class="sn-row-note">
            <Notice tone="info" action={<TextButton onClick={() => perception({ innerLifeEnabled: true })}>{t("打开内心")}</TextButton>}>
              {t("听到的声音要靠内心来整理，内心关着时 TA 听了也记不住")}</Notice>
          </div>
        )}
        <FeatureRow icon="heart" hue="rose" title={t("内心")} description={t("把看到听到的整理成影音总结，闲下来自己想想") + lastHour("inner")}
          checked={inner} onChange={(value) => perception({ innerLifeEnabled: value })} />
      </div>

      <div class="sn-card">
        <div class="sn-group"><span>{t("说话和声音")}</span></div>
        <PermissionStrip features={blocked.microphone} kind="microphone" startGrant={startGrant} waiting={waiting} setSection={setSection} />
        <FeatureRow icon="mic" hue="blue" title={t("麦克风收音")} needs={t("麦克风")} description={t("喊名字就能和 TA 说话，还能随时打断")}
          checked={microphone} onChange={(value) => voice({ microphone: value })} />
        <FeatureRow icon="sound" hue="slate" title={t("声音播放")} description={t("TA 说话时出声")}
          checked={playback} onChange={(value) => voice({ playback: value })} />
        <OwnerVoiceRow view={view} change={change} reload={reload} />
        <div class="sn-feature">
          <span class="sn-feature__icon ui-hue-slate"><Icon name="person" size={16} /></span>
          <span class="sn-feature__text">
            <span class="sn-feature__title">{t("唤醒词")}</span>
            <span class="sn-feature__description">{t("唤醒词跟着角色走，每个角色可以有几个")}</span>
          </span>
          <PillButton size="sm" variant="secondary" onClick={() => openCharacter("profile")}>{t("在 角色 › 性格 里改")}</PillButton>
        </div>
      </div>
    </div>
  );
}

function PermissionsSection({ permissions, startGrant, waiting, appName, settingsNames }: SectionProps) {
  const mac = navigator.userAgent.includes("Mac");
  const shown = PERMISSION_ORDER.filter((kind) => kind === "systemAudio"
    // Normally covered by Screen Recording; listed only when audio alone is refused.
    ? permissions?.systemAudio === "denied" && usable(permissions.screen)
    : permissions?.[kind] !== "not-required");
  // The "can't find tulpa" help only matters while something is still off.
  const missing = shown.some((kind) => !usable(permissions?.[kind]));
  return (
    <div class="sn-stack">
      {shown.map((kind) => {
        const state = permissions?.[kind];
        const ok = usable(state);
        const info = PERMISSION_INFO[kind];
        return (
          <div class={`sn-card sn-card--service${ok ? "" : " sn-card--attention"}`}>
            <div class="sn-card__head">
              <span class={`sn-card__icon ui-hue-${ok ? "green" : waiting === kind ? "blue" : "amber"}`}><Icon name={info.icon} size={19} weight={1.9} /></span>
              <span class="sn-card__text">
                <span class="sn-card__title">{t(info.title)}
                  {state === "not-required" ? <Badge tone="muted">{t("不需要")}</Badge>
                    : ok ? <Badge tone="ok">{t("已开启")}</Badge>
                    : waiting === kind ? <Badge tone="info">{t("等你打开开关")}</Badge>
                    : <Badge tone="warn">{t("未开启")}</Badge>}
                </span>
                <span class="sn-card__subtitle">{t("用于：{{features}}", { features: t(info.usedBy) })}</span>
              </span>
              {!ok && state !== undefined && (
                <PillButton size="sm" icon="check" onClick={() => startGrant(kind)}>{waiting === kind ? t("重新打开设置") : t("去开启")}</PillButton>
              )}
            </div>
          </div>
        );
      })}
      {mac && missing && (
        <div class="sn-card sn-help">
          <div class="sn-help__art" aria-hidden="true">
            <div><i />{t("微信")}</div><div><i />Chrome</div>
            <div class="sn-help__bar"><b>+</b><span>−</span></div>
          </div>
          <div>
            <strong>{t("系统设置的列表里找不到「{{app}}」？", { app: appName })}</strong>
            <ol>
              <li>{t("在「{{place}}」里，点列表左下角的「+」", { place: `${t(settingsNames.privacy)} › ${t(settingsNames.screen)}` })}{settingsNames.audioOnly && t("（「电脑声音」在同一页下方的「{{permission}}」里）", { permission: t(settingsNames.audioOnly) })}</li>
              <li>{t("在「应用程序」里选 {{app}}，点「打开」", { app: appName })}</li>
              <li>{t("系统问要不要「退出并重新打开」，选「以后」就行，不用重启")}</li>
            </ol>
          </div>
        </div>
      )}
      <p class="sn-hint">{t("权限只决定 TA 能不能用这些设备；用不用、什么时候用，由「陪伴」里的开关决定。")}</p>
    </div>
  );
}

function SettingRow({ title, description, children }: { title: string; description: string; children: ComponentChildren }) {
  return (
    <div class="sn-setting">
      <span class="sn-feature__text">
        <span class="sn-feature__title">{title}</span>
        <span class="sn-feature__description">{description}</span>
      </span>
      {children}
    </div>
  );
}

/**
 * Counts up each time the local service comes up, so what a page reads from
 * it is read again instead of staying empty from before it was running.
 */
function useRuntimeReady(): number {
  const [epoch, setEpoch] = useState(0);
  useEffect(() => {
    const api = window.desktopPet;
    let up = false;
    const apply = (status: DesktopRuntimeStatus | null | undefined) => {
      const now = status?.phase === "ready" || status?.phase === "starting-companion";
      if (now && !up) setEpoch((value) => value + 1);
      up = now;
    };
    void api.getRuntimeStatus().then(apply).catch(() => undefined);
    return api.onRuntimeStatus(apply);
  }, []);
  return epoch;
}

/** 用量: tokens by purpose, and which of them run without being asked. */
function UsageSection({ setSection }: SectionProps) {
  const api = window.desktopPet;
  const [summary, setSummary] = useState<UsageSummary | null | undefined>(undefined);
  const load = () => void api.getUsage().then(setSummary).catch(() => setSummary(null));
  // Opened while TA was still starting: it comes in by itself once TA is up.
  const runtimeReady = useRuntimeReady();
  useEffect(load, [runtimeReady]);
  if (summary === undefined) return <p class="sn-hint">{t("正在统计…")}</p>;
  if (summary === null) return <p class="sn-hint">{t("TA 正在启动，好了这里会自己显示。")}</p>;
  const cell = ({ tokens, calls }: UsageTotals) => calls === 0
    ? <span class="sn-usage__cell sn-usage__cell--empty">—</span>
    : (
      <span class="sn-usage__cell">
        {tokens > 0 ? tokenText(tokens, getLocale()) : t("{{count}} 次", { count: calls })}
        {tokens > 0 && <small>{t("{{count}} 次", { count: calls })}</small>}
      </span>
    );
  return (
    <div class="sn-stack">
      <div class="sn-card sn-usage-total">
        <div>
          <span class="sn-usage-total__label">{t("今天一共用了")}</span>
          <span class="sn-usage-total__value">{tokenText(summary.today.tokens, getLocale())} <small>token</small></span>
        </div>
        <PillButton size="sm" variant="secondary" onClick={load}>{t("刷新")}</PillButton>
      </div>
      <p class="sn-hint sn-usage-note">
        {t("token 是 AI 服务计费的单位。标着「后台」的几项你不开口也会用，想省一点可以")}<TextButton onClick={() => setSection("companion")}>{t("去陪伴里关掉")}</TextButton>
      </p>
      <div class="sn-card sn-usage">
        <div class="sn-usage__head">
          <span>{t("用途")}</span><span>{t("今天")}</span><span>{t("最近 1 小时")}</span><span>{t("近 7 天")}</span>
        </div>
        {summary.groups.map((item) => {
          const info = USAGE_GROUPS.find((group) => group.group === item.group)!;
          return (
            <div class="sn-usage__row">
              <span class="sn-usage__name">
                <span class="sn-usage__title">{t(item.label)}{item.background && <span class="ui-tag">{t("后台")}</span>}</span>
                <small>{t(info.detail)}{item.background && item.active ? t("，最近 10 分钟还在用") : ""}</small>
              </span>
              {cell(item.today)}{cell(item.hour)}{cell(item.week)}
            </div>
          );
        })}
      </div>
      <p class="sn-hint">{t("这里只统计仍在本机保留的调用日志，已删除的记录不计入。")}</p>
    </div>
  );
}

function GeneralSection({ view, change, save, openLogs, locale }: SectionProps) {
  const api = window.desktopPet;
  const [scene, setScene] = useState<ScenePreference>(scenePreference());
  const [bubbles, setBubbles] = useState(true);
  const [savingLocale, setSavingLocale] = useState(false);
  const [localeError, setLocaleError] = useState<string | null>(null);
  const selectLocale = async (next: DesktopUiLocale) => {
    if (savingLocale || next === locale) return;
    setSavingLocale(true);
    setLocaleError(null);
    try { await changeLocale(next); }
    catch (error) { setLocaleError(readableError(error)); }
    finally { setSavingLocale(false); }
  };
  useEffect(() => { void api.getSpeechBubbles().then(setBubbles).catch(() => undefined); }, []);
  return (
    <div class="sn-card sn-list">
      <SettingRow title="语言 / Language" description={t("切换所有窗口的界面语言，不改变角色和聊天内容")}>
        <LanguagePicker locale={locale} busy={savingLocale} onChange={(next) => void selectLocale(next)} showLabel={false} />
      </SettingRow>
      {localeError && <Notice tone="warn">{t("语言没有保存成功，请重试")} {localeError}</Notice>}
      <SettingRow title={t("外观")} description={t("白天、黄昏、夜晚的天空和面板颜色")}>
        <Segmented label={t("外观")} value={scene}
          options={[{ id: "auto", label: t("跟随时间") }, { id: "day", label: t("白天") }, { id: "dusk", label: t("黄昏") }, { id: "night", label: t("夜晚") }]}
          onChange={(value) => { setScene(value); setScenePreference(value); }} />
      </SettingRow>
      <SettingRow title={t("形象清晰度")} description={t("越清晰越吃内存；放大时自动提高")}>
        <Segmented label={t("形象清晰度")} value={String(view.rendering.maxTextureDimension) as "2048" | "4096" | "8192"}
          options={TEXTURE_OPTIONS}
          onChange={(value) => change({ rendering: { ...view.rendering, maxTextureDimension: Number(value) as 2048 | 4096 | 8192 } })} />
      </SettingRow>
      <SettingRow title={t("开机时陪着你")} description={t("登录电脑后自动出现在桌面")}>
        <Switch checked={view.launchAtLogin} label={t("开机时陪着你")} onChange={(value) => change({ launchAtLogin: value })} />
      </SettingRow>
      <SettingRow title={t("说话时显示气泡")} description={t("在 TA 身边显示正在说的话")}>
        <Switch checked={bubbles} label={t("说话时显示气泡")} onChange={(value) => { setBubbles(value); void api.setSpeechBubbles(value); }} />
      </SettingRow>
      <SettingRow title={t("运行日志")} description={t("遇到问题时，可以查看 TA 和 AI 服务之间的记录")}>
        <PillButton size="sm" variant="secondary" onClick={openLogs}>{t("打开日志")}</PillButton>
      </SettingRow>
      <LogRetentionRow days={view.llmLogRetentionDays ?? 7}
        onCommit={(days) => save(withChanges(view, { llmLogRetentionDays: days }))} />
    </div>
  );
}

function LogRetentionRow({ days, onCommit }: { days: number; onCommit: (days: number) => Promise<boolean> }) {
  const [text, setText] = useState(String(days));
  const [saving, setSaving] = useState(false);
  useEffect(() => setText(String(days)), [days]);
  const commit = async () => {
    if (saving) return;
    const value = Number(text);
    if (!Number.isInteger(value) || value < 1 || value > 365 || value === days) {
      setText(String(days));
      return;
    }
    setSaving(true);
    try { if (!await onCommit(value)) setText(String(days)); }
    finally { setSaving(false); }
  };
  return (
    <SettingRow title={t("日志保留天数")} description={t("请求、回复和图片保存在本机，超过此天数自动删除；聊天记录和长期记忆不受影响")}>
      <span class="sn-number">
        <input class="ui-input" type="number" min={1} max={365} step={1} value={text} disabled={saving} aria-label={t("日志保留天数")}
          onInput={(event) => setText(event.currentTarget.value)}
          onBlur={() => void commit()} />
        <small>{t("天")}</small>
      </span>
    </SettingRow>
  );
}

/** A tuning number: what it does on the left, a small field with its unit on the right. */
function NumberRow({ title, description, unit, value, step, min, max, onCommit }: {
  title: string; description: string; unit?: string; value: number; step: number; min: number; max: number; onCommit: (value: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  return (
    <SettingRow title={title} description={description}>
      <span class="sn-number">
        <input class="ui-input" type="number" step={step} min={min} max={max} value={text} aria-label={title}
          onInput={(event) => setText(event.currentTarget.value)}
          onBlur={() => {
            const parsed = Number(text);
            if (Number.isFinite(parsed) && parsed >= min && parsed <= max && parsed !== value) onCommit(parsed);
            else setText(String(value));
          }} />
        {/* Always there, so every field lines up whether it has a unit or not. */}
        <small>{unit ?? ""}</small>
      </span>
    </SettingRow>
  );
}

function AdvancedSection({ view, change }: SectionProps) {
  const advanced = view.advanced;
  const set = (patch: Partial<DesktopConfiguration["advanced"]>) => change({ advanced: { ...advanced, ...patch } });
  const thoughts = view.experimentalCompanion ?? { model: "jev-1.13.0", wakeSeconds: 5 };
  return (
    <div class="sn-stack">
      <div class="sn-strip">
        <Icon name="warn" size={14} />
        <span>{t("这些是调试用的参数，一般不用动。改乱了可以一键恢复。")}</span>
        <PillButton size="sm" variant="secondary" onClick={() => set(ADVANCED_DEFAULTS)}>{t("恢复默认")}</PillButton>
      </div>
      <div class="sn-card sn-list">
        <NumberRow title={t("人声检测阈值")} description={t("越高越不容易把杂音当成说话")} value={advanced.vadThreshold} step={0.01} min={0} max={1}
          onCommit={(value) => set({ vadThreshold: value })} />
        <NumberRow title={t("说完后停多久算说完")} description={t("停顿超过这么久，就当你这句说完了")} unit={t("毫秒")} value={advanced.vadMinSilenceMs} step={10} min={10} max={5000}
          onCommit={(value) => set({ vadMinSilenceMs: value })} />
        <NumberRow title={t("最短人声")} description={t("比这更短的声音不算在说话")} unit={t("毫秒")} value={advanced.vadMinSpeechMs} step={10} min={10} max={5000}
          onCommit={(value) => set({ vadMinSpeechMs: value })} />
        <NumberRow title={t("轮次判断阈值")} description={t("越高越要确定你说完了才开口")} value={advanced.smartTurnThreshold} step={0.01} min={0} max={1}
          onCommit={(value) => set({ smartTurnThreshold: value })} />
        <NumberRow title={t("内心多久想一次")} description={t("内心打开时，闲下来隔多久想一想")} unit={t("秒")} value={thoughts.thoughtSeconds ?? 2} step={1} min={1} max={3600}
          onCommit={(value) => change({ experimentalCompanion: { ...thoughts, thoughtSeconds: value } })} />
        <NumberRow title={t("多久看一次要不要开口")} description={t("主动陪伴时，空闲下来隔多久考虑一次")} unit={t("秒")} value={thoughts.wakeSeconds} step={1} min={1} max={3600}
          onCommit={(value) => change({ experimentalCompanion: { ...thoughts, wakeSeconds: value } })} />
      </div>
      <div class="sn-card sn-list">
        <SettingRow title={t("等表情就绪再开口")} description={t("最多等 2 秒，让表情和语音同步")}>
          <Switch checked={view.speechFacialWaitForReady ?? false} label={t("等表情就绪再开口")} onChange={(value) => change({ speechFacialWaitForReady: value })} />
        </SettingRow>
        <SettingRow title={t("语音诊断记录")} description={t("保存语音链路的调试数据")}>
          <Switch checked={advanced.diagnosticsEnabled} label={t("语音诊断记录")} onChange={(value) => set({ diagnosticsEnabled: value })} />
        </SettingRow>
      </div>
    </div>
  );
}

/** 只听我的声音: record, replace or delete the voiceprint, then switch it on. */
function OwnerVoiceRow({ view, change, reload }: Pick<SectionProps, "view" | "change" | "reload">) {
  const api = window.desktopPet;
  const [status, setStatus] = useState<OwnerVoiceStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [managing, setManaging] = useState(false);
  const [confirmAll, setConfirmAll] = useState(false);
  const [recording, setRecording] = useState<null | { append: boolean }>(null);
  const [missed, setMissed] = useState(false);
  const [playing, setPlaying] = useState<string | null>(null);
  const player = useRef<HTMLAudioElement | null>(null);
  const stop = () => { player.current?.pause(); player.current = null; setPlaying(null); };
  const listen = async (clip: string) => {
    if (playing === clip) { stop(); return; }
    stop();
    try {
      const audio = new Audio(await api.getOwnerVoiceClip(clip));
      player.current = audio;
      audio.onended = () => { if (player.current === audio) stop(); };
      setPlaying(clip);
      await audio.play();
    } catch (failure) {
      stop();
      setError(readableError(failure));
    }
  };
  const refresh = () => void api.getOwnerVoiceStatus().then(setStatus).catch(() => undefined);
  useEffect(() => stop, []);
  useEffect(() => {
    refresh();
    // Sentences TA just ignored: offered here as well as next to the pet.
    void api.chat.senses().then((value) => setMissed(Boolean(value?.ownerVoiceMissed))).catch(() => undefined);
    const off = api.chat.onSenses((value) => {
      const now = Boolean(value?.ownerVoiceMissed);
      setMissed((before) => { if (before && !now) refresh(); return now; });
    });
    return off;
  }, []);
  const enabled = view.audio.ownerRecognitionEnabled;
  const backend = view.advanced.audioBackend;
  const controls = ownerVoiceControls(status, { phase: "idle", status: null, detail: "" }, backend, backend, false, enabled);
  const remove = async (createdAt?: string) => {
    setError(null);
    try {
      setStatus(await api.deleteOwnerVoice(createdAt));
      reload();
    } catch (failure) {
      setError(readableError(failure));
    }
  };
  const count = status?.enrollmentCount ?? 1;
  const description = status?.enrolled
    ? t("已录 {{count}} 段你的声音。打开后，别人说话 TA 不理；换了麦克风或位置认不准时，再录一段", { count })
    : t("录一段你的声音，打开后别人说话 TA 就不理了");
  return (
    <div class="sn-owner">
      <FeatureRow icon="shield" hue="slate" title={t("只听我的声音")} description={description}
        checked={enabled} disabled={controls.checkboxDisabled}
        onChange={(value) => change({ audio: { ...view.audio, ownerRecognitionEnabled: value } })} />
      <div class="sn-owner__body">
        {missed && (
          <Notice tone="info" action={
            <PillButton size="sm" variant="secondary" onClick={() => void api.chat.senseAction({ type: "owner-voice-add" })}>{t("是我，加进声纹")}</PillButton>
          }>{t("刚才有几句没认出是你。确认是你自己说的，加进来 TA 下次就认得了")}</Notice>
        )}
        <div class="sn-owner__actions">
          {status?.enrolled ? (
            <>
              <PillButton size="sm" variant="secondary" icon="mic" onClick={() => setRecording({ append: true })}>{t("再录一段")}</PillButton>
              <TextButton onClick={() => setManaging(!managing)}>{managing ? t("收起") : t("管理声纹（{{count}} 份）", { count })}</TextButton>
            </>
          ) : (
            <>
              <PillButton size="sm" variant="secondary" icon="mic" onClick={() => setRecording({ append: false })}>{t("录入声纹")}</PillButton>
              {!status?.modelReady && <span class="sn-hint">{t("第一次会先下载约 28 MB 的声纹模型")}</span>}
            </>
          )}
        </div>
        {managing && controls.canDelete && (
          <ul class="sn-owner__list">
            {(status?.enrollments ?? []).map((item) => (
              <li>
                <span class="sn-owner__item">
                  <b>{item.source === "added" ? t("加进来的没认出的话") : t("你录的一段")}</b>
                  <small>{recordedAt(item.createdAt)}</small>
                </span>
                <span class="sn-owner__item-actions">
                  {item.clip
                    ? <TextButton onClick={() => void listen(item.clip!)}>{playing === item.clip ? t("停止") : t("试听")}</TextButton>
                    : <span class="sn-hint">{t("没有录音")}</span>}
                  <TextButton onClick={() => { if (playing === item.clip) stop(); void remove(item.createdAt); }}>{t("删除")}</TextButton>
                </span>
              </li>
            ))}
            <li class="sn-owner__all">
              <TextButton onClick={() => setRecording({ append: false })}>{t("全部重录")}</TextButton>
              <TextButton onClick={() => setConfirmAll(true)}>{t("全部删除")}</TextButton>
            </li>
          </ul>
        )}
        {error && <Notice>{t(error)}</Notice>}
      </div>
      {recording && (
        <OwnerVoiceRecording append={recording.append} recognitionOn={enabled}
          onTurnOn={() => change({ audio: { ...view.audio, ownerRecognitionEnabled: true } })}
          onClose={(result) => { setRecording(null); if (result) { setStatus(result); reload(); } }} />
      )}
      {confirmAll && (
        <Dialog title={t("删掉全部 {{count}} 份声纹？", { count })} confirmLabel={t("全部删除")}
          onCancel={() => setConfirmAll(false)}
          onConfirm={() => { setConfirmAll(false); setManaging(false); void remove(); }}>
          <p>{t("删完「只听我的声音」会关掉，要用的话得重新录。只想去掉认错的那一份，可以在列表里单独删。")}</p>
        </Dialog>
      )}
    </div>
  );
}

/** Something to read aloud, so nobody wonders what to say for 15 seconds. */
const OWNER_VOICE_SCRIPT = "今天天气还不错，我想出去走走，顺便买点水果回来。你还记得上次我们聊的那本书吗？结局挺出乎意料的，等有空了我再慢慢跟你说。";

/**
 * Recording a voiceprint, step by step in one window: the model download
 * with its progress, a 15-second countdown with something to read, then
 * making the voiceprint. Closing it cancels.
 */
function OwnerVoiceRecording({ append, recognitionOn, onTurnOn, onClose }: {
  append: boolean;
  recognitionOn: boolean;
  onTurnOn: () => void;
  onClose: (result: OwnerVoiceStatus | null) => void;
}) {
  const api = window.desktopPet;
  const [state, setState] = useState<OwnerVoiceEnrollmentState>({ phase: "preparing", status: null, detail: t("正在准备…") });
  const [outcome, setOutcome] = useState<{ result: OwnerVoiceStatus } | { error: string } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [left, setLeft] = useState(15);
  useEffect(() => api.onOwnerVoiceEnrollmentState(setState), []);
  useEffect(() => {
    let live = true;
    setOutcome(null);
    setState({ phase: "preparing", status: null, detail: t("正在准备…") });
    api.enrollOwnerVoice(append)
      .then((result) => { if (live) setOutcome({ result }); })
      .catch((failure) => { if (live) setOutcome({ error: readableError(failure) }); });
    return () => { live = false; };
  }, [attempt]);
  useEffect(() => {
    if (state.phase !== "recording") return;
    const total = state.durationSeconds ?? 15;
    const started = Date.now();
    setLeft(total);
    const timer = window.setInterval(() => setLeft(Math.max(0, total - Math.floor((Date.now() - started) / 1000))), 200);
    return () => window.clearInterval(timer);
  }, [state.phase]);
  const working = outcome === null;
  const cancel = () => {
    if (working) void api.cancelOwnerVoiceEnrollment();
    onClose(null);
  };
  const done = outcome && "result" in outcome ? outcome.result : null;
  const failed = outcome && "error" in outcome && !outcome.error.includes("已取消") ? outcome.error : null;
  const total = state.durationSeconds ?? 15;

  let body;
  if (done) {
    body = (
      <div class="ov-step">
        <span class="ov-mark ov-mark--ok"><Icon name="check" size={26} weight={3} /></span>
        <p class="ov-lead">{t("录好了")}</p>
        <p class="ov-note">{t("现在 TA 认得 {{count}} 段你的声音", { count: done.enrollmentCount ?? 1 })}{recognitionOn ? "" : t("。打开「只听我的声音」后，别人说话 TA 就不理了")}</p>
      </div>
    );
  } else if (failed) {
    body = (
      <div class="ov-step">
        <Notice>{failed}</Notice>
      </div>
    );
  } else if (state.phase === "recording") {
    body = (
      <div class="ov-step">
        <span class="ov-ring" style={{ "--ov-seconds": `${total}s` }}>
          <svg viewBox="0 0 120 120" aria-hidden="true"><circle cx="60" cy="60" r="54" /><circle class="ov-ring__run" cx="60" cy="60" r="54" /></svg>
          <span class="ov-ring__count">{left}</span>
          <span class="ov-ring__mic"><Icon name="mic" size={14} weight={2.4} /></span>
        </span>
        <p class="ov-lead">{t("正在录音，请正常说话")}</p>
        <p class="ov-script">{t(OWNER_VOICE_SCRIPT)}</p>
        <p class="ov-note">{t("照着念就行，也可以随便聊；在安静的地方，用平常的音量，别停太久")}</p>
      </div>
    );
  } else if (state.phase === "preparing" && state.download) {
    const ratio = state.download.received / state.download.total;
    body = (
      <div class="ov-step">
        <p class="ov-lead">{t("正在下载声纹模型")}</p>
        <span class="ov-bar"><span style={{ width: `${Math.round(ratio * 100)}%` }} /></span>
        <p class="ov-note">{(state.download.received / 1048576).toFixed(1)} / {(state.download.total / 1048576).toFixed(1)} {t("MB · 只有第一次需要下载，下完就开始录音")}</p>
      </div>
    );
  } else {
    const lead = state.phase === "processing" ? t("正在生成你的声纹…")
      : state.phase === "restoring" ? t("马上就好…")
      : t("正在准备麦克风…");
    body = (
      <div class="ov-step">
        <span class="ov-spinner" aria-hidden="true" />
        <p class="ov-lead">{lead}</p>
        <p class="ov-note">{state.phase === "preparing" ? t("准备好就会开始 15 秒的录音") : t("录好的声音只保存在这台电脑上")}</p>
      </div>
    );
  }
  const footer = done ? (
    <>
      {!recognitionOn && <PillButton variant="secondary" onClick={() => onClose(done)}>{t("先不打开")}</PillButton>}
      <PillButton icon="check" autoFocus onClick={() => { if (!recognitionOn) onTurnOn(); onClose(done); }}>
        {recognitionOn ? t("完成") : t("打开「只听我的声音」")}
      </PillButton>
    </>
  ) : failed ? (
    <>
      <PillButton variant="secondary" onClick={() => onClose(null)}>{t("关闭")}</PillButton>
      <PillButton icon="mic" onClick={() => setAttempt(attempt + 1)}>{t("再试一次")}</PillButton>
    </>
  ) : (
    <PillButton variant="secondary" icon="close" disabled={state.phase === "restoring"} onClick={cancel}>{t("取消录入")}</PillButton>
  );
  return (
    // While it works only 取消录入 stops it: a stray click outside must not.
    <Dialog title={append ? t("再录一段你的声音") : t("录入你的声音")} onCancel={working ? () => undefined : () => onClose(done)} footer={footer}>
      {body}
    </Dialog>
  );
}

/** "9月28日 20:40" in local time. */
function recordedAt(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return t("时间未知");
  return date.toLocaleString(getLocale(), { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
}

/**
 * 帮你做事: what TA can do for the person when asked (look things up, the
 * smart home, an outside Agent), then what other programs hand TA to pass on.
 */
function ExtensionsSection({ view, change, save, setSection }: SectionProps) {
  return (
    <div class="sn-stack">
      <div class="sn-stack__title">{t("TA 能替你做")}</div>
      <WebSearchCard view={view} change={change} setSection={setSection} />
      <HomeCard view={view} change={change} save={save} />
      <AgentCard view={view} change={change} save={save} />
      <div class="sn-stack__title">{t("别的程序告诉 TA")}</div>
      <CodexCard view={view} change={change} />
      <GatewayCard view={view} change={change} />
    </div>
  );
}

function WebSearchCard({ view, change, setSection }: Pick<SectionProps, "view" | "change" | "setSection">) {
  const enabled = view.webSearchEnabled !== false;
  return (
    <ServiceCard icon="search" hue="blue" title={t("联网查资料")}
      subtitle={t("遇到新闻、天气这类要最新消息的事，TA 会先上网查了再回你")}
      status={<Switch checked={enabled} label={t("联网查资料")} onChange={(value) => change({ webSearchEnabled: value })} />}>
      {enabled && !searchCapable(view) && (
        <Notice tone="info" action={<TextButton onClick={() => setSection("ai")}>{t("去 AI 服务 换模型")}</TextButton>}>
          {t("你用的推理大脑不能联网，开着也查不了。现在能联网的是 {{services}}", { services: WEB_SEARCH_SERVICES })}
        </Notice>
      )}
    </ServiceCard>
  );
}

function UrlField({ label, value, placeholder, onCommit }: {
  label: string; value: string; placeholder: string; onCommit: (value: string) => void;
}) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  // Same row as the key under it: label on the left, field beside it.
  return (
    <div class="sn-key">
      <label>{label}</label>
      <input class="ui-input" placeholder={placeholder} value={text} aria-label={label}
        onInput={(event) => setText(event.currentTarget.value)}
        onBlur={() => { if (text.trim() && text.trim() !== value) onCommit(text.trim()); else setText(value); }} />
    </div>
  );
}

function HomeCard({ view, change, save }: Pick<SectionProps, "view" | "change" | "save">) {
  const home = view.homeAssistant;
  const hasToken = view.secrets.homeAssistantToken;
  const [editing, setEditing] = useState(false);
  const store = async (token: string): Promise<Check> => {
    const saved = await save(withChanges(view, {}, { homeAssistantToken: token }));
    if (saved) setEditing(false);
    return saved ? { phase: "ok" } : { phase: "error", message: "保存没有成功，请再试一次" };
  };
  return (
    <ServiceCard icon="home" hue="green" title={t("智能家居")} tag="Home Assistant"
      subtitle={t("让 TA 帮你开灯、关空调、看家里的设备状态")}
      status={<Switch checked={home.enabled} label={t("智能家居")} disabled={!hasToken && !home.enabled}
        onChange={(value) => change({ homeAssistant: { ...home, enabled: value } })} />}>
      <UrlField label={t("地址")} value={home.url} placeholder="http://homeassistant.local:8123"
        onCommit={(url) => change({ homeAssistant: { ...home, url } })} />
      {hasToken && !editing ? (
        <div class="sn-key">
          <span class="sn-key__saved">{t("访问令牌已保存")}</span>
          <PillButton size="sm" variant="secondary" onClick={() => setEditing(true)}>{t("更换")}</PillButton>
        </div>
      ) : (
        <>
          <KeyField id="sn-home-token" label={t("访问令牌")} placeholder={t("在 Home Assistant 个人资料页创建的长期访问令牌")}
            actionLabel={t("保存")} onVerify={store} />
          {editing && <div class="sn-inline"><TextButton onClick={() => setEditing(false)}>{t("取消")}</TextButton></div>}
        </>
      )}
      {!hasToken && <p class="sn-hint">{t("先填好地址和访问令牌，再打开开关")}</p>}
    </ServiceCard>
  );
}

function AgentCard({ view, change, save }: Pick<SectionProps, "view" | "change" | "save">) {
  const agent = view.agent;
  const hasKey = view.secrets.agentApiKey;
  const [editing, setEditing] = useState(false);
  const store = async (apiKey: string): Promise<Check> => {
    const saved = await save(withChanges(view, {}, { agentApiKey: apiKey }));
    if (saved) setEditing(false);
    return saved ? { phase: "ok" } : { phase: "error", message: "保存没有成功，请再试一次" };
  };
  return (
    <ServiceCard icon="send" hue="blue" title={t("Agent 任务")} tag="Hermes"
      subtitle={t("复杂的事交给外部 Agent 去做，做完 TA 把结果告诉你")}
      status={<Switch checked={agent.enabled} label={t("Agent 任务")} disabled={!agent.url && !agent.enabled}
        onChange={(value) => change({ agent: { ...agent, enabled: value } })} />}>
      <UrlField label={t("地址")} value={agent.url} placeholder="http://127.0.0.1:8642"
        onCommit={(url) => change({ agent: { ...agent, url } })} />
      {hasKey && !editing ? (
        <div class="sn-key">
          <span class="sn-key__saved">{t("API Key 已保存")}</span>
          <PillButton size="sm" variant="secondary" onClick={() => setEditing(true)}>{t("更换")}</PillButton>
          <TextButton onClick={() => void save(withChanges(view, {}, { agentApiKey: null }))}>{t("移除")}</TextButton>
        </div>
      ) : (
        <>
          <KeyField id="sn-agent-key" label="API Key" placeholder={t("粘贴 Agent 的 API Key，没有可以不填")} actionLabel={t("保存")} onVerify={store} />
          {editing && <div class="sn-inline"><TextButton onClick={() => setEditing(false)}>{t("取消")}</TextButton></div>}
        </>
      )}
    </ServiceCard>
  );
}

function CodexCard({ view, change }: Pick<SectionProps, "view" | "change">) {
  const enabled = view.codexNotificationsEnabled ?? false;
  const [status, setStatus] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) { setStatus(null); return; }
    const read = () => void window.desktopPet.getCodexNotificationStatus()
      .then((result) => setStatus(result.error ?? (result.enabled ? "正在留意本机的 Codex 任务" : "还没开始留意")))
      .catch(() => setStatus("暂时读不到状态"));
    read();
    const timer = window.setInterval(() => { if (!document.hidden) read(); }, 5_000);
    return () => window.clearInterval(timer);
  }, [enabled]);
  return (
    <ServiceCard icon="check" hue="violet" title={t("Codex 任务完成通知")}
      subtitle={t("本机 Codex 任务做完时，结果会进到当前角色的聊天里；刚打开时不补发以前的")}
      status={<Switch checked={enabled} label={t("Codex 任务完成通知")} onChange={(value) => change({ codexNotificationsEnabled: value })} />}>
      {status && <p class="sn-hint">{t(status)}</p>}
    </ServiceCard>
  );
}

const DEFAULT_GATEWAY = { enabled: false, port: 17821 };

/** 外部消息: other programs send TA a message over HTTP, with this card's key. */
function GatewayCard({ view, change }: Pick<SectionProps, "view" | "change">) {
  const gateway = view.gateway ?? DEFAULT_GATEWAY;
  const [status, setStatus] = useState<GatewayStatus | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [note, setNote] = useState("");
  const [testing, setTesting] = useState(false);
  useEffect(() => {
    if (!gateway.enabled) { setStatus(null); return; }
    let live = true;
    const read = () => void window.desktopPet.getGatewayStatus().then((value) => { if (live) setStatus(value); }).catch(() => {});
    read();
    const timer = window.setInterval(() => { if (!document.hidden) read(); }, 5_000);
    return () => { live = false; window.clearInterval(timer); };
  }, [gateway.enabled, gateway.port]);
  const copy = (text: string, what: string) => void navigator.clipboard.writeText(text)
    .then(() => setNote(t("已复制{{what}}", { what }))).catch(() => setNote("没能复制，请手动选中复制"));
  // Shown with the key hidden like above; copied with the real one.
  const example = (key: string) => status ? [
    `curl -X POST ${status.addresses[0]} \\`,
    `  -H "Authorization: Bearer ${key}" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '${JSON.stringify({ source: t("我的脚本"), text: t("快递到楼下了") })}'`,
  ].join("\n") : "";
  return (
    <ServiceCard icon="bolt" hue="amber" title={t("外部消息")}
      subtitle={t("让脚本、快捷指令、智能家居等程序给 TA 发消息，由 TA 转告你")}
      status={<Switch checked={gateway.enabled} label={t("外部消息")}
        onChange={(value) => change({ gateway: { ...gateway, enabled: value } })} />}>
      {gateway.enabled && status && (
        <div class="sn-stack sn-gateway">
          {status.error ? <Notice>{status.error}</Notice> : <p class="sn-hint">{t("正在接收。带上下面的钥匙，从哪里都能发")}</p>}
          <div class="sn-gateway__row">
            <span class="sn-gateway__label">{t("地址")}</span>
            <div class="sn-gateway__values">
              {status.addresses.map((address, index) => (
                <div class="sn-gateway__value" key={address}>
                  <code>{address}</code>
                  <small>{index === 0 ? t("这台电脑") : t("同一网络的其他设备")}</small>
                  <TextButton onClick={() => copy(address, t("地址"))}>{t("复制")}</TextButton>
                </div>
              ))}
            </div>
          </div>
          <div class="sn-gateway__row">
            <span class="sn-gateway__label">{t("钥匙")}</span>
            <div class="sn-gateway__value">
              <code>{showKey ? status.key : "•".repeat(24)}</code>
              <TextButton onClick={() => setShowKey(!showKey)}>{showKey ? t("隐藏") : t("显示")}</TextButton>
              <TextButton onClick={() => copy(status.key, t("钥匙"))}>{t("复制")}</TextButton>
              {confirmReset ? (
                <>
                  <TextButton onClick={() => void window.desktopPet.resetGatewayKey().then((value) => {
                    setStatus(value); setConfirmReset(false); setNote("换好了，用旧钥匙的程序要改成新的");
                  })}>{t("确定换")}</TextButton>
                  <TextButton onClick={() => setConfirmReset(false)}>{t("取消")}</TextButton>
                </>
              ) : <TextButton onClick={() => setConfirmReset(true)}>{t("换一把")}</TextButton>}
            </div>
          </div>
          <div class="sn-gateway__row">
            <span class="sn-gateway__label">{t("端口")}</span>
            <PortInput value={gateway.port} onCommit={(port) => change({ gateway: { ...gateway, port } })} />
          </div>
          <div class="sn-gateway__row sn-gateway__row--block">
            <span class="sn-gateway__label">{t("怎么发")}</span>
            <p class="sn-hint">{t("POST 一段 JSON：text 是内容；source 写是谁发的；想让 TA 只记下、不说出来，加上 \"speak\": false。")}</p>
            <pre class="sn-gateway__code">{example(showKey && status ? status.key : t("<钥匙>"))}</pre>
            <div class="sn-gateway__actions">
              <PillButton size="sm" variant="secondary" disabled={testing || !status.listening} onClick={() => {
                setTesting(true); setNote("");
                void window.desktopPet.testGateway()
                  .then((result) => setNote(result.message))
                  .finally(() => setTesting(false));
              }}>{testing ? t("正在发…") : t("发条试试")}</PillButton>
              <TextButton onClick={() => copy(example(status.key), t("示例（已带上钥匙）"))}>{t("复制示例")}</TextButton>
            </div>
          </div>
          {note && <p class="sn-hint">{t(note)}</p>}
        </div>
      )}
    </ServiceCard>
  );
}

function PortInput({ value, onCommit }: { value: number; onCommit: (value: number) => void }) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  return (
    <input class="ui-input sn-gateway__port" type="number" min={1024} max={65535} step={1} value={text} aria-label={t("端口")}
      onInput={(event) => setText(event.currentTarget.value)}
      onBlur={() => {
        const port = Number(text);
        if (Number.isInteger(port) && port >= 1024 && port <= 65535 && port !== value) onCommit(port);
        else setText(String(value));
      }} />
  );
}

/** 运行日志: the request/response log, shown inside settings. */
function LogsView({ retentionDays }: { retentionDays: number }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!host.current) return;
    const panel = new LlmLogPanel(host.current);
    panel.setVisible(true);
    return () => panel.setVisible(false);
  }, []);
  return (
    <div class="sn-stack">
      <p class="sn-hint">{t("本机日志包含请求、回复及图片，保留 {{days}} 天；运行期间定期清理过期记录。用量统计只包含保留期间的调用。", { days: retentionDays })}</p>
      <div ref={host} class="sn-logs" />
    </div>
  );
}
