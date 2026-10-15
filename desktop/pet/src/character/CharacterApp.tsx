import { t, getLocale } from "../i18n";
import { useLocale } from "../i18n/preact";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { openLook, recall, recallValue, remember } from "../ui/remembered";

import type {
  CharacterDetails,
  CharacterMemory,
  CharacterSummary,
  CharacterVoice,
  EntityFact,
  ForgetPlan,
  ForgetTarget,
  MemoryItem,
  ModelRenderBundle,
} from "../shared/contracts";
import { readableError } from "../onboarding/flow";
import { withChanges } from "../settings-next/model";
import { cloudVoiceService } from "../shared/voice-service";
import { CharacterStage } from "../onboarding/CharacterStage";
import { Badge, Chip, Dialog, Field, Notice, PillButton, TextButton } from "../ui/components";
import { Icon } from "../ui/icons";
import { MainNav } from "../ui/MainNav";
import { SkyStage } from "../ui/SkyStage";
import {
  CHARACTER_TABS,
  MAX_WAKE_WORDS,
  RELATIONS,
  TRAITS,
  changedFields,
  characterSummary,
  daysSince,
  draftFrom,
  profileInput,
  relationShown,
  relationStored,
  relationText,
  selectedRelation,
  selectedTraits,
  toggleTrait,
  traitBlock,
  traitNote,
  traitsText,
  wakeWordError,
  type CharacterDraft,
  type CharacterTab,
} from "./model";
import { WavRecorder } from "./recorder";
import { memoryText } from "../shared/memory-wording";

const CLONE_SCRIPT = "你好，很高兴认识你。接下来的日子里，我会用自然、清晰的声音陪你聊天，分享生活里每一个值得记住的瞬间。";

const TAB_IDS: readonly CharacterTab[] = ["profile", "voice", "memory"];

export function CharacterApp() {
  const locale = useLocale();
  const api = window.desktopPet;
  const [characters, setCharacters] = useState<CharacterSummary[] | null>(null);
  const [profile, setProfile] = useState<CharacterDetails | null>(null);
  const [draft, setDraft] = useState<CharacterDraft | null>(null);
  const [tab, showTab] = useState<CharacterTab>(() => recall("character-tab", TAB_IDS, "profile"));
  const setTab = (next: CharacterTab) => { remember("character-tab", next); showTab(next); };
  const [bundle, setBundle] = useState<ModelRenderBundle | null>(null);
  const [switcher, setSwitcher] = useState(false);
  const [creating, setCreating] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const current = characters?.find((item) => item.current) ?? null;

  const loadCharacters = async () => {
    try {
      const list = await api.characters.list();
      setCharacters(list);
      const active = list.find((item) => item.current);
      if (active) {
        const details = await api.characters.profile(active.id);
        setProfile(details);
        setDraft(draftFrom(details));
      }
    } catch (error) {
      setNotice(readableError(error));
      window.setTimeout(() => void loadCharacters(), 3000);
    }
  };
  const loadBundle = () => void api.getSelectedModelRenderBundle().then(setBundle).catch(() => undefined);
  useEffect(() => {
    void loadCharacters();
    loadBundle();
    // The 形象 can also change from the tray or the pet's menu.
    window.addEventListener("focus", loadBundle);
    return () => window.removeEventListener("focus", loadBundle);
  }, []);

  const saved = useMemo(() => profile && draftFrom(profile), [profile]);
  const changes = saved && draft ? changedFields(saved, draft) : 0;

  const save = async () => {
    if (!profile || !draft || !saved) return;
    if (!draft.name.trim()) { setNotice("名字不能空着"); return; }
    if (!draft.wakeWords.length) { setNotice("至少留一个唤醒词"); return; }
    setSaving(true);
    try {
      await api.characters.saveProfile(profile.id, profileInput(draft, saved));
      const details = await api.characters.profile(profile.id);
      setProfile(details);
      setDraft(draftFrom(details));
      setCharacters((list) => list?.map((item) => item.id === profile.id ? { ...item, name: details.name, role: details.role } : item) ?? list);
      setNotice(null);
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      setSaving(false);
    }
  };

  const switchTo = async (id: string) => {
    setSwitcher(false);
    if (changes && !window.confirm(t("还有没保存的修改，切换角色会丢掉它们。继续吗？"))) return;
    try {
      await api.characters.bind(id);
      await loadCharacters();
      loadBundle();
    } catch (error) {
      setNotice(readableError(error));
    }
  };

  const name = draft?.name.trim() || current?.name || "TA";
  const tabInfo = CHARACTER_TABS.find((item) => item.id === tab)!;
  const days = daysSince(profile?.createdAt ?? null);
  const summary = draft ? characterSummary(draft) : "";

  return (
    <div class="sn cp">
      <SkyStage />
      <MainNav current="character" />

      <section class="cp-stage" aria-label={t("角色舞台")}>
        <button type="button" class="cp-switch" aria-haspopup="menu" aria-expanded={switcher} onClick={() => setSwitcher(!switcher)}>
          <span class="cp-switch__avatar">{bundle?.model.previewUrl ? <img src={bundle.model.previewUrl} alt="" /> : <Icon name="person" size={16} />}</span>
          <span class="cp-switch__name">{name}</span>
          <span class="cp-switch__hint">{t("切换角色")}</span>
          <Icon name="back" size={12} weight={2.4} />
        </button>
        {switcher && characters && (
          <div class="cp-menu" role="menu" aria-label={t("角色")}>
            {characters.map((item) => (
              <button type="button" role="menuitemradio" aria-checked={item.current} class="cp-menu__item" onClick={() => void switchTo(item.id)}>
                <span class="cp-menu__initial">{(item.name || "?").slice(0, 1)}</span>
                <span class="cp-menu__text">
                  <b>{item.name || t("没起名字的角色")}</b>
                  <small>{item.current ? t("当前") : item.lastActiveAt ? t("上次聊天 {{p0}}", { p0: new Date(item.lastActiveAt).toLocaleDateString(locale) }) : t("还没聊过")}</small>
                </span>
                {item.current && <span class="cp-menu__check"><Icon name="check" size={11} weight={3.2} /></span>}
              </button>
            ))}
            <div class="cp-menu__divider" />
            <button type="button" role="menuitem" class="cp-menu__action" onClick={() => { setSwitcher(false); setCreating(true); }}>
              <Icon name="plus" size={15} />{t("新建角色")}<small>{t("起名字 → 定性格")}</small>
            </button>
          </div>
        )}
        <button type="button" class="cp-stage__character" title={t("换形象、调表情和装扮")} onClick={openLook}>
          <CharacterStage bundle={bundle} emotion="neutral" ready={bundle !== null} fallback={t("正在请 TA 出来…")} />
          <span class="cp-stage__edit"><Icon name="face" size={13} weight={2.4} />{t("换形象、调表情和装扮")}</span>
        </button>
        <div class="cp-bubble">{t(tabInfo.tip)}</div>
        <div class="cp-plate">
          <b>{name}</b>
          <span>{[summary, days ? t("陪你 {{p0}} 天", { p0: days }) : ""].filter(Boolean).join(" · ")}</span>
        </div>
      </section>

      <section class="sn-panel cp-panel" aria-labelledby="cp-title">
        <header class="sn-panel__header">
          <div>
            <h1 id="cp-title">{tab === "profile" ? name : t("{{p0}}的{{p1}}", { p0: name, p1: t(tabInfo.label) })}</h1>
            <p>{subtitle(tab)}</p>
          </div>
          <span class="sn-latin">{tabInfo.latin}</span>
        </header>
        <div class="cp-tabs" role="tablist" aria-label={t("角色分类")}>
          {CHARACTER_TABS.map((item) => (
            <button type="button" role="tab" aria-selected={item.id === tab} onClick={() => setTab(item.id)}>{t(item.label)}</button>
          ))}
        </div>
        <div class="sn-panel__body">
          {notice && <Notice tone="warn">{t(notice)}</Notice>}
          {!profile || !draft ? <p class="sn-hint">{t("正在读取角色…")}</p> : (
            <>
              {tab === "profile" && <ProfileTab draft={draft} setDraft={setDraft} effectiveSoul={profile.effectiveSoul} />}
              {tab === "voice" && <VoiceTab characterId={profile.id} name={name} />}
              {tab === "memory" && <MemoryTab characterId={profile.id} name={name} />}
            </>
          )}
        </div>
        <footer class="sn-panel__footer">
          <span class="cp-footer__note">{footerNote(tab)}</span>
          {tab === "profile" && changes > 0 && (
            <>
              <span class="cp-footer__dirty">{t("有 {{count}} 处修改", { count: changes })}</span>
              <PillButton size="sm" variant="secondary" onClick={() => saved && setDraft(saved)}>{t("撤销")}</PillButton>
              <PillButton size="sm" icon="check" disabled={saving} onClick={() => void save()}>{saving ? t("正在保存…") : t("保存")}</PillButton>
            </>
          )}
        </footer>
      </section>

      {creating && <CreateCharacter onDone={async (id) => {
        setCreating(false);
        if (id) await switchTo(id);
      }} />}
    </div>
  );
}

function subtitle(tab: CharacterTab): string {
  return {
    profile: t("名字、唤醒词、人设背景、和你的关系、说话方式"),
    voice: t("TA 开口时用哪个音色"),
    memory: t("TA 记得的关于你的事：可以让 TA 忘掉，但不能改写"),
  }[tab];
}

function footerNote(tab: CharacterTab): string {
  return {
    profile: t("改设定不会影响 TA 的记忆；保存后下一句话就生效"),
    voice: t("选中后下一句话就用新音色"),
    memory: t("记忆只保存在这台电脑上"),
  }[tab];
}

// ---------- 性格 ----------

function ProfileTab({ draft, setDraft, effectiveSoul }: {
  draft: CharacterDraft;
  setDraft: (draft: CharacterDraft) => void;
  effectiveSoul: string;
}) {
  const [adding, setAdding] = useState(false);
  const [word, setWord] = useState("");
  const [wordError, setWordError] = useState(false);
  const [showSoul, setShowSoul] = useState(false);
  const update = (patch: Partial<CharacterDraft>) => setDraft({ ...draft, ...patch });
  const relation = selectedRelation(draft.role);
  const traits = selectedTraits(draft.setting);
  const addWord = () => {
    const error = wakeWordError(word, draft.wakeWords);
    if (error) { setWordError(true); return; }
    update({ wakeWords: [...draft.wakeWords, word.trim()] });
    setWord("");
    setWordError(false);
    setAdding(false);
  };

  return (
    <div class="sn-stack">
      <div class="sn-card sn-card--service">
        <div class="cp-row">
          <label class="cp-row__label" for="cp-name">{t("名字")}</label>
          <input id="cp-name" class="ui-input cp-name" maxLength={64} value={draft.name} onInput={(event) => update({ name: event.currentTarget.value })} />
          <span class="cp-row__label cp-row__label--gap">{t("唤醒词")}</span>
          <div class="cp-words">
            {draft.wakeWords.map((item) => (
              <span class="cp-word">{item}
                <button type="button" aria-label={t("删除唤醒词 {{p0}}", { p0: item })} onClick={() => update({ wakeWords: draft.wakeWords.filter((value) => value !== item) })}>
                  <Icon name="close" size={10} weight={3} />
                </button>
              </span>
            ))}
            {adding ? (
              <span class="ui-chip-input">
                <input aria-label={t("新的唤醒词")} value={word} maxLength={8} placeholder={t("2 到 8 个汉字")} ref={(element) => element?.focus()}
                  onInput={(event) => { setWord(event.currentTarget.value); setWordError(false); }}
                  onKeyDown={(event) => { if (event.key === "Enter") addWord(); if (event.key === "Escape") setAdding(false); }} />
                <button type="button" aria-label={t("添加")} onClick={addWord}><Icon name="check" size={12} weight={3} /></button>
              </span>
            ) : draft.wakeWords.length < MAX_WAKE_WORDS && (
              <button type="button" class="cp-add" onClick={() => setAdding(true)}><Icon name="plus" size={11} weight={3} />{t("添加")}</button>
            )}
          </div>
        </div>
        {wordError ? <Notice>{wakeWordError(word, draft.wakeWords)}</Notice>
          : <span class="sn-hint">{t("喊其中任何一个都能叫醒 TA，最多 {{p0}} 个；2 到 4 个字最容易听准", { p0: MAX_WAKE_WORDS })}</span>}
      </div>

      <div class="sn-card">
        <div class="cp-card-head"><span>{t("人设背景")}</span><TextButton onClick={() => setShowSoul(true)}>{t("看看 TA 实际读到的完整设定")}</TextButton></div>
        <div class="cp-text">
          <textarea aria-label={t("人设背景")} rows={3} maxLength={1000} placeholder={t("TA 是谁、从哪来、经历过什么、喜欢什么")}
            value={draft.background} onInput={(event) => update({ background: event.currentTarget.value })} />
          <small>{draft.background.length}/1000</small>
        </div>
      </div>

      <div class="sn-card">
        <div class="cp-card-head"><span>{t("和你的关系")}</span><small>{t("点一个会填好描述，也可以自己改")}</small></div>
        <div class="cp-chips">
          {RELATIONS.map((item) => <Chip selected={relation === item} onClick={() => update({ role: relationText(item) })}>{t(item)}</Chip>)}
        </div>
        <div class="cp-text cp-text--line">
          <input aria-label={t("和你的关系")} maxLength={200} value={relationShown(draft.role)} onInput={(event) => update({ role: relationStored(event.currentTarget.value) })} />
          <small>{draft.role.length}/200</small>
        </div>
      </div>

      <div class="sn-card">
        <div class="cp-card-head"><span>{t("性格与说话方式")}</span><small>{traitNote(traits, TRAITS)}</small></div>
        <TraitChips traits={traits} onChange={(next) => update({ setting: traitsText(next) })} />
        <div class="cp-text">
          <textarea aria-label={t("性格与说话方式")} rows={2} maxLength={200} value={draft.setting} onInput={(event) => update({ setting: event.currentTarget.value })} />
          <small>{draft.setting.length}/200</small>
        </div>
      </div>

      {showSoul && (
        <Dialog title={t("TA 实际读到的设定")} confirmLabel={t("知道了")} onConfirm={() => setShowSoul(false)} onCancel={() => setShowSoul(false)}>
          <pre class="cp-soul">{effectiveSoul}</pre>
          <p class="sn-hint">{t("这是保存后的版本；没保存的修改不会出现在这里。")}</p>
        </Dialog>
      )}
    </div>
  );
}

// ---------- 声音 ----------

/** 通义 or 本机 for this character only, and the 通义 key right there when it's missing. */
function VoiceService({ voice, name, onSwitch, onKey }: {
  voice: CharacterVoice; name: string;
  onSwitch: (service: "cloud" | "local") => Promise<void>;
  onKey: (apiKey: string) => Promise<string | null>;
}) {
  const cloud = voice.provider !== "local";
  const [askingKey, setAskingKey] = useState(false);
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const services = [
    { id: "cloud" as const, label: t("通义语音"), description: t("更自然，有多种音色，还能复刻；按用量计费") },
    { id: "local" as const, label: t("本机语音"), description: t("免费、不联网，只有一个音色") },
  ];
  const pick = async (id: "cloud" | "local") => {
    setMessage(null);
    if (id === "cloud" && !voice.cloud.ready) { setAskingKey(true); return; }
    setAskingKey(false);
    await onSwitch(id);
  };
  const saveKey = async () => {
    if (key.trim().length < 8) { setMessage("Key 看起来不完整"); return; }
    setBusy(true);
    const error = await onKey(key.trim());
    setBusy(false);
    if (error) setMessage(error);
    else { setKey(""); setAskingKey(false); }
  };
  const showKey = askingKey || (cloud && !voice.cloud.ready);

  return (
    <div class="sn-card">
      <div class="cp-card-head"><span>{t("用哪种语音")}</span><small>{t("换了下一句就生效，只影响{{name}}", { name })}</small></div>
      <div class="cp-services" role="radiogroup" aria-label={t("用哪种语音")}>
        {services.map((item) => {
          const checked = (item.id === "cloud") === cloud;
          return (
            <button type="button" role="radio" aria-checked={checked} class="cp-service" onClick={() => void pick(item.id)}>
              <span class="cp-service__dot" aria-hidden="true" />
              <span class="cp-service__text"><b>{item.label}</b><small>{item.description}</small></span>
            </button>
          );
        })}
      </div>
      {showKey ? (
        <div class="cp-row">
          <label class="cp-row__label" for="cp-voice-key">{t("阿里云百炼 Key")}</label>
          <input id="cp-voice-key" class="ui-input" type="password" autocomplete="off" placeholder={t("粘贴通义语音用的 Key")} value={key}
            onInput={(event) => { setKey(event.currentTarget.value); setMessage(null); }}
            onKeyDown={(event) => { if (event.key === "Enter") void saveKey(); }} />
          <PillButton size="sm" icon="check" disabled={busy || !key.trim()} onClick={() => void saveKey()}>{busy ? t("正在保存…") : t("保存并换上")}</PillButton>
          {askingKey && <TextButton onClick={() => { setAskingKey(false); setMessage(null); }}>{t("取消")}</TextButton>}
          <a class="ui-text-button" href="https://bailian.console.aliyun.com/?tab=model#/api-key" target="_blank" rel="noopener">{t("去阿里云百炼获取 Key")}</a>
        </div>
      ) : cloud && (
        <div class="cp-row">
          <span class="cp-row__label">{t("阿里云百炼 Key")}</span>
          <Badge tone="ok">{t("可用")}</Badge>
          <TextButton onClick={() => setAskingKey(true)}>{t("更换")}</TextButton>
        </div>
      )}
      {message && <Notice>{t(message)}</Notice>}
    </div>
  );
}

function parsedVoice(value: string | null): { model: string; voice: string; kind: string } | null {
  try {
    const parsed = JSON.parse(value ?? "null") as Record<string, unknown> | null;
    return parsed && typeof parsed.model === "string" && typeof parsed.voice === "string" && typeof parsed.kind === "string"
      ? { model: parsed.model, voice: parsed.voice, kind: parsed.kind } : null;
  } catch {
    return null;
  }
}

function VoiceTab({ characterId, name }: { characterId: string; name: string }) {
  const api = window.desktopPet;
  const [voice, setVoice] = useState<CharacterVoice | null>(null);
  const [custom, setCustom] = useState<Array<{ id: string; model: string; ready: boolean }>>([]);
  const [sentence, setSentence] = useState<string | null>(null);
  const sampleSentence = sentence ?? t("今天也辛苦啦，要不要休息一下？");
  const [playing, setPlaying] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const load = () => {
    void api.characters.voice(characterId).then((value) => {
      setVoice(value);
      if (value.provider !== "local") void api.characters.customVoices().then(setCustom).catch(() => setCustom([]));
    }).catch((error) => setNotice(readableError(error)));
  };
  useEffect(load, [characterId]);
  useEffect(() => () => audio.current?.pause(), []);
  if (!voice) return <p class="sn-hint">{notice ? t(notice) : t("正在读取音色…")}</p>;

  const local = voice.provider === "local";
  const model = voice.options.find((option) => option.model === voice.model) ?? voice.options[0];
  const choices = [
    ...(model?.voices ?? []).map((item) => ({ id: item.id, label: item.label, kind: local ? "local" : "preset" })),
    ...custom.filter((item) => item.ready && item.model === model?.model).map((item, index) => ({ id: item.id, label: t("我的音色 {{p0}}", { p0: index + 1 }), kind: "custom" })),
  ];
  const choose = async (id: string, kind: string) => {
    if (!model) return;
    try {
      await api.characters.saveVoice(characterId, { model: model.model, voice: id, kind });
      setVoice({ ...voice, voice: id, kind });
    } catch (error) {
      setNotice(readableError(error));
    }
  };
  /** Switches 通义/本机; the 通义 voice picked before comes back when switching back. */
  const switchService = async (service: "cloud" | "local", current = voice) => {
    const cloudKey = `character-cloud-voice:${characterId}`;
    if (service === "local" && current.provider !== "local" && current.voice) {
      remember(cloudKey, JSON.stringify({ model: current.model, voice: current.voice, kind: current.kind }));
    }
    const option = (service === "local" ? current.local : current.cloud).options[0];
    const first = option?.voices[0];
    if (!option || !first) { setNotice("这种语音现在没有可用的音色"); return; }
    const fallback = { model: option.model, voice: first.id, kind: service === "local" ? "local" : "preset" };
    const previous = service === "cloud" ? parsedVoice(recallValue(cloudKey)) : null;
    try {
      // A remembered voice may be gone (e.g. a deleted clone); then use the first one.
      await api.characters.saveVoice(characterId, previous ?? fallback)
        .catch((error) => previous ? api.characters.saveVoice(characterId, fallback) : Promise.reject(error));
      load();
    } catch (error) {
      setNotice(readableError(error));
    }
  };
  /** Saves the 通义 key, waits until the voice service can use it, then switches to 通义. */
  const saveKey = async (apiKey: string): Promise<string | null> => {
    try {
      const view = await api.getConfiguration();
      const { models, id } = cloudVoiceService(view);
      await api.saveConfiguration(withChanges(view, { models }, { modelApiKeys: { [id]: apiKey } }));
      for (let attempt = 0; attempt < 20; attempt += 1) {
        const next = await api.characters.voice(characterId);
        if (next.cloud.ready) { await switchService("cloud", next); return null; }
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      return "Key 已保存，但语音服务还没准备好，稍后再选一次通义语音";
    } catch (error) {
      return readableError(error);
    }
  };
  const play = async (id: string) => {
    if (!model) return;
    audio.current?.pause();
    if (playing === id) { setPlaying(null); return; }
    setPlaying(id);
    try {
      const wav = await api.characters.previewVoice({ model: model.model, voice: id }, sampleSentence.trim() || t("你好"));
      const element = new Audio(URL.createObjectURL(new Blob([wav], { type: "audio/wav" })));
      audio.current = element;
      element.onended = () => setPlaying(null);
      await element.play();
    } catch (error) {
      setPlaying(null);
      setNotice(readableError(error));
    }
  };

  return (
    <div class="sn-stack">
      <VoiceService voice={voice} name={name} onSwitch={(service) => switchService(service)} onKey={saveKey} />
      {notice && <Notice tone="warn">{t(notice)}</Notice>}
      <div class="sn-card">
        <div class="cp-card-head"><span>{t("挑一个音色")}</span><small>{local ? t("本机语音") : t("通义语音")} · {t("{{count}} 个音色", { count: choices.length })}</small></div>
        <div class="cp-voices">
          {choices.map((item) => (
            <div class="cp-voice" aria-current={item.id === voice.voice ? "true" : undefined}>
              <button type="button" class="cp-voice__pick" aria-pressed={item.id === voice.voice} onClick={() => void choose(item.id, item.kind)}>
                <b>{item.label}</b>{item.kind === "custom" && <small>{t("复刻")}</small>}
              </button>
              <button type="button" class="cp-voice__play" aria-label={t("试听 {{p0}}", { p0: item.label })} onClick={() => void play(item.id)}>
                <Icon name={playing === item.id ? "pause" : "play"} size={11} weight={2.4} />
              </button>
            </div>
          ))}
        </div>
        <div class="cp-row">
          <label class="cp-row__label" for="cp-sample">{t("试听这句")}</label>
          <input id="cp-sample" class="ui-input" value={sampleSentence} maxLength={60} onInput={(event) => setSentence(event.currentTarget.value)} />
        </div>
      </div>
      {!local && model
        ? <CloneVoice targetModel={model.model} onCloned={async (id) => { await choose(id, "custom"); load(); }} />
        : <p class="sn-hint">{t("想用自己录的声音？在上面换成通义语音，就能复刻一个。")}</p>}
    </div>
  );
}

function CloneVoice({ targetModel, onCloned }: { targetModel: string; onCloned: (id: string) => Promise<void> }) {
  const recorder = useRef<WavRecorder | null>(null);
  const [phase, setPhase] = useState<"idle" | "recording" | "uploading">("idle");
  const [seconds, setSeconds] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const stop = async () => {
    const current = recorder.current;
    if (!current) return;
    recorder.current = null;
    const wav = await current.stop();
    if (current.seconds < 8) { setPhase("idle"); setMessage("录得太短了，至少读 10 秒"); return; }
    setPhase("uploading");
    try {
      const id = await window.desktopPet.characters.cloneVoice(wav, targetModel);
      await onCloned(id);
      setMessage("复刻好了，已经换上这个音色");
    } catch (error) {
      setMessage(readableError(error));
    } finally {
      setPhase("idle");
    }
  };
  const start = async () => {
    setMessage(null);
    const next = new WavRecorder();
    try {
      await next.start((value) => {
        setSeconds(value);
        if (value >= 20) void stop();
      });
      recorder.current = next;
      setPhase("recording");
    } catch (error) {
      setMessage(readableError(error));
    }
  };
  useEffect(() => () => recorder.current?.cancel(), []);

  return (
    <div class="sn-card">
      <div class="sn-card__head">
        <span class="sn-card__icon ui-hue-rose"><Icon name="mic" size={17} /></span>
        <span class="sn-card__text">
          <span class="sn-card__title">{t("复刻一个声音")}</span>
          <span class="sn-card__subtitle">{t("朗读下面这段文字 10 到 20 秒，生成专属音色")}</span>
        </span>
        {phase === "recording"
          ? <PillButton size="sm" icon="check" onClick={() => void stop()}>{t("读完了（{{seconds}} 秒）", { seconds: Math.floor(seconds) })}</PillButton>
          : <PillButton size="sm" variant="secondary" disabled={phase === "uploading"} onClick={() => void start()}>{phase === "uploading" ? t("正在生成…") : t("开始录音")}</PillButton>}
      </div>
      <p class="cp-script">{t(CLONE_SCRIPT)}</p>
      {message && <span class="sn-hint">{t(message)}</span>}
    </div>
  );
}

// ---------- 记忆 ----------

function MemoryTab({ characterId, name }: { characterId: string; name: string }) {
  const api = window.desktopPet;
  const [memory, setMemory] = useState<CharacterMemory | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const [facts, setFacts] = useState<EntityFact[] | null>(null);
  const load = () => void api.characters.memory(characterId).then(setMemory).catch((error) => setNotice(readableError(error)));
  useEffect(load, [characterId]);
  const toggleEntity = (id: string) => {
    if (open === id) { setOpen(null); return; }
    setOpen(id);
    setFacts(null);
    void api.characters.entityFacts(characterId, id).then(setFacts).catch((error) => setNotice(readableError(error)));
  };
  if (!memory) return <p class="sn-hint">{notice ? t(notice) : t("正在读取记忆…")}</p>;
  const days = daysSince(memory.createdAt);
  const needle = query.trim();
  const entities = memory.entities.filter((entity) => !needle || entity.aliases.some((alias) => alias.includes(needle)));
  const forgotten = () => { load(); if (open) toggleEntity(open); };

  return (
    <div class="sn-stack">
      {notice && <Notice tone="warn">{t(notice)}</Notice>}
      <div class="cp-stats">
        <div class="sn-card cp-stat"><span>{t("认识你")}</span><b>{days ?? "—"}<small>{t("天")}</small></b></div>
        <div class="sn-card cp-stat"><span>{t("记得的人和事")}</span><b>{memory.entities.length.toLocaleString(getLocale())}<small>{t("个")}</small></b></div>
        <div class="sn-card cp-stat"><span>{t("长期印象")}</span><b>{memory.impressions.length.toLocaleString(getLocale())}<small>{t("条")}</small></b></div>
      </div>
      <div class="cp-memory-columns">
        <div class="sn-card">
          <div class="cp-card-head"><span>{t("共同记忆")}</span><small>{t("关于你、关于 TA、你们之间的长期印象 · 可以忘掉，不能改")}</small></div>
          {memory.impressions.length === 0 ? <p class="sn-hint">{t("还没形成长期印象，多聊聊就有了。")}</p> : (
            <ul class="cp-list">
              {memory.impressions.map((item) => <ForgettableRow characterId={characterId} text={memoryText(item.text, name)} target={{ itemId: item.id }} onForgotten={forgotten} />)}
            </ul>
          )}
          <span class="sn-hint">{t("说得不对？直接告诉 TA，TA 会自己更正")}</span>
        </div>
        <div class="sn-card">
          <div class="cp-card-head"><span>{t("最近在聊")}</span><small>{t("这几天")}</small></div>
          {memory.recent.length === 0 ? <p class="sn-hint">{t("最近还没聊什么。")}</p> : (
            <ul class="cp-list cp-list--plain">{memory.recent.slice(-6).map((item: MemoryItem) => <li>{memoryText(item.text, name)}</li>)}</ul>
          )}
        </div>
      </div>
      <div class="sn-card">
        <div class="cp-card-head">
          <span>{t("记得的人和事")}</span>
          <input class="ui-input cp-search" type="search" aria-label={t("搜索记忆")} placeholder={t("搜一搜 TA 记得什么")} value={query} onInput={(event) => setQuery(event.currentTarget.value)} />
        </div>
        {entities.length === 0 ? <p class="sn-hint">{needle ? t("没找到相关的人和事") : t("还没记住具体的人和事。")}</p> : (
          <ul class="cp-list">
            {entities.map((entity) => (
              <li class="cp-entity">
                <button type="button" class="cp-entity__head" aria-expanded={open === entity.id} onClick={() => toggleEntity(entity.id)}>
                  <b>{entity.aliases[0] ?? t("未命名")}</b>
                  <span>{entity.aliases.slice(1).join(" · ")}</span>
                  <small>{t("{{count}} 条", { count: entity.factCount })}</small>
                </button>
                {open === entity.id && (
                  <ul class="cp-list cp-list--nested">
                    {facts === null ? <li class="sn-hint">{t("正在读取…")}</li> : facts.map((fact) => (
                      <ForgettableRow characterId={characterId} text={memoryText(fact.text, name)} target={{ factId: fact.id }} onForgotten={forgotten} />
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function ForgettableRow({ characterId, text, target, onForgotten }: {
  characterId: string;
  text: string;
  target: ForgetTarget;
  onForgotten: () => void;
}) {
  const [plan, setPlan] = useState<ForgetPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ask = async () => {
    setError(null);
    try {
      setPlan(await window.desktopPet.characters.forgetPreview(characterId, target));
    } catch (failure) {
      setError(readableError(failure));
    }
  };
  const forget = async () => {
    setBusy(true);
    try {
      await window.desktopPet.characters.forget(characterId, target);
      setPlan(null);
      onForgotten();
    } catch (failure) {
      setError(readableError(failure));
    } finally {
      setBusy(false);
    }
  };
  const extra = plan && ("itemId" in target ? plan.itemCount - 1 : plan.itemCount);
  return (
    <li class="cp-forget">
      <span class="cp-forget__row">
        <span>{text}</span>
        <button type="button" class="cp-forget__button" aria-label={t("忘掉：{{p0}}", { p0: text })} title={t("忘掉这条")} aria-pressed={plan !== null} onClick={() => plan ? setPlan(null) : void ask()}>
          <Icon name="close" size={12} weight={2.4} />
        </button>
      </span>
      {plan && (
        <span class="sn-strip cp-forget__confirm">
          <span>{t("会连同 {{count}} 条相关细节一起忘掉", { count: plan.factCount })}{extra ? t("，另外 {{p0}} 条印象也提到了它们，会一起删掉", { p0: extra }) : ""}{t("，找不回来")}</span>
          <PillButton size="sm" variant="secondary" onClick={() => setPlan(null)}>{t("取消")}</PillButton>
          <PillButton size="sm" disabled={busy} onClick={() => void forget()}>{t("忘掉")}</PillButton>
        </span>
      )}
      {error && <Notice>{t(error)}</Notice>}
    </li>
  );
}

// ---------- 新建角色 ----------

function CreateCharacter({ onDone }: { onDone: (id: string | null) => void }) {
  const [name, setName] = useState("");
  const [relation, setRelation] = useState<string>("朋友");
  const [traits, setTraits] = useState<string[]>(["温柔"]);
  const [error, setError] = useState<string | null>(null);
  const create = async () => {
    const trimmed = name.trim();
    if (!trimmed) { setError("给 TA 起个名字吧"); return; }
    try {
      const id = await window.desktopPet.characters.create({
        name: trimmed,
        role: relationText(relation),
        setting: traitsText(traits),
        background: "",
        wakeWords: wakeWordError(trimmed, []) ? undefined : [trimmed],
      });
      onDone(id);
    } catch (failure) {
      setError(readableError(failure));
    }
  };
  return (
    <Dialog title={t("新建角色")} confirmLabel={t("创建并换上")} onConfirm={() => void create()} onCancel={() => onDone(null)}>
      <div class="sn-stack">
        <Field label={t("名字")} hint={t("也会成为第一个唤醒词")}>
          <input class="ui-input" maxLength={64} value={name} onInput={(event) => setName(event.currentTarget.value)} />
        </Field>
        <div class="cp-chips">{RELATIONS.map((item) => <Chip selected={relation === item} onClick={() => setRelation(item)}>{t(item)}</Chip>)}</div>
        <TraitChips traits={traits} onChange={setTraits} />
        <p class="sn-hint">{traitNote(traits, TRAITS)}</p>
        <p class="sn-hint">{t("新角色先穿现在这个形象，之后在「形象」里换。人设背景可以创建后再写。")}</p>
        {error && <Notice>{t(error)}</Notice>}
      </div>
    </Dialog>
  );
}

/** Trait chips that grey out what clashes or overflows and say why on hover. */
function TraitChips({ traits, onChange }: { traits: string[]; onChange: (traits: string[]) => void }) {
  return (
    <div class="cp-chips">
      {TRAITS.map((item) => {
        const blocked = traitBlock(traits, item);
        return (
          <Chip selected={traits.includes(item)} disabled={Boolean(blocked)} title={blocked ?? undefined}
            onClick={() => onChange(toggleTrait(traits, item))}>{t(item)}</Chip>
        );
      })}
    </div>
  );
}
