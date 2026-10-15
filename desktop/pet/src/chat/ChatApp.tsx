import { t, getLocale } from "../i18n";
import { useLocale } from "../i18n/preact";
import { messageText, type UiMessage } from "../i18n/message";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";

import type { ChatAttachment, ChatMessage } from "../shared/chat";
import {
  MAX_ATTACHMENTS, contentWithAttachments, contentWithoutAttachments, readAttachment, type PendingAttachment,
} from "../shared/chat-attachments";
import type { CharacterSummary, ModelRenderBundle, PetState } from "../shared/contracts";
import { PET_STATES } from "../shared/contracts";
import { PetChatProgress, terminalProgress, type ChatProgress } from "../shared/pet-chat-progress";
import { needsSettings, replyFailureNote, settingsSectionFor } from "../shared/service-failures";
import type { SenseKey, SensesSnapshot, SenseView } from "../shared/pet-senses";
import { readableError } from "../onboarding/flow";
import { CharacterStage } from "../onboarding/CharacterStage";
import { Notice, TextButton } from "../ui/components";
import { Icon, type IconName } from "../ui/icons";
import { MainNav } from "../ui/MainNav";
import { SkyStage } from "../ui/SkyStage";
import { mergeMessages, newConversationLabel, originLabel, snippet, stateHint, stateLabel, STARTERS, timelineRows } from "./model";

/** How often the page looks for new messages while it is on screen. */
const POLL_MS = 1500;
const HEARING_FADE_MS = 6000;
const HEARING_SHOWN = 80;
const SENSE_ICONS: Record<SenseKey, IconName> = { screen: "eye", audio: "sound", microphone: "mic", inner: "spark" };
const SENSE_LABELS: Record<SenseKey, string> = { screen: "屏幕", audio: "电脑声音", microphone: "麦克风", inner: "内心" };

interface Pending { id: string; content: string; sentAt: number; attachments: PendingAttachment[] }

/**
 * 主窗口 · 聊天: the same conversation as the pet's 「聊两句」. The pet speaks
 * the replies; this page shows them as text, following along while TA talks.
 */
export function ChatApp() {
  const locale = useLocale();
  const api = window.desktopPet;
  const [characters, setCharacters] = useState<CharacterSummary[] | null>(null);
  const [wakeWords, setWakeWords] = useState<string[]>([]);
  const [bundle, setBundle] = useState<ModelRenderBundle | null>(null);
  const [switcher, setSwitcher] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[] | null>(null);
  const [hasBefore, setHasBefore] = useState(false);
  const [atLatest, setAtLatest] = useState(true);
  const [focus, setFocus] = useState<string | null>(null);
  const [pending, setPending] = useState<Pending[]>([]);
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const [imagesAllowed, setImagesAllowed] = useState(true);
  const [dragging, setDragging] = useState(false);
  const filePicker = useRef<HTMLInputElement>(null);
  const [notice, setNotice] = useState<UiMessage | null>(null);
  const [petState, setPetState] = useState<PetState | null>(null);
  // What is being heard right now, before it is sent as a message.
  const [hearing, setHearing] = useState("");
  const [micStarting, setMicStarting] = useState(false);
  const [micNote, setMicNote] = useState<UiMessage>("");
  const [progress, setProgress] = useState<ChatProgress | null>(null);
  const [senses, setSenses] = useState<SensesSnapshot | null>(null);
  const [searching, setSearching] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ChatMessage[] | null>(null);
  // 「新对话」 just started and nothing has been said in it yet.
  const [fresh, setFresh] = useState<{ sessionId: string; at: Date } | null>(null);
  const [starting, setStarting] = useState(false);
  const list = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const loadingOlder = useRef(false);
  const fishId = useRef<string | null>(null);
  const tracker = useRef(new PetChatProgress());
  const current = characters?.find((item) => item.current) ?? null;
  const name = current?.name || "TA";

  const loadCharacters = async () => {
    try {
      const all = await api.characters.list();
      setCharacters(all);
      const active = all.find((item) => item.current);
      if (active) setWakeWords((await api.characters.profile(active.id)).wakeWords);
    } catch { window.setTimeout(() => void loadCharacters(), 3000); }
  };
  const loadBundle = () => void api.getSelectedModelRenderBundle().then(setBundle).catch(() => undefined);

  const refresh = async () => {
    try {
      const page = await api.chat.page();
      if (fishId.current !== page.fishId) {
        // Another character: a different conversation.
        fishId.current = page.fishId;
        tracker.current.reset();
        setProgress(null);
        setMessages(page.items);
        setHasBefore(page.hasBefore);
        return;
      }
      setMessages((old) => mergeMessages(old ?? [], page.items));
      setHasBefore((old) => old || page.hasBefore);
      // What was sent shows up as a real message; drop its stand-in.
      setPending((old) => old.filter((item) => !page.items.some((message) => message.role === "user"
        && message.content === item.content && Date.parse(message.createdAt) >= item.sentAt - 60_000)
        && Date.now() - item.sentAt < 20_000));
      setNotice((old) => old === "暂时连不上 TA，正在重试…" ? null : old);
    } catch {
      setNotice("暂时连不上 TA，正在重试…");
    }
  };

  useEffect(() => {
    void loadCharacters();
    loadBundle();
    void refresh();
    void api.chat.senses().then(setSenses).catch(() => undefined);
    void api.chat.petState().then(acceptPetEvent).catch(() => undefined);
    const stopPet = api.chat.onPetEvent(acceptPetEvent);
    const stopSenses = api.chat.onSenses(setSenses);
    const stopNew = api.chat.onNewConversation((sessionId) => setFresh({ sessionId, at: new Date() }));
    window.addEventListener("focus", loadBundle);
    return () => { stopPet(); stopSenses(); stopNew(); window.removeEventListener("focus", loadBundle); };
  }, []);

  useEffect(() => {
    if (!atLatest) return;
    const timer = window.setInterval(() => { if (document.visibilityState === "visible") void refresh(); }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [atLatest]);

  function acceptPetEvent(value: unknown) {
    const event = value as { type?: string; payload?: { state?: unknown; text?: unknown } } | null;
    if (event?.type === "pet.state") {
      const state = PET_STATES.find((item) => item === event.payload?.state) ?? null;
      if (state) setPetState(state);
      if (state !== "listening" && state !== "acknowledging") setHearing("");
      return;
    }
    if (event?.type === "pet.voice.hearing") {
      setHearing(typeof event.payload?.text === "string" ? event.payload.text : "");
      return;
    }
    const next = tracker.current.accept(value, fishId.current ?? undefined);
    if (next) setProgress(next);
  }

  // A sentence that was dropped (too quiet, only noise) never becomes a message.
  useEffect(() => {
    if (!micNote) return;
    const timer = window.setTimeout(() => setMicNote(""), 12_000);
    return () => window.clearTimeout(timer);
  }, [micNote]);

  useEffect(() => {
    if (!hearing) return;
    const timer = window.setTimeout(() => setHearing(""), HEARING_FADE_MS);
    return () => window.clearTimeout(timer);
  }, [hearing]);

  // Keep the newest message in view unless the person scrolled up to read.
  useEffect(() => {
    const element = list.current;
    if (element && stickToBottom.current && atLatest) element.scrollTop = element.scrollHeight;
  }, [messages, pending, progress, petState, atLatest]);

  // The list can change size without new messages: the input box shrinks
  // after sending, a notice appears, a bubble finishes rendering. Stay on the
  // newest line through those too, unless the person scrolled up.
  useEffect(() => {
    const element = list.current;
    if (!element) return;
    const pin = () => { if (stickToBottom.current && !searching) element.scrollTop = element.scrollHeight; };
    const sizes = new ResizeObserver(pin);
    sizes.observe(element);
    const watch = () => { for (const child of element.children) sizes.observe(child); };
    watch();
    const changes = new MutationObserver(() => { watch(); pin(); });
    changes.observe(element, { childList: true });
    return () => { sizes.disconnect(); changes.disconnect(); };
  }, [searching]);

  useEffect(() => {
    if (!focus || !list.current) return;
    list.current.querySelector(`[data-message="${focus}"]`)?.scrollIntoView({ block: "center" });
  }, [focus, messages]);

  const onScroll = async () => {
    const element = list.current;
    if (!element) return;
    stickToBottom.current = element.scrollHeight - element.scrollTop - element.clientHeight < 60;
    if (element.scrollTop > 40 || !hasBefore || loadingOlder.current || !messages?.length) return;
    loadingOlder.current = true;
    const height = element.scrollHeight;
    try {
      const page = await api.chat.page(messages[0]!.sequence);
      setMessages((old) => mergeMessages(old ?? [], page.items));
      setHasBefore(page.hasBefore);
      requestAnimationFrame(() => { element.scrollTop += element.scrollHeight - height; });
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      loadingOlder.current = false;
    }
  };

  useEffect(() => {
    // A file let go outside the chat box would replace the page with the file itself.
    const hold = (event: DragEvent) => event.preventDefault();
    window.addEventListener("dragover", hold);
    window.addEventListener("drop", hold);
    return () => { window.removeEventListener("dragover", hold); window.removeEventListener("drop", hold); };
  }, []);
  useEffect(() => {
    // Only a model known to refuse pictures keeps them out; unknown ones may try.
    void api.getModelImageInput().then((input) => setImagesAllowed(input?.chat !== "unsupported")).catch(() => undefined);
  }, []);

  const addFiles = async (files: Iterable<File>) => {
    const chosen = [...files];
    for (const file of chosen) {
      const read = await readAttachment(file, { imagesAllowed, locale: getLocale() });
      if ("error" in read) { setNotice(read.error); continue; }
      let full = false;
      setAttachments((old) => {
        full = old.length >= MAX_ATTACHMENTS;
        return full ? old : [...old, read];
      });
      if (full) { setNotice({ key: "一次最多发 {{p0}} 个附件", values: { p0: MAX_ATTACHMENTS } }); break; }
    }
  };

  /**
   * The mic button is 麦克风收音 itself: off, a press turns it on (saying so)
   * and TA starts listening; on, TA starts listening as if called by name;
   * while TA listens, a press stops it.
   */
  const talk = async () => {
    if (micStarting) return;
    try {
      if (listening) {
        await api.chat.stopListening();
        return;
      }
      if (micBlocked) {
        await api.chat.senseAction({ type: "fix", kind: "microphone" });
        return;
      }
      if (micOff) {
        setMicStarting(true);
        await api.chat.senseAction({ type: "toggle", key: "microphone" });
        setMicNote({ key: "麦克风收音打开了：之后喊「{{p0}}」也能直接说。想关的话点左下「TA 现在能」里的麦克风", values: { p0: wakeWords[0] ?? name } });
      }
      // Just turned on, the voice service can take a moment to take it.
      for (let attempt = 0; ; attempt += 1) {
        try {
          await api.chat.listen();
          return;
        } catch (error) {
          if (!micOff || attempt >= 10) throw error;
          await new Promise((resolve) => setTimeout(resolve, 800));
        }
      }
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      setMicStarting(false);
    }
  };

  const send = async (text = draft) => {
    const typed = text.trim();
    const attached = text === draft ? attachments : [];
    if (!typed && !attached.length) return;
    const content = contentWithAttachments(typed, attached.map((item) => item.upload));
    const item: Pending = { id: crypto.randomUUID(), content, sentAt: Date.now(), attachments: attached };
    const restore = () => {
      setPending((old) => old.filter((entry) => entry.id !== item.id));
      setDraft((old) => old || typed);
      if (attached.length) setAttachments((old) => old.length ? old : attached);
    };
    setPending((old) => [...old, item]);
    setDraft("");
    setAttachments([]);
    stickToBottom.current = true;
    if (!atLatest) await backToLatest();
    try {
      const ids: string[] = [];
      for (const entry of attached) {
        const uploaded = await api.chat.attach(entry.upload);
        if (!uploaded.ok) { restore(); setNotice(uploaded.error); return; }
        ids.push(uploaded.id);
      }
      const result = await api.chat.send(item.id, content, ids);
      if (!result.ok) {
        restore();
        setNotice(result.error ?? "没发出去，再试一次");
        return;
      }
      setNotice(null);
      void refresh();
    } catch (error) {
      restore();
      setNotice(readableError(error));
    }
  };

  const backToLatest = async () => {
    setFocus(null);
    setAtLatest(true);
    stickToBottom.current = true;
    const page = await api.chat.page();
    setMessages(page.items);
    setHasBefore(page.hasBefore);
  };

  const jumpTo = async (hit: ChatMessage) => {
    try {
      const page = await api.chat.page(hit.sequence + 1);
      setAtLatest(false);
      setMessages(page.items);
      setHasBefore(page.hasBefore);
      setFocus(hit.id);
      setSearching(false);
      setResults(null);
    } catch (error) {
      setNotice(readableError(error));
    }
  };

  useEffect(() => {
    if (!searching) return;
    const text = query.trim();
    if (!text) { setResults(null); return; }
    const timer = window.setTimeout(() => {
      void api.chat.search(text).then(setResults).catch((error) => setNotice(readableError(error)));
    }, 300);
    return () => window.clearTimeout(timer);
  }, [query, searching]);

  const switchTo = async (id: string) => {
    setSwitcher(false);
    try {
      await api.characters.bind(id);
      await loadCharacters();
      loadBundle();
      await refresh();
    } catch (error) {
      setNotice(readableError(error));
    }
  };

  const microphone = senses?.senses.find((item) => item.key === "microphone") ?? null;
  const micBlocked = microphone?.status === "noperm" || microphone?.status === "waiting";
  const micOff = microphone ? !microphone.on : false;
  const hears = microphone ? microphone.on && !micBlocked : true;
  const listening = hears && (petState === "listening" || petState === "acknowledging");
  const shownState = !hears && (petState === "listening" || petState === "acknowledging") ? null : petState;
  const speakingId = progress && !terminalProgress(progress.status) ? progress.messageId : null;
  const rows = useMemo(() => timelineRows(messages ?? [], new Date()), [messages, locale]);
  const freshShown = fresh !== null && !(messages ?? []).some((message) => message.sessionId === fresh.sessionId);
  const startNew = async () => {
    setStarting(true);
    try {
      const result = await api.chat.newConversation();
      if (!result.ok) setNotice(result.error ?? "没能开始新对话，请再试一次");
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      setStarting(false);
    }
  };
  const last = messages?.at(-1);
  const waitingForReply = petState === "thinking" && (!last || last.role === "user" || pending.length > 0);
  const calls = wakeWords.length ? wakeWords.slice(0, 2).map((word) => `「${word}」`).join("") : `「${name}」`;

  return (
    <div class="sn ch">
      <SkyStage />
      <MainNav current="chat" />

      <section class="ch-stage" aria-label={t("角色舞台")}>
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
          </div>
        )}
        {/* Only while TA is doing something; at rest it would just sit on the character. */}
        {(listening || petState === "speaking" || petState === "thinking") && (
          <div class="ch-state" role="status" title={stateHint(shownState, !senses?.companionPaused)}>
            <span><Icon name={listening ? "ear" : petState === "speaking" ? "sound" : "spark"} size={11} weight={2.4} /></span>
            {stateLabel(shownState, !senses?.companionPaused)}
          </div>
        )}
        {/* Asleep TA looks like idle TA: say what wakes TA. */}
        {petState === "standby" && senses?.sleeping && (
          <div class="ch-state ch-state--asleep" role="status">
            <span><Icon name="moon" size={11} weight={2.4} /></span>
            {t("睡着了 · 喊“{{word}}”或点麦克风", { word: senses.sleeping.wakeWord })}
          </div>
        )}
        <div class="ch-stage__character">
          <CharacterStage bundle={bundle} emotion="neutral" ready={bundle !== null} fallback={t("正在请 TA 出来…")} />
        </div>
        <div class="ch-senses">
          <span class="ch-senses__head"><b>{t("TA 现在能")}</b><TextButton onClick={() => void api.chat.senseAction({ type: "settings" })}>{t("调整")}</TextButton></span>
          <div class="ch-senses__grid">
            {(senses?.senses ?? []).map((sense) => <SenseButton sense={sense} onClick={() => void api.chat.senseAction(
              sense.status === "noperm" && sense.fix ? { type: "fix", kind: sense.fix } : { type: "toggle", key: sense.key })} />)}
          </div>
        </div>
      </section>

      <section class="sn-panel ch-panel" aria-labelledby="ch-title">
        <header class="sn-panel__header ch-header">
          <div>
            <h1 id="ch-title">{t("和{{name}}聊天", { name })}</h1>
            <p>{t("和桌面上「聊两句」是同一段对话 · 打字，或者喊{{calls}}", { calls })}</p>
          </div>
          {searching ? (
            <div class="ch-search">
              <Icon name="search" size={14} weight={2.2} />
              <label for="ch-search" class="ch-hidden">{t("搜索聊天记录")}</label>
              <input id="ch-search" type="search" autoFocus value={query} placeholder={t("搜一搜以前说过的话")}
                onInput={(event) => setQuery(event.currentTarget.value)}
                onKeyDown={(event) => { if (event.key === "Escape") { setSearching(false); setQuery(""); setResults(null); } }} />
              {results && <small>{t("找到 {{count}} 条", { count: results.length })}</small>}
              <button type="button" aria-label={t("关闭搜索")} onClick={() => { setSearching(false); setQuery(""); setResults(null); }}><Icon name="close" size={11} weight={3} /></button>
            </div>
          ) : (
            <div class="ch-header__actions">
              <button type="button" class="ch-new" disabled={starting} title={t("从头开始聊：刚才的话不再接着说，TA 记得的事不会忘")}
                onClick={() => void startNew()}><Icon name="plus" size={12} weight={2.6} />{t("新对话")}</button>
              <button type="button" class="ch-icon-button" aria-label={t("搜索聊天记录")} title={t("搜索聊天记录")} onClick={() => setSearching(true)}><Icon name="search" size={15} weight={2.2} /></button>
            </div>
          )}
        </header>

        <div class="ch-list" ref={list} onScroll={() => void onScroll()}>
          {searching && results ? (
            <div class="ch-results">
              {results.length === 0 && <p class="sn-hint">{t("没找到含有「{{query}}」的话", { query: query.trim() })}</p>}
              {results.map((hit) => {
                const part = snippet(hit.content, query.trim());
                return (
                  <button type="button" class="ch-hit" onClick={() => void jumpTo(hit)}>
                    <span class="ch-hit__meta"><b>{hit.role === "user" ? t("你") : name}</b>{new Date(hit.createdAt).toLocaleString(getLocale(), { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}<em>{t("跳到这里")}</em></span>
                    <span>{part.pre}<mark>{part.hit}</mark>{part.post}</span>
                  </button>
                );
              })}
            </div>
          ) : (
            <>
              {hasBefore && <p class="ch-more">{t("往上翻，看更早的聊天")}</p>}
              {messages === null && <p class="sn-hint">{t("正在读取聊天…")}</p>}
              {rows.map((row) => row.kind === "divider"
                ? <div class="ch-divider" key={row.key}><span>{row.label}</span></div>
                : <Message key={row.key} message={row.message} time={row.time} name={name}
                  focused={row.message.id === focus}
                  speaking={row.message.id === speakingId ? progress : null}
                  spoken={progress?.messageId === row.message.id ? progress.visibleContent : ""}
                  onInterrupt={() => void api.chat.interrupt(row.message.id).catch((error) => setNotice(readableError(error)))} />)}
              {freshShown && (
                <div class="ch-divider ch-divider--fresh" key="fresh">
                  <span>{newConversationLabel(fresh.at, new Date())} · {t("TA 记得的事都还在")}</span>
                </div>
              )}
              {pending.map((item) => (
                <div class="ch-row ch-row--user" key={item.id}>
                  <div class="ch-bubble ch-bubble--user">
                    {contentWithoutAttachments(item.content, item.attachments.map((entry) => entry.upload))}
                    {item.attachments.length > 0 && <div class="ch-attachments">
                      {item.attachments.map((entry) => <AttachmentChip kind={entry.upload.kind} name={entry.upload.name} preview={entry.preview} />)}
                    </div>}
                  </div>
                  <span class="ch-meta">{t("发送中…")}</span>
                </div>
              ))}
              {waitingForReply && (
                <div class="ch-row"><div class="ch-bubble ch-thinking"><span aria-hidden="true"><i /><i /><i /></span>{t("{{name}}在想…", { name })}</div></div>
              )}
              {messages?.length === 0 && pending.length === 0 && (
                <div class="ch-starters">
                  <p class="sn-hint">{t("和{{name}}说点什么吧。聊过的事 TA 会记住，只存在这台电脑上。", { name })}</p>
                  <div>{STARTERS.map((text) => <button type="button" onClick={() => void send(t(text))}>{t(text)}</button>)}</div>
                </div>
              )}
            </>
          )}
        </div>

        {!atLatest && !searching && (
          <button type="button" class="ch-latest" onClick={() => void backToLatest()}>{t("回到最新")}<Icon name="chevronDown" size={12} weight={2.4} /></button>
        )}
        {notice && <div class="ch-notice"><Notice tone="warn">{messageText(notice)}</Notice></div>}
        {micNote && (
          <div class="ch-notice">
            <Notice tone="info" action={<TextButton onClick={() => setMicNote("")}>{t("知道了")}</TextButton>}>{messageText(micNote)}</Notice>
          </div>
        )}
        {micBlocked && (
          <div class="ch-banner">
            <Icon name="mic" size={15} />
            <span>{t("麦克风还没开：TA 听不到你说话，但打字聊天不受影响")}</span>
            <button type="button" onClick={() => void api.chat.senseAction({ type: "fix", kind: "microphone" })}>{t("去开启")}</button>
          </div>
        )}
        {listening && (
          <div class="ch-listening" role="status">
            <span class="ch-listening__wave" aria-hidden="true"><i /><i /><i /><i /><i /></span>
            <span class="ch-listening__text">
              {t("{{name}}在听你说 · 说完停一下就会自动发出", { name })}{hearing && <span class="ch-listening__heard">{hearing.length > HEARING_SHOWN ? `…${hearing.slice(-HEARING_SHOWN)}` : hearing}</span>}
            </span>
            <TextButton onClick={() => void api.chat.stopListening().catch((error) => setNotice(readableError(error)))}>{t("不说了")}</TextButton>
          </div>
        )}
        {attachments.length > 0 && (
          <div class="ch-attachments ch-attachments--draft">
            {attachments.map((entry) => (
              <AttachmentChip kind={entry.upload.kind} name={entry.upload.name} preview={entry.preview}
                onRemove={() => setAttachments((old) => old.filter((other) => other.key !== entry.key))} />
            ))}
          </div>
        )}
        <form class={`ch-composer${dragging ? " ch-composer--drop" : ""}`} onSubmit={(event) => { event.preventDefault(); void send(); }}
          onDragOver={(event) => { if (event.dataTransfer?.types.includes("Files")) { event.preventDefault(); setDragging(true); } }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => { event.preventDefault(); setDragging(false); if (event.dataTransfer?.files.length) void addFiles(event.dataTransfer.files); }}>
          <label for="ch-input" class="ch-hidden">{t("给{{name}}发消息", { name })}</label>
          <input ref={filePicker} type="file" multiple class="ch-hidden"
            onChange={(event) => { const files = event.currentTarget.files; if (files) void addFiles(files); event.currentTarget.value = ""; }} />
          <button type="button" class="ch-icon-button" aria-label={t("发图片或文件")} title={t("发图片或文本文件，也可以直接拖进来或粘贴图片")}
            onClick={() => filePicker.current?.click()}>
            <Icon name="clip" size={18} />
          </button>
          <textarea id="ch-input" rows={1} value={draft} placeholder={t("跟{{p0}}说点什么…  回车发送", { p0: name })}
            onPaste={(event) => {
              const files = [...(event.clipboardData?.files ?? [])];
              if (files.length) { event.preventDefault(); void addFiles(files); }
            }}
            onInput={(event) => setDraft(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); void send(); }
            }} />
          <button type="button" class={`ch-icon-button ch-mic${listening || micStarting ? " ch-mic--on" : micOff || micBlocked ? " ch-mic--off" : ""}`}
            aria-pressed={listening}
            aria-label={micBlocked ? t("打开麦克风权限") : listening ? t("停止收听") : micOff ? t("打开麦克风收音，开始说话") : t("让{{p0}}开始听", { p0: name })}
            title={micBlocked ? t("还没给麦克风权限，点一下去打开") : listening ? t("正在听，说完停一下就会发出。再点一下停止收听")
              : micOff ? t("麦克风收音关着。点一下打开，TA 就开始听你说") : t("麦克风收音开着。点一下让{{p0}}开始听，和喊名字一样", { p0: name })}
            onClick={() => void talk()}>
            <Icon name={micOff || micBlocked ? "micOff" : "mic"} size={18} />
          </button>
          <button type="submit" class="ch-send" disabled={!draft.trim() && !attachments.length}><span><Icon name="send" size={14} weight={2.6} /></span>{t("发送")}</button>
        </form>
      </section>
    </div>
  );
}

function SenseButton({ sense, onClick }: { sense: SenseView; onClick: () => void }) {
  const warn = sense.status === "noperm" || sense.status === "waiting";
  return (
    <button type="button" class="ch-sense" aria-pressed={sense.on} title={`${t(sense.title)}: ${t(sense.label)}${sense.detail ? `, ${t(sense.detail)}` : ""}`} onClick={onClick}>
      <Icon name={SENSE_ICONS[sense.key]} size={16} />
      {t(SENSE_LABELS[sense.key])}
      {warn && <span class="ch-sense__warn" aria-label={t("缺少权限")} />}
    </button>
  );
}

function Message({ message, time, name, focused, speaking, spoken, onInterrupt }: {
  message: ChatMessage; time: string; name: string; focused: boolean;
  speaking: ChatProgress | null;
  /** What the pet has said of this message so far; ahead of the stored text right after it speaks. */
  spoken: string;
  onInterrupt: () => void;
}) {
  if (message.role === "user") {
    return (
      <div class="ch-row ch-row--user" data-message={message.id}>
        <div class={`ch-bubble ch-bubble--user${focused ? " ch-bubble--focus" : ""}`}>
          {message.attachments ? contentWithoutAttachments(message.content, message.attachments) : message.content}
          {message.attachments && <div class="ch-attachments">
            {message.attachments.map((entry) => <SentAttachment attachment={entry} />)}
          </div>}
        </div>
        <span class="ch-meta">{message.spoken && <span class="ch-meta__spoken"><Icon name="mic" size={11} weight={2.2} />{t("语音")}</span>}{time}</span>
      </div>
    );
  }
  const text = speaking || spoken.length > message.content.length ? spoken : message.content;
  const thinking = !text && (message.status === "generating" || speaking !== null);
  return (
    <div class="ch-row" data-message={message.id}>
      {message.origin && <span class="ch-reason"><Icon name="spark" size={12} weight={2.2} />{originLabel(message.origin, name)}</span>}
      {(text || thinking || (!speaking && message.unsaid)) && (
        <div class={`ch-bubble${speaking ? " ch-bubble--speaking" : ""}${focused ? " ch-bubble--focus" : ""}`}>
          {thinking ? <span class="ch-thinking"><span aria-hidden="true"><i /><i /><i /></span>{t("{{name}}在想…", { name })}</span> : text}
          {!speaking && message.unsaid && <s class="ch-unsaid" aria-label={t("没说出口：{{p0}}", { p0: message.unsaid })}>{message.unsaid}</s>}
        </div>
      )}
      <span class="ch-meta">
        {time}
        {speaking && (
          <>
            <span class="ch-speaking" aria-hidden="true"><i /><i /><i /><i /></span>
            <b>{t("正在说")}</b>
            <button type="button" onClick={onInterrupt}>{t("打断")}</button>
          </>
        )}
        {!speaking && message.status === "interrupted" && (
          <span class="ch-note">{t("说到这里被你打断了")}</span>
        )}
        {!speaking && (message.status === "failed" || message.failure) && (
          <FailureNote message={message} name={name} said={Boolean(text)} />
        )}
      </span>
    </div>
  );
}

/** Why a reply did not reach the user, and where to fix it when only they can. */
function FailureNote({ message, name, said }: { message: ChatMessage; name: string; said: boolean }) {
  const failure = message.failure ?? null;
  return (
    <span class="ch-note ch-note--warn">
      {replyFailureNote(failure, name, { said }, getLocale())}
      {failure && needsSettings(failure) && (
        <button type="button" class="ch-note__fix" onClick={() => window.desktopPet.openSettings(settingsSectionFor(failure))}>{t("去设置")}</button>
      )}
    </span>
  );
}

function AttachmentChip({ kind, name, preview, onRemove }: {
  kind: "image" | "text"; name: string; preview?: string; onRemove?: () => void;
}) {
  return (
    <span class={`ch-attachment${kind === "image" && preview ? " ch-attachment--image" : ""}`} title={name}>
      {kind === "image" && preview ? <img src={preview} alt={name} /> : <><Icon name="file" size={14} /><span>{name}</span></>}
      {onRemove && <button type="button" class="ch-attachment__remove" aria-label={t("移除 {{p0}}", { p0: name })} onClick={onRemove}>
        <Icon name="close" size={10} weight={2.6} />
      </button>}
    </span>
  );
}

const sentImages = new Map<string, Promise<string | null>>();

/** A picture or file in a sent message; pictures load once and stay cached. */
function SentAttachment({ attachment }: { attachment: ChatAttachment }) {
  const [preview, setPreview] = useState<string | undefined>();
  useEffect(() => {
    if (attachment.kind !== "image") return;
    let current = true;
    if (!sentImages.has(attachment.id)) sentImages.set(attachment.id, window.desktopPet.chat.attachmentImage(attachment.id));
    void sentImages.get(attachment.id)!.then((url) => { if (current && url) setPreview(url); });
    return () => { current = false; };
  }, [attachment.id]);
  return <AttachmentChip kind={attachment.kind} name={attachment.name} preview={preview} />;
}
