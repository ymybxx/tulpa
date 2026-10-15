import { t } from "../i18n";
import { useLocale } from "../i18n/preact";
import { useEffect, useRef, useState } from "preact/hooks";
import type { ModelPackage } from "../shared/contracts";
import { renderLookThumbnail } from "../renderer/look-thumbnail";
import { readableError } from "../onboarding/flow";
import { Notice, PillButton } from "../ui/components";
import { Icon } from "../ui/icons";
import { MainNav } from "../ui/MainNav";
import { SkyStage } from "../ui/SkyStage";

/**
 * 形象: every look the current character can wear, each shown whole. Which
 * one TA wears is marked; 换上 wears another; 编辑 opens the look editor.
 */
export function LooksApp() {
  useLocale();
  const api = window.desktopPet;
  const [models, setModels] = useState<ModelPackage[] | null>(null);
  const [worn, setWorn] = useState<string | null>(null);
  const [character, setCharacter] = useState("TA");
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  // Looks whose card picture is missing or out of date (装扮 changed) get a new
  // one, one at a time; each is tried once per visit.
  const [outdated, setOutdated] = useState<string[]>([]);
  const [drawing, setDrawing] = useState<string | null>(null);
  const tried = useRef(new Set<string>());
  const refresh = () => {
    void Promise.all([api.listModels(), api.getSelectedModel()])
      .then(([list, selected]) => { setModels(list); setWorn(selected?.id ?? null); })
      .catch((error) => { setModels([]); setNotice(readableError(error)); });
    void api.outdatedModelPreviews()
      .then((ids) => setOutdated(ids.filter((id) => !tried.current.has(id))))
      .catch(() => undefined);
  };
  useEffect(() => {
    refresh();
    void api.getCurrentCharacter().then((value) => { if (value) setCharacter(value.name); }).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!models || drawing) return;
    const next = models.find((model) => outdated.includes(model.id) && !tried.current.has(model.id));
    if (!next) return;
    tried.current.add(next.id);
    setDrawing(next.id);
    void (async () => {
      try {
        const png = await renderLookThumbnail(await api.getModelRenderBundle(next.id));
        const saved = png ? await api.saveModelPreview(next.id, png) : null;
        if (saved) setModels((list) => list?.map((model) => model.id === saved.id ? saved : model) ?? list);
      } catch (error) {
        console.warn(`形象「${next.name}」的预览图没生成出来:`, error);
      } finally {
        setDrawing(null);
      }
    })();
  }, [models, outdated, drawing]);

  const wear = async (id: string) => {
    setBusy(id);
    setNotice(null);
    try {
      await api.selectModel(id);
      setWorn(id);
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      setBusy(null);
    }
  };
  const edit = (id: string) => {
    window.dispatchEvent(new CustomEvent("look-editor:edit", { detail: id }));
    api.showMainView("models");
  };
  const importModel = async () => {
    setNotice(null);
    try {
      const result = await api.importModel("any");
      if (!result) return;
      if (!result.model) setNotice(result.report.errors[0]?.message ?? "这个模型没法导入");
      refresh();
    } catch (error) {
      setNotice(readableError(error));
    }
  };

  // The worn look first, then the others as they were added.
  const ordered = models ? [...models].sort((a, b) => Number(b.id === worn) - Number(a.id === worn)) : [];
  return (
    <div class="sn lk">
      <SkyStage />
      <MainNav current="look" />
      <section class="sn-panel lk-panel" aria-labelledby="lk-title">
        <header class="sn-panel__header">
          <div>
            <h1 id="lk-title">{t("{{name}}的形象", { name: character })}</h1>
            <p>{t("可以放好几套，TA 穿哪套，桌面上就是哪套；每套的表情、动作和装扮都能单独调")}</p>
          </div>
          <span class="sn-latin">LOOKS</span>
        </header>
        <div class="sn-panel__body">
          {notice && <Notice>{t(notice)}</Notice>}
          {models === null ? <p class="sn-hint">{t("正在读取形象…")}</p> : (
            <div class="lk-grid">
              {ordered.map((model) => {
                const wearing = model.id === worn;
                return (
                  <article class={`lk-card${wearing ? " lk-card--worn" : ""}`} key={model.id}>
                    <button type="button" class="lk-card__preview" onClick={() => edit(model.id)} title={t("编辑「{{p0}}」", { p0: model.name })}>
                      {model.previewUrl ? <img src={model.previewUrl} alt={model.name} />
                        : drawing === model.id ? <span class="lk-card__drawing">{t("正在生成预览…")}</span>
                        : <Icon name="person" size={40} />}
                      {wearing && <span class="lk-card__badge"><Icon name="check" size={11} weight={3} />{t("正在穿")}</span>}
                    </button>
                    <div class="lk-card__foot">
                      <b class="lk-card__name">{model.name}</b>
                      <div class="lk-card__actions">
                        {!wearing && (
                          <PillButton size="sm" disabled={busy !== null} onClick={() => void wear(model.id)}>
                            {busy === model.id ? t("正在换…") : t("换上")}
                          </PillButton>
                        )}
                        <PillButton size="sm" variant="secondary" icon="sliders" onClick={() => edit(model.id)}>{t("编辑")}</PillButton>
                      </div>
                    </div>
                  </article>
                );
              })}
              {/* The whole card is the button: people click the plus, not a line of small text. */}
              <button type="button" class="lk-card lk-card--add" onClick={() => void importModel()}>
                <span class="lk-add__icon"><Icon name="plus" size={22} weight={2.6} /></span>
                <b>{t("添加形象")}</b>
                <small>{t("选一个 Live2D 模型：整个文件夹，或打包好的 .zip")}</small>
              </button>
            </div>
          )}
        </div>
        <footer class="sn-panel__footer"><Icon name="lock" size={12} /><span>{t("换上后桌面立即更新；换形象不影响 TA 的名字、性格、声音和记忆")}</span></footer>
      </section>
    </div>
  );
}
