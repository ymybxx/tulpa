import { t } from "../i18n";
import { useEffect, useRef, useState } from "preact/hooks";

import type { ModelRenderBundle, PetEmotion } from "../shared/contracts";
import { loadCubismCore } from "../renderer/cubism-core";
import { characterSnapshot } from "../renderer/preview-snapshot";
import type { Live2DRenderer } from "../renderer/live2d-renderer";
import { CharacterPlatform } from "../ui/SkyStage";

/**
 * Live preview of the chosen 形象, always showing the full body. The renderer is
 * created lazily once runtime assets are ready and reused across model switches.
 */
export function CharacterStage({ bundle, emotion, ready, fallback, onSnapshot }: {
  bundle: ModelRenderBundle | null;
  emotion: PetEmotion;
  ready: boolean;
  fallback: string;
  /** Called once per model that has no preview image yet, with a PNG of the stage. */
  onSnapshot?: (modelId: string, png: ArrayBuffer) => void;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const renderer = useRef<Live2DRenderer | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => () => {
    renderer.current?.destroy();
    renderer.current = null;
  }, []);

  useEffect(() => {
    if (!ready || !bundle || !canvas.current || !viewport.current) return;
    let cancelled = false;
    setShown(false);
    void (async () => {
      try {
        await loadCubismCore();
        if (!renderer.current) {
          const { Live2DRenderer } = await import("../renderer/live2d-renderer");
          if (cancelled) return;
          renderer.current = new Live2DRenderer(canvas.current!, viewport.current!, {
            maxTextureDimension: 2048,
            preserveDrawingBuffer: true,
            onFatalError: (fatal) => setError(fatal.message),
          });
        }
        await renderer.current.loadModel(bundle);
        if (!cancelled) {
          setError(null);
          setShown(true);
        }
        if (!cancelled && onSnapshot && !bundle.model.previewUrl && canvas.current) {
          // Capture the model's default pose (eyes open, no idle motion), not a random idle frame.
          // setEditingPreview draws that pose synchronously, so no frame wait is needed (and rAF
          // would never fire while the window is in the background).
          const live = renderer.current;
          live.setEditingPreview(true);
          try {
            const blob = await characterSnapshot(canvas.current);
            if (blob && !cancelled) onSnapshot(bundle.model.id, await blob.arrayBuffer());
          } finally {
            live.setEditingPreview(false);
          }
        }
      } catch (failure) {
        if (!cancelled) setError(failure instanceof Error ? failure.message : String(failure));
      }
    })();
    return () => { cancelled = true; };
  }, [ready, bundle]);

  useEffect(() => {
    renderer.current?.setEmotion(emotion);
  }, [emotion, shown]);

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const observer = new ResizeObserver(() => renderer.current?.resize());
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div class="onb-character">
      <CharacterPlatform />
      <div class="onb-character__viewport" ref={viewport}>
        <canvas ref={canvas} class={shown ? "shown" : ""} aria-label={t("形象实时预览")} />
      </div>
      {!shown && <div class="onb-character__placeholder">{error ? t(error) : fallback}</div>}
    </div>
  );
}
