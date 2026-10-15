#!/usr/bin/env node

import { cp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { PNG } from "pngjs";

import { inspectCubismModel } from "./lib/inspect-cubism-model.mjs";

const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const PET_ROOT = path.resolve(SCRIPT_DIRECTORY, "..");
const source = path.resolve(process.argv[2] ?? "");
const destination = path.resolve(
  process.argv[3] ?? path.join(PET_ROOT, ".local", "master-templates", "epsilon-free-dev-v2"),
);

if (!process.argv[2]) {
  throw new Error(
    "用法: node scripts/prepare-epsilon-master.mjs <模型目录> [输出目录] [model3.json] [母版ID] [母版名称] <full-body|half-body>",
  );
}

const MODEL_ENTRY = process.argv[4] ?? await findModelEntry(source);
const TEMPLATE_ID = process.argv[5] ?? `${path.basename(MODEL_ENTRY, ".model3.json").toLowerCase()}-dev-v1`;
const TEMPLATE_NAME = process.argv[6] ?? `${path.basename(MODEL_ENTRY, ".model3.json")} 开发母版`;
const COVERAGE = process.argv[7];
if (COVERAGE !== "full-body" && COVERAGE !== "half-body") {
  throw new Error("必须明确指定母版覆盖范围 full-body 或 half-body");
}
const CORE_FILE = path.join(PET_ROOT, "public/vendor/live2d/live2dcubismcore.min.js");

await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });
await cp(source, destination, { recursive: true, force: false });

const modelJson = JSON.parse(await readFile(path.join(destination, MODEL_ENTRY), "utf8"));
const references = modelJson.FileReferences ?? {};
const textureFiles = references.Textures ?? [];
const model = await inspectCubismModel(CORE_FILE, path.join(destination, references.Moc));
const displayInfo = references.DisplayInfo
  ? JSON.parse(await readFile(path.join(destination, references.DisplayInfo), "utf8"))
  : { Parts: [] };
const pose = references.Pose
  ? JSON.parse(await readFile(path.join(destination, references.Pose), "utf8"))
  : { Groups: [] };
const hiddenPoseParts = new Set((pose.Groups ?? []).flatMap((group) =>
  Array.isArray(group) ? group.slice(1).map((item) => item.Id) : []
));
const partNames = new Map((displayInfo.Parts ?? []).map((part) => [part.Id, part.Name]));
const atlases = await Promise.all(textureFiles.map(async (file) => {
  const image = PNG.sync.read(await readFile(path.join(destination, file)));
  return { file, width: image.width, height: image.height, image };
}));

const alignmentGuideFile = "alignment-guide.png";
const guideSize = scaledGuideSize(model.canvas.width, model.canvas.height);
await writeFile(
  path.join(destination, alignmentGuideFile),
  PNG.sync.write(renderNeutralGuide(
    model,
    atlases,
    guideSize.width,
    guideSize.height,
    hiddenPoseParts,
  )),
);

await rm(path.join(destination, "masks"), { recursive: true, force: true });
await mkdir(path.join(destination, "masks"), { recursive: true });

const slots = [];
for (const drawable of model.drawables) {
  const atlas = atlases[drawable.texture];
  if (!atlas) throw new Error(`Drawable ${drawable.id} 指向不存在的纹理 ${drawable.texture}`);
  const part = model.parts[drawable.partIndex] ?? { id: "UNBOUND" };
  const canvasBounds = boundsOfPairs(drawable.vertices);
  if (
    canvasBounds.width <= 1e-6 || canvasBounds.height <= 1e-6 ||
    drawable.indices.length < 3
  ) continue;
  const semantic = classifyDrawable(drawable, part.id);
  const neutralVisible = drawable.opacity > 0.001 &&
    !hiddenPoseParts.has(part.id) && !excludedFromGuide(part.id);
  const source = semantic.source === "template"
    ? "template"
    : neutralVisible
      ? "visible"
      : "completion";
  const region = uvRegion(drawable.uvs, atlas.width, atlas.height);
  const mask = rasterizeDrawableMask(drawable, region, atlas.image);
  const safeId = drawable.id.toLowerCase().replaceAll("_", "-");
  const silhouetteMask = `masks/${safeId}.png`;
  await writeFile(path.join(destination, silhouetteMask), PNG.sync.write(mask));
  slots.push({
    id: safeId,
    name: `${partNames.get(part.id) ?? semantic.label} · ${drawable.id}`,
    drawableId: drawable.id,
    partId: part.id,
    group: semantic.group,
    source,
    // The first preview only needs drawables that actually contribute to the
    // neutral render. Hidden/alternate drawables are generated on demand.
    required: source === "visible",
    zIndex: drawable.renderOrder,
    atlas: { texture: drawable.texture, ...region },
    anchor: { x: region.x + region.width / 2, y: region.y + region.height / 2 },
    canvasBounds,
    mesh: {
      vertices: rounded(drawable.vertices),
      uvs: rounded(drawable.uvs),
      indices: drawable.indices,
    },
    silhouetteMask,
    completionPrompt: semantic.prompt,
  });
}

const eyeBlink = groupIds(modelJson, "EyeBlink");
const lipSync = groupIds(modelJson, "LipSync");
await writeFile(path.join(destination, "master.template.json"), `${JSON.stringify({
  schemaVersion: 2,
  id: TEMPLATE_ID,
  name: TEMPLATE_NAME,
  modelEntry: MODEL_ENTRY,
  alignmentGuide: alignmentGuideFile,
  coverage: COVERAGE,
  canvas: model.canvas,
  parameters: { eyeBlink, lipSync },
  textureAtlases: atlases.map(({ file, width, height }) => ({ file, width, height })),
  slots,
}, null, 2)}\n`);

await writeFile(path.join(destination, "pet.manifest.json"), `${JSON.stringify({
  schemaVersion: 3,
  id: TEMPLATE_ID,
  name: TEMPLATE_NAME,
  renderer: "live2d-cubism",
  entry: MODEL_ENTRY,
  display: { scale: 0.95, x: 0.5, y: 0.98 },
  motionAssets: {},
  expressionAssets: {},
  behaviors: {
    states: emptyPools(["standby", "acknowledging", "listening", "thinking", "click"]),
    emotions: emptyPools(["neutral", "angry", "happy", "shy", "sad"]),
  },
  lipsync: {
    mode: "open-form",
    source: "automatic",
    needsConfirmation: false,
    mouthOpenParameter: lipSync[0] ?? "PARAM_MOUTH_OPEN_Y",
    mouthFormParameter: "PARAM_MOUTH_FORM",
    jawParameter: null,
    puckerWidenParameter: null,
    funnelParameter: null,
    vowelParameters: { a: null, e: null, i: null, o: null, u: null },
    maxOpen: 0.7,
  },
}, null, 2)}\n`);

const generatedGroups = Object.fromEntries([...new Set(slots.map((slot) => slot.group))]
  .map((group) => [group, slots.filter((slot) => slot.group === group).length]));
console.log(`${destination}\n${slots.length} 个 Drawable 槽位: ${JSON.stringify(generatedGroups)}`);

function classifyDrawable(drawable, partId) {
  const normalizedPart = partId.replaceAll("_", "").toUpperCase();
  const bounds = boundsOfPairs(drawable.vertices);
  const centerX = bounds.x + bounds.width / 2;
  if (["EFFECT", "CORE", "ROUGH", "INK", "SMOKE", "LIGHT", "HEART", "RABBIT", "WAND", "AURA"]
    .some((value) => normalizedPart.includes(value))) {
    return semantic("template", "template", "母版效果", "keep the original template effect texture");
  }
  if (normalizedPart.includes("FOOT") || normalizedPart.includes("LEG")) {
    const side = centerX < 0 ? "left" : "right";
    return semantic(`leg-${side}`, "completion", `${side === "left" ? "左" : "右"}腿与脚`,
      `complete the ${side} leg, foot and footwear in the exact neutral full-body template pose`);
  }
  if (normalizedPart.includes("ARML")) return semantic("arm-left", "completion", "左臂", "complete the left arm and hand behind any clothing in the exact template pose");
  if (normalizedPart.includes("ARMR")) return semantic("arm-right", "completion", "右臂", "complete the right arm and hand behind any clothing in the exact template pose");
  if (["CLOTHES", "ROBE", "HOODIE"].some((value) => normalizedPart.includes(value))) return semantic("clothing", "completion", "服装", "complete the full garment, including all regions hidden by arms and hair");
  if (normalizedPart.includes("BODY") || normalizedPart.includes("NECK")) return semantic("body", "completion", "身体", "complete the neck and torso skin beneath hair and clothing in the exact template pose");
  if (normalizedPart.includes("HAIRBACK") || normalizedPart.includes("HAIRSIDE")) return semantic("hair", "completion", "遮挡头发", "complete this hair layer from root to tip, including the region hidden behind face and clothing");
  if (normalizedPart.includes("HAIRFRONT")) return semantic("hair", "visible", "前发", "isolate this front hair layer with clean transparent edges");
  if (normalizedPart.includes("MOUTH")) return semantic("face", "completion", "嘴部", "generate the matching upper lip, lower lip, mouth interior, teeth and tongue states for lip sync");
  if (normalizedPart.includes("FACE")) return semantic("face", "completion", "脸部底色", "complete clean face skin beneath hair, eyes, nose and mouth without changing visible skin color");
  if (["EYE", "BROW", "NOSE", "EAR"].some((value) => normalizedPart.includes(value))) {
    return semantic("face", "visible", "五官", "isolate this facial feature with clean transparent edges and preserve the exact character identity");
  }
  if (normalizedPart.includes("HOHO") || normalizedPart.includes("CHEEK")) return semantic("effect", "visible", "面部效果", "isolate the cheek or blush effect with transparent background");
  return semantic("accessory", "visible", "配饰", "isolate this accessory with clean transparent edges");
}

function semantic(group, source, label, prompt) {
  return { group, source, label, prompt };
}

function groupIds(modelJson, name) {
  const group = (modelJson.Groups ?? []).find((item) => item.Target === "Parameter" && item.Name === name);
  return Array.isArray(group?.Ids) ? group.Ids : [];
}

function uvRegion(uvs, width, height) {
  const points = [];
  for (let index = 0; index < uvs.length; index += 2) {
    points.push({ x: uvs[index] * width, y: (1 - uvs[index + 1]) * height });
  }
  const minX = Math.max(0, Math.floor(Math.min(...points.map((point) => point.x))));
  const minY = Math.max(0, Math.floor(Math.min(...points.map((point) => point.y))));
  const maxX = Math.min(width, Math.ceil(Math.max(...points.map((point) => point.x))));
  const maxY = Math.min(height, Math.ceil(Math.max(...points.map((point) => point.y))));
  return { x: minX, y: minY, width: Math.max(1, maxX - minX), height: Math.max(1, maxY - minY) };
}

function boundsOfPairs(values) {
  const xs = [];
  const ys = [];
  for (let index = 0; index < values.length; index += 2) {
    xs.push(values[index]);
    ys.push(values[index + 1]);
  }
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function rasterizeDrawableMask(drawable, region, atlas) {
  const output = new PNG({ width: region.width, height: region.height });
  const points = [];
  for (let index = 0; index < drawable.uvs.length; index += 2) {
    points.push({
      x: drawable.uvs[index] * atlas.width - region.x,
      y: (1 - drawable.uvs[index + 1]) * atlas.height - region.y,
    });
  }
  for (let index = 0; index < drawable.indices.length; index += 3) {
    fillTriangle(output, points[drawable.indices[index]], points[drawable.indices[index + 1]], points[drawable.indices[index + 2]], region, atlas);
  }
  return output;
}

function fillTriangle(output, a, b, c, region, atlas) {
  const minX = Math.max(0, Math.floor(Math.min(a.x, b.x, c.x)));
  const maxX = Math.min(output.width - 1, Math.ceil(Math.max(a.x, b.x, c.x)));
  const minY = Math.max(0, Math.floor(Math.min(a.y, b.y, c.y)));
  const maxY = Math.min(output.height - 1, Math.ceil(Math.max(a.y, b.y, c.y)));
  const area = edge(a, b, c);
  if (Math.abs(area) < 1e-8) return;
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const point = { x: x + 0.5, y: y + 0.5 };
      const w0 = edge(b, c, point) / area;
      const w1 = edge(c, a, point) / area;
      const w2 = edge(a, b, point) / area;
      if (w0 < -1e-5 || w1 < -1e-5 || w2 < -1e-5) continue;
      const atlasOffset = ((region.y + y) * atlas.width + region.x + x) * 4;
      const targetOffset = (y * output.width + x) * 4;
      output.data[targetOffset] = 255;
      output.data[targetOffset + 1] = 255;
      output.data[targetOffset + 2] = 255;
      output.data[targetOffset + 3] = atlas.data[atlasOffset + 3];
    }
  }
}

function edge(a, b, point) {
  return (point.x - a.x) * (b.y - a.y) - (point.y - a.y) * (b.x - a.x);
}

function rounded(values) {
  return values.map((value) => Math.round(value * 1_000_000) / 1_000_000);
}

function emptyPools(keys) {
  return Object.fromEntries(keys.map((key) => [key, []]));
}

function renderNeutralGuide(model, atlases, width, height, hiddenPoseParts) {
  const output = new PNG({ width, height });
  for (let offset = 0; offset < output.data.length; offset += 4) {
    output.data[offset] = 255;
    output.data[offset + 1] = 255;
    output.data[offset + 2] = 255;
    output.data[offset + 3] = 255;
  }
  const drawables = [...model.drawables]
    .filter((drawable) => {
      if (drawable.opacity <= 0.001) return false;
      const partId = model.parts[drawable.partIndex]?.id ?? "";
      return !hiddenPoseParts.has(partId) && !excludedFromGuide(partId);
    })
    .sort((left, right) => left.renderOrder - right.renderOrder);
  for (const drawable of drawables) {
    const atlas = atlases[drawable.texture]?.image;
    if (!atlas) continue;
    const destination = pointPairs(drawable.vertices).map(([x, y]) => ({
      x: ((model.canvas.originX + x * model.canvas.pixelsPerUnit) / model.canvas.width) * width,
      y: ((model.canvas.originY - y * model.canvas.pixelsPerUnit) / model.canvas.height) * height,
    }));
    const source = pointPairs(drawable.uvs).map(([u, v]) => ({
      x: u * atlas.width,
      y: (1 - v) * atlas.height,
    }));
    for (let index = 0; index < drawable.indices.length; index += 3) {
      const a = drawable.indices[index];
      const b = drawable.indices[index + 1];
      const c = drawable.indices[index + 2];
      paintGuideTriangle(
        output,
        atlas,
        [destination[a], destination[b], destination[c]],
        [source[a], source[b], source[c]],
        drawable.opacity,
      );
    }
  }
  return output;
}

function excludedFromGuide(partId) {
  const normalized = partId.replaceAll("_", "").toUpperCase();
  return [
    "RABBIT", "EFFECT", "INK", "SMOKE", "EXPLOSION", "AURA", "LIGHT", "HEART",
    "WAND", "HAT", "COLORSAMPLE",
  ].some((value) => normalized.includes(value));
}

async function findModelEntry(directory) {
  const entries = (await readdir(directory)).filter((file) => file.endsWith(".model3.json"));
  if (entries.length !== 1) {
    throw new Error(`模型目录必须正好包含一个 .model3.json，实际找到 ${entries.length} 个`);
  }
  return entries[0];
}

function scaledGuideSize(width, height) {
  const scale = Math.min(1, 2000 / width, 2000 / height);
  return {
    width: Math.max(512, Math.round(width * scale / 16) * 16),
    height: Math.max(512, Math.round(height * scale / 16) * 16),
  };
}

function paintGuideTriangle(output, atlas, destination, source, opacity) {
  const [a, b, c] = destination;
  const area = edge(a, b, c);
  if (Math.abs(area) < 1e-8) return;
  const minX = Math.max(0, Math.floor(Math.min(a.x, b.x, c.x)));
  const maxX = Math.min(output.width - 1, Math.ceil(Math.max(a.x, b.x, c.x)));
  const minY = Math.max(0, Math.floor(Math.min(a.y, b.y, c.y)));
  const maxY = Math.min(output.height - 1, Math.ceil(Math.max(a.y, b.y, c.y)));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const point = { x: x + 0.5, y: y + 0.5 };
      const weights = [
        edge(b, c, point) / area,
        edge(c, a, point) / area,
        edge(a, b, point) / area,
      ];
      if (weights.some((weight) => weight < -1e-5)) continue;
      const sourceX = source[0].x * weights[0] + source[1].x * weights[1] + source[2].x * weights[2];
      const sourceY = source[0].y * weights[0] + source[1].y * weights[1] + source[2].y * weights[2];
      const sx = Math.max(0, Math.min(atlas.width - 1, Math.round(sourceX)));
      const sy = Math.max(0, Math.min(atlas.height - 1, Math.round(sourceY)));
      const sourceOffset = (sy * atlas.width + sx) * 4;
      const alpha = atlas.data[sourceOffset + 3] / 255 * opacity;
      if (alpha <= 0) continue;
      const targetOffset = (y * output.width + x) * 4;
      output.data[targetOffset] = Math.round(
        atlas.data[sourceOffset] * alpha + output.data[targetOffset] * (1 - alpha),
      );
      output.data[targetOffset + 1] = Math.round(
        atlas.data[sourceOffset + 1] * alpha + output.data[targetOffset + 1] * (1 - alpha),
      );
      output.data[targetOffset + 2] = Math.round(
        atlas.data[sourceOffset + 2] * alpha + output.data[targetOffset + 2] * (1 - alpha),
      );
    }
  }
}

function pointPairs(values) {
  const output = [];
  for (let index = 0; index < values.length; index += 2) {
    output.push([values[index], values[index + 1]]);
  }
  return output;
}
