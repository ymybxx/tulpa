import type { MasterTemplatePartGroup } from "../src/shared/character-generation";
import type { ModelEditPartGroup } from "../src/shared/contracts";

export interface SemanticPart {
  id: string;
  label: string;
  group: ModelEditPartGroup;
  templateGroup: MasterTemplatePartGroup;
}

export interface PartNode {
  id: string;
  name: string;
}

interface Rule {
  id: string;
  label: string;
  group: ModelEditPartGroup;
  template: MasterTemplatePartGroup | "arm" | "leg" | "shoe";
  /** Each entry is a set of English words that must all appear as whole tokens. */
  words?: string[][];
  /** Substrings for Chinese and Japanese names. */
  cjk?: string[];
  /** CJK substrings that veto this rule (e.g. 发光 is not hair). */
  not?: string[];
}

// Order matters: specific accessories and effects come before the body parts
// whose characters they contain (发卡 before 发, 耳环 before 耳, 眼镜 before 眼).
const RULES: Rule[] = [
  { id: "accessory-glasses", label: "眼镜", group: "accessory", template: "accessory", words: [["glass"], ["glasses"], ["megane"]], cjk: ["眼镜", "眼鏡", "メガネ", "めがね"] },
  { id: "accessory-mask", label: "眼罩与面饰", group: "accessory", template: "accessory", words: [["mask"], ["eyepatch"], ["eye", "patch"], ["veil"]], cjk: ["眼罩", "面具", "面饰", "面纱", "眼部羽", "マスク", "眼帯"] },
  { id: "accessory-hairpin", label: "发饰", group: "accessory", template: "accessory", words: [["hairpin"], ["hair", "clip"], ["hair", "pin"], ["hairclip"], ["hair", "ornament"]], cjk: ["发卡", "发夹", "发饰", "发带", "发圈", "头饰", "髪飾", "ヘアピン", "カチューシャ"] },
  { id: "accessory-earring", label: "耳饰", group: "accessory", template: "accessory", words: [["earring"], ["earrings"]], cjk: ["耳环", "耳饰", "耳坠", "耳钉", "イヤリング", "ピアス"] },
  { id: "accessory-hat", label: "帽子", group: "accessory", template: "accessory", words: [["hat"], ["cap"], ["beret"], ["hood"]], cjk: ["帽", "ハット", "キャップ", "フード"] },
  { id: "accessory-ribbon", label: "蝴蝶结与缎带", group: "accessory", template: "accessory", words: [["ribbon"], ["bow"]], cjk: ["蝴蝶结", "缎带", "丝带", "领结", "リボン"] },
  { id: "accessory-necklace", label: "项链与颈饰", group: "accessory", template: "accessory", words: [["necklace"], ["choker"], ["pendant"]], cjk: ["项链", "项圈", "颈饰", "吊坠", "ネックレス", "チョーカー"] },
  { id: "accessory-ears", label: "兽耳", group: "accessory", template: "accessory", words: [["kemomimi"], ["animal", "ear"], ["cat", "ear"], ["fox", "ear"]], cjk: ["兽耳", "猫耳", "狐耳", "兔耳", "ケモミミ", "猫耳"] },
  { id: "accessory-tail", label: "尾巴", group: "accessory", template: "accessory", words: [["tail"]], cjk: ["尾巴", "しっぽ", "尻尾"] },
  { id: "accessory-wing", label: "翅膀", group: "accessory", template: "accessory", words: [["wing"], ["wings"]], cjk: ["翅膀", "翼", "羽根"] },
  { id: "accessory-halo", label: "光环", group: "accessory", template: "accessory", words: [["halo"]], cjk: ["光环", "天使环", "光輪"] },
  { id: "accessory-held", label: "手持物", group: "accessory", template: "accessory", words: [["wand"], ["staff"], ["sword"], ["weapon"], ["umbrella"], ["book"], ["phone"], ["microphone"], ["mic"]], cjk: ["杖", "剑", "剣", "武器", "伞", "書", "手机", "话筒", "麦克风", "マイク"] },
  { id: "effect", label: "特效", group: "effect", template: "effect", words: [["effect"], ["fx"], ["aura"], ["glow"], ["light"], ["smoke"], ["sparkle"], ["explosion"], ["ink"]], cjk: ["特效", "发光", "光效", "光", "エフェクト", "オーラ", "煙", "烟", "爆", "闪", "キラ"], not: ["高光", "光环", "天使环"] },
  { id: "iris", label: "眼珠与高光", group: "iris", template: "face", words: [["eyeball"], ["eye", "ball"], ["iris"], ["pupil"], ["eye", "highlight"], ["highlight"]], cjk: ["眼珠", "眼球", "瞳", "虹膜", "高光", "目玉", "ハイライト"] },
  { id: "brow", label: "眉毛", group: "brow", template: "face", words: [["brow"], ["eyebrow"]], cjk: ["眉"] },
  { id: "eye", label: "眼睛与睫毛", group: "eye", template: "face", words: [["eye"], ["eyes"], ["lash"], ["eyelash"], ["eyelid"]], cjk: ["眼", "睫", "目", "まぶた", "まつ毛"] },
  { id: "mouth", label: "嘴", group: "mouth", template: "face", words: [["mouth"], ["lip"], ["lips"], ["teeth"], ["tooth"], ["tongue"]], cjk: ["嘴", "口", "唇", "牙", "舌"], not: ["袖口", "领口", "口袋", "衣口", "开口"] },
  { id: "hair-front", label: "刘海与前发", group: "hair-front", template: "hair", words: [["hair", "front"], ["front", "hair"], ["hairfront"], ["bang"], ["bangs"], ["ahoge"]], cjk: ["刘海", "劉海", "前发", "前髪", "呆毛", "アホ毛"] },
  { id: "hair-side", label: "侧发", group: "hair-side", template: "hair", words: [["hair", "side"], ["side", "hair"], ["hairside"]], cjk: ["侧发", "側發", "鬓", "横髪", "サイド"] },
  { id: "hair-back", label: "后发", group: "hair-back", template: "hair", words: [["hair", "back"], ["back", "hair"], ["hairback"], ["ponytail"], ["twintail"], ["braid"], ["hair"]], cjk: ["后发", "後髪", "後ろ髪", "马尾", "辫", "頭髪", "头发", "髪", "发"], not: ["发光", "发射", "发卡", "发夹", "发饰", "发带", "发圈"] },
  { id: "face", label: "脸", group: "face", template: "face", words: [["face"], ["nose"], ["ear"], ["ears"], ["cheek"], ["blush"], ["head"], ["skin", "face"]], cjk: ["脸", "臉", "顔", "面部", "鼻", "耳", "腮", "颊", "頬", "红晕", "头部", "頭"] },
  { id: "upper-clothes", label: "上衣与外套", group: "clothing", template: "clothing", words: [["coat"], ["jacket"], ["robe"], ["hoodie"], ["shirt"], ["blouse"], ["sweater"], ["vest"], ["uniform"], ["cape"], ["cloak"], ["sleeve"], ["collar"], ["tie"], ["necktie"]], cjk: ["外套", "上衣", "衬衫", "卫衣", "毛衣", "制服", "水手服", "披风", "披肩", "斗篷", "袖", "领", "ローブ", "パーカー", "コート", "シャツ", "セーラー", "上着"] },
  { id: "lower-clothes", label: "裙子与裤子", group: "clothing", template: "clothing", words: [["skirt"], ["pants"], ["shorts"], ["trousers"]], cjk: ["裙", "裤", "スカート", "ズボン", "パンツ"] },
  { id: "clothes", label: "服装", group: "clothing", template: "clothing", words: [["cloth"], ["clothes"], ["clothing"], ["dress"], ["outfit"], ["costume"]], cjk: ["衣", "服", "装", "ドレス"], not: ["装饰", "衣架"] },
  { id: "shoe", label: "鞋袜", group: "shoe", template: "shoe", words: [["shoe"], ["shoes"], ["boot"], ["boots"], ["sock"], ["socks"], ["stocking"], ["stockings"]], cjk: ["鞋", "靴", "袜", "襪", "ソックス", "タイツ"] },
  { id: "arm", label: "手臂与手", group: "arm", template: "arm", words: [["arm"], ["arms"], ["hand"], ["hands"], ["finger"]], cjk: ["手", "臂", "腕", "指"] },
  { id: "leg", label: "腿", group: "leg", template: "leg", words: [["leg"], ["legs"], ["foot"], ["feet"], ["thigh"]], cjk: ["腿", "脚", "足"] },
  { id: "body", label: "身体", group: "body", template: "body", words: [["body"], ["neck"], ["chest"], ["torso"], ["shoulder"], ["skin"]], cjk: ["身体", "躯干", "胸", "脖", "颈", "首", "肩", "体"] },
];

const CONTAINER_RULES = new Set(["body", "face", "clothes"]);

/** Splits `PartHairFront`, `D_EYE_BALL_00` and `hair_front-L` into lower-case words. */
export function englishTokens(value: string): Set<string> {
  const words = value
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .split(/[^A-Za-z]+/)
    .map((word) => word.toLowerCase())
    .filter(Boolean);
  return new Set(words);
}

function matches(rule: Rule, text: string): boolean {
  const stripped = (rule.not ?? []).reduce((value, word) => value.replaceAll(word, ""), text);
  if (rule.cjk?.some((word) => stripped.includes(word))) return true;
  const tokens = englishTokens(text);
  return rule.words?.some((words) => words.every((word) => tokens.has(word))) ?? false;
}

function templateFor(rule: Rule, side: "left" | "right"): MasterTemplatePartGroup {
  if (rule.template === "arm") return side === "left" ? "arm-left" : "arm-right";
  if (rule.template === "leg") return side === "left" ? "leg-left" : "leg-right";
  if (rule.template === "shoe") return side === "left" ? "shoe-left" : "shoe-right";
  return rule.template;
}

/**
 * Names a drawable's part from, in order: the rigger's UserData tag, the
 * drawable ID, then its Part and every ancestor Part (nearest first). The
 * first level that says something recognizable wins, so an unnamed 「左」 under
 * 「后发」 is still back hair. Nothing recognizable groups by the outermost
 * named Part so the user sees a real name they can rename.
 */
export function classifyPart(options: {
  drawableId: string;
  chain: readonly PartNode[];
  tag?: string | null;
  side: "left" | "right";
}): SemanticPart {
  const levels: Array<{ text: string; ancestor: boolean }> = [
    ...(options.tag ? [{ text: options.tag, ancestor: false }] : []),
    { text: options.drawableId, ancestor: false },
    ...options.chain.map((node, index) => ({ text: `${node.id} ${node.name}`.trim(), ancestor: index > 0 })),
  ];
  for (const { text, ancestor } of levels) {
    // Riggers nest clothes and hair under broad 「身体」/「头部」 folders; a
    // container that far up says too little, so those pieces stay for the
    // picture model instead of all landing in 身体 or 脸.
    const rule = RULES.find((candidate) => !(ancestor && CONTAINER_RULES.has(candidate.id)) && matches(candidate, text));
    if (rule) {
      return { id: rule.id, label: rule.label, group: rule.group, templateGroup: templateFor(rule, options.side) };
    }
  }
  if (options.tag) {
    return { id: `tag-${slug(options.tag)}`, label: options.tag.trim().slice(0, 20), group: "unknown", templateGroup: "accessory" };
  }
  const outer = [...options.chain].reverse().find((node) => node.name.trim() || node.id);
  const label = outer ? (outer.name.trim() || outer.id) : "其他";
  return { id: `other-${slug(outer?.id ?? "unbound")}`, label, group: "unknown", templateGroup: "accessory" };
}

function slug(value: string): string {
  const ascii = value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  if (ascii) return ascii.slice(0, 40);
  let hash = 0;
  for (const char of value) hash = (hash * 31 + char.codePointAt(0)!) >>> 0;
  return hash.toString(36);
}
