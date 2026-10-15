import type { TextureEditUnit } from "../shared/contracts";

export interface NamedPiece {
  unit: TextureEditUnit;
  name: string;
}

export interface PieceSection {
  /** The sub-part the model's author put these pieces in, e.g. 「后发下」「双马尾长」. */
  name: string;
  pieces: NamedPiece[];
  /** Pieces the character is not showing right now (another hairstyle, an expression's extra). */
  hidden?: boolean;
}

export const HIDDEN_SECTION = "现在看不到的";

/** Layer names that only say which side or which length. */
const MODIFIER_CHARS = "左右中前后上下长短";

/**
 * The pieces of one part, sorted into the sub-parts the model's author made —
 * hair is cut into dozens of strands for sway, and a flat grid of 「纹理 35」
 * says nothing. Inside a section a piece is named by its side and length
 * (「长 · 左」), numbered when several share that. Biggest sections and
 * pieces come first.
 */
export function pieceSections(
  units: readonly TextureEditUnit[],
  groupName: string,
  shown: (unit: TextureEditUnit) => boolean = () => true,
): PieceSection[] {
  const visible = units.filter(shown);
  const hidden = units.filter((unit) => !shown(unit));
  const sections = group(visible, groupName);
  if (!hidden.length) return sections;
  // What is not on screen goes last, each piece still saying which sub-part it is from.
  const pieces = group(hidden, groupName).flatMap((section) => section.pieces.map((piece) => ({
    unit: piece.unit,
    name: piece.name.startsWith(section.name) ? piece.name : `${section.name} · ${piece.name}`,
  })));
  return [...sections, { name: HIDDEN_SECTION, pieces, hidden: true }];
}

function group(units: readonly TextureEditUnit[], groupName: string): PieceSection[] {
  const sections = new Map<string, Array<{ unit: TextureEditUnit; label: string }>>();
  for (const unit of units) {
    const { base, modifiers } = describe(unit, groupName);
    const list = sections.get(base) ?? [];
    list.push({ unit, label: modifiers.join(" · ") });
    sections.set(base, list);
  }
  const single = sections.size === 1;
  return [...sections].map(([name, members]) => {
    members.sort((left, right) => area(right.unit) - area(left.unit));
    const seen = new Map<string, number>();
    const pieces = members.map(({ unit, label }) => {
      const shared = members.filter((other) => other.label === label).length > 1;
      if (label && !shared) return { unit, name: label };
      const count = (seen.get(label) ?? 0) + 1;
      seen.set(label, count);
      if (label) return { unit, name: `${label} ${count}` };
      const plain = single ? groupName : name;
      return { unit, name: members.length === 1 ? plain : `${plain} ${count}` };
    });
    return { name, pieces };
  }).sort((left, right) => total(right) - total(left));
}

/** The nearest real layer name above the piece, and the side/length words below it. */
function describe(unit: TextureEditUnit, groupName: string): { base: string; modifiers: string[] } {
  // searchText lists the drawable, then each Part from the nearest outward.
  const names = (unit.searchText ?? "").split(" ").filter((word) => /[぀-ヿ一-鿿]/u.test(word));
  const modifiers: string[] = [];
  for (const name of names) {
    if (name === groupName) break;
    if ([...name].every((char) => MODIFIER_CHARS.includes(char))) {
      modifiers.unshift(name);
      continue;
    }
    return { base: name, modifiers };
  }
  return { base: groupName, modifiers };
}

function area(unit: TextureEditUnit): number {
  return unit.modelBounds.width * unit.modelBounds.height;
}

function total(section: PieceSection): number {
  return section.pieces.reduce((sum, piece) => sum + area(piece.unit), 0);
}
