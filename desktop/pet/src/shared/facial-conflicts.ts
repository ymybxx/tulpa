/**
 * Which Jev-driven parts cannot show at the same time, read from what each
 * parameter does to the model's drawables rather than from names, so the
 * rules hold for any model.
 *
 * - Parts that switch drawables (a hand pose, crying eyes, a tongue) and touch
 *   the same drawables exclude each other: the stronger answer wins.
 * - While a part that replaces drawables (hides some, shows others) is on,
 *   parts that move what it shows hold still: a walking-arm parameter must
 *   not drag a hand pose's arm around. Moving what it hid is harmless.
 * - Live2D's standard head, body, face and eye parameters are made to combine
 *   with everything and are never held back; overlays that only add drawables
 *   (an anger mark, a blush) replace nothing and hold nothing back.
 */

export interface ParameterEffect {
  shown: number[];
  hidden: number[];
  moved: number[];
}
export type ParameterEffects = Record<string, ParameterEffect>;

/** A part as the rules see it; `nativeId` is the model parameter it drives. */
export interface ConflictPart {
  id: string;
  nativeId: string | null;
  group: string;
  neutral: number;
  min: number;
  max: number;
}

/** Standard groups authored to blend with any pose or expression. */
const BLENDING_GROUPS = new Set(["head", "body", "overall", "eyes", "gaze", "brows", "mouth", "cheek"]);
/** A part counts as on once it is this far from rest, as a share of its range on that side. */
const ACTIVE = 0.1;

export interface FacialConflictPlan {
  /** Pairs of part IDs that never show together. */
  exclusive: Array<[string, string]>;
  /** Part ID → parts held at rest while it is on. */
  holds: Record<string, string[]>;
}

export function facialConflictPlan(parts: readonly ConflictPart[], effects: ParameterEffects | undefined): FacialConflictPlan {
  const plan: FacialConflictPlan = { exclusive: [], holds: {} };
  if (!effects) return plan;
  const governed = parts.flatMap(part => {
    const effect = part.nativeId ? effects[part.nativeId] : undefined;
    if (!effect || BLENDING_GROUPS.has(part.group)) return [];
    const switched = new Set([...effect.shown, ...effect.hidden]);
    return [{ part, switched, shown: new Set(effect.shown), moved: new Set(effect.moved), replaces: effect.shown.length > 0 && effect.hidden.length > 0 }];
  });
  const touches = (drawables: Set<number>, other: Set<number>) => [...drawables].some(index => other.has(index));
  for (const [i, a] of governed.entries()) {
    for (const b of governed.slice(i + 1)) {
      if (a.part.nativeId === b.part.nativeId) continue;
      if (a.switched.size && b.switched.size
        && (touches(a.switched, b.switched) || touches(a.switched, b.moved) || touches(b.switched, a.moved))) {
        plan.exclusive.push([a.part.id, b.part.id]);
      }
    }
  }
  for (const a of governed) {
    if (!a.replaces) continue;
    const held = governed.filter(b => b !== a && !b.switched.size && touches(a.shown, b.moved)).map(b => b.part.id);
    if (held.length) plan.holds[a.part.id] = held;
  }
  return plan;
}

/** Applies the plan to one sentence's values (keyed by part ID); parts that lose go back to rest. */
export function resolveFacialConflicts(values: Readonly<Record<string, number>>, parts: readonly ConflictPart[], plan: FacialConflictPlan): Record<string, number> {
  if (!plan.exclusive.length && !Object.keys(plan.holds).length) return { ...values };
  const byId = new Map(parts.map(part => [part.id, part]));
  const strength = (id: string) => {
    const part = byId.get(id), value = values[id];
    if (!part || value === undefined) return 0;
    const span = value >= part.neutral ? part.max - part.neutral : part.neutral - part.min;
    return span > 0 ? Math.abs(value - part.neutral) / span : 0;
  };
  const result = { ...values };
  const rest = (id: string) => { const part = byId.get(id); if (part && id in result) result[id] = part.neutral; };
  // Strongest first; each part stays only if nothing already kept excludes it.
  const contenders = [...new Set(plan.exclusive.flat())].filter(id => strength(id) >= ACTIVE).sort((a, b) => strength(b) - strength(a));
  const kept: string[] = [];
  for (const id of contenders) {
    if (kept.some(other => plan.exclusive.some(([x, y]) => (x === id && y === other) || (x === other && y === id)))) rest(id);
    else kept.push(id);
  }
  for (const [id, held] of Object.entries(plan.holds)) {
    if (strength(id) >= ACTIVE && result[id] !== byId.get(id)?.neutral) for (const other of held) rest(other);
  }
  return result;
}
