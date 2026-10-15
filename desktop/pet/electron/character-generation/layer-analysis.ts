interface SlotNode {
  id: string;
  name: string;
  required: boolean;
}

export function validateRequiredSlots(
  slots: SlotNode[],
  artifacts: Record<string, string>,
): string[] {
  return slots.flatMap((slot) => slot.required && !artifacts[slot.id] ? [slot.id] : []);
}
