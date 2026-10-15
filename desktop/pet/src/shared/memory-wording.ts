/**
 * Memories are stored in the words the memory model wrote them in ("用户认为…",
 * "助手于…"). Shown to the person they are about, they read as "你" and the
 * character's name; the stored memory never changes.
 */
export function memoryText(text: string, name: string): string {
  return text
    .replace(/该用户|用户/g, "你")
    .replace(/该助手|助手/g, name);
}
