import type { PetAudioFrame } from "./contracts";

export function decodeAudioFrame(raw: Uint8Array): PetAudioFrame | null {
  if (raw.byteLength < 25 || raw[0] !== 1) return null;
  const bytes = Uint8Array.from(raw);
  const generation = uuidFromBytes(bytes.subarray(1, 17));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const samplePosition = Number(view.getBigUint64(17, true));
  const pcm = bytes.slice(25).buffer;
  return { generation, samplePosition, pcm };
}

function uuidFromBytes(bytes: Uint8Array): string {
  const hex = [...bytes]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20),
  ].join("-");
}
