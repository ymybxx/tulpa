import { PNG } from "pngjs";

const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const IEND = 0x49454e44;

/** Decode the PNG stream, tolerating trailing file data just as the renderer does. */
export function decodePng(bytes: Buffer): PNG {
  if (bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    let offset = PNG_SIGNATURE.length;
    while (offset + 12 <= bytes.length) {
      const length = bytes.readUInt32BE(offset);
      const end = offset + length + 12;
      if (end > bytes.length) break;
      if (length === 0 && bytes.readUInt32BE(offset + 4) === IEND) {
        // Keep every chunk and its CRC intact; only data after IEND is ignored.
        return PNG.sync.read(bytes.subarray(0, end));
      }
      offset = end;
    }
  }
  // Let pngjs report malformed signatures, chunks, checksums and truncation.
  return PNG.sync.read(bytes);
}
