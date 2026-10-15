import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import { decodePng } from "./png-reader";

function fixture(): Buffer {
  const image = new PNG({ width: 2, height: 1 });
  image.data.set([255, 40, 80, 255, 10, 20, 30, 128]);
  return PNG.sync.write(image);
}

describe("imported PNG decoding", () => {
  it("preserves pixels and source bytes when a valid PNG has trailing data", () => {
    const png = fixture();
    const file = Buffer.concat([png, Buffer.from([0x8f, 0x57, 0x86, 0x98]), png]);
    const original = Buffer.from(file);
    expect(() => PNG.sync.read(file)).toThrow("unrecognised content at end of stream");
    const decoded = decodePng(file);
    expect([decoded.width, decoded.height]).toEqual([2, 1]);
    expect(decoded.data).toEqual(PNG.sync.read(png).data);
    expect(file).toEqual(original);
  });

  it("keeps ordinary PNG decoding unchanged", () => {
    const png = fixture();
    expect(decodePng(png).data).toEqual(PNG.sync.read(png).data);
  });

  it("still rejects damaged CRCs even when trailing data is present", () => {
    const damaged = Buffer.concat([fixture(), Buffer.from("extra")]);
    damaged[29] = damaged[29]! ^ 1; // IHDR checksum
    expect(() => decodePng(damaged)).toThrow();
  });

  it("rejects truncation and invalid signatures", () => {
    const png = fixture();
    expect(() => decodePng(png.subarray(0, png.length - 5))).toThrow();
    expect(() => decodePng(Buffer.from("not a PNG"))).toThrow();
  });
});
