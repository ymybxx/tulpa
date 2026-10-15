/** Test actual triangles, not the rectangular bounds of overlapping parts. */
export function meshContainsPoint(vertices: ArrayLike<number>, indices: ArrayLike<number>, x: number, y: number): boolean {
  for (let i = 0; i + 2 < indices.length; i += 3) {
    const a = indices[i]! * 2, b = indices[i + 1]! * 2, c = indices[i + 2]! * 2;
    const ax = vertices[a]!, ay = vertices[a + 1]!, bx = vertices[b]!, by = vertices[b + 1]!, cx = vertices[c]!, cy = vertices[c + 1]!;
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (Math.abs(area) < 1e-12) continue;
    const u = ((bx - x) * (cy - y) - (by - y) * (cx - x)) / area;
    const v = ((cx - x) * (ay - y) - (cy - y) * (ax - x)) / area;
    if (u >= -1e-6 && v >= -1e-6 && u + v <= 1 + 1e-6) return true;
  }
  return false;
}
