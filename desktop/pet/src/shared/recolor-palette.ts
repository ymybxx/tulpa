/** Ready colors for 换色; any other color comes from the color picker. */
export const RECOLOR_PALETTE: ReadonlyArray<{ name: string; hex: string }> = [
  { name: "樱粉", hex: "#F4A7C0" },
  { name: "玫红", hex: "#D6336C" },
  { name: "正红", hex: "#C8323C" },
  { name: "橘橙", hex: "#F08C3C" },
  { name: "奶黄", hex: "#F5D98B" },
  { name: "薄荷", hex: "#8FD3B6" },
  { name: "墨绿", hex: "#2F6B4F" },
  { name: "天蓝", hex: "#7CC4F0" },
  { name: "藏青", hex: "#2C3E73" },
  { name: "薰衣草", hex: "#B9A2E0" },
  { name: "深紫", hex: "#5B3A8C" },
  { name: "栗棕", hex: "#7A5234" },
  { name: "米白", hex: "#F3EBDD" },
  { name: "银灰", hex: "#B8BCC6" },
  { name: "乌黑", hex: "#2B2B35" },
  { name: "肤色", hex: "#F6D9CC" },
];

/** The palette name of a color, or the color itself. */
export function recolorName(hex: string): string {
  const upper = hex.toUpperCase();
  return RECOLOR_PALETTE.find((color) => color.hex === upper)?.name ?? upper;
}
