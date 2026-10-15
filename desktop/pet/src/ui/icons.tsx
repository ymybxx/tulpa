const PATHS = {
  language: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM3 12h18M12 3c-5 5-5 13 0 18M12 3c5 5 5 13 0 18",
  check: "M5 12.5l4.5 4.5L19 7.5",
  back: "M15 6l-6 6 6 6",
  close: "M6 6l12 12M18 6L6 18",
  plus: "M12 5v14M5 12h14",
  pause: "M8 5v14M16 5v14",
  play: "M7 5l12 7-12 7z",
  chat: "M4 5h16v11H9l-5 4z",
  mic: "M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3zM5 11a7 7 0 0 0 14 0M12 18v3",
  micOff: "M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3zM5 11a7 7 0 0 0 14 0M12 18v3M4 4l16 16",
  eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
  moon: "M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z",
  ear: "M7 9a5 5 0 0 1 10 0c0 3.5-3 4.5-3 7.5a2.5 2.5 0 0 1-5 0M10 10a2 2 0 0 1 4 0c0 1-1 1.5-1 2.5",
  sound: "M4 10v4h4l5 4V6L8 10zM16 9a4 4 0 0 1 0 6M19 6a8 8 0 0 1 0 12",
  heart: "M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z",
  screen: "M3 5h18v11H3zM8 20h8M12 16v4",
  upload: "M12 16V4M7 9l5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3",
  clip: "M16 7l-6.5 6.5a2 2 0 0 0 2.8 2.8L19 9.6a4 4 0 0 0-5.7-5.7l-6.8 6.8a6 6 0 0 0 8.5 8.5L20 14.2",
  file: "M6 3h8l4 4v14H6zM14 3v4h4M9 12h6M9 16h6",
  card: "M4 5h16v14H4zM8 10h8M8 14h5",
  lock: "M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 0 1 7 0v3",
  external: "M14 5h5v5M19 5l-8 8M18 14v5H5V6h5",
  chevronDown: "M6 9l6 6 6-6",
  sun: "M12 3v3M12 18v3M3 12h3M18 12h3M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z",
  home: "M4 11l8-7 8 7v9H4zM10 20v-6h4v6",
  person: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21a8 8 0 0 1 16 0",
  gear: "M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1",
  spark: "M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6",
  shield: "M12 3l8 3v6c0 4.5-3.4 8-8 9-4.6-1-8-4.5-8-9V6z",
  sliders: "M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M14 4v4M8 10v4M16 16v4",
  chart: "M5 20v-7M12 20V5M19 20v-11M3 20h18",
  warn: "M12 4l9 16H3zM12 10v4M12 17v.5",
  search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-4-4",
  send: "M5 12h13M13 6l6 6-6 6",
  bolt: "M13 3L5 14h6l-1 7 8-11h-6z",
  face: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM9 10h.01M15 10h.01M8.5 14.5a4.5 4.5 0 0 0 7 0",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16, weight = 2 }: { name: IconName; size?: number; weight?: number }) {
  return (
    <svg class="ui-icon" width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d={PATHS[name]} stroke="currentColor" stroke-width={weight} stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  );
}
