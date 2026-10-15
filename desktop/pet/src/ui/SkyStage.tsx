import type { ComponentChildren } from "preact";

const STARS: ReadonlyArray<readonly [number, number, number]> = [
  [150, 90, 1], [260, 30, 1.2], [420, 70, 1], [520, 30, 1.5], [640, 110, 1], [700, 50, 1.2], [900, 120, 1], [340, 140, 0.9],
];

/**
 * The A+ sky: layered islands, haze and clouds drawn with static gradients and
 * paths. Colors come from the active scene's tokens; nothing animates.
 */
export function SkyStage({ children }: { children?: ComponentChildren }) {
  return (
    <div class="ui-sky">
      <div class="ui-sky__sun" />
      <svg class="ui-sky__layer" viewBox="0 0 960 640" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
        <g class="ui-sky__stars">{STARS.map(([x, y, r]) => <circle cx={x} cy={y} r={r} />)}</g>
        <g class="ui-sky__islands">
          <path d="M560 262 c20 -30 60 -34 90 -18 c18 -14 46 -12 58 6 l-10 10 c-30 6 -60 8 -90 6 c-18 0 -34 0 -48 -4z" />
          <path d="M600 272 l24 26 l20 -26z" />
          <path d="M760 236 c14 -18 40 -20 56 -8 c12 -6 28 -4 34 8 l-6 6 c-26 4 -54 4 -84 0z" />
          <path d="M790 242 l14 16 l12 -16z" />
          <path d="M40 300 c18 -22 52 -26 76 -12 c12 -6 30 -4 36 8 l-6 6 c-34 4 -68 4 -106 -2z" />
        </g>
      </svg>
      <div class="ui-sky__haze" />
      <svg class="ui-sky__layer" viewBox="0 0 960 640" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
        <g class="ui-sky__clouds-far">
          <ellipse cx="120" cy="420" rx="160" ry="34" /><ellipse cx="330" cy="430" rx="140" ry="28" />
          <ellipse cx="560" cy="416" rx="180" ry="30" /><ellipse cx="820" cy="426" rx="180" ry="32" />
        </g>
        <g class="ui-sky__clouds-mid">
          <ellipse cx="60" cy="480" rx="120" ry="40" /><ellipse cx="250" cy="492" rx="150" ry="42" />
          <ellipse cx="480" cy="484" rx="120" ry="36" /><ellipse cx="720" cy="494" rx="170" ry="40" />
          <ellipse cx="930" cy="480" rx="110" ry="36" />
        </g>
      </svg>
      {children}
    </div>
  );
}

/** Halo ring and cloud platform a character stands on. */
export function CharacterPlatform() {
  return (
    <>
      <svg class="ui-halo" viewBox="0 0 440 440" aria-hidden="true">
        <circle cx="220" cy="220" r="200" class="ui-halo__outer" />
        <circle cx="220" cy="220" r="168" class="ui-halo__inner" />
        <circle cx="220" cy="220" r="138" class="ui-halo__dots" />
      </svg>
      <svg class="ui-platform" viewBox="0 0 400 80" preserveAspectRatio="xMidYMax meet" aria-hidden="true">
        <ellipse cx="200" cy="18" rx="56" ry="6" class="ui-platform__shadow" />
        <ellipse cx="200" cy="38" rx="190" ry="24" class="ui-platform__shadow" />
        <ellipse cx="136" cy="28" rx="84" ry="16" /><ellipse cx="266" cy="30" rx="88" ry="16" />
        <ellipse cx="200" cy="36" rx="160" ry="20" />
      </svg>
    </>
  );
}
