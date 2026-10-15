import { render } from "preact";

import "../ui/theme.css";
import "../ui/ui.css";
import "../ui/sky.css";
import "./onboarding.css";
import { followScene } from "../ui/scene";
import { OnboardingApp } from "./OnboardingApp";
import type { OnboardingStep } from "./flow";

let mounted: { root: HTMLElement; stopScene: () => void } | null = null;

export function mountOnboarding(root: HTMLElement): void {
  if (mounted?.root === root) return;
  unmountOnboarding();
  root.classList.add("ui-theme");
  mounted = { root, stopScene: followScene(root) };
  // Development only: main.html?onboardingStep=permissions jumps straight to a step for testing.
  const step = import.meta.env.DEV
    ? new URLSearchParams(location.search).get("onboardingStep") as OnboardingStep | null
    : null;
  render(<OnboardingApp initialStep={step ?? "look"} />, root);
}

export function unmountOnboarding(): void {
  if (!mounted) return;
  render(null, mounted.root);
  mounted.stopScene();
  mounted = null;
}
