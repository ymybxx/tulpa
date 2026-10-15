import type { DesktopMainView } from "../shared/contracts";

export function shouldActivateModelManager(
  view: DesktopMainView,
  _documentHidden: boolean,
  windowVisible: boolean,
): boolean {
  return view === "models" && windowVisible;
}
