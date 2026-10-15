import { useEffect, useState } from "preact/hooks";
import { getLocale, initializeLocale, subscribeLocale } from "./index";

/** Rerender in place: changing language must not discard a form or recording. */
export function useLocale() {
  const [locale, setLocale] = useState(getLocale);
  useEffect(() => {
    const update = () => setLocale(getLocale());
    const stop = subscribeLocale(update);
    update();
    void initializeLocale();
    return stop;
  }, []);
  return locale;
}
