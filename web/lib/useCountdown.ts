"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Per-round countdown. Restarts whenever `resetKey` (e.g. the round number), `seconds` or `enabled` changes.
 * Calls `onExpire` exactly once per countdown when it reaches zero; `paused` freezes it (e.g. while a request
 * is in flight). The remaining time is stored together with the key it belongs to, so a stale "0" from the
 * previous round can never trigger a second expiry.
 */
export function useCountdown({
  enabled,
  seconds,
  resetKey,
  onExpire,
  paused = false,
}: {
  enabled: boolean;
  seconds: number;
  resetKey: string | number;
  onExpire: () => void;
  paused?: boolean;
}): { remaining: number; fraction: number } {
  const key = `${resetKey}|${seconds}|${enabled ? 1 : 0}`;
  const [state, setState] = useState({ key, left: seconds });
  const remaining = state.key === key ? state.left : seconds;
  const firedKey = useRef<string | null>(null);
  const expireRef = useRef(onExpire);
  useEffect(() => {
    expireRef.current = onExpire;
  });

  useEffect(() => {
    if (!enabled || paused) return;
    const id = setInterval(() => {
      setState((s) => ({ key, left: Math.max(0, (s.key === key ? s.left : seconds) - 1) }));
    }, 1000);
    return () => clearInterval(id);
  }, [enabled, paused, seconds, key]);

  useEffect(() => {
    if (enabled && !paused && remaining <= 0 && firedKey.current !== key) {
      firedKey.current = key;
      expireRef.current();
    }
  }, [enabled, paused, remaining, key]);

  return { remaining, fraction: seconds > 0 ? remaining / seconds : 0 };
}
