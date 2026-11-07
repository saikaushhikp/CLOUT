import { describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useCountdown } from "@/lib/useCountdown";
import { createHttpClient } from "@/lib/api";
import { ApiError } from "@/lib/errors";

describe("useCountdown", () => {
  it("fires once per round and restarts when the round changes", () => {
    vi.useFakeTimers();
    try {
      const onExpire = vi.fn();
      const { result, rerender } = renderHook((p: { round: number; paused: boolean }) =>
        useCountdown({ enabled: true, seconds: 3, resetKey: p.round, onExpire, paused: p.paused }),
      { initialProps: { round: 1, paused: false } });
      expect(result.current.remaining).toBe(3);
      act(() => void vi.advanceTimersByTime(3000));
      expect(onExpire).toHaveBeenCalledTimes(1);
      act(() => void vi.advanceTimersByTime(5000));
      expect(onExpire).toHaveBeenCalledTimes(1);
      rerender({ round: 2, paused: false });
      expect(result.current.remaining).toBe(3);
      expect(onExpire).toHaveBeenCalledTimes(1);
      rerender({ round: 2, paused: true });
      act(() => void vi.advanceTimersByTime(10_000));
      expect(onExpire).toHaveBeenCalledTimes(1);
      rerender({ round: 2, paused: false });
      act(() => void vi.advanceTimersByTime(3000));
      expect(onExpire).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("http client", () => {
  const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });

  it("maps every route of the contract", async () => {
    const calls: [string, string, string | undefined][] = [];
    const fetchImpl = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      calls.push([String(url), init?.method ?? "GET", init?.body as string | undefined]);
      return ok({});
    });
    const c = createHttpClient("", fetchImpl as unknown as typeof fetch);
    await c.health();
    await c.createGame({ num_voters: 1000 });
    await c.observation("g 1", "p0");
    await c.query("g", "p0", { sector_a: "Edu", sector_b: "Env", region: "N" });
    await c.commit("g", "p0", { manifesto: [0.2, 0.2, 0.2, 0.2, 0.2] });
    await c.allocate("g", "p0", { workers_by_region: [1, 2, 3, 4], extra_workers: 0, media_regions: [true, false, false, false] });
    await c.forfeit("g", "p0");
    await c.graphSample("g");
    await c.debrief("g", "p0");
    await c.deleteGame("g");
    expect(calls.map(([u, m]) => `${m} ${u}`)).toEqual([
      "GET /api/health",
      "POST /api/games",
      "GET /api/games/g%201/players/p0/observation",
      "POST /api/games/g/players/p0/query",
      "POST /api/games/g/players/p0/commit",
      "POST /api/games/g/players/p0/allocate",
      "POST /api/games/g/players/p0/forfeit",
      "GET /api/games/g/graph_sample?max_nodes=1500",
      "GET /api/games/g/players/p0/debrief",
      "DELETE /api/games/g",
    ]);
    expect(JSON.parse(calls[3][2]!)).toEqual({ sector_a: "Edu", sector_b: "Env", region: "N" });
  });

  it("turns HTTP errors into ApiError with the FastAPI detail", async () => {
    const c = createHttpClient("", (async () => new Response(JSON.stringify({ detail: "Unknown game" }), { status: 404 })) as unknown as typeof fetch);
    await expect(c.observation("x", "p0")).rejects.toMatchObject({ status: 404, message: "Unknown game" });
    const down = createHttpClient("", (async () => {
      throw new TypeError("fetch failed");
    }) as unknown as typeof fetch);
    await expect(down.health()).rejects.toBeInstanceOf(ApiError);
  });
});
