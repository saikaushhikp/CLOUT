import { readFile } from "node:fs/promises";
import path from "node:path";
import { GameApp } from "@/components/GameApp";

/** HOW_TO_PLAY.md lives at the repo root (one level above web/); it is read when the page is built. */
async function howToPlay(): Promise<string | undefined> {
  try {
    return await readFile(path.join(process.cwd(), "..", "HOW_TO_PLAY.md"), "utf8");
  } catch {
    return undefined;
  }
}

export default async function Home() {
  return <GameApp howToPlay={await howToPlay()} />;
}
