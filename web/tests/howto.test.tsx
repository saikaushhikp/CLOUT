import { readFileSync } from "node:fs";
import path from "node:path";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { HowToPlay, parseHowTo } from "@/components/HowToPlay";
import { GameApp } from "@/components/GameApp";
import { createMockClient } from "@/lib/mockApi";

// The setup screen shows the repo's HOW_TO_PLAY.md (read by app/page.tsx from one level above web/).
const DOC = readFileSync(path.join(__dirname, "..", "..", "HOW_TO_PLAY.md"), "utf8");

describe("how to play", () => {
  it("parses the doc's Markdown subset: headings, paragraphs, lists with wrapped items, bold", () => {
    const md = "# Title\n\nOne\nline.\n\n## Part\n\n1. **First** step\n2. Second\n\n- item\n  continued\n- next\n\nAfter.";
    expect(parseHowTo(md)).toEqual([
      { kind: "h1", text: "Title" },
      { kind: "p", text: "One line." },
      { kind: "h2", text: "Part" },
      { kind: "ol", items: ["**First** step", "Second"] },
      { kind: "ul", items: ["item continued", "next"] },
      { kind: "p", text: "After." },
    ]);
  });

  it("renders HOW_TO_PLAY.md with its headings and bold terms", () => {
    render(<HowToPlay markdown={DOC} />);
    const box = screen.getByRole("article", { name: "How to play" });
    expect(within(box).getByRole("heading", { level: 1, name: "How to Play Clout" })).toBeInTheDocument();
    for (const h of ["The goal", "4. Campaign", "Tips"]) expect(within(box).getByRole("heading", { level: 2, name: h })).toBeInTheDocument();
    expect(within(box).getByText("more than half of all votes").tagName).toBe("B");
    expect(box.textContent).not.toContain("**");
  });

  it("setup screen: big title, easter-egg tagline link, and the guide in a 'How to CLOUT?' dialog", () => {
    render(<GameApp client={createMockClient()} howToPlay={DOC} />);
    expect(screen.getByRole("heading", { level: 1, name: "CLOUT" })).toBeInTheDocument();
    const tagline = screen.getByRole("link", { name: "The winner takes it all...." });
    expect(tagline).toHaveAttribute("href", "https://youtu.be/92cwKCU8Z5c?t=66&autoplay=1");
    expect(tagline).toHaveAttribute("target", "_blank");
    expect(screen.getByRole("form", { name: /new game setup/i })).toBeInTheDocument();

    expect(screen.queryByRole("article", { name: "How to play" })).toBeNull(); // only in the dialog
    fireEvent.click(screen.getByRole("button", { name: "How to CLOUT?" }));
    const dialog = screen.getByRole("dialog", { name: "How to CLOUT?" });
    expect(within(dialog).getByRole("article", { name: "How to play" })).toBeInTheDocument();
    expect(within(dialog).getByRole("heading", { level: 2, name: "The goal" })).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("without the doc, the setup screen has no guide button", () => {
    render(<GameApp client={createMockClient()} />);
    expect(screen.queryByRole("button", { name: "How to CLOUT?" })).toBeNull();
    expect(screen.getByRole("heading", { level: 1, name: "CLOUT" })).toBeInTheDocument();
  });
});
