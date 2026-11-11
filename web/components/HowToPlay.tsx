"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

/**
 * Renders HOW_TO_PLAY.md on the setup screen. The doc only uses a small Markdown subset, so this handles exactly
 * that: `#`/`##` headings, paragraphs, `-` and `1.` lists (indented lines continue an item) and **bold**.
 */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) => (part.startsWith("**") && part.endsWith("**") ? <b key={i}>{part.slice(2, -2)}</b> : part));
}

type Block = { kind: "h1" | "h2" | "p"; text: string } | { kind: "ul" | "ol"; items: string[] };

export function parseHowTo(md: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: { kind: "ul" | "ol"; items: string[] } | null = null;
  const flush = () => {
    if (para.length) blocks.push({ kind: "p", text: para.join(" ") });
    if (list) blocks.push(list);
    para = [];
    list = null;
  };
  for (const raw of md.split("\n")) {
    const line = raw.trimEnd();
    const item = /^(-|\d+\.)\s+(.*)$/.exec(line);
    if (!line.trim()) {
      flush();
    } else if (line.startsWith("#")) {
      flush();
      const h2 = line.startsWith("## ");
      blocks.push({ kind: h2 ? "h2" : "h1", text: line.replace(/^#+\s+/, "") });
    } else if (item) {
      const kind = item[1] === "-" ? "ul" : "ol";
      if (para.length || (list && list.kind !== kind)) flush();
      list ??= { kind, items: [] };
      list.items.push(item[2]);
    } else if (list && /^\s+/.test(raw)) list.items[list.items.length - 1] += ` ${line.trim()}`;
    else {
      if (list) flush();
      para.push(line.trim());
    }
  }
  flush();
  return blocks;
}

export function HowToPlay({ markdown }: { markdown: string }) {
  // Fade an edge only when there is more text beyond it (no fade over the first or last lines).
  const ref = useRef<HTMLElement>(null);
  const [edge, setEdge] = useState({ top: true, bottom: false });
  const update = () => {
    const e = ref.current;
    if (e) setEdge({ top: e.scrollTop <= 2, bottom: e.scrollTop + e.clientHeight >= e.scrollHeight - 2 });
  };
  useEffect(update, [markdown]);
  return (
    <article
      ref={ref}
      className={`howto-scroll${edge.top ? " at-top" : ""}${edge.bottom ? " at-bottom" : ""}`}
      tabIndex={0}
      aria-label="How to play"
      onScroll={update}
    >
      {parseHowTo(markdown).map((b, i) => {
        if ("items" in b) {
          const List = b.kind;
          return (
            <List key={i}>
              {b.items.map((t, k) => (
                <li key={k}>{inline(t)}</li>
              ))}
            </List>
          );
        }
        const Tag = b.kind;
        return <Tag key={i}>{inline(b.text)}</Tag>;
      })}
    </article>
  );
}
