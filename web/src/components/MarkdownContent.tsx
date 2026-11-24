"use client";

import React from "react";

function renderInline(text: string, keyPrefix: string): React.ReactNode[] {
  // Tokenise inline code, bold, and italics safely without dangerouslySetInnerHTML.
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  const nodes: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let idx = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      nodes.push(
        <code
          key={`${keyPrefix}-c-${idx++}`}
          className="rounded bg-stone-200/80 px-1.5 py-0.5 font-mono text-[0.85em] text-stone-900 dark:bg-zinc-800 dark:text-zinc-100"
        >
          {token.slice(1, -1)}
        </code>
      );
    } else if (token.startsWith("**") && token.endsWith("**")) {
      nodes.push(
        <strong key={`${keyPrefix}-b-${idx++}`} className="font-semibold text-stone-950 dark:text-zinc-50">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith("*") && token.endsWith("*")) {
      nodes.push(
        <em key={`${keyPrefix}-i-${idx++}`} className="italic">
          {token.slice(1, -1)}
        </em>
      );
    } else {
      nodes.push(token);
    }
    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

export function MarkdownContent({
  content,
  compact = false,
}: {
  content: string;
  compact?: boolean;
}) {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  let blockKey = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i++;
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}\s*$/.test(line.trim())) {
      blocks.push(
        <hr key={blockKey++} className="my-4 border-stone-200 dark:border-zinc-800" />
      );
      i++;
      continue;
    }

    // Headings
    const headingMatch = /^(#{1,6})\s+(.+)$/.exec(line.trim());
    if (headingMatch) {
      const level = headingMatch[1].length;
      const text = headingMatch[2];
      const size = compact
        ? level === 1
          ? "text-sm"
          : "text-[0.8125rem]"
        : level === 1
          ? "text-lg"
          : level === 2
            ? "text-base"
            : "text-sm";
      const cls = `mt-3 mb-1 font-semibold tracking-tight text-stone-950 first:mt-0 dark:text-zinc-50 ${size}`;
      blocks.push(
        <h3 key={blockKey++} className={cls}>
          {renderInline(text, `h-${blockKey}`)}
        </h3>
      );
      i++;
      continue;
    }

    // Blockquote (verbatim corpus quotations)
    if (line.trimStart().startsWith(">")) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trimStart().startsWith(">")) {
        quoteLines.push(lines[i].trimStart().replace(/^>\s?/, ""));
        i++;
      }
      blocks.push(
        <blockquote
          key={blockKey++}
          className="my-3 rounded-r-lg border-l-[3px] border-amber-700/70 bg-stone-100/80 px-4 py-2.5 text-[0.94rem] leading-relaxed text-stone-800 dark:border-amber-500/60 dark:bg-zinc-900/80 dark:text-zinc-200"
        >
          {quoteLines.map((ql, qIdx) => (
            <p key={qIdx} className={qIdx > 0 ? "mt-1.5" : undefined}>
              {renderInline(ql, `q-${blockKey}-${qIdx}`)}
            </p>
          ))}
        </blockquote>
      );
      continue;
    }

    // Unordered list
    if (/^\s*[-*•]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*•]\s+/, ""));
        i++;
      }
      blocks.push(
        <ul key={blockKey++} className="my-2.5 list-disc space-y-1.5 pl-5 text-stone-800 dark:text-zinc-200">
          {items.map((item, idx) => (
            <li key={idx} className="leading-relaxed">
              {renderInline(item, `ul-${blockKey}-${idx}`)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // Ordered list
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+[.)]\s+/, ""));
        i++;
      }
      blocks.push(
        <ol key={blockKey++} className="my-2.5 list-decimal space-y-1.5 pl-5 text-stone-800 dark:text-zinc-200">
          {items.map((item, idx) => (
            <li key={idx} className="leading-relaxed">
              {renderInline(item, `ol-${blockKey}-${idx}`)}
            </li>
          ))}
        </ol>
      );
      continue;
    }

    // Paragraph
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^#{1,3}\s+/.test(lines[i].trim()) &&
      !lines[i].trimStart().startsWith(">") &&
      !/^\s*[-*•]\s+/.test(lines[i]) &&
      !/^\s*\d+[.)]\s+/.test(lines[i]) &&
      !/^[-*_]{3,}\s*$/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    blocks.push(
      <p key={blockKey++} className="my-2.5 leading-[1.68] text-stone-800 first:mt-0 last:mb-0 dark:text-zinc-200">
        {renderInline(paraLines.join(" "), `p-${blockKey}`)}
      </p>
    );
  }

  return <div className={compact ? "text-[0.8125rem] leading-relaxed" : "text-[0.95rem]"}>{blocks}</div>;
}
