import { memo, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import { Check, Copy } from "lucide-react";

// ── Block-level memoization ─────────────────────────────────────
//
// During streaming, react-markdown re-parses the entire message on every
// token. For long answers the page lags badly. Pattern lifted from
// Vercel's `streamdown` and the AI SDK cookbook: split the content into
// independent blocks, memoize each block, and only the actively
// streaming (last) block re-parses while everything before it stays
// stable.
//
// Splitting must respect fenced code blocks — we never split inside one
// or the renderer would see a half-open ``` and render gibberish.

function splitIntoBlocks(content: string): string[] {
  const lines = content.split("\n");
  const blocks: string[] = [];
  let buf: string[] = [];
  let inFence = false;

  const flush = () => {
    if (buf.length === 0) return;
    const block = buf.join("\n").replace(/^\n+|\n+$/g, "");
    if (block) blocks.push(block);
    buf = [];
  };

  for (const line of lines) {
    if (/^```/.test(line.trimStart())) {
      inFence = !inFence;
      buf.push(line);
      continue;
    }
    if (line.trim() === "" && !inFence) {
      flush();
      continue;
    }
    buf.push(line);
  }
  flush();
  return blocks;
}

// ── Components ──────────────────────────────────────────────────

function CodeBlock({ language, children }: { language?: string; children: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      // ignore — clipboard may be blocked in non-secure contexts
    }
  };
  return (
    <div className="my-3 rounded-field overflow-hidden border border-glass-border bg-surface-1000/60">
      <div className="flex items-center justify-between px-3 py-1.5 text-[11px] font-mono text-surface-400 border-b border-glass-border">
        <span className="lowercase tracking-wider">{language || "code"}</span>
        <button
          onClick={onCopy}
          className="inline-flex items-center gap-1 hover:text-surface-100 transition-colors px-1.5 py-0.5 rounded"
          aria-label="Copy code"
        >
          {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-[12.5px] leading-relaxed font-mono text-surface-100">
        <code>{children}</code>
      </pre>
    </div>
  );
}

const components: Components = {
  pre({ children }) {
    return <>{children}</>;
  },
  code({ className, children, ...props }) {
    const text = String(children).replace(/\n$/, "");
    const match = /language-(\w+)/.exec(className || "");
    const isFenced = !!match || text.includes("\n");
    if (isFenced) {
      return <CodeBlock language={match?.[1]}>{text}</CodeBlock>;
    }
    return (
      <code
        className="bg-surface-900/70 text-accent px-1.5 py-0.5 rounded text-[12px] font-mono border border-glass-border"
        {...props}
      >
        {children}
      </code>
    );
  },
  p({ children }) {
    return <p className="mb-2 last:mb-0">{children}</p>;
  },
  ul({ children }) {
    return <ul className="list-disc list-outside ml-5 mb-2 space-y-1">{children}</ul>;
  },
  ol({ children }) {
    return <ol className="list-decimal list-outside ml-5 mb-2 space-y-1">{children}</ol>;
  },
  li({ children }) {
    return <li className="leading-relaxed pl-1">{children}</li>;
  },
  h1({ children }) {
    return (
      <h1 className="font-display text-[18px] font-semibold text-surface-50 mb-2 mt-4 first:mt-0 tracking-[-0.01em]">
        {children}
      </h1>
    );
  },
  h2({ children }) {
    return (
      <h2 className="font-display text-[15px] font-semibold text-surface-50 mb-1.5 mt-4 first:mt-0">
        {children}
      </h2>
    );
  },
  h3({ children }) {
    return (
      <h3 className="font-display text-[13px] font-semibold uppercase tracking-wider text-surface-300 mb-1 mt-3 first:mt-0">
        {children}
      </h3>
    );
  },
  strong({ children }) {
    return <strong className="font-semibold text-surface-50">{children}</strong>;
  },
  em({ children }) {
    return <em className="italic text-surface-200">{children}</em>;
  },
  a({ href, children }) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-accent underline decoration-accent/40 hover:decoration-accent/80 underline-offset-2"
      >
        {children}
      </a>
    );
  },
  blockquote({ children }) {
    return (
      <blockquote className="border-l-2 border-accent/40 pl-3 my-2 text-surface-300 italic">
        {children}
      </blockquote>
    );
  },
  hr() {
    return <hr className="border-glass-border my-4" />;
  },
  table({ children }) {
    return (
      <div className="overflow-x-auto my-3 rounded-field border border-glass-border">
        <table className="text-[12px] w-full border-collapse">{children}</table>
      </div>
    );
  },
  thead({ children }) {
    return <thead className="bg-surface-900/60">{children}</thead>;
  },
  th({ children }) {
    return (
      <th className="border-b border-glass-border px-3 py-2 text-left font-semibold text-surface-200">
        {children}
      </th>
    );
  },
  td({ children }) {
    return (
      <td className="border-b border-glass-border px-3 py-2 text-surface-300 align-top">
        {children}
      </td>
    );
  },
};

const remarkPlugins = [remarkGfm];

const Block = memo(function Block({ content }: { content: string }) {
  return (
    <ReactMarkdown remarkPlugins={remarkPlugins} components={components}>
      {content}
    </ReactMarkdown>
  );
});

function MarkdownInner({ content }: { content: string }) {
  const blocks = useMemo(() => splitIntoBlocks(content), [content]);
  return (
    <>
      {blocks.map((b, i) => (
        // Block content is the key — when a stable block recurs unchanged
        // across renders, React reuses the same instance and memo skips
        // re-parsing. Only the actively streaming last block re-parses.
        <Block key={`${i}:${b.length}:${b.slice(0, 32)}`} content={b} />
      ))}
    </>
  );
}

const Markdown = memo(MarkdownInner);
export default Markdown;
