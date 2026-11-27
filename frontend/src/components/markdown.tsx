import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

export function Markdown({ children }: { children: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-2 text-sm leading-relaxed break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          ul: (p) => <ul className="ml-5 list-disc space-y-1" {...p} />,
          ol: (p) => <ol className="ml-5 list-decimal space-y-1" {...p} />,
          a: (p) => <a className="underline" target="_blank" rel="noreferrer" {...p} />,
          code: (p) => <code className="bg-muted px-1 font-mono text-xs" {...p} />,
          pre: (p) => <pre className="overflow-x-auto border bg-muted p-3 font-mono text-xs leading-relaxed [&_code]:bg-transparent [&_code]:p-0" {...p} />,
          blockquote: (p) => <blockquote className="border-l-2 pl-3 text-muted-foreground" {...p} />,
          table: (p) => <div className="overflow-x-auto"><table className="text-xs [&_td]:border [&_td]:px-2 [&_th]:border [&_th]:px-2" {...p} /></div>,
          h1: (p) => <h3 className="font-heading font-semibold" {...p} />,
          h2: (p) => <h3 className="font-heading font-semibold" {...p} />,
          h3: (p) => <h3 className="font-heading font-semibold" {...p} />,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
