import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

export function Markdown({ children }: { children: string }) {
  return (
    <div className="flex flex-col gap-2 text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          ul: (p) => <ul className="ml-5 list-disc space-y-1" {...p} />,
          ol: (p) => <ol className="ml-5 list-decimal space-y-1" {...p} />,
          a: (p) => <a className="underline" target="_blank" rel="noreferrer" {...p} />,
          code: (p) => <code className="bg-muted px-1 font-mono text-xs" {...p} />,
          pre: (p) => <pre className="overflow-x-auto bg-muted p-3 text-xs" {...p} />,
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
