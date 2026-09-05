import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Turn [Source n] markers into clickable markdown links. */
export function linkifyCitations(content) {
  if (!content) return "";
  return String(content).replace(
    /\[Source\s+(\d+)\]/gi,
    (_, n) => `[Source ${n}](#cite-${n})`
  );
}

export function MarkdownMessage({ content, onSourceClick }) {
  const linked = linkifyCitations(content);

  return (
    <div className="md-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children, ...props }) {
            const m = typeof href === "string" ? href.match(/^#cite-(\d+)$/i) : null;
            if (m) {
              const n = Number(m[1]);
              return (
                <button
                  type="button"
                  className="cite-chip"
                  onClick={(e) => {
                    e.preventDefault();
                    onSourceClick?.(n);
                  }}
                  title={`Open Source ${n}`}
                >
                  {children}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer" {...props}>
                {children}
              </a>
            );
          },
        }}
      >
        {linked}
      </ReactMarkdown>
    </div>
  );
}
