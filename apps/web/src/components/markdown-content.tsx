"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Renders assistant answers as GitHub-flavored markdown (headings,
 * lists, tables, code fences). Raw HTML in the model output is NOT
 * rendered — react-markdown's safe-by-default behaviour.
 */
export default function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="prose prose-sm max-w-none break-words dark:prose-invert prose-headings:mb-2 prose-headings:mt-3 prose-p:my-2 prose-table:my-2">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}
