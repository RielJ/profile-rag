import type { Citation } from "../api/client.ts";
import { ATTRIBUTION_LABELS, evidenceId, safeHttpUrl } from "./labels.ts";

// The cited chunks, numbered to match the answer's markers. Every value is
// rendered as text (React escapes it); nothing is parsed as HTML or Markdown.
export function EvidencePanel({ citations }: { citations: Citation[] }) {
  return (
    <section className="evidence" aria-labelledby="evidence-heading">
      <h2 id="evidence-heading">Evidence</h2>
      <ol>
        {citations.map((citation, index) => {
          const url = safeHttpUrl(citation.url);
          return (
            <li key={citation.chunk_id} id={evidenceId(citation.chunk_id)} tabIndex={-1}>
              <span className="evidence-number" aria-hidden="true">
                {index + 1}
              </span>
              <div className="evidence-body">
                <p className="evidence-title">
                  <span className="visually-hidden">Source {index + 1}: </span>
                  {url === null ? (
                    citation.source_title
                  ) : (
                    <a href={url} rel="noopener noreferrer">
                      {citation.source_title}
                    </a>
                  )}
                </p>
                {citation.section === null ? null : (
                  <p className="evidence-section">{citation.section}</p>
                )}
                <p className="evidence-attribution">{ATTRIBUTION_LABELS[citation.attribution]}</p>
                <blockquote className="evidence-excerpt">{citation.excerpt}</blockquote>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
