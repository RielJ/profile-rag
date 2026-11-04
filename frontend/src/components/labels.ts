// Human labels for API values. `satisfies` makes a new enum value in the
// generated types a compile error here until it has a label.

import type { Attribution } from "../api/client.ts";

export const ATTRIBUTION_LABELS = {
  self_authored: "Written by the person",
  self_report: "Resume / self-reported",
  project_doc: "Project documentation",
  third_party: "Third-party source",
} as const satisfies Record<Attribution, string>;

/** `url` if it parses as an http(s) URL, else null. The API already nulls others. */
export function safeHttpUrl(url: string | null): string | null {
  if (url === null) return null;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
}

/** The id of a citation's evidence item; markers link to `#` + this. */
export function evidenceId(chunkId: string): string {
  return `evidence-${chunkId}`;
}
