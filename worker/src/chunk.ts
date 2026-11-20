/**
 * A retrieved chunk: its text and the registry metadata the prompt and the
 * citations need. Retrieval (src/retrieve.ts) builds these from database rows.
 */

import type { Attribution } from "./contract";

export interface Chunk {
  chunkId: string;
  sourceId: string;
  title: string;
  attribution: Attribution;
  /** The registry's `url`; not necessarily a public link (see citations.publicUrl). */
  url: string | null;
  /** The headings above the chunk, top first. */
  headingPath: string[];
  /** The chunk's text: its heading line, a blank line, then its body. */
  text: string;
}
