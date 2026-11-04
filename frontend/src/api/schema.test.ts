// Checks that the generated API types (schema.d.ts, from `make types`) keep
// the backend's discriminated union: `status` alone tells the variants apart.
// The type checker does most of the work: `describe` would not compile if a
// variant were missing or `status` did not narrow.

import type { components } from "./schema";

type AskResponse = components["schemas"]["AskResponse"];

function summary(response: AskResponse): string {
  switch (response.status) {
    case "answered":
      return `${response.answer} [${response.citations.map((c) => c.chunk_id).join(", ")}]`;
    case "unknown":
      return `unknown: ${response.reason}`;
    case "failed":
      return `failed: ${response.error} (${response.request_id})`;
    default: {
      const unreachable: never = response;
      return unreachable;
    }
  }
}

describe("AskResponse", () => {
  it("narrows on status", () => {
    const answered: AskResponse = {
      status: "answered",
      answer: "Dana maintains lantern-queue.",
      citations: [
        {
          chunk_id: "lantern:v1:a",
          source_id: "lantern",
          source_title: "lantern-queue README",
          url: null,
          attribution: "project_doc",
          excerpt: "Dana maintains lantern-queue.",
        },
      ],
      model: null,
    };
    const failed: AskResponse = {
      status: "failed",
      error: "pi_timeout",
      message: "The answer model took too long to reply. Try again later.",
      request_id: "abc",
    };

    expect(summary(answered)).toBe("Dana maintains lantern-queue. [lantern:v1:a]");
    expect(summary({ status: "unknown", reason: "No evidence.", model: null })).toBe(
      "unknown: No evidence.",
    );
    expect(summary(failed)).toBe("failed: pi_timeout (abc)");
  });
});
