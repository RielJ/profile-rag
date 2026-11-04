import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { AskResult, FailedCode } from "../api/client.ts";
import { FAILED_STATUS, answered, citation, failed, unknown } from "../test/fixtures.ts";
import { AnswerView } from "./AnswerView.tsx";

function response(body: Extract<AskResult, { kind: "response" }>["response"], httpStatus = 200) {
  const result: AskResult = { kind: "response", response: body, httpStatus, requestId: null };
  return render(<AnswerView result={result} />);
}

const FAILED_CODES: FailedCode[] = [
  "retrieval_failed",
  "pi_timeout",
  "pi_failed",
  "bad_reply",
  "invented_citation",
  "internal_error",
];

describe("AnswerView: answered", () => {
  it("shows the answer with one marker per citation, linked to its evidence item", () => {
    const body = answered();
    const { container } = response(body);
    expect(screen.getByRole("heading", { name: "Answer" })).toBeInTheDocument();
    expect(screen.getByText(/Dana led the job queue migration\./)).toBeInTheDocument();

    const markers = screen.getAllByRole("link", { name: /^Evidence \d/ });
    expect(markers.map((marker) => marker.textContent)).toEqual(["[1]", "[2]"]);
    body.citations.forEach((cited, index) => {
      const href = `#evidence-${cited.chunk_id}`;
      expect(markers[index]).toHaveAttribute("href", href);
      // The target exists, so the link works without JavaScript.
      expect(container.querySelector(`[id="evidence-${cited.chunk_id}"]`)).not.toBeNull();
    });
  });

  it("focuses the evidence item when its marker is clicked", async () => {
    response(answered());
    await userEvent.click(screen.getByRole("link", { name: /^Evidence 2/ }));
    expect(document.activeElement?.id).toBe("evidence-blog:v2:b2");
  });

  it("numbers the evidence and shows title, attribution label, excerpt, and link", () => {
    response(answered());
    const items = within(screen.getByRole("region", { name: "Evidence" })).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    const [first, second] = items;
    if (first === undefined || second === undefined) throw new Error("two items expected");
    expect(first).toHaveTextContent("1");
    expect(within(first).getByRole("link", { name: "Resume 2026" })).toHaveAttribute(
      "href",
      "https://example.com/resume",
    );
    expect(within(first).getByRole("link", { name: "Resume 2026" })).toHaveAttribute(
      "rel",
      "noopener noreferrer",
    );
    expect(first).toHaveTextContent("Resume / self-reported");
    expect(first).toHaveTextContent("Led the migration of the job queue to PostgreSQL.");
    // No url: the title is plain text.
    expect(within(second).queryByRole("link")).toBeNull();
    expect(second).toHaveTextContent("Why we moved our job queue");
    expect(second).toHaveTextContent("Written by the person");
  });

  it("shows the model that answered", () => {
    response(answered());
    expect(screen.getByText("provider/model-x")).toBeInTheDocument();
  });

  it("leaves out the model line when no model was reported", () => {
    response(answered({ model: null }));
    expect(screen.queryByText(/Answered by/)).toBeNull();
  });

  it("renders HTML in the answer and the evidence as text", () => {
    const html = '<img src=x onerror="alert(1)"><b>bold</b>';
    const { container } = response(
      answered({
        answer: `Answer ${html}`,
        citations: [citation({ source_title: `Title ${html}`, excerpt: `Excerpt ${html}` })],
      }),
    );
    expect(container.querySelector("img, b, script")).toBeNull();
    expect(screen.getByText(`Answer ${html}`, { exact: false })).toBeInTheDocument();
    expect(screen.getByText(`Excerpt ${html}`)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: `Title ${html}` })).toBeInTheDocument();
  });

  it.each([
    "javascript:alert(1)",
    "data:text/html,<b>x</b>",
    "ftp://example.com/file",
    "file:///etc/passwd",
    "/relative/path",
    "not a url",
    "",
  ])("never links the non-http URL %j", (url) => {
    response(answered({ citations: [citation({ url })] }));
    const evidence = screen.getByRole("region", { name: "Evidence" });
    expect(within(evidence).queryByRole("link")).toBeNull();
    expect(evidence).toHaveTextContent("Resume 2026");
  });
});

describe("AnswerView: unknown", () => {
  it("says the answer is not in the evidence and gives the reason, not as an error", () => {
    const { container } = response(unknown());
    expect(screen.getByRole("heading", { name: "Not in the evidence" })).toBeInTheDocument();
    expect(screen.getByText("The evidence does not mention a salary.")).toBeInTheDocument();
    expect(container.querySelector(".notice-problem")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("region", { name: "Evidence" })).toBeNull();
  });

  it("renders HTML in the reason as text", () => {
    const { container } = response(unknown({ reason: "<script>alert(1)</script>" }));
    expect(container.querySelector("script")).toBeNull();
    expect(screen.getByText("<script>alert(1)</script>")).toBeInTheDocument();
  });
});

describe("AnswerView: failed", () => {
  it.each(FAILED_CODES)("%s shows the safe message and the request id", (code) => {
    response(failed(code), FAILED_STATUS[code]);
    expect(screen.getByRole("heading", { name: "No answer this time" })).toBeInTheDocument();
    expect(screen.getByText(`Safe message for ${code}.`)).toBeInTheDocument();
    expect(screen.getByText(`req-${code}`)).toBeInTheDocument();
    // The error code itself is internal; only the message is shown.
    expect(screen.queryByText(code)).toBeNull();
  });
});

describe("AnswerView: client-side problems", () => {
  it("shows a 422 as a rejected question with the server's messages", () => {
    render(
      <AnswerView
        result={{
          kind: "invalid",
          messages: ["String should have at most 1000 characters"],
          requestId: null,
        }}
      />,
    );
    expect(
      screen.getByRole("heading", { name: "The question was not accepted" }),
    ).toBeInTheDocument();
    expect(screen.getByText("String should have at most 1000 characters")).toBeInTheDocument();
  });

  it("shows a network error as a connection problem, without a status", () => {
    render(
      <AnswerView
        result={{ kind: "client_error", reason: "network", httpStatus: null, requestId: null }}
      />,
    );
    expect(screen.getByRole("heading", { name: "Connection problem" })).toBeInTheDocument();
    expect(screen.getByText(/Could not reach the server/)).toBeInTheDocument();
    expect(screen.queryByText(/HTTP status/)).toBeNull();
  });

  it("shows an unreadable response with its HTTP status and request id", () => {
    render(
      <AnswerView
        result={{ kind: "client_error", reason: "not_json", httpStatus: 502, requestId: "r9" }}
      />,
    );
    expect(screen.getByRole("heading", { name: "Connection problem" })).toBeInTheDocument();
    expect(screen.getByText("502")).toBeInTheDocument();
    expect(screen.getByText("r9")).toBeInTheDocument();
  });
});
