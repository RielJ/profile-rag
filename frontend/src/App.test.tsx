import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App.tsx";
import { answered, failed, jsonResponse, ndjsonResponse, unknown } from "./test/fixtures.ts";

const HEALTHY = { status: "ok", database: "ok" };

// A fetch that answers /api/health with `health` and /api/ask/stream with `askReply()`.
function mockFetch(
  health: () => Promise<Response>,
  askReply: () => Promise<Response> = async () =>
    ndjsonResponse([{ type: "result", response: unknown() }]),
) {
  const fetchMock = vi.fn<typeof fetch>(async (input) =>
    String(input) === "/api/health" ? health() : askReply(),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("App: server status line", () => {
  it("shows a healthy backend", async () => {
    mockFetch(async () => jsonResponse(200, HEALTHY));
    render(<App />);
    expect(await screen.findByText("Server and database are up.")).toBeInTheDocument();
  });

  it("shows a degraded backend (HTTP 503 still has a body)", async () => {
    mockFetch(async () => jsonResponse(503, { status: "degraded", database: "unavailable" }));
    render(<App />);
    expect(await screen.findByText(/^The database is unavailable/)).toBeInTheDocument();
  });

  it("says when the backend cannot be reached", async () => {
    mockFetch(async () => {
      throw new TypeError("Failed to fetch");
    });
    render(<App />);
    expect(await screen.findByText(/^Can’t reach the server/)).toBeInTheDocument();
  });

  it("tries again while the backend is unreachable, and recovers without a reload", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let up = false;
    const fetchMock = mockFetch(async () => {
      if (!up) throw new TypeError("Failed to fetch");
      return jsonResponse(200, HEALTHY);
    });
    render(<App />);
    expect(await screen.findByText(/^Can’t reach the server/)).toBeInTheDocument();

    up = true;
    await vi.advanceTimersByTimeAsync(5000);
    expect(await screen.findByText("Server and database are up.")).toBeInTheDocument();

    const checks = fetchMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(20000);
    expect(fetchMock.mock.calls.length).toBe(checks); // healthy: no more polling
  });

  it("checks the server again after each answer", async () => {
    let up = true;
    mockFetch(async () => {
      if (!up) throw new TypeError("Failed to fetch");
      return jsonResponse(200, HEALTHY);
    });
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByText("Server and database are up.")).toBeInTheDocument();

    up = false;
    await user.type(screen.getByLabelText("Your question"), "Salary?{Enter}");
    expect(await screen.findByText(/^Can’t reach the server/)).toBeInTheDocument();
  });
});

// A stream response whose lines the test sends one at a time with `send`.
function controlledStream() {
  let stream!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      stream = controller;
    },
  });
  return {
    response: new Response(body, { status: 200 }),
    send(event: unknown) {
      stream.enqueue(new TextEncoder().encode(JSON.stringify(event) + "\n"));
    },
    close() {
      stream.close();
    },
  };
}

describe("App: asking", () => {
  it("shows a pending state, then the draft as it streams, then the checked answer", async () => {
    const reply = controlledStream();
    mockFetch(
      async () => jsonResponse(200, HEALTHY),
      async () => reply.response,
    );
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Your question"), "What did Dana build?{Enter}");

    expect(await screen.findByText("Searching the evidence…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Asking…" })).toBeDisabled();
    expect(screen.getByRole("region", { name: "Result" })).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("status")).toHaveTextContent("Asking…");

    reply.send({ type: "draft", text: "Dana built " });
    reply.send({ type: "draft", text: "a job queue" });
    expect(await screen.findByText("Dana built a job queue")).toBeInTheDocument();
    expect(screen.getByText(/^Writing…/)).toBeInTheDocument();
    expect(screen.queryByText("Searching the evidence…")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Asking…"); // drafts are not announced
    expect(screen.queryByRole("link")).not.toBeInTheDocument(); // no citations yet

    reply.send({ type: "result", response: answered() });
    reply.close();
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Answer ready. It cites 2 excerpts.",
    );
    expect(screen.queryByText(/^Writing…/)).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Result" })).toHaveAttribute("aria-busy", "false");
    expect(screen.getByRole("button", { name: "Ask" })).toBeEnabled();
  });

  it("drops the draft when the checked result is a failure", async () => {
    mockFetch(
      async () => jsonResponse(200, HEALTHY),
      async () =>
        ndjsonResponse([
          { type: "draft", text: "An answer citing a made-up source" },
          { type: "result", response: failed("invented_citation") },
        ]),
    );
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Your question"), "Anything{Enter}");
    expect(await screen.findByRole("status")).toHaveTextContent(
      "No answer. Safe message for invented_citation.",
    );
    expect(screen.queryByText("An answer citing a made-up source")).not.toBeInTheDocument();
  });

  it("announces an unknown answer as not in the evidence", async () => {
    mockFetch(async () => jsonResponse(200, HEALTHY));
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Your question"), "Salary?{Enter}");
    expect(await screen.findByRole("heading", { name: "Not in the evidence" })).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("Not in the evidence.");
  });

  it("announces a network error as a connection problem", async () => {
    mockFetch(
      async () => jsonResponse(200, HEALTHY),
      async () => {
        throw new TypeError("Failed to fetch");
      },
    );
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Your question"), "Anything{Enter}");
    expect(await screen.findByRole("heading", { name: "Connection problem" })).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("Connection problem.");
  });
});
