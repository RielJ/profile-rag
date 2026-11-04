import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App.tsx";
import { answered, jsonResponse, unknown } from "./test/fixtures.ts";

const HEALTHY = { status: "ok", database: "ok" };

// A fetch that answers /api/health with `health` and /api/ask with `ask()`.
function mockFetch(
  health: () => Promise<Response>,
  askReply: () => Promise<Response> = async () => jsonResponse(200, unknown()),
) {
  const fetchMock = vi.fn<typeof fetch>(async (input) =>
    String(input) === "/api/health" ? health() : askReply(),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App: server status line", () => {
  it("shows a healthy backend", async () => {
    mockFetch(async () => jsonResponse(200, HEALTHY));
    render(<App />);
    expect(await screen.findByText("Server: ok. Database: ok.")).toBeInTheDocument();
  });

  it("shows a degraded backend (HTTP 503 still has a body)", async () => {
    mockFetch(async () => jsonResponse(503, { status: "degraded", database: "unavailable" }));
    render(<App />);
    expect(await screen.findByText("Server: degraded. Database: unavailable.")).toBeInTheDocument();
  });

  it("says when the backend cannot be reached", async () => {
    mockFetch(async () => {
      throw new TypeError("Failed to fetch");
    });
    render(<App />);
    expect(await screen.findByText("Server unreachable.")).toBeInTheDocument();
  });
});

describe("App: asking", () => {
  it("shows a pending state while the question is answered, then the answer", async () => {
    let reply: (response: Response) => void = () => {};
    mockFetch(
      async () => jsonResponse(200, HEALTHY),
      () =>
        new Promise<Response>((resolve) => {
          reply = resolve;
        }),
    );
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Your question"), "What did Dana build?{Enter}");

    expect(screen.getByText("Searching the evidence…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Asking…" })).toBeDisabled();
    expect(screen.getByRole("region", { name: "Result" })).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("status")).toHaveTextContent("Asking…");

    reply(jsonResponse(200, answered()));
    expect(await screen.findByRole("heading", { name: "Answer" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Answer ready, with 2 evidence items.");
    expect(screen.getByRole("region", { name: "Result" })).toHaveAttribute("aria-busy", "false");
    expect(screen.getByRole("button", { name: "Ask" })).toBeEnabled();
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
