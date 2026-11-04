import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App.tsx";

// Replace the browser's fetch with one that returns a fixed response.
function mockFetch(status: number, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status })),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("shows a healthy backend", async () => {
    mockFetch(200, { status: "ok", database: "ok" });
    render(<App />);
    expect(await screen.findByText("Status")).toBeInTheDocument();
    expect(screen.getAllByText("ok")).toHaveLength(2);
  });

  it("shows a degraded backend (HTTP 503 still has a body)", async () => {
    mockFetch(503, { status: "degraded", database: "unavailable" });
    render(<App />);
    expect(await screen.findByText("degraded")).toBeInTheDocument();
    expect(screen.getByText("unavailable")).toBeInTheDocument();
  });

  it("shows an error when the backend cannot be reached", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Failed to fetch");
  });

  it("shows an error for an unexpected status", async () => {
    mockFetch(500, { detail: "boom" });
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Unexpected status 500");
  });
});
