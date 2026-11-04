import { useEffect, useRef, useState } from "react";
import { ask, type AskResponse, type AskResult } from "./api/client.ts";
import { fetchHealth, type HealthResponse } from "./api/health.ts";
import { AnswerView } from "./components/AnswerView.tsx";
import { AskForm } from "./components/AskForm.tsx";

// One question at a time. A union rather than separate `loading` and
// `result` variables, so "loading with a stale result" cannot happen.
type AskState = { kind: "idle" } | { kind: "loading" } | { kind: "done"; result: AskResult };

export default function App() {
  const [state, setState] = useState<AskState>({ kind: "idle" });
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => () => inFlight.current?.abort(), []);

  async function handleAsk(question: string) {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setState({ kind: "loading" });
    const result = await ask(question, { signal: controller.signal });
    if (controller.signal.aborted) return;
    setState({ kind: "done", result });
  }

  const loading = state.kind === "loading";

  return (
    <div className="page">
      <header className="intro">
        <h1>Ask about this person’s work</h1>
        <p>
          Answers come only from approved sources, such as a resume, project documentation, and
          posts. Each answer lists the evidence it used. When the sources do not say, the page tells
          you so instead of guessing.
        </p>
      </header>
      <main>
        <AskForm loading={loading} onAsk={(question) => void handleAsk(question)} />
        {/* Screen readers hear one short line per result, not the whole evidence list. */}
        <p role="status" aria-live="polite" className="visually-hidden">
          {statusLine(state)}
        </p>
        <section className="result" aria-label="Result" aria-busy={loading}>
          {state.kind === "loading" ? (
            <p className="pending">Searching the evidence…</p>
          ) : state.kind === "done" ? (
            <AnswerView result={state.result} />
          ) : null}
        </section>
      </main>
      <footer>
        <HealthLine />
      </footer>
    </div>
  );
}

function statusLine(state: AskState): string {
  switch (state.kind) {
    case "idle":
      return "";
    case "loading":
      return "Asking…";
    case "done":
      return resultLine(state.result);
    default: {
      const unreachable: never = state;
      return unreachable;
    }
  }
}

function resultLine(result: AskResult): string {
  switch (result.kind) {
    case "invalid":
      return "The question was not accepted.";
    case "client_error":
      return "Connection problem. No answer.";
    case "response":
      return responseLine(result.response);
    default: {
      const unreachable: never = result;
      return unreachable;
    }
  }
}

function responseLine(response: AskResponse): string {
  switch (response.status) {
    case "answered": {
      const count = response.citations.length;
      return `Answer ready, with ${count} evidence ${count === 1 ? "item" : "items"}.`;
    }
    case "unknown":
      return "Not in the evidence. No answer given.";
    case "failed":
      return `No answer. ${response.message}`;
    default: {
      const unreachable: never = response;
      return unreachable;
    }
  }
}

// The page has exactly one of these states at a time.
type HealthState =
  | { kind: "loading" }
  | { kind: "loaded"; health: HealthResponse }
  | { kind: "failed" };

function HealthLine() {
  const [state, setState] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    // Abort the request if the component unmounts before it finishes.
    const controller = new AbortController();
    fetchHealth(controller.signal)
      .then((health) => setState({ kind: "loaded", health }))
      .catch(() => {
        if (!controller.signal.aborted) setState({ kind: "failed" });
      });
    return () => controller.abort();
  }, []);

  switch (state.kind) {
    case "loading":
      return <p className="health">Checking the server…</p>;
    case "failed":
      return <p className="health">Server unreachable.</p>;
    case "loaded":
      return (
        <p className="health">
          Server: {state.health.status}. Database: {state.health.database}.
        </p>
      );
    default: {
      const unreachable: never = state;
      return unreachable;
    }
  }
}
