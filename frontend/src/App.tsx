import { useCallback, useEffect, useRef, useState } from "react";
import { ask, type AskResponse, type AskResult } from "./api/client.ts";
import { fetchHealth, type HealthResponse } from "./api/health.ts";
import { AnswerView } from "./components/AnswerView.tsx";
import { AskForm } from "./components/AskForm.tsx";

// One question at a time. A union rather than separate `loading` and
// `result` variables, so "loading with a stale result" cannot happen.
type AskState = { kind: "idle" } | { kind: "loading" } | { kind: "done"; result: AskResult };

export default function App() {
  const [state, setState] = useState<AskState>({ kind: "idle" });
  const [health, checkHealth] = useHealth();
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
    checkHealth(); // the answer may have failed because the server went down
  }

  const loading = state.kind === "loading";

  return (
    <div className="page">
      <header className="intro">
        <h1>Ask about this person’s work</h1>
        <p>
          Answers come only from sources the owner has reviewed and approved, such as a resume,
          project docs, and blog posts. Each answer links to the excerpts it used. If the sources
          don’t cover a question, you get no answer instead of a guess.
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
        <p className="health">{healthLine(health)}</p>
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
      return `Answer ready. It cites ${count} ${count === 1 ? "excerpt" : "excerpts"}.`;
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

const HEALTH_RETRY_MS = 5000;

// GET /api/health, checked when the page loads and whenever the returned
// function is called. While the server is unreachable or degraded, it checks
// again every HEALTH_RETRY_MS, so the footer recovers without a reload.
function useHealth(): [HealthState, () => void] {
  const [state, setState] = useState<HealthState>({ kind: "loading" });
  const checkNow = useRef<() => void>(() => {});

  useEffect(() => {
    const controller = new AbortController();
    let retry: ReturnType<typeof setTimeout> | undefined;
    let latest = 0; // a newer check makes the replies of older ones stale

    function check() {
      clearTimeout(retry);
      const id = ++latest;
      fetchHealth(controller.signal)
        .then((health) => {
          if (id !== latest) return;
          setState({ kind: "loaded", health });
          if (health.status !== "ok") retry = setTimeout(check, HEALTH_RETRY_MS);
        })
        .catch(() => {
          if (controller.signal.aborted || id !== latest) return;
          setState({ kind: "failed" });
          retry = setTimeout(check, HEALTH_RETRY_MS);
        });
    }

    checkNow.current = check;
    check();
    return () => {
      controller.abort();
      clearTimeout(retry);
    };
  }, []);

  const recheck = useCallback(() => checkNow.current(), []);
  return [state, recheck];
}

const RETRY_SECONDS = HEALTH_RETRY_MS / 1000;

function healthLine(state: HealthState): string {
  switch (state.kind) {
    case "loading":
      return "Checking the server…";
    case "failed":
      return `Can’t reach the server. Trying again every ${RETRY_SECONDS} seconds.`;
    case "loaded":
      return state.health.database === "ok"
        ? "Server and database are up."
        : `The database is unavailable, so questions will fail. Trying again every ${RETRY_SECONDS} seconds.`;
    default: {
      const unreachable: never = state;
      return unreachable;
    }
  }
}
