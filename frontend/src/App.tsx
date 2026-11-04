import { useEffect, useState } from "react";
import { fetchHealth, type HealthResponse } from "./api/health.ts";

// The page has exactly one of these states at a time. Modeling them as a
// union (instead of separate `loading`, `error`, and `data` variables) makes
// impossible combinations, such as "loading with an error", unrepresentable.
type HealthState =
  | { kind: "loading" }
  | { kind: "loaded"; health: HealthResponse }
  | { kind: "failed"; message: string };

export default function App() {
  const [state, setState] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    // Abort the request if the component unmounts before it finishes.
    const controller = new AbortController();
    fetchHealth(controller.signal)
      .then((health) => setState({ kind: "loaded", health }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const message = error instanceof Error ? error.message : String(error);
        setState({ kind: "failed", message });
      });
    return () => controller.abort();
  }, []);

  return (
    <main>
      <h1>profile-rag</h1>
      <section aria-labelledby="health-heading">
        <h2 id="health-heading">Backend health</h2>
        <HealthView state={state} />
      </section>
    </main>
  );
}

function HealthView({ state }: { state: HealthState }) {
  switch (state.kind) {
    case "loading":
      return <p>Checking…</p>;
    case "failed":
      return <p role="alert">Backend unreachable: {state.message}</p>;
    case "loaded":
      return (
        <dl>
          <dt>Status</dt>
          <dd>{state.health.status}</dd>
          <dt>Database</dt>
          <dd>{state.health.database}</dd>
        </dl>
      );
  }
}
