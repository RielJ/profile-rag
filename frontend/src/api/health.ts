// Client for GET /api/health.
//
// The types are written by hand and must match
// backend/src/profile_rag/api/health.py.

export type DependencyStatus = "ok" | "unavailable";

export interface HealthResponse {
  status: "ok" | "degraded";
  database: DependencyStatus;
}

function isHealthResponse(value: unknown): value is HealthResponse {
  // `in` checks narrow `value` without a type assertion.
  if (typeof value !== "object" || value === null) return false;
  if (!("status" in value) || !("database" in value)) return false;
  return (
    (value.status === "ok" || value.status === "degraded") &&
    (value.database === "ok" || value.database === "unavailable")
  );
}

// The backend returns 200 when healthy and 503 when degraded. Both carry a
// HealthResponse body, so both count as a successful *request*. Any other
// status, or a body of the wrong shape, is an error.
export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch("/api/health", signal ? { signal } : {});
  if (response.status !== 200 && response.status !== 503) {
    throw new Error(`Unexpected status ${response.status} from /api/health`);
  }
  const body: unknown = await response.json();
  if (!isHealthResponse(body)) {
    throw new Error("Unexpected response body from /api/health");
  }
  return body;
}
