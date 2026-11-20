/**
 * Who may ask: the access-key check in front of the ask route.
 *
 * The portfolio's server-side proxy holds the key and sends it in
 * `X-Access-Key`, so visitors reach the API through the proxy only.
 * `GET /api/health` stays open for uptime checks.
 *
 * Unlike the Python API, which skips the check when no key is set, the Worker
 * fails closed: with no `ACCESS_KEY` secret, every question is refused.
 */

export const ACCESS_KEY_HEADER = "X-Access-Key";
export const VISITOR_IP_HEADER = "X-Visitor-IP";

const encoder = new TextEncoder();

/** True if a key is configured and `request` carries it. Constant-time in the key's content. */
export async function hasAccessKey(
  request: Request,
  expected: string | undefined,
): Promise<boolean> {
  if (!expected) return false;
  const given = request.headers.get(ACCESS_KEY_HEADER) ?? "";
  // Compare digests, so the comparison takes the same time whatever the lengths.
  const [a, b] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(given)),
    crypto.subtle.digest("SHA-256", encoder.encode(expected)),
  ]);
  return crypto.subtle.timingSafeEqual(a, b);
}

/** The visitor's address as the proxy reports it, at most 100 characters, or null. */
export function visitorOf(request: Request): string | null {
  const visitor = request.headers.get(VISITOR_IP_HEADER)?.trim();
  return visitor ? visitor.slice(0, 100) : null;
}
