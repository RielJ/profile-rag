/**
 * Who may ask: the access-key check in front of the ask route.
 *
 * The portfolio's server-side proxy holds the key and sends it in
 * `X-Access-Key`, so visitors reach the API through the proxy only.
 * `GET /api/health` stays open for uptime checks.
 *
 * The check fails closed: with no `ACCESS_KEY` secret, every question is
 * refused. The visitor's address header is read in src/limits.ts.
 */

export const ACCESS_KEY_HEADER = "X-Access-Key";

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
