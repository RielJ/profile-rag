import { afterEach, describe, expect, it, vi } from "vitest";
import {
  FAILED_STATUS,
  answered,
  citation,
  failed,
  jsonResponse,
  unknown,
} from "../test/fixtures.ts";
import { ask, getSource, type FailedCode } from "./client.ts";

function respondWith(response: Response | (() => Promise<Response>)) {
  const fetchMock = vi.fn<typeof fetch>(async () =>
    typeof response === "function" ? response() : response,
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ask", () => {
  it("posts the trimmed question with mode and k", async () => {
    const fetchMock = respondWith(jsonResponse(200, answered(), { "X-Request-ID": "r1" }));
    await ask("  What did Dana build?  \n", { mode: "hybrid", k: 4 });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] ?? [];
    expect(url).toBe("/api/ask");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({
      question: "What did Dana build?",
      mode: "hybrid",
      k: 4,
    });
  });

  it("omits mode and k when they are not given", async () => {
    const fetchMock = respondWith(jsonResponse(200, unknown()));
    await ask("q");
    const [, init] = fetchMock.mock.calls[0] ?? [];
    expect(JSON.parse(String(init?.body))).toEqual({ question: "q" });
  });

  it("parses an answered 200 and keeps the request id", async () => {
    respondWith(jsonResponse(200, answered(), { "X-Request-ID": "r1" }));
    expect(await ask("q")).toEqual({
      kind: "response",
      response: answered(),
      httpStatus: 200,
      requestId: "r1",
    });
  });

  it("parses an unknown 200", async () => {
    respondWith(jsonResponse(200, unknown({ model: null })));
    const result = await ask("q");
    expect(result.kind === "response" && result.response).toEqual(unknown({ model: null }));
  });

  const failedCases: [FailedCode, number][] = [
    ["retrieval_failed", FAILED_STATUS.retrieval_failed],
    ["pi_timeout", FAILED_STATUS.pi_timeout],
    ["pi_failed", FAILED_STATUS.pi_failed],
    ["bad_reply", FAILED_STATUS.bad_reply],
    ["invented_citation", FAILED_STATUS.invented_citation],
    ["internal_error", FAILED_STATUS.internal_error],
  ];

  it.each(failedCases)("parses a failed %s body on HTTP %i", async (code, status) => {
    const body = failed(code);
    respondWith(jsonResponse(status, body, { "X-Request-ID": body.request_id }));
    expect(await ask("q")).toEqual({
      kind: "response",
      response: body,
      httpStatus: status,
      requestId: body.request_id,
    });
  });

  it("drops extra fields instead of passing them through", async () => {
    respondWith(jsonResponse(200, { ...unknown(), html: "<b>x</b>" }));
    const result = await ask("q");
    expect(result.kind === "response" && Object.keys(result.response)).toEqual([
      "status",
      "reason",
      "model",
    ]);
  });

  it("turns a 422 into a validation result with only the messages", async () => {
    respondWith(
      jsonResponse(422, {
        detail: [
          {
            loc: ["body", "question"],
            msg: "String should have at most 1000 characters",
            type: "string_too_long",
            input: "the question echoed back",
          },
        ],
      }),
    );
    expect(await ask("q")).toEqual({
      kind: "invalid",
      messages: ["String should have at most 1000 characters"],
      requestId: null,
    });
  });

  it("reports a network error", async () => {
    respondWith(async () => {
      throw new TypeError("Failed to fetch");
    });
    expect(await ask("q")).toEqual({
      kind: "client_error",
      reason: "network",
      httpStatus: null,
      requestId: null,
    });
  });

  it("reports an aborted request as aborted, not as a network error", async () => {
    respondWith(async () => {
      throw new DOMException("The operation was aborted.", "AbortError");
    });
    expect(await ask("q")).toMatchObject({ kind: "client_error", reason: "aborted" });
  });

  it("reports a body that is not JSON, with its status", async () => {
    respondWith(new Response("<html>Bad Gateway</html>", { status: 502 }));
    expect(await ask("q")).toEqual({
      kind: "client_error",
      reason: "not_json",
      httpStatus: 502,
      requestId: null,
    });
  });

  it.each([
    ["no status", 200, { answer: "x" }],
    ["an unknown status", 200, { status: "maybe" }],
    ["answered without citations", 200, answered({ citations: [] })],
    [
      "a citation with a bad attribution",
      200,
      { ...answered(), citations: [{ ...citation(), attribution: "rumor" }] },
    ],
    ["failed on HTTP 200", 200, failed("pi_failed")],
    ["answered on HTTP 502", 502, answered()],
    ["a failed body with an unknown code", 502, { ...failed("pi_failed"), error: "oops" }],
    ["an array", 200, []],
    ["a 422 without detail", 422, { error: "x" }],
  ])("reports %s as a bad shape", async (_name, status, body) => {
    respondWith(jsonResponse(status, body));
    expect(await ask("q")).toMatchObject({ kind: "client_error", reason: "bad_shape" });
  });

  it("reports an undocumented status code", async () => {
    respondWith(jsonResponse(404, { detail: "Not Found" }));
    expect(await ask("q")).toMatchObject({
      kind: "client_error",
      reason: "unexpected_status",
      httpStatus: 404,
    });
  });
});

describe("getSource", () => {
  const source = {
    id: "resume",
    kind: "resume",
    title: "Resume 2026",
    url: null,
    attribution: "self_report",
    visibility: "public",
    approved_at: "2026-01-01T00:00:00Z",
  };

  it("encodes the id and parses the source", async () => {
    const fetchMock = respondWith(jsonResponse(200, source));
    expect(await getSource("a/b c")).toEqual({ kind: "source", source });
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/sources/a%2Fb%20c");
  });

  it("reports a missing source", async () => {
    respondWith(jsonResponse(404, { detail: "source not found" }));
    expect(await getSource("x")).toEqual({ kind: "not_found" });
  });

  it("reports a bad source body", async () => {
    respondWith(jsonResponse(200, { ...source, kind: "diary" }));
    expect(await getSource("x")).toMatchObject({ kind: "client_error", reason: "bad_shape" });
  });
});
