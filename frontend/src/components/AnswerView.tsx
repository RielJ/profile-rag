import type {
  AnsweredResponse,
  AskResponse,
  AskResult,
  ClientError,
  FailedResponse,
  UnknownResponse,
} from "../api/client.ts";
import { EvidencePanel } from "./EvidencePanel.tsx";
import { evidenceId } from "./labels.ts";

// One AskResult on screen. The outer switch is on the client's result kind,
// the inner one on the API's `status`. Both end in a `never` check, so a new
// variant in the generated types is a compile error until it is rendered.
// All text from the model or the corpus is rendered as text, never as HTML.
export function AnswerView({ result }: { result: AskResult }) {
  switch (result.kind) {
    case "response":
      return <ResponseView response={result.response} />;
    case "invalid":
      return (
        <div className="notice notice-problem">
          <h2>The question was not accepted</h2>
          <ul>
            {result.messages.map((message, index) => (
              <li key={index}>{message}</li>
            ))}
          </ul>
        </div>
      );
    case "client_error":
      return <ClientErrorView error={result} />;
    default: {
      const unreachable: never = result;
      return unreachable;
    }
  }
}

function ResponseView({ response }: { response: AskResponse }) {
  switch (response.status) {
    case "answered":
      return <Answered response={response} />;
    case "unknown":
      return <Unknown response={response} />;
    case "failed":
      return <Failed response={response} />;
    default: {
      const unreachable: never = response;
      return unreachable;
    }
  }
}

function focusEvidence(chunkId: string) {
  // The href alone scrolls to the item; focusing it as well moves keyboard
  // and screen-reader position there. The default navigation still runs.
  document.getElementById(evidenceId(chunkId))?.focus();
}

function ModelLine({ model, verb }: { model: string | null; verb: string }) {
  if (model === null) return null;
  return (
    <p className="model-line">
      {verb} by <span className="model-name">{model}</span>
    </p>
  );
}

function Answered({ response }: { response: AnsweredResponse }) {
  return (
    <article className="answer" aria-labelledby="answer-heading">
      <h2 id="answer-heading">Answer</h2>
      <p className="answer-text">
        {response.answer}{" "}
        {response.citations.map((citation, index) => (
          <a
            key={citation.chunk_id}
            className="marker"
            href={`#${evidenceId(citation.chunk_id)}`}
            aria-label={`Evidence ${index + 1}: ${citation.source_title}`}
            onClick={() => focusEvidence(citation.chunk_id)}
          >
            [{index + 1}]
          </a>
        ))}
      </p>
      <ModelLine model={response.model} verb="Answered" />
      <EvidencePanel citations={response.citations} />
    </article>
  );
}

function Unknown({ response }: { response: UnknownResponse }) {
  return (
    <article className="notice notice-unknown" aria-labelledby="unknown-heading">
      <h2 id="unknown-heading">Not in the evidence</h2>
      <p>The approved sources do not answer this question, so no answer is given.</p>
      <p className="reason">
        <span className="reason-label">Reason:</span> {response.reason}
      </p>
      <ModelLine model={response.model} verb="Checked" />
    </article>
  );
}

function Failed({ response }: { response: FailedResponse }) {
  return (
    <div className="notice notice-problem">
      <h2>No answer this time</h2>
      <p>{response.message}</p>
      <p className="request-id">
        Request ID: <code>{response.request_id}</code>
      </p>
    </div>
  );
}

const CLIENT_ERROR_TEXT = {
  network: "Could not reach the server. Check your connection and try again.",
  aborted: "The request was cancelled.",
  not_json: "The server sent a response this page cannot read.",
  bad_shape: "The server sent a response this page cannot read.",
  unexpected_status: "The server sent a response this page does not expect.",
} as const satisfies Record<ClientError["reason"], string>;

function ClientErrorView({ error }: { error: ClientError }) {
  return (
    <div className="notice notice-problem">
      <h2>Connection problem</h2>
      <p>{CLIENT_ERROR_TEXT[error.reason]}</p>
      {error.httpStatus === null ? null : (
        <p className="request-id">
          HTTP status: <code>{error.httpStatus}</code>
          {error.requestId === null ? null : (
            <>
              {" "}
              · Request ID: <code>{error.requestId}</code>
            </>
          )}
        </p>
      )}
    </div>
  );
}
