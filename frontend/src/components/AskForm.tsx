import { useId, useState, type SubmitEvent, type KeyboardEvent } from "react";
import { QUESTION_MAX_LENGTH } from "../api/client.ts";

interface AskFormProps {
  loading: boolean;
  onAsk: (question: string) => void;
}

// The question box. Enter submits; Shift+Enter adds a new line. While a
// question is being answered the button is disabled and the form is
// aria-busy; the text stays editable-looking but read-only, so focus is not
// lost the way it would be on a disabled textarea.
export function AskForm({ loading, onAsk }: AskFormProps) {
  const [question, setQuestion] = useState("");
  const inputId = useId();
  const counterId = useId();
  const trimmed = question.trim();
  const canSubmit = !loading && trimmed.length > 0;

  function submit() {
    if (canSubmit) onAsk(trimmed);
  }

  function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    submit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // isComposing: Enter confirms an IME composition (Japanese, Chinese…), not the form.
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    submit();
  }

  const nearLimit = question.length >= QUESTION_MAX_LENGTH - 50;

  return (
    <form className="ask-form" onSubmit={handleSubmit} aria-busy={loading}>
      <label htmlFor={inputId}>Your question</label>
      <textarea
        id={inputId}
        name="question"
        rows={3}
        maxLength={QUESTION_MAX_LENGTH}
        value={question}
        readOnly={loading}
        aria-describedby={counterId}
        onChange={(event) => setQuestion(event.target.value)}
        onKeyDown={handleKeyDown}
      />
      <div className="ask-form-row">
        <p id={counterId} className={nearLimit ? "counter counter-near" : "counter"}>
          {question.length} / {QUESTION_MAX_LENGTH} characters
        </p>
        <button type="submit" disabled={!canSubmit} aria-busy={loading}>
          {loading ? "Asking…" : "Ask"}
        </button>
      </div>
      <p className="hint">Enter to ask, Shift+Enter for a new line.</p>
    </form>
  );
}
