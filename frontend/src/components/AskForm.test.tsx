import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { QUESTION_MAX_LENGTH } from "../api/client.ts";
import { AskForm } from "./AskForm.tsx";

function setup(loading = false) {
  const onAsk = vi.fn<(question: string) => void>();
  const user = userEvent.setup();
  const view = render(<AskForm loading={loading} onAsk={onAsk} />);
  const box = screen.getByLabelText("Your question");
  return { onAsk, user, box, view };
}

describe("AskForm", () => {
  it("submits the trimmed question with the button", async () => {
    const { onAsk, user, box } = setup();
    await user.type(box, "  What did Dana build?  ");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    expect(onAsk).toHaveBeenCalledExactlyOnceWith("What did Dana build?");
  });

  it("submits on Enter and adds a new line on Shift+Enter", async () => {
    const { onAsk, user, box } = setup();
    await user.type(box, "line one{Shift>}{Enter}{/Shift}line two");
    expect(onAsk).not.toHaveBeenCalled();
    expect(box).toHaveValue("line one\nline two");
    await user.type(box, "{Enter}");
    expect(onAsk).toHaveBeenCalledExactlyOnceWith("line one\nline two");
  });

  it("does not submit a blank question", async () => {
    const { onAsk, user, box } = setup();
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    await user.type(box, "   {Enter}");
    expect(onAsk).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
  });

  it("is disabled and busy while loading", async () => {
    const { onAsk, user, box, view } = setup(true);
    const button = screen.getByRole("button", { name: "Asking…" });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-busy", "true");
    expect(view.container.querySelector("form")).toHaveAttribute("aria-busy", "true");
    expect(box).toHaveAttribute("readonly");
    await user.type(box, "more{Enter}");
    expect(onAsk).not.toHaveBeenCalled();
  });

  it(`stops at ${QUESTION_MAX_LENGTH} characters and counts them`, async () => {
    const { onAsk, user, box } = setup();
    expect(screen.getByText(`0 / ${QUESTION_MAX_LENGTH} characters`)).toBeInTheDocument();
    await user.click(box);
    await user.paste("x".repeat(QUESTION_MAX_LENGTH + 25));
    expect(box).toHaveValue("x".repeat(QUESTION_MAX_LENGTH));
    expect(
      screen.getByText(`${QUESTION_MAX_LENGTH} / ${QUESTION_MAX_LENGTH} characters`),
    ).toBeInTheDocument();
    expect(box).toHaveAccessibleDescription(
      `${QUESTION_MAX_LENGTH} / ${QUESTION_MAX_LENGTH} characters`,
    );
    await user.keyboard("{Enter}");
    expect(onAsk).toHaveBeenCalledExactlyOnceWith("x".repeat(QUESTION_MAX_LENGTH));
  });
});
