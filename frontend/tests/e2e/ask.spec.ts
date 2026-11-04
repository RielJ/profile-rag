// The ask flow in a real browser, against backend/tests/e2e_server.py: the
// real API and answer chain, corpus-example/ in a throwaway database, fake
// embeddings, and a fake Pi (backend/tests/e2e_fake_pi.py) that cites the
// first two retrieved chunks, answers "unknown" to salary questions, and
// fails on questions that contain E2E-FAIL.

import { expect, test, type Page } from "@playwright/test";

async function askQuestion(page: Page, question: string) {
  await page.goto("/");
  await page.getByLabel("Your question").fill(question);
  await page.getByLabel("Your question").press("Enter");
}

test("an answered question links each marker to its evidence item", async ({ page }) => {
  await askQuestion(page, "What job queue work has Dana done?");

  await expect(page.getByRole("heading", { name: "Answer" })).toBeVisible();
  await expect(page.getByText("A scripted answer from the e2e fake model")).toBeVisible();
  // The model name the fake Pi reports (fake_pi.assistant_end).
  await expect(page.getByText("test/fake-model")).toBeVisible();

  const evidence = page.getByRole("region", { name: "Evidence" });
  await expect(evidence.getByRole("listitem")).toHaveCount(2);

  const marker = page.getByRole("link", { name: /^Evidence 2:/ });
  await expect(marker).toHaveText("[2]");
  const href = await marker.getAttribute("href");
  expect(href).toMatch(/^#evidence-.+/);
  const target = page.locator(`[id="${href?.slice(1)}"]`);

  await marker.click();
  await expect(target).toBeFocused();
  await expect(target).toBeInViewport();
  expect(decodeURIComponent(new URL(page.url()).hash)).toBe(href);
  // The item shows a source title and a human attribution label.
  await expect(target.locator(".evidence-title")).not.toBeEmpty();
  await expect(target.locator(".evidence-attribution")).toHaveText(
    /Written by the person|Resume \/ self-reported|Project documentation|Third-party source/,
  );
  await expect(page.getByRole("status")).toHaveText("Answer ready, with 2 evidence items.");
});

test("an unanswerable question is shown as not in the evidence", async ({ page }) => {
  await askQuestion(page, "What is Dana's salary?");

  await expect(page.getByRole("heading", { name: "Not in the evidence" })).toBeVisible();
  await expect(page.getByText("The evidence does not mention a salary.")).toBeVisible();
  await expect(page.getByRole("region", { name: "Evidence" })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "No answer this time" })).toHaveCount(0);
});

test("a failed answer shows the safe message and the request id", async ({ page }) => {
  await askQuestion(page, "E2E-FAIL: what happens when the model fails?");

  await expect(page.getByRole("heading", { name: "No answer this time" })).toBeVisible();
  await expect(
    page.getByText("The answer model could not be reached. Try again later.", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".request-id code")).toHaveText(/^[0-9a-f]{32}$/);
});

test("the page fits a 375px-wide screen without sideways scrolling", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 740 });
  await askQuestion(page, "What job queue work has Dana done?");
  await expect(page.getByRole("heading", { name: "Answer" })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
