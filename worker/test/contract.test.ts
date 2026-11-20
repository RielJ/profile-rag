import { describe, expect, it } from "vitest";

import { ATTRIBUTIONS, FAILED_CODES, FAILURES } from "../src/contract";
import contract from "../src/generated/backend-contract.json";

// TypeScript widens the generated JSON's keys and lists to string, so these
// tests are what catch a code or attribution added on the Python side only.
describe("the generated contract", () => {
  it("has a message for exactly the failure codes the Worker knows", () => {
    expect(Object.keys(contract.failures).sort()).toEqual([...FAILED_CODES].sort());
  });

  it("has every message filled in", () => {
    for (const code of FAILED_CODES) expect(FAILURES[code].length).toBeGreaterThan(0);
  });

  it("lists exactly the attributions the Worker knows", () => {
    expect([...contract.attributions].sort()).toEqual([...ATTRIBUTIONS].sort());
  });
});
