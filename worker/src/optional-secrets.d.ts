// Secrets the Worker runs without. `secrets.required` in wrangler.jsonc
// types only the required ones, so the optional ones are declared here.
// `declare global` because tsconfig's moduleDetection "force" makes this file a module.
declare global {
  interface Env {
    /** A Groq API key. When set, Groq answers if Workers AI fails (src/model.ts). */
    GROQ_API_KEY?: string;
  }
}

export {};
