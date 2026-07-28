/**
 * Gemini call. Small on purpose.
 *
 * The language model never computes anything here. Every number in its output
 * has already been produced by the booster and the policy engine and is handed
 * to it as fact; its only job is to say what those numbers mean in a sentence
 * a planner can act on. That split is what keeps the system honest — a wrong
 * word is a wrong word, not a wrong risk score.
 */

const ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models";

/**
 * Model choice, measured against this project's own key rather than assumed:
 *
 *   gemini-2.5-flash / -lite   closed to new accounts ("no longer available
 *                              to new users") — the obvious default is a trap
 *   gemini-2.0-flash           returns quota-exceeded on a fresh free key
 *   gemini-3.6-flash           1.26 s, generally available   <- primary
 *   gemini-3-flash-preview     1.37 s                        <- fallback
 *   gemini-flash-latest        1.74 s, floating alias        <- last resort
 *
 * The alias is last on purpose: it silently follows Google's newest model, so
 * it is the right safety net and the wrong default.
 *
 * Gemini 3 takes `thinkingLevel`, not `thinkingBudget` — passing the latter is
 * rejected outright with "Request contains an invalid argument".
 */
export const MODELS = ["gemini-3.6-flash", "gemini-3-flash-preview", "gemini-flash-latest"];

/** Thinking tokens are drawn from maxOutputTokens, so the ceiling has to cover
 *  both or the answer comes back empty with finishReason MAX_TOKENS. */
const THINKING_HEADROOM = 700;

/**
 * Cloudflare's own inference, used only when Gemini's daily free quota is gone.
 * Runs on the same Worker, needs no key and no card. See riskFallbackPrompt()
 * for why it gets its own prompt and why the site assistant never uses it.
 */
export const WORKERS_AI_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";

export class LLMError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "LLMError";
    this.status = status;
  }
}

/** @returns {Promise<string>} */
export async function generateOnWorkersAI({ ai, system, user, maxOutputTokens = 400 }) {
  if (!ai) throw new LLMError("Workers AI bağlantısı yok", 503);

  const out = await ai.run(WORKERS_AI_MODEL, {
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    max_tokens: maxOutputTokens,
    temperature: 0.1,
  });

  const text = (out?.response ?? "").trim();
  if (!text) throw new LLMError("Workers AI metin döndürmedi", 502);
  return text;
}

export async function generate({
  apiKey,
  models = MODELS,
  system,
  user,
  maxOutputTokens = 500,
  temperature = 0.2,
  signal,
}) {
  if (!apiKey) throw new LLMError("GEMINI_API_KEY tanımlı değil", 503);

  let last = null;
  for (const model of models) {
    try {
      return await once({ apiKey, model, system, user, maxOutputTokens, temperature, signal });
    } catch (err) {
      last = err;
      // Retry the next model only for transient or model-specific failures.
      // A blocked prompt or a bad key will fail identically everywhere.
      const transient = err instanceof LLMError &&
        (err.status === 429 || err.status === 502 || err.status === 503);
      if (!transient) throw err;
    }
  }
  throw last;
}

async function once({ apiKey, model, system, user, maxOutputTokens, temperature, signal }) {
  const res = await fetch(`${ENDPOINT}/${model}:generateContent`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      // In the header, not the query string: keys in URLs end up in logs.
      "x-goog-api-key": apiKey,
    },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: system }] },
      contents: [{ role: "user", parts: [{ text: user }] }],
      generationConfig: {
        temperature,
        maxOutputTokens: maxOutputTokens + THINKING_HEADROOM,
        topP: 0.9,
        // These are short, grounded rewrites; the latency shows up directly in
        // the demo, so keep deliberation to the minimum the API allows.
        thinkingConfig: { thinkingLevel: "low" },
      },
    }),
    signal,
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new LLMError(
      `Gemini ${model} ${res.status}: ${detail.slice(0, 300)}`,
      res.status === 429 ? 429 : res.status >= 500 ? 503 : 502
    );
  }

  const data = await res.json();
  const candidate = data?.candidates?.[0];
  const text = candidate?.content?.parts?.map((p) => p.text ?? "").join("").trim();
  const reason = candidate?.finishReason;

  if (!text) {
    const why = reason ?? data?.promptFeedback?.blockReason ?? "boş yanıt";
    // MAX_TOKENS with no text means thinking ate the whole budget: worth
    // another model rather than surfacing an empty answer.
    throw new LLMError(`Gemini ${model} metin döndürmedi (${why})`,
      why === "MAX_TOKENS" ? 503 : 502);
  }

  // A truncated answer is worse than none: it reads as finished and stops
  // mid-sentence, and the sentence it swallows is often the guardrail. Seen in
  // testing — "...Planlamacının belirlediği" and nothing after it.
  if (reason === "MAX_TOKENS") {
    throw new LLMError(`Gemini ${model} yanıtı yarıda kesildi (MAX_TOKENS)`, 503);
  }
  return text;
}
