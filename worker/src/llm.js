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

export class LLMError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "LLMError";
    this.status = status;
  }
}

export async function generate({
  apiKey,
  model = "gemini-2.5-flash",
  system,
  user,
  maxOutputTokens = 500,
  temperature = 0.2,
  signal,
}) {
  if (!apiKey) throw new LLMError("GEMINI_API_KEY tanımlı değil", 503);

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
        maxOutputTokens,
        topP: 0.9,
        // No "thinking" budget: these are short, grounded rewrites and the
        // latency shows up directly in the demo.
        thinkingConfig: { thinkingBudget: 0 },
      },
      safetySettings: [],
    }),
    signal,
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new LLMError(
      `Gemini ${res.status}: ${detail.slice(0, 300)}`,
      res.status === 429 ? 429 : 502
    );
  }

  const data = await res.json();
  const candidate = data?.candidates?.[0];
  const text = candidate?.content?.parts?.map((p) => p.text ?? "").join("").trim();

  if (!text) {
    const reason = candidate?.finishReason ?? data?.promptFeedback?.blockReason ?? "boş yanıt";
    throw new LLMError(`Gemini metin döndürmedi (${reason})`, 502);
  }
  return text;
}
