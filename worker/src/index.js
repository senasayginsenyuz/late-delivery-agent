/**
 * late-delivery-agent — Cloudflare Worker
 *
 *   GET  /api/health   what is loaded, what it measured
 *   GET  /api/schema   the fields /api/risk accepts and their allowed values
 *   POST /api/risk     score an order, choose a threshold, explain the call
 *   POST /api/ask      grounded questions about the site
 *
 * The model runs here in JavaScript. Gemini is only ever asked to put the
 * already-computed numbers into a sentence — see llm.js.
 */

import model from "../../model/export/model.json";
import encoders from "../../model/export/encoders.json";
import metrics from "../../model/export/metrics.json";

import { predict } from "./booster.js";
import { buildVector, allowedValues, ValidationError, CANONICAL_DAYS } from "./features.js";
import { choosePolicy, counterfactuals, display } from "./policy.js";
import { generate, generateOnWorkersAI, LLMError, WORKERS_AI_MODEL } from "./llm.js";
import { assistantSystemPrompt, riskSystemPrompt, riskFallbackPrompt } from "./knowledge.js";

const ALLOWED_ORIGINS = new Set([
  "https://senasayginsenyuz.com",
  "https://www.senasayginsenyuz.com",
  "https://senasayginsenyuz.github.io",
]);

// Local preview servers pick whatever port is free, so allow any loopback
// origin in development. Deployed, only the list above gets CORS headers.
function isAllowedOrigin(origin) {
  if (!origin) return false;
  if (ALLOWED_ORIGINS.has(origin)) return true;
  return /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin);
}

const LIMITS = {
  question: 500,      // characters
  body: 8 * 1024,     // bytes
  perMinute: 12,
  llmPerMinute: 6,
};

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const origin = request.headers.get("Origin");
    const cors = corsHeaders(origin);

    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });

    try {
      switch (`${request.method} ${url.pathname}`) {
        case "GET /api/health":
          return json(health(), 200, cors);
        case "GET /api/schema":
          return json(schema(), 200, cors);
        case "POST /api/risk":
          return json(await handleRisk(request, env), 200, cors);
        case "POST /api/ask":
          return json(await handleAsk(request, env), 200, cors);
        default:
          return json({ error: "not_found", paths: ["/api/health", "/api/schema", "/api/risk", "/api/ask"] }, 404, cors);
      }
    } catch (err) {
      return json(toErrorBody(err), toStatus(err), cors);
    }
  },
};

// ------------------------------------------------------------------ handlers

function health() {
  const [parent, operational, lookup] = metrics.comparison;
  return {
    ok: true,
    model: {
      trees: model.n_trees,
      features: model.features,
      runs_in: "pure JavaScript at the edge",
      parity_vs_xgboost: metrics.parity,
    },
    measured: { parent, operational, lookup },
    dataset: metrics.dataset,
  };
}

function schema() {
  return {
    endpoint: "POST /api/risk",
    required: ["shipping_mode", "market", "region", "category", "payment_type"],
    optional: {
      scheduled_days: "defaults to the window that ships with the chosen mode",
      quantity: "1–5, default 1",
      unit_price: "0–5000, default 100",
      discount_rate: "0–0.25, default 0",
      order_date: "ISO date; only month and weekday are used",
      costs: { missed_late: "default 10", false_alarm: "default 1" },
      explain: "true (default) asks Gemini for a written call; false skips it",
      lang: "tr (default) | en",
    },
    allowed_values: allowedValues(encoders),
    market_regions: MARKET_REGIONS,
    canonical_days: CANONICAL_DAYS,
    note:
      "Shipping Mode fixes the scheduled window in this dataset. Supplying a " +
      "different scheduled_days is allowed but is flagged as off-distribution.",
  };
}

async function handleRisk(request, env) {
  await rateLimit(request, "risk", LIMITS.perMinute);
  const body = await readJson(request);

  const order = body.order ?? body;
  const lang = body.lang === "en" ? "en" : "tr";
  const wantsExplanation = body.explain !== false;
  const costs = {
    missed_late: body.costs?.missed_late ?? 10,
    false_alarm: body.costs?.false_alarm ?? 1,
  };

  const { x, resolved, warnings } = buildVector(encoders, order);
  const probability = round4(predict(model, x));

  const policy = choosePolicy(metrics, costs, lang);
  const threshold = policy.chosen.threshold;
  const flagged = probability >= threshold;

  const score = (candidate) => predict(model, buildVector(encoders, candidate).x);
  const options = counterfactuals(
    { ...order, shipping_mode: resolved.shipping_mode },
    score, probability, threshold, CANONICAL_DAYS
  );

  const analysis = {
    order: resolved,
    probability,
    decision: {
      flagged,
      threshold,
      label: flagged
        ? lang === "en" ? "expedite / warn the customer" : "hızlandır / müşteriyi uyar"
        : lang === "en" ? "leave on the normal plan" : "normal planda bırak",
    },
    costs,
    policy: {
      chosen: policy.chosen,
      recommendation: policy.recommendation,
      baselines: policy.baselines,
      options: policy.options,
    },
    counterfactuals: options,
    guardrails: policy.guardrails,
    warnings,
    observed_late_rate_by_mode: metrics.collinearity.modes,
  };

  if (!wantsExplanation) return { ...analysis, explanation: null };

  const prompt = JSON.stringify(forTheModel(analysis, lang));

  try {
    await rateLimit(request, "llm", LIMITS.llmPerMinute);
    const explanation = await generate({
      apiKey: env.GEMINI_API_KEY,
      system: riskSystemPrompt(lang),
      user: prompt,
      maxOutputTokens: 900,
    });
    return { ...analysis, explanation, explained_by: "gemini" };
  } catch (err) {
    // Gemini's free tier allows 20 requests per day per model. When that runs
    // out, Cloudflare's own inference takes over rather than the section going
    // wordless. Its prompt is a separate, much blunter one — see knowledge.js.
    if (err?.name !== "RateLimited") {
      try {
        const explanation = await generateOnWorkersAI({
          ai: env.AI,
          system: riskFallbackPrompt(lang),
          user: prompt,
          maxOutputTokens: 400,
        });
        return { ...analysis, explanation, explained_by: WORKERS_AI_MODEL };
      } catch { /* fall through to the wordless answer */ }
    }
    // The numbers are the product; the sentence is a convenience. Losing every
    // language model must not lose the analysis.
    return {
      ...analysis,
      explanation: null,
      explained_by: null,
      explanation_error: err instanceof LLMError ? err.message : "açıklama üretilemedi",
    };
  }
}

async function handleAsk(request, env) {
  await rateLimit(request, "ask", LIMITS.llmPerMinute);
  const body = await readJson(request);

  const question = String(body.question ?? "").trim();
  if (!question) throw new ValidationError("question", "question boş olamaz");
  if (question.length > LIMITS.question) {
    throw new ValidationError("question", `question en fazla ${LIMITS.question} karakter`);
  }
  const lang = body.lang === "en" ? "en" : "tr";

  const answer = await generate({
    apiKey: env.GEMINI_API_KEY,
    system: assistantSystemPrompt(lang),
    // Fencing the user's text keeps instructions inside it visibly separate
    // from the system's own; rule 6 of the prompt tells the model what to do
    // when it finds any.
    user: `<ziyaretci_sorusu>\n${question}\n</ziyaretci_sorusu>`,
    maxOutputTokens: 400,
    temperature: 0.3,
  });

  return { answer, lang, grounded: true };
}

// ------------------------------------------------------------------- helpers

/**
 * Trim the analysis to what the sentence needs — a smaller prompt drifts less —
 * and hand over finished strings rather than raw numbers.
 *
 * Telling a model not to compute is weaker than giving it nothing to compute.
 * With raw values it turned 0.4109 into "%41,09" while the panel beside it read
 * "%41,1"; with pre-formatted strings there is nothing left to disagree about.
 *
 * The guardrails are passed as short codes, not as their full text: the page
 * already prints them verbatim in their own block, so a note that repeats them
 * is duplication the reader has to wade through.
 */
function forTheModel(a, lang) {
  const prob = (x) => display.probability(x, lang);
  const pc = (x) => display.percent(x, lang);
  const cost = (x) => display.cost(x, lang);

  return {
    order: a.order,
    late_probability: prob(a.probability),
    decision: {
      flagged: a.decision.flagged,
      threshold: display.ratio(a.decision.threshold, lang),
      label: a.decision.label,
    },
    costs: a.costs,
    chosen_threshold: {
      expected_cost_per_order: cost(a.policy.chosen.expected_cost),
      share_of_orders_flagged: pc(a.policy.chosen.flag_rate),
    },
    recommendation: a.policy.recommendation,
    best_rule_without_a_model: {
      name: a.policy.baselines.best_blanket_name,
      expected_cost_per_order: cost(a.policy.baselines.best_blanket),
      model_saves_per_order: cost(a.policy.baselines.saving_per_order),
    },
    counterfactuals: a.counterfactuals.slice(0, 3).map((c) => ({
      change: c.change,
      scheduled_days: c.scheduled_days,
      new_probability: prob(c.probability),
      clears_threshold: c.crosses_threshold,
    })),
    guardrail_codes: a.guardrails.map((g) => g.code),
    warnings: a.warnings,
  };
}

const MARKET_REGIONS = {
  Africa: ["Central Africa", "East Africa", "North Africa", "Southern Africa", "West Africa"],
  Europe: ["Eastern Europe", "Northern Europe", "Southern Europe", "Western Europe"],
  LATAM: ["Caribbean", "Central America", "South America"],
  "Pacific Asia": ["Central Asia", "Eastern Asia", "Oceania", "South Asia", "Southeast Asia", "West Asia"],
  USCA: ["Canada", "East of USA", "South of  USA ", "US Center ", "West of USA "],
};

async function readJson(request) {
  const raw = await request.text();
  if (raw.length > LIMITS.body) throw new ValidationError("body", "istek gövdesi çok büyük");
  try {
    return JSON.parse(raw || "{}");
  } catch {
    throw new ValidationError("body", "geçerli JSON gönderin");
  }
}

/**
 * Per-IP token bucket, held in the isolate.
 *
 * Honest about what it is: Cloudflare may run several isolates for one Worker,
 * so this caps abuse rather than enforcing an exact quota. It exists to stop a
 * loop from burning the Gemini free tier. The hard ceiling is a rate-limiting
 * rule in front of the Worker, which is where a real quota belongs.
 */
const buckets = new Map();

async function rateLimit(request, bucket, perMinute) {
  const ip = request.headers.get("CF-Connecting-IP") ?? "local";
  const key = `${bucket}:${ip}`;
  const now = Date.now();

  if (buckets.size > 5000) {
    for (const [k, v] of buckets) if (now - v.start > 60_000) buckets.delete(k);
  }

  const entry = buckets.get(key);
  if (!entry || now - entry.start > 60_000) {
    buckets.set(key, { start: now, count: 1 });
    return;
  }
  if (++entry.count > perMinute) {
    const err = new Error(`dakikada en fazla ${perMinute} istek — biraz bekleyin`);
    err.name = "RateLimited";
    throw err;
  }
}

function corsHeaders(origin) {
  const headers = {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    vary: "Origin",
  };
  if (isAllowedOrigin(origin)) {
    headers["access-control-allow-origin"] = origin;
    headers["access-control-allow-methods"] = "GET, POST, OPTIONS";
    headers["access-control-allow-headers"] = "content-type";
    headers["access-control-max-age"] = "86400";
  }
  return headers;
}

function toStatus(err) {
  if (err instanceof ValidationError) return 400;
  if (err?.name === "RateLimited") return 429;
  if (err instanceof LLMError) return err.status ?? 502;
  return 500;
}

function toErrorBody(err) {
  if (err instanceof ValidationError) {
    return { error: "validation", field: err.field, message: err.message, allowed: err.allowed };
  }
  if (err?.name === "RateLimited") return { error: "rate_limited", message: err.message };
  if (err instanceof LLMError) return { error: "llm", message: err.message };
  return { error: "internal", message: err?.message ?? "bilinmeyen hata" };
}

function json(body, status, headers) {
  return new Response(JSON.stringify(body), { status, headers });
}

function round4(n) {
  return Math.round(n * 1e4) / 1e4;
}
