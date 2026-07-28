# Late Delivery Decision Agent

*A trained XGBoost model running in JavaScript at the edge, a cost-driven
threshold policy on top of it, and a language model that is allowed to write
the sentence but never to compute the number.*

Live demo: **[senasayginsenyuz.com](https://senasayginsenyuz.com)** ·
Parent study: **[supply-chain-late-delivery-ml](https://github.com/senasayginsenyuz/supply-chain-late-delivery-ml)**

---

## What this is

The parent study answered *"can we predict which orders arrive late?"* It could,
to a point, and it explained why the point was where it was.

This project answers the next question, which is the one a planner actually
asks: **"so what do I do about this order?"**

A probability is not a decision. Turning one into the other needs a threshold,
and a threshold is a business question, not a modelling question — it depends
on what a missed late delivery costs you against what a needless expedite
costs you. That is the whole job of this agent.

---

## Architecture

```
order  ─▶  feature encoding  ─▶  XGBoost (200 trees, pure JS)  ─▶  probability
                                                                      │
                     cost of a miss  ─┐                               │
              cost of a false alarm  ─┴─▶  threshold policy  ◀────────┘
                                                │
                                                ├─▶  decision + counterfactuals
                                                ├─▶  guardrails
                                                └─▶  Gemini writes it up
```

Three deliberate boundaries:

**The model runs where the request lands.** The 200 trees are flattened out of
XGBoost's own JSON dump into parallel arrays and walked by ~30 lines of
JavaScript. No Python at runtime, no model server, no cold start. A scoring
call is well under a millisecond.

**The policy is arithmetic, not a prompt.** Expected cost per order is computed
from the measured precision/recall curve at every threshold, and the cheapest
one wins. The same inputs always give the same threshold.

**The language model never touches a number.** Every figure is computed before
Gemini is called and handed to it **already formatted as a string** — telling a
model not to compute is weaker than leaving it nothing to compute. With raw
values it rendered `0.4109` as "%41,09" while the panel beside it read "%41,1".
If Gemini is unreachable, the API still returns the full analysis with
`explanation: null` — the numbers are the product, the sentence is a convenience.

### Picking the model, by measurement

The obvious default is a trap. Probed against a fresh free-tier key:

| Model | Result |
|---|---|
| `gemini-2.5-flash` / `-lite` | "no longer available to new users" |
| `gemini-2.0-flash` | quota exceeded |
| **`gemini-3.6-flash`** | 1.26 s — **primary** |
| `gemini-3-flash-preview` | 1.37 s — fallback |
| `gemini-flash-latest` | 1.74 s — last resort (floating alias) |

Two API details that fail loudly rather than gracefully: Gemini 3 takes
`thinkingLevel`, not `thinkingBudget` (the latter is rejected as an invalid
argument), and thinking tokens are drawn from `maxOutputTokens`, so a tight
ceiling returns a *truncated* answer — one test run stopped at
"…Planlamacının belirlediği" and nothing after it. A half-sentence reads as
finished and the clause it swallows is usually the guardrail, so `MAX_TOKENS`
is treated as a failure and retried on the next model, not returned.

---

## The finding that shaped the design

The parent study reported that two features carry most of the signal:
`Shipping Mode` (~45% of importance) and `Days for shipment (scheduled)` (~38%).
Retrained on the fields a planner has at order time, that concentration gets
sharper — 93.1% across the two.

Checking why turned up something the importance chart cannot show. In this
dataset each shipping mode carries **exactly one** scheduled window:

| Shipping Mode | Scheduled window | Observed late rate | Orders |
|---|---|---|---|
| First Class | 1 day | 95.3% | 27,814 |
| Second Class | 2 days | 76.6% | 35,216 |
| Same Day | 0 days | 45.7% | 9,737 |
| Standard Class | 4 days | 38.1% | 107,752 |

The two dominant features are one variable wearing two hats. So the blunt test:
learn those four numbers from the training split and use them as the entire
model.

| Model | Features | Accuracy | Weighted F1 | ROC-AUC |
|---|---|---|---|---|
| Parent study | 25 | 0.7134 | 0.7077 | 0.7761 |
| Operational (this agent) | 11 | 0.6972 | 0.6919 | 0.7464 |
| **Lookup table** | **1** | **0.6967** | **0.6905** | **0.7295** |

**Four numbers score 0.6967 where two hundred trees score 0.6972, and the two
agree on 99.03% of test orders.**

Note the counterintuitive ordering: faster shipping is *more* likely to be
late. First Class is late 95.3% of the time and Standard Class 38.1%, because
lateness here is measured against the promise, and the tighter promise is the
harder one to keep. Anyone reading the importance chart alone would have taken
"Shipping Mode matters most" and moved on.

This is not a reason to throw the model away — its ranking is genuinely better
(ROC-AUC 0.746 against 0.730), which is what matters when a fixed expedite
budget has to be spread across a book of orders. It is a reason for the agent
to say so out loud, which it does. Every response carries the comparison.

---

## What the agent returns

`POST /api/risk`

```json
{
  "order": { "shipping_mode": "First Class", "market": "Europe",
             "region": "Western Europe", "category": "Cleats",
             "payment_type": "DEBIT", "unit_price": 327.75 },
  "costs": { "missed_late": 1, "false_alarm": 1 }
}
```

```jsonc
{
  "probability": 0.9971,
  "decision":   { "flagged": true, "threshold": 0.5 },
  "policy": {
    "chosen":         { "threshold": 0.5, "expected_cost": 0.3028, … },
    "recommendation": { "use": "model", "rule": "eşik 0.5", … },
    "baselines":      { "expedite_nothing": 0.5483, "expedite_everything": 0.4517,
                        "saving_per_order": 0.1489, "saving_share": 0.3296 }
  },
  "counterfactuals": [
    { "change": "Shipping Mode: First Class → Standard Class",
      "probability": 0.4211, "delta": -0.576, "crosses_threshold": true }
  ],
  "guardrails": [ … ],
  "explanation": "…"
}
```

**Counterfactuals, not feature importance.** Importance describes the dataset;
it does not tell a planner what to do. Each alternative shipping mode is
re-scored through the same model, with the scheduled window following the mode
as it does in the data, so the output is the lever the planner actually holds.

**Guardrails are not optional.** Three can fire, and the prompt requires the
language model to relay every one:

- `lookup_parity` — always present: four numbers match the model on 99.03% of
  orders, and here is where the model still earns its place.
- `no_selectivity` — the chosen threshold flags 95% or more of the book, which
  is "expedite everything" with extra steps.
- `no_edge_over_blanket` / `worse_than_blanket` — the model's expected cost
  against the better of the two model-free policies.

That last one has teeth. Price a miss at 100× a false alarm and the agent
answers:

> Modelin beklenen maliyeti sipariş başına 0.5949; "hepsini hızlandır"
> kuralının maliyeti 0.4517. Bu maliyet oranında model modelsiz kuraldan
> pahalı. Öneri: modeli devreden çıkarın ve doğrudan "hepsini hızlandır"
> uygulayın.

The measured curve bottoms out at threshold 0.30 and still lets 0.6% of late
orders slip. Priced high enough, that residue costs more than expediting the
entire book — so the honest recommendation is to stop using the model. An agent
that cannot reach that conclusion is not a decision support system.

---

## Porting XGBoost to JavaScript

Two precision traps, both silent, both caught by tests.

**Thresholds.** XGBoost holds split conditions in float32 but serialises them
as the shortest decimal that round-trips, so a threshold stored as
`0.0299999993` is written `"0.03"`. Read that back as float64 and every value
between the two goes down the wrong branch. NumPy hides this in Python — weak
promotion casts the float64 threshold back to float32 — which is exactly why it
survives to bite a port. `train.py` forces every threshold and leaf back
through float32 before export, and a test asserts all 151 KB of them survive
`Math.fround` unchanged.

**Feature values.** A caller sends `327.75` and `0.03` as float64; XGBoost
compares in float32. Measured on this model with realistic orders, **1233 of
4000** took a different branch without the rounding, moving the probability by
up to 0.038. `Float32Array` assignment does it.

With both in place the JavaScript reproduces XGBoost to **1.9 × 10⁻⁷** across
5,000 held-out rows. `test/parity.test.js` replays 300 of those rows against
the probabilities XGBoost itself produced, so the port is checked against the
real thing rather than against a second copy of the same logic.

---

## Repository

```
model/
  train.py               training, comparison, export — one command, reproducible
  export/
    model.json           200 trees, flattened, float32-exact       (151 KB)
    encoders.json        category -> integer maps used in training
    metrics.json         measured performance and threshold curve
    fixture.json         300 held-out rows + XGBoost's own answers
worker/
  src/booster.js         the tree walk
  src/features.js        order -> feature vector, with real validation
  src/policy.js          threshold choice, counterfactuals, guardrails
  src/llm.js             Gemini
  src/knowledge.js       grounding for the site assistant
  src/index.js           routing, CORS, rate limiting
  test/                  30 tests, no dependencies
```

## Running it

```bash
python3 model/train.py --csv path/to/DataCoSupplyChainDataset.csv
```

```bash
cd worker && npm install && npm test
```

```bash
cd worker && npx wrangler dev
```

Deployment needs a Gemini API key as a secret — never a variable, never in the
repo:

```bash
npx wrangler secret put GEMINI_API_KEY
```

The dataset is not in this repository. It is
[DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
on Kaggle, 180,519 orders, and it is worth noting that the raw file ships with
a `Customer Password` column — a reminder that production data arrives carrying
things that must be identified and dropped before any modelling starts.

## Limitations

- **The ceiling is the data.** Recall for late orders plateaus near 55%. The
  parent study established that adding engineered features does not move it;
  the real drivers — weather, traffic, carrier reliability, actual distance —
  were never collected.
- **The threshold curve is coarse.** Nine points, 0.30 to 0.70 in steps of
  0.05. The policy engine picks among those nine, not from a continuum.
- **Costs are relative.** `missed_late` and `false_alarm` are ratios, not
  currency. Only their proportion changes the decision.
- **Rate limiting is per-isolate.** Cloudflare may run several isolates for one
  Worker, so the in-memory bucket caps abuse rather than enforcing an exact
  quota. A real quota belongs in a rate-limiting rule in front of the Worker.
- **The site assistant is closed-book by construction.** It answers only from
  `knowledge.js`. Ask it something outside that and it says the site does not
  cover it — which is the intended behaviour, not a gap. It carries no phone
  number or e-mail address, so it cannot become the hole in a privacy policy
  the rest of the site keeps. Adversarially tested against instruction
  override, forged "SYSTEM:" turns, urgency pretexts for contact details, and
  invented employers; it also reports date ranges rather than computing
  durations, and emits plain text because the page prints the answer verbatim.

---

**Sena Saygın Şenyüz** — Industrial Engineer · AI & Data · Business Analysis
[senasayginsenyuz.com](https://senasayginsenyuz.com) ·
[LinkedIn](https://linkedin.com/in/senasayginsenyuz)
