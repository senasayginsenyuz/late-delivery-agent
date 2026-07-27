/**
 * The decision layer is where a wrong answer costs money, so it is tested
 * against hand-checked cost structures rather than against itself.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { choosePolicy, counterfactuals } from "../src/policy.js";
import { predict } from "../src/booster.js";
import { buildVector, CANONICAL_DAYS } from "../src/features.js";

const load = (name) =>
  JSON.parse(readFileSync(fileURLToPath(new URL(`../../model/export/${name}`, import.meta.url)), "utf8"));

const metrics = load("metrics.json");
const model = load("model.json");
const encoders = load("encoders.json");

test("balanced costs land on the threshold the study selected", () => {
  const { chosen } = choosePolicy(metrics, { missed_late: 1, false_alarm: 1 });
  assert.equal(chosen.threshold, 0.5);
});

test("expensive misses push the threshold down", () => {
  const cheap = choosePolicy(metrics, { missed_late: 1, false_alarm: 1 }).chosen.threshold;
  const dear = choosePolicy(metrics, { missed_late: 20, false_alarm: 1 }).chosen.threshold;
  assert.ok(dear < cheap, `${dear} should be below ${cheap}`);
});

test("expensive false alarms push the threshold up", () => {
  const a = choosePolicy(metrics, { missed_late: 1, false_alarm: 1 }).chosen.threshold;
  const b = choosePolicy(metrics, { missed_late: 1, false_alarm: 20 }).chosen.threshold;
  assert.ok(b >= a, `${b} should be at or above ${a}`);
});

test("expected cost is arithmetic, not vibes", () => {
  const costs = { missed_late: 7, false_alarm: 3 };
  const { options } = choosePolicy(metrics, costs);
  const base = metrics.dataset.late_share;

  for (const option of options) {
    const point = metrics.threshold_curve.find((p) => p.threshold === option.threshold);
    const tp = point.recall_late * base;
    const fp = Math.max(0, point.flag_rate - tp);
    const fn = Math.max(0, base - tp);
    const expected = costs.missed_late * fn + costs.false_alarm * fp;
    assert.ok(Math.abs(option.expected_cost - expected) < 1e-3,
      `t=${option.threshold}: ${option.expected_cost} vs ${expected}`);
  }
});

test("the chosen option really is the cheapest one offered", () => {
  for (const costs of [
    { missed_late: 1, false_alarm: 1 }, { missed_late: 50, false_alarm: 1 },
    { missed_late: 1, false_alarm: 50 }, { missed_late: 4, false_alarm: 3 },
    { missed_late: 0, false_alarm: 1 }, { missed_late: 1, false_alarm: 0 },
  ]) {
    const { chosen, options } = choosePolicy(metrics, costs);
    const min = Math.min(...options.map((o) => o.expected_cost));
    assert.ok(Math.abs(chosen.expected_cost - min) < 1e-9, JSON.stringify(costs));
  }
});

test("it admits when a blanket rule is just as good", () => {
  // Misses three times dearer than false alarms: flagging everything is within
  // a rounding error of the best threshold. The agent has to say so.
  const { guardrails, baselines, chosen } = choosePolicy(metrics, { missed_late: 3, false_alarm: 1 });
  const codes = guardrails.map((g) => g.code);
  assert.ok(codes.includes("no_edge_over_blanket"), JSON.stringify(guardrails, null, 1));
  assert.ok(baselines.saving_share < 0.02);
  assert.ok(chosen.flag_rate > 0.9);
});

test("it admits when it has stopped being selective", () => {
  const { guardrails } = choosePolicy(metrics, { missed_late: 100, false_alarm: 1 });
  assert.ok(guardrails.some((g) => g.code === "no_selectivity"));
});

test("it says so outright when a blanket rule is cheaper", () => {
  // The measured curve bottoms out at threshold 0.30 and still lets 0.6% of
  // late orders through. Price a miss high enough and that residue costs more
  // than expediting everything — the model simply cannot reach full coverage.
  const { guardrails, baselines, recommendation } =
    choosePolicy(metrics, { missed_late: 100, false_alarm: 1 });

  assert.ok(guardrails.some((g) => g.code === "worse_than_blanket"),
    JSON.stringify(guardrails.map((g) => g.code)));
  assert.ok(!guardrails.some((g) => g.code === "no_edge_over_blanket"),
    "worse_than_blanket and no_edge_over_blanket are mutually exclusive");
  assert.ok(baselines.saving_per_order < 0);
  assert.equal(recommendation.use, "blanket_rule");
  assert.equal(recommendation.rule, "hepsini hızlandır");
});

test("English requests get English guardrails and English decimals", () => {
  // The client must not be able to drop a guardrail by failing to translate
  // it, so both languages are produced here.
  const { guardrails, recommendation } =
    choosePolicy(metrics, { missed_late: 100, false_alarm: 1 }, "en");

  const codes = guardrails.map((g) => g.code).sort();
  assert.deepEqual(codes, ["lookup_parity", "no_selectivity", "worse_than_blanket"]);
  assert.equal(recommendation.rule, "expedite everything");

  for (const text of [...guardrails.map((g) => g.message), recommendation.reason]) {
    assert.doesNotMatch(text, /\d,\d/, `Turkish decimal comma in English text: ${text}`);
    assert.doesNotMatch(text, /[şğıçöü]/i, `Turkish letters in English text: ${text}`);
  }
});

test("Turkish requests never leak a dot decimal into prose", () => {
  for (const costs of [
    { missed_late: 1, false_alarm: 1 }, { missed_late: 3, false_alarm: 1 },
    { missed_late: 100, false_alarm: 1 }, { missed_late: 1, false_alarm: 30 },
  ]) {
    const { guardrails, recommendation } = choosePolicy(metrics, costs, "tr");
    for (const text of [...guardrails.map((g) => g.message), recommendation.reason]) {
      assert.doesNotMatch(text, /\d\.\d/, `dot decimal in Turkish text: ${text}`);
    }
  }
});

test("both languages carry the same guardrail codes for the same costs", () => {
  for (const costs of [
    { missed_late: 1, false_alarm: 1 }, { missed_late: 3, false_alarm: 1 },
    { missed_late: 100, false_alarm: 1 }, { missed_late: 1, false_alarm: 30 },
  ]) {
    const tr = choosePolicy(metrics, costs, "tr");
    const en = choosePolicy(metrics, costs, "en");
    assert.deepEqual(tr.guardrails.map((g) => g.code), en.guardrails.map((g) => g.code),
      JSON.stringify(costs));
    assert.equal(tr.recommendation.use, en.recommendation.use);
    assert.equal(tr.chosen.threshold, en.chosen.threshold);
  }
});

test("the 99.03% agreement figure keeps its precision", () => {
  // Rounded to one decimal this reads 99.0, which understates the finding.
  const { guardrails } = choosePolicy(metrics, { missed_late: 1, false_alarm: 1 });
  const lookup = guardrails.find((g) => g.code === "lookup_parity");
  assert.match(lookup.message, /%99,03/, lookup.message);
});

test("it recommends the model only when the model earns it", () => {
  const good = choosePolicy(metrics, { missed_late: 1, false_alarm: 1 }).recommendation;
  assert.equal(good.use, "model");
  assert.equal(good.rule, "eşik 0,50");

  for (const costs of [
    { missed_late: 3, false_alarm: 1 },
    { missed_late: 100, false_alarm: 1 },
    { missed_late: 1, false_alarm: 30 },
  ]) {
    assert.equal(choosePolicy(metrics, costs).recommendation.use, "blanket_rule",
      JSON.stringify(costs));
  }
});

test("no guardrail sentence ends a number with a Turkish suffix", () => {
  // "%99,4'i" is wrong for a number read "doksan dokuz virgül dört" — the
  // suffix depends on the reading. The messages are phrased to avoid it.
  for (const costs of [
    { missed_late: 1, false_alarm: 1 }, { missed_late: 3, false_alarm: 1 },
    { missed_late: 100, false_alarm: 1 }, { missed_late: 1, false_alarm: 30 },
  ]) {
    const { guardrails, recommendation } = choosePolicy(metrics, costs);
    for (const text of [...guardrails.map((g) => g.message), recommendation.reason]) {
      assert.doesNotMatch(text, /\d['’]/, `suffix after a number: ${text}`);
      assert.doesNotMatch(text, /%-|%−/, `negative percentage in prose: ${text}`);
    }
  }
});

test("the lookup-table comparison is always disclosed", () => {
  for (const costs of [
    { missed_late: 1, false_alarm: 1 }, { missed_late: 10, false_alarm: 1 },
  ]) {
    const { guardrails } = choosePolicy(metrics, costs);
    assert.ok(guardrails.some((g) => g.code === "lookup_parity"), JSON.stringify(costs));
  }
});

test("balanced costs do beat both blanket rules", () => {
  const { baselines, chosen } = choosePolicy(metrics, { missed_late: 1, false_alarm: 1 });
  assert.ok(chosen.expected_cost < baselines.expedite_nothing);
  assert.ok(chosen.expected_cost < baselines.expedite_everything);
  assert.ok(baselines.saving_share > 0.02);
});

test("nonsense costs are refused", () => {
  for (const bad of [
    { missed_late: -1, false_alarm: 1 }, { missed_late: 0, false_alarm: 0 },
    { missed_late: "çok", false_alarm: 1 }, {},
  ]) {
    assert.throws(() => choosePolicy(metrics, bad), JSON.stringify(bad));
  }
});

test("counterfactuals offer the other three modes, cheapest risk first", () => {
  const order = {
    shipping_mode: "First Class", market: "Europe", region: "Western Europe",
    category: "Cleats", payment_type: "DEBIT", order_date: "2026-07-27",
  };
  const score = (candidate) => predict(model, buildVector(encoders, candidate).x);
  const current = score(order);
  const options = counterfactuals(order, score, current, 0.5, CANONICAL_DAYS);

  assert.equal(options.length, 3);
  assert.ok(!options.some((o) => o.shipping_mode === "First Class"));
  for (let i = 1; i < options.length; i++) {
    assert.ok(options[i - 1].probability <= options[i].probability, "not sorted");
  }
  // First Class is the worst mode in this data, so a switch must help and the
  // best switch must clear the threshold.
  assert.ok(options.every((o) => o.delta < 0), JSON.stringify(options, null, 1));
  assert.ok(options[0].crosses_threshold, JSON.stringify(options[0]));
});

test("counterfactuals let the window follow the mode", () => {
  const order = {
    shipping_mode: "Standard Class", scheduled_days: 4, market: "LATAM",
    region: "South America", category: "Cleats", payment_type: "TRANSFER",
  };
  const score = (candidate) => predict(model, buildVector(encoders, candidate).x);
  const options = counterfactuals(order, score, score(order), 0.5, CANONICAL_DAYS);
  for (const option of options) {
    assert.equal(option.scheduled_days, CANONICAL_DAYS[option.shipping_mode]);
  }
});
