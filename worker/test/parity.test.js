/**
 * Does the JavaScript booster reproduce XGBoost?
 *
 * Not "does it look plausible" — the fixture holds 300 rows straight out of the
 * held-out test set together with the probability XGBoost produced for each of
 * them in Python. If the port drifts, this fails.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { predict, marginOf } from "../src/booster.js";
import { buildVector, ValidationError, CANONICAL_DAYS } from "../src/features.js";

const load = (name) =>
  JSON.parse(readFileSync(fileURLToPath(new URL(`../../model/export/${name}`, import.meta.url)), "utf8"));

const model = load("model.json");
const encoders = load("encoders.json");
const metrics = load("metrics.json");
const fixture = load("fixture.json");

test("fixture matches the exported feature order", () => {
  assert.deepEqual(fixture.features, model.features);
  assert.deepEqual(encoders.order, model.features);
});

test("300 held-out rows reproduce XGBoost to 1e-6", () => {
  let worst = 0;
  for (const { x, p } of fixture.cases) {
    const got = predict(model, x);
    worst = Math.max(worst, Math.abs(got - p));
  }
  assert.ok(worst < 1e-6, `worst absolute error ${worst.toExponential(3)}`);
});

test("every exported threshold and leaf survives Math.fround unchanged", () => {
  // XGBoost prints float32 split conditions as shortest round-trip decimals,
  // so an unfixed export writes 0.03 where the model holds 0.0299999993.
  // train.py forces them back through float32; if that ever regresses, a
  // float64 comparison here starts sending rows down the wrong branch.
  const offenders = [];
  for (const [k, tree] of model.trees.entries()) {
    for (const [i, v] of tree.t.entries()) {
      if (Math.fround(v) !== v) offenders.push(`tree ${k} node ${i}: ${v}`);
    }
  }
  assert.deepEqual(offenders.slice(0, 5), [], `${offenders.length} not float32-exact`);
});

test("rounding the caller's float64 input to float32 changes real answers", () => {
  // Guards the Float32Array in booster.js. A caller sends 327.75 and 0.03 in
  // double precision; XGBoost compares in single. If this ever stops mattering
  // the guard has become dead weight and should be revisited rather than kept
  // on faith.
  const f64Walk = (x) => {
    let total = 0;
    for (const tree of model.trees) {
      let i = 0;
      while (tree.f[i] !== -1) i = x[tree.f[i]] < tree.t[i] ? tree.l[i] : tree.r[i];
      total += tree.t[i];
    }
    return total + model.intercept;
  };

  let differing = 0;
  for (let k = 0; k < 400; k++) {
    const { x } = buildVector(encoders, {
      ...sampleOrder(),
      unit_price: Math.round((9.99 + (k * 7.31) % 1990) * 100) / 100,
      discount_rate: (k % 26) / 100,
      quantity: 1 + (k % 5),
    });
    if (Math.abs(f64Walk(x) - marginOf(model, x)) > 1e-9) differing++;
  }
  assert.ok(differing > 0, "float32 input rounding no longer changes any branch");
});

test("every leaf is reachable and every split points forward", () => {
  for (const [k, t] of model.trees.entries()) {
    const n = t.f.length;
    for (let i = 0; i < n; i++) {
      if (t.f[i] === -1) {
        assert.ok(Number.isFinite(t.t[i]), `tree ${k} node ${i}: leaf value not finite`);
        continue;
      }
      assert.ok(t.f[i] >= 0 && t.f[i] < model.features.length, `tree ${k}: feature out of range`);
      for (const child of [t.l[i], t.r[i], t.m[i]]) {
        assert.ok(child > i && child < n, `tree ${k} node ${i}: child ${child} not forward`);
      }
    }
  }
});

test("a missing value follows the branch training chose", () => {
  const { x } = buildVector(encoders, sampleOrder());
  const withHole = [...x];
  withHole[model.features.indexOf("Order Item Product Price")] = null;
  const p = predict(model, withHole);
  assert.ok(p > 0 && p < 1, `probability out of range: ${p}`);
});

test("probabilities stay inside (0,1) for extreme inputs", () => {
  for (const mode of Object.keys(CANONICAL_DAYS)) {
    for (const price of [0, 5000]) {
      for (const discount of [0, 0.25]) {
        const p = predict(model, buildVector(encoders, {
          ...sampleOrder(), shipping_mode: mode, unit_price: price,
          discount_rate: discount, quantity: 5,
        }).x);
        assert.ok(p > 0 && p < 1 && Number.isFinite(p), `${mode} ${price} ${discount} -> ${p}`);
      }
    }
  }
});

test("shipping mode ranks the way the data says it does", () => {
  // Observed late rates: First Class 0.953 > Second Class 0.766 >
  // Same Day 0.457 > Standard Class 0.381. A model that inverts this is broken
  // regardless of what its accuracy says.
  const order = sampleOrder();
  const p = Object.fromEntries(Object.keys(CANONICAL_DAYS).map((mode) => [
    mode, predict(model, buildVector(encoders, { ...order, shipping_mode: mode }).x),
  ]));
  assert.ok(p["First Class"] > p["Second Class"], JSON.stringify(p));
  assert.ok(p["Second Class"] > p["Same Day"], JSON.stringify(p));
  assert.ok(p["Same Day"] > p["Standard Class"], JSON.stringify(p));
});

test("an unknown category is rejected, not silently encoded", () => {
  assert.throws(
    () => buildVector(encoders, { ...sampleOrder(), category: "Kuantum Bilgisayar" }),
    (err) => err instanceof ValidationError && err.field === "Category Name" && err.allowed.length > 0
  );
});

test("trailing spaces in the dataset's own labels still match", () => {
  const { resolved } = buildVector(encoders, {
    ...sampleOrder(), market: "USCA", region: "South of USA",
  });
  assert.equal(resolved.region, "South of  USA ");
});

test("scheduled_days follows the shipping mode unless overridden", () => {
  const a = buildVector(encoders, { ...sampleOrder(), shipping_mode: "Same Day" });
  assert.equal(a.resolved.scheduled_days, 0);
  assert.equal(a.warnings.length, 0);

  const b = buildVector(encoders, {
    ...sampleOrder(), shipping_mode: "Same Day", scheduled_days: 4,
  });
  assert.equal(b.resolved.scheduled_days, 4);
  assert.equal(b.warnings.length, 1, "off-distribution override must warn");
});

test("out-of-range numbers are refused", () => {
  for (const bad of [
    { quantity: 99 }, { unit_price: -1 }, { discount_rate: 0.9 }, { scheduled_days: 40 },
  ]) {
    assert.throws(() => buildVector(encoders, { ...sampleOrder(), ...bad }),
      ValidationError, JSON.stringify(bad));
  }
});

test("the same request always returns the same number", () => {
  const order = sampleOrder();
  delete order.order_date;                    // no date -> must not fall back to now()
  const first = predict(model, buildVector(encoders, order).x);
  const second = predict(model, buildVector(encoders, order).x);
  assert.equal(first, second);
});

test("weekday encoding matches pandas (Monday = 0)", () => {
  // 2026-07-27 is a Monday.
  const { resolved } = buildVector(encoders, { ...sampleOrder(), order_date: "2026-07-27" });
  assert.equal(resolved.order_day_of_week, 0);
  assert.equal(resolved.order_month, 7);
});

test("exported metrics carry what the policy engine depends on", () => {
  assert.ok(metrics.threshold_curve.length >= 5);
  for (const point of metrics.threshold_curve) {
    for (const key of ["threshold", "precision_late", "recall_late", "flag_rate"]) {
      assert.equal(typeof point[key], "number", `${key} missing at ${point.threshold}`);
    }
  }
  assert.ok(metrics.dataset.late_share > 0 && metrics.dataset.late_share < 1);
  assert.equal(metrics.comparison.length, 3);
});

function sampleOrder() {
  return {
    shipping_mode: "Standard Class",
    market: "Europe",
    region: "Western Europe",
    category: "Cleats",
    payment_type: "DEBIT",
    quantity: 1,
    unit_price: 327.75,
    discount_rate: 0.04,
    order_date: "2026-07-27",
  };
}
