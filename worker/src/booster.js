/**
 * XGBoost booster evaluator, pure JavaScript, no dependencies.
 *
 * The trees come from model/train.py, flattened out of XGBoost's own JSON dump
 * into parallel arrays. Walking them here means the trained model runs at the
 * edge in under a millisecond with no Python runtime anywhere.
 *
 * Single precision is the whole difficulty, on both sides of the comparison:
 *
 *   Thresholds — XGBoost holds them as float32 but prints the shortest decimal
 *     that round-trips, so 0.0299999993 is written "0.03". model/train.py
 *     forces them back through float32 before export; test/parity.test.js
 *     asserts every one of them survives Math.fround unchanged.
 *
 *   Feature values — a caller sends 327.75 and 0.03 as float64. XGBoost
 *     compares in float32, so we must too. Measured on this model with
 *     realistic orders: 1233 of 4000 took a different branch without the
 *     rounding, shifting the probability by up to 0.038. Float32Array
 *     assignment does it for us.
 *
 * test/parity.test.js replays 300 real held-out rows against the probabilities
 * XGBoost itself produced for them.
 */

/** Feature vector -> raw sum of leaf values (log-odds before the intercept). */
export function marginOf(model, x) {
  const x32 = toFloat32(x);
  const trees = model.trees;
  let total = 0;

  for (let k = 0; k < trees.length; k++) {
    const { f, t, l, r, m } = trees[k];
    let i = 0;
    while (f[i] !== -1) {
      const v = x32[f[i]];
      // NaN marks a value the caller did not supply; XGBoost sends those down
      // the branch it learned during training, which is what m[i] holds.
      i = Number.isNaN(v) ? m[i] : v < t[i] ? l[i] : r[i];
    }
    total += t[i];
  }
  return total + model.intercept;
}

/** Feature vector -> probability that the order arrives late. */
export function predict(model, x) {
  const z = marginOf(model, x);
  // Branch on the sign so the exponent never overflows for extreme margins.
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

function toFloat32(x) {
  if (x instanceof Float32Array) return x;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    out[i] = v === null || v === undefined ? NaN : v;
  }
  return out;
}
