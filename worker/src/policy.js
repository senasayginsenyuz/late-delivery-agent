/**
 * The decision layer.
 *
 * A probability is not a decision. Turning one into the other needs a cost
 * structure, and that belongs to the planner, not to the model. This module
 * takes the measured precision/recall curve and the planner's two costs, works
 * out the expected cost of every operating point, and picks the cheapest.
 *
 * It also does the part most demos skip: it checks whether the model is worth
 * running at all. Two blanket policies need no model — expedite nothing, or
 * expedite everything. If neither is meaningfully beaten, the honest answer is
 * to say so.
 */

/**
 * Wording lives here rather than in the client, because the client must not be
 * able to drop a guardrail by failing to translate it. Both languages carry
 * the same sentences and the same numbers.
 */
const WORDS = {
  tr: {
    expediteNothing: "hiçbirini hızlandırma",
    expediteEverything: "hepsini hızlandır",
    threshold: (t) => `eşik ${fixed(t, 2, "tr")}`,
    noSelectivity: (flag, all) =>
      `Seçilen eşikte işaretlenen sipariş oranı %${flag}. Bu, "${all}" demekle ` +
      `neredeyse aynı — model burada seçicilik üretmiyor.`,
    worseThanBlanket: (model, blanket, name) =>
      `Modelin beklenen maliyeti sipariş başına ${model}; "${name}" kuralının ` +
      `maliyeti ${blanket}. Bu maliyet oranında model modelsiz kuraldan pahalı. ` +
      `Öneri: modeli devreden çıkarın ve doğrudan "${name}" uygulayın.`,
    noEdge: (model, blanket, name, share) =>
      `Modelin beklenen maliyeti sipariş başına ${model}; en iyi modelsiz kural ` +
      `("${name}") ${blanket}. Kazanç yalnızca %${share} — modeli çalıştırmak ` +
      `kendini karşılamıyor.`,
    lookupParity: (lutAcc, modelAcc, agree, modelAuc, lutAuc) =>
      `Sadece Shipping Mode'a bakan 4 satırlık bir tablo ${lutAcc} doğruluk veriyor; ` +
      `200 ağaçlı model ${modelAcc}. İkisi test siparişlerinin %${agree} oranında ` +
      `aynı kararı veriyor. Modelin tek gerçek üstünlüğü sıralama kalitesi ` +
      `(ROC-AUC ${modelAuc} / ${lutAuc}) — yani sabit bir hızlandırma bütçesini ` +
      `dağıtırken işe yarar, tek siparişin evet/hayır kararında değil.`,
    useModel: (saving, share, flag) =>
      `Model sipariş başına ${saving} birim tasarruf ediyor (%${share}), ` +
      `işaretlenen sipariş oranı %${flag}.`,
    useBlanket: (name) =>
      `Bu maliyet oranında model, "${name}" kuralının üzerine anlamlı bir şey koymuyor.`,
  },
  en: {
    expediteNothing: "expedite nothing",
    expediteEverything: "expedite everything",
    threshold: (t) => `threshold ${fixed(t, 2, "en")}`,
    noSelectivity: (flag, all) =>
      `At the chosen threshold ${flag}% of orders get flagged. That is almost ` +
      `exactly "${all}" — the model is producing no selectivity here.`,
    worseThanBlanket: (model, blanket, name) =>
      `The model's expected cost is ${model} per order; the "${name}" rule costs ` +
      `${blanket}. At this cost ratio the model is more expensive than no model ` +
      `at all. Recommendation: switch it off and apply "${name}" directly.`,
    noEdge: (model, blanket, name, share) =>
      `The model's expected cost is ${model} per order; the best model-free rule ` +
      `("${name}") costs ${blanket}. The gain is only ${share}% — running the ` +
      `model does not pay for itself.`,
    lookupParity: (lutAcc, modelAcc, agree, modelAuc, lutAuc) =>
      `A four-row lookup table on Shipping Mode alone scores ${lutAcc}; the ` +
      `200-tree model scores ${modelAcc}. They make the same call on ${agree}% of ` +
      `test orders. The model's only real advantage is ranking quality ` +
      `(ROC-AUC ${modelAuc} / ${lutAuc}) — useful for spreading a fixed expedite ` +
      `budget, not for a single order's yes/no.`,
    useModel: (saving, share, flag) =>
      `The model saves ${saving} units per order (${share}%), flagging ${flag}% of them.`,
    useBlanket: (name) =>
      `At this cost ratio the model adds nothing over the "${name}" rule.`,
  },
};

/** Expected per-order outcome rates at one threshold, from measured numbers. */
function ratesAt(point, baseRate) {
  const tp = point.recall_late * baseRate;
  const fp = Math.max(0, point.flag_rate - tp);
  const fn = Math.max(0, baseRate - tp);
  const tn = Math.max(0, 1 - baseRate - fp);
  return { tp, fp, fn, tn };
}

/**
 * @param {object} metrics  model/export/metrics.json
 * @param {{missed_late:number, false_alarm:number}} costs
 * @param {"tr"|"en"} lang
 */
export function choosePolicy(metrics, costs, lang = "tr") {
  const L = WORDS[lang] ?? WORDS.tr;
  const n = (v, digits = 4) => num(v, lang, digits);
  const cMiss = Number(costs?.missed_late);
  const cFalse = Number(costs?.false_alarm);
  if (!Number.isFinite(cMiss) || cMiss < 0) throw new Error("missed_late must be >= 0");
  if (!Number.isFinite(cFalse) || cFalse < 0) throw new Error("false_alarm must be >= 0");
  if (cMiss === 0 && cFalse === 0) throw new Error("at least one cost must be > 0");

  const baseRate = metrics.dataset.late_share;

  const options = metrics.threshold_curve.map((point) => {
    const { tp, fp, fn, tn } = ratesAt(point, baseRate);
    return {
      threshold: point.threshold,
      precision_late: point.precision_late,
      recall_late: point.recall_late,
      flag_rate: point.flag_rate,
      expected_cost: round4(cMiss * fn + cFalse * fp),
      rates: { tp: round4(tp), fp: round4(fp), fn: round4(fn), tn: round4(tn) },
    };
  });

  // Ties go to the higher threshold: fewer orders disturbed for the same money.
  const best = options.reduce((a, b) =>
    b.expected_cost < a.expected_cost - 1e-12 ? b : a);

  const expediteNothing = round4(cMiss * baseRate);
  const expediteEverything = round4(cFalse * (1 - baseRate));
  const bestBlanket = Math.min(expediteNothing, expediteEverything);
  const blanketName = expediteNothing <= expediteEverything
    ? L.expediteNothing
    : L.expediteEverything;

  const saving = bestBlanket - best.expected_cost;
  const savingShare = bestBlanket > 0 ? saving / bestBlanket : 0;

  const guardrails = [];
  if (best.flag_rate >= 0.95) {
    guardrails.push({
      code: "no_selectivity",
      message: L.noSelectivity(pct(best.flag_rate, lang), L.expediteEverything),
    });
  }

  if (saving < 0) {
    // The measured curve stops at threshold 0.30, which still lets 0.6% of
    // late orders through. When a miss is dear enough, that residue costs more
    // than expediting the entire book — the model cannot reach full coverage.
    guardrails.push({
      code: "worse_than_blanket",
      message: L.worseThanBlanket(n(best.expected_cost), n(bestBlanket), blanketName),
    });
  } else if (savingShare < 0.02) {
    guardrails.push({
      code: "no_edge_over_blanket",
      message: L.noEdge(n(best.expected_cost), n(bestBlanket), blanketName,
        pct(savingShare, lang)),
    });
  }

  const lut = metrics.comparison.find((m) => m.model.startsWith("lookup"));
  const full = metrics.comparison.find((m) => m.model.startsWith("operational"));
  if (lut && full) {
    guardrails.push({
      code: "lookup_parity",
      message: L.lookupParity(
        n(lut.accuracy), n(full.accuracy),
        pct(metrics.collinearity.lookup_vs_model_agreement, lang),
        n(full.roc_auc), n(lut.roc_auc)
      ),
    });
  }

  const useModel = saving > 0 && savingShare >= 0.02 && best.flag_rate < 0.95;

  return {
    chosen: best,
    options,
    baselines: {
      expedite_nothing: expediteNothing,
      expedite_everything: expediteEverything,
      best_blanket: bestBlanket,
      best_blanket_name: blanketName,
      saving_per_order: round4(saving),
      saving_share: round4(savingShare),
    },
    recommendation: {
      use: useModel ? "model" : "blanket_rule",
      rule: useModel ? L.threshold(best.threshold) : blanketName,
      reason: useModel
        ? L.useModel(n(round4(saving)), pct(savingShare, lang), pct(best.flag_rate, lang))
        : L.useBlanket(blanketName),
    },
    guardrails,
  };
}

/**
 * What would actually change this order's risk.
 *
 * Not a feature-importance chart — importance describes the dataset. This
 * re-scores the same order under each shipping mode the planner could pick,
 * which is the only lever they actually hold.
 *
 * @param {(order:object)=>number} score  re-scores a modified order
 */
export function counterfactuals(order, score, current, threshold, canonicalDays) {
  const out = [];
  for (const mode of Object.keys(canonicalDays)) {
    if (mode === order.shipping_mode) continue;
    let p;
    try {
      // Let the scheduled window follow the mode, as it does in the data.
      p = score({ ...order, shipping_mode: mode, scheduled_days: undefined });
    } catch {
      continue;
    }
    out.push({
      change: `Shipping Mode: ${order.shipping_mode} → ${mode}`,
      shipping_mode: mode,
      scheduled_days: canonicalDays[mode],
      probability: round4(p),
      delta: round4(p - current),
      crosses_threshold: current >= threshold && p < threshold,
    });
  }
  out.sort((a, b) => a.probability - b.probability);
  return out;
}

function round4(n) {
  return Math.round(n * 1e4) / 1e4;
}

/** A decimal in the reader's own notation — Turkish uses a comma. */
function num(x, lang, digits = 4) {
  return fixed(trimZeros(Number(x).toFixed(digits)), null, lang);
}

/** Same, but keeping every requested decimal: a threshold reads 0,50 not 0,5. */
function fixed(x, digits, lang) {
  const s = digits === null ? String(x) : Number(x).toFixed(digits);
  return lang === "en" ? s : s.replace(".", ",");
}

/**
 * Percentage, two significant decimals, trailing zeros dropped so 99.03 stays
 * 99,03 while 36.00 reads 36.
 *
 * Turkish sentences here are also phrased so none of them ever needs an
 * apostrophe suffix after a number — the correct suffix depends on how the
 * number is read aloud ("%99,4'ü" but "%99,03'ünde"), and getting that wrong
 * in front of a recruiter is worse than rephrasing around it.
 */
function pct(x, lang) {
  return num(x * 100, lang, 2);
}

function trimZeros(s) {
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
}
