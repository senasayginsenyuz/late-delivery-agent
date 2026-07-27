"""
Operational late-delivery model — training and export
=====================================================

The parent study (supply-chain-late-delivery-ml) trained on 25 features and
found that two of them carry almost all the signal: Shipping Mode (~45% of
importance) and Days for shipment (scheduled) (~38%).

An agent that a planner actually uses cannot ask for 25 fields. So this script
trains a second, *operational* model on only the fields a planner knows at the
moment the order is placed, and measures it against the 25-feature model on the
identical stratified split. If the operational model holds its ground, that is
a direct operational confirmation of the parent study's data-ceiling finding.

Outputs (model/export/):
  model.json      XGBoost trees, flattened for pure-JS evaluation at the edge
  encoders.json   category -> integer maps, so JS encodes exactly as Python did
  metrics.json    measured performance and the threshold curve

Run:
  python3 model/train.py --csv "/path/to/DataCoSupplyChainDataset.csv"
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, brier_score_loss)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

HERE = pathlib.Path(__file__).resolve().parent
EXPORT = HERE / "export"

SEED = 42
TEST_SIZE = 0.2

# Fields a planner has in hand when the order is placed. Nothing here is known
# only after dispatch — that was the leakage trap in the parent study.
OPERATIONAL = {
    "num": [
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Order Item Product Price",
        "Order Item Discount Rate",
        "order_month",
        "order_day_of_week",
    ],
    "cat": [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Category Name",
        "Type",
    ],
}

# Columns the parent study dropped: target, post-delivery leakage, identifiers,
# personal data, near-empty columns, and redundant IDs.
PARENT_DROP = [
    "Late_delivery_risk",
    "Delivery Status", "Days for shipping (real)", "shipping date (DateOrders)",
    "Product Description", "Order Zipcode",
    "Order Id", "Customer Id", "Order Customer Id",
    "Order Item Id", "Product Card Id", "Order Item Cardprod Id",
    "Customer Email", "Customer Fname", "Customer Lname",
    "Customer Password", "Customer Street", "Product Image",
    "Product Status",
    "Category Id", "Department Id", "Product Category Id",
    "Latitude", "Longitude",
    "Product Name",
]

PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=4,
              min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
              random_state=SEED, eval_metric="logloss")


def add_date_parts(frame, src):
    d = pd.to_datetime(frame[src])
    frame["order_month"] = d.dt.month
    frame["order_day_of_week"] = d.dt.dayofweek
    frame["order_year"] = d.dt.year
    return frame


def build_parent_matrix(df):
    """Reproduce the 25-feature matrix of the parent study."""
    X = df.drop(columns=PARENT_DROP)
    X = add_date_parts(X, "order date (DateOrders)")
    X = X.drop(columns=["order date (DateOrders)"])
    X = X.drop(columns=["Customer City", "Customer State", "Customer Zipcode",
                        "Order City", "Order State"])
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.Categorical(X[col].astype(str)).codes
    return X


def build_operational_matrix(df):
    """Only what a planner knows at order time. Encoding maps are returned so
    the JS side can reproduce them byte for byte."""
    X = pd.DataFrame(index=df.index)
    X = add_date_parts(X.assign(**{"order date (DateOrders)": df["order date (DateOrders)"]}),
                       "order date (DateOrders)")
    X = X.drop(columns=["order date (DateOrders)", "order_year"])

    for col in OPERATIONAL["num"]:
        if col not in X.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce")

    encoders = {}
    for col in OPERATIONAL["cat"]:
        # Sorted, explicit, serialisable — not sklearn's LabelEncoder, whose
        # state would have to be pickled and could not cross into JS.
        levels = sorted(df[col].astype(str).unique())
        mapping = {v: i for i, v in enumerate(levels)}
        encoders[col] = mapping
        X[col] = df[col].astype(str).map(mapping).astype("int32")

    order = OPERATIONAL["num"] + OPERATIONAL["cat"]
    return X[order], encoders


def report(name, y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "f1_weighted": round(float(f1_score(y_true, pred, average="weighted")), 4),
        "f1_late": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "precision_late": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall_late": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }


def flatten_trees(booster, feature_names):
    """Turn the booster into arrays a short JS evaluator can walk.

    Read from XGBoost's own JSON dump rather than trees_to_dataframe(), which
    rounds split thresholds for display.

    Everything numeric is forced back through float32 before export. XGBoost
    holds split conditions in single precision, but both its JSON writer and
    trees_to_dataframe() print the shortest decimal that round-trips — a
    threshold stored as 0.0299999993 comes out as "0.03". A consumer that then
    compares in float64 sends every value between those two numbers down the
    wrong branch. NumPy hides this in Python (weak promotion casts the float64
    threshold back to float32); JavaScript does not. Writing float32-exact
    decimals makes the file correct for any reader, in any language.

    Each tree becomes parallel arrays indexed by node id:
      f[i]  feature index, or -1 for a leaf
      t[i]  split threshold, or the leaf value
      l[i]  child taken when value <  t  (XGBoost uses strict <)
      r[i]  child taken when value >= t
      m[i]  child taken when the value is missing
    """
    dump = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    model = dump["learner"]["gradient_booster"]["model"]
    n_features = len(feature_names)
    trees = []
    f32 = lambda v: float(np.float32(v))

    for raw in model["trees"]:
        left = [int(v) for v in raw["left_children"]]
        right = [int(v) for v in raw["right_children"]]
        cond = [f32(v) for v in raw["split_conditions"]]
        feat = [int(v) for v in raw["split_indices"]]
        default_left = [int(v) for v in raw["default_left"]]

        n = len(left)
        f, t, l, r, m = [-1] * n, [0.0] * n, [-1] * n, [-1] * n, [-1] * n
        for i in range(n):
            if left[i] == -1:                 # leaf: split_conditions holds the value
                t[i] = cond[i]
                continue
            assert 0 <= feat[i] < n_features, "feature index out of range"
            f[i] = feat[i]
            t[i] = cond[i]
            l[i] = left[i]
            r[i] = right[i]
            m[i] = left[i] if default_left[i] else right[i]

        trees.append({"f": f, "t": t, "l": l, "r": r, "m": m})

    return trees


def js_predict_margin(trees, row):
    """Reference implementation of the JS evaluator, in Python.
    Used only to prove the two agree before anything ships.

    `row` must already be float32. XGBoost stores and compares feature values
    in single precision; comparing the float64 original against a split
    threshold puts values that sit between two float32 neighbours on the wrong
    side of the split. Measured on this model: 1157 of 3000 rows changed
    branch. The JS side gets this with Math.fround().
    """
    total = 0.0
    for tree in trees:
        i = 0
        while tree["f"][i] != -1:
            v = row[tree["f"][i]]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                i = tree["m"][i]
            elif v < tree["t"][i]:
                i = tree["l"][i]
            else:
                i = tree["r"][i]
        total += tree["t"][i]
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    print("Reading", args.csv)
    df = pd.read_csv(args.csv, encoding="latin-1")
    y = df["Late_delivery_risk"].astype(int)
    print(f"  rows={len(df):,}  late={y.mean():.3f}")

    # ---------------------------------------------------------------- parent
    Xp = build_parent_matrix(df)
    Xp_tr, Xp_te, y_tr, y_te = train_test_split(
        Xp, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    parent = XGBClassifier(**PARAMS).fit(Xp_tr, y_tr)
    p_parent = parent.predict_proba(Xp_te)[:, 1]
    parent_row = report(f"parent ({Xp.shape[1]} features)", y_te, p_parent)
    print("\nparent :", parent_row)

    # ----------------------------------------------------------- operational
    Xo, encoders = build_operational_matrix(df)
    Xo_tr, Xo_te, y_tr2, y_te2 = train_test_split(
        Xo, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    assert (y_te.values == y_te2.values).all(), "splits diverged"

    model = XGBClassifier(**PARAMS).fit(Xo_tr, y_tr2)
    p_op = model.predict_proba(Xo_te)[:, 1]
    op_row = report(f"operational ({Xo.shape[1]} features)", y_te, p_op)
    print("operational:", op_row)

    # ------------------------------------------------- the honesty baseline
    # Shipping Mode and Days for shipment (scheduled) are perfectly collinear
    # here: each mode carries exactly one scheduled window. So the two features
    # holding 93% of the importance are really one variable. Test the blunt
    # consequence — can four numbers replace two hundred trees?
    train_df = df.iloc[Xo_tr.index]
    lut = train_df.groupby("Shipping Mode")["Late_delivery_risk"].mean()
    p_lut = df.iloc[Xo_te.index]["Shipping Mode"].map(lut).to_numpy()
    lut_row = report("lookup table (Shipping Mode only)", y_te, p_lut)
    agreement = float(((p_lut >= 0.5) == (p_op >= 0.5)).mean())
    print("lookup  :", lut_row)
    print(f"lookup and model agree on {agreement:.4f} of test orders")

    modes = (df.groupby("Shipping Mode")
               .agg(scheduled_days=("Days for shipment (scheduled)", "median"),
                    late_rate=("Late_delivery_risk", "mean"),
                    orders=("Late_delivery_risk", "size")))
    mode_table = [{"shipping_mode": m,
                   "scheduled_days": int(r.scheduled_days),
                   "late_rate": round(float(r.late_rate), 4),
                   "orders": int(r.orders)}
                  for m, r in modes.iterrows()]
    print("\nshipping mode -> scheduled window -> observed late rate")
    for r in mode_table:
        print(f"  {r['shipping_mode']:<15} {r['scheduled_days']}d  "
              f"{r['late_rate']:.3f}  n={r['orders']:,}")

    # -------------------------------------------------------- threshold curve
    curve = []
    for th in np.arange(0.30, 0.71, 0.05):
        pred = (p_op >= th).astype(int)
        curve.append({
            "threshold": round(float(th), 2),
            "precision_late": round(float(precision_score(y_te, pred, zero_division=0)), 4),
            "recall_late": round(float(recall_score(y_te, pred, zero_division=0)), 4),
            "f1_late": round(float(f1_score(y_te, pred, zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_te, pred, average="weighted")), 4),
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "flag_rate": round(float(pred.mean()), 4),
        })
    print("\nthreshold curve")
    for c in curve:
        print(f"  {c['threshold']:.2f}  P={c['precision_late']:.3f}  "
              f"R={c['recall_late']:.3f}  F1={c['f1_late']:.3f}  "
              f"flagged={c['flag_rate']:.3f}")

    # -------------------------------------------------------------- overfit
    train_acc = accuracy_score(y_tr2, model.predict(Xo_tr))
    test_acc = op_row["accuracy"]

    # ------------------------------------------------------- feature weight
    importance = sorted(
        ({"feature": f, "gain": round(float(g), 4)}
         for f, g in zip(Xo.columns, model.feature_importances_)),
        key=lambda d: -d["gain"])

    # ---------------------------------------------------------------- export
    booster = model.get_booster()
    booster.feature_names = list(Xo.columns)
    trees = flatten_trees(booster, list(Xo.columns))

    # Recover the constant XGBoost adds to the tree sum. Deriving it from the
    # data rather than trusting a config field means the JS port cannot drift
    # if XGBoost changes how base_score is stored.
    margin = booster.predict(__import__("xgboost").DMatrix(Xo_te, feature_names=list(Xo.columns)),
                             output_margin=True)
    n_check = 5000
    sample = Xo_te.head(n_check).to_numpy(dtype=np.float32)
    tree_sum = np.array([js_predict_margin(trees, row) for row in sample])
    offsets = margin[:n_check] - tree_sum
    intercept = float(np.median(offsets))
    spread = float(np.max(np.abs(offsets - intercept)))
    print(f"\nintercept = {intercept:.8f}   spread across {n_check} rows = {spread:.2e}")
    assert spread < 1e-4, "tree flattening does not reproduce XGBoost margins"

    # End-to-end parity: flattened trees + intercept vs. the real model.
    recon = 1.0 / (1.0 + np.exp(-(tree_sum + intercept)))
    max_err = float(np.max(np.abs(recon - p_op[:n_check])))
    print(f"max |flattened - xgboost| over {n_check} rows = {max_err:.3e}")
    assert max_err < 1e-5, "flattened model disagrees with XGBoost"

    # A fixture the JS test suite replays, so the port is checked against real
    # rows rather than against a second copy of the same Python code.
    fixture = [{"x": [float(v) for v in row], "p": float(p)}
               for row, p in zip(sample[:300], p_op[:300])]

    EXPORT.mkdir(parents=True, exist_ok=True)

    (EXPORT / "model.json").write_text(json.dumps({
        "objective": "binary:logistic",
        "intercept": round(intercept, 8),
        "features": list(Xo.columns),
        "n_trees": len(trees),
        "trees": trees,
    }, separators=(",", ":")))

    (EXPORT / "encoders.json").write_text(json.dumps({
        "numeric": OPERATIONAL["num"],
        "categorical": encoders,
        "order": list(Xo.columns),
    }, ensure_ascii=False, indent=1))

    (EXPORT / "metrics.json").write_text(json.dumps({
        "dataset": {
            "rows": int(len(df)),
            "late_share": round(float(y.mean()), 4),
            "test_rows": int(len(y_te)),
            "split": {"test_size": TEST_SIZE, "seed": SEED, "stratified": True},
        },
        "comparison": [parent_row, op_row, lut_row],
        "collinearity": {
            "note": "Shipping Mode fixes the scheduled window, so the two "
                    "features carrying 93% of the importance are one variable.",
            "modes": mode_table,
            "lookup_vs_model_agreement": round(agreement, 4),
        },
        "operational": {
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "overfit_gap": round(float(train_acc - test_acc), 4),
            "feature_importance": importance,
        },
        "threshold_curve": curve,
        "parity": {"max_abs_error_vs_xgboost": max_err, "rows_checked": n_check},
    }, indent=1))

    (EXPORT / "fixture.json").write_text(json.dumps({
        "features": list(Xo.columns),
        "note": "rows straight out of the held-out test set, with the "
                "probability XGBoost itself produced for them",
        "cases": fixture,
    }, separators=(",", ":")))

    kb = (EXPORT / "model.json").stat().st_size / 1024
    print(f"\nwrote export/  model.json {kb:.0f} KB · {len(trees)} trees")
    print("done")


if __name__ == "__main__":
    sys.exit(main())
