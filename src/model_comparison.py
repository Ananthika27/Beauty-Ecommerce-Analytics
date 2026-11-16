"""Compare gradient-boosting libraries with honest, leak-free hyperparameter tuning.

* Tuning: random search; train on Nov+Dec snapshots, score on the Jan snapshot.
* Final: refit best settings on Nov+Dec+Jan, evaluate ONCE on the untouched Feb snapshot.

Usage:  python src/model_comparison.py
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import churn_model as cm

N_TRIALS = 12


def sample_params(rng):
    return {
        "depth": int(rng.integers(3, 8)),
        "lr": float(10 ** rng.uniform(-2.0, -0.8)),
        "n": int(rng.choice([150, 300, 500])),
        "l2": float(10 ** rng.uniform(-1, 1.5)),
        "min_leaf": int(rng.choice([20, 50, 100, 200])),
    }


def build(name, p):
    if name == "sklearn HistGradientBoosting":
        return HistGradientBoostingClassifier(max_depth=p["depth"], learning_rate=p["lr"], max_iter=p["n"],
                                              l2_regularization=p["l2"], min_samples_leaf=p["min_leaf"], random_state=42)
    if name == "LightGBM":
        return lgb.LGBMClassifier(max_depth=p["depth"], learning_rate=p["lr"], n_estimators=p["n"], reg_lambda=p["l2"],
                                  min_child_samples=p["min_leaf"], num_leaves=2 ** p["depth"], random_state=42, verbose=-1)
    return xgb.XGBClassifier(max_depth=p["depth"], learning_rate=p["lr"], n_estimators=p["n"], reg_lambda=p["l2"],
                             min_child_weight=max(p["min_leaf"] / 20, 1), random_state=42, tree_method="hist", verbosity=0)


def main():
    purchases, eng = cm.load_purchases(), cm.load_engagement()
    feats = cm.ALL_FEATURES
    train = cm.make_dataset(purchases, ["2019-11-01", "2019-12-01"], eng)
    val = cm.make_dataset(purchases, ["2020-01-01"], eng)
    test = cm.make_dataset(purchases, ["2020-02-01"], eng)
    full = pd.concat([train, val])

    rows = []
    for name in ["sklearn HistGradientBoosting", "LightGBM", "XGBoost"]:
        rng = np.random.default_rng(7)
        t0 = time.time()
        # default settings first, so tuning has to beat them
        trials = [{"depth": 4, "lr": 0.06, "n": 300, "l2": 1.0, "min_leaf": 20}] + [sample_params(rng) for _ in range(N_TRIALS)]
        scored = []
        for p in trials:
            m = build(name, p).fit(train[feats], train["label"])
            scored.append((roc_auc_score(val["label"], m.predict_proba(val[feats])[:, 1]), p))
        best_val, best = max(scored, key=lambda x: x[0])
        default_val = scored[0][0]

        final = build(name, best).fit(full[feats], full["label"])
        p_test = final.predict_proba(test[feats])[:, 1]
        default_final = build(name, trials[0]).fit(full[feats], full["label"])
        p_def = default_final.predict_proba(test[feats])[:, 1]
        lo, hi = cm.bootstrap_ci(test["label"], p_test, n=100)
        rows.append({
            "model": name,
            "test_roc_auc_default": round(roc_auc_score(test["label"], p_def), 4),
            "test_roc_auc_tuned": round(roc_auc_score(test["label"], p_test), 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "test_pr_auc_tuned": round(average_precision_score(test["label"], p_test), 4),
            "top_decile_lift_tuned": round(cm.top_decile_lift(test["label"].values, p_test), 2),
            "val_auc_default": round(default_val, 4), "val_auc_best": round(best_val, 4),
            "best_params": str(best), "seconds": round(time.time() - t0),
        })
        print(rows[-1])

    pd.DataFrame(rows).to_csv(cm.OUT / "model_library_comparison.csv", index=False)


if __name__ == "__main__":
    main()
