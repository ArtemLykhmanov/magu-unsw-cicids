from __future__ import annotations

import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import warnings

# Make project src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.pipeline import Pipeline

from src.config import Paths
from src.data_io import load_csv, unsw_files
from src.preprocess import build_preprocessor
from src.models import make_models
from src.metrics_ext import pr_auc, roc_auc, f1_at_threshold
from src.eval_utils import mcnemar_test


def main():
    parser = argparse.ArgumentParser(description="Train RF/XGB pipelines on UNSW and save for McNemar eval")
    parser.add_argument("--out-dir", type=Path, default=Path("out/models"))
    args = parser.parse_args()

    # Silence noisy library warnings and replace them with concise log notes.
    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories in columns .* during transform. These unknown categories will be encoded as all zeros",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Parameters: { \"use_label_encoder\" } are not used.*",
        category=UserWarning,
    )
    print(
        "Note: sklearn OneHotEncoder may encounter unseen categories at transform time; "
        "they are encoded as all-zero vectors and do not break the model."
    )
    print(
        "Note: XGBoost parameter 'use_label_encoder' is deprecated/ignored; "
        "we suppress its warning as it does not affect results."
    )

    paths = Paths.from_root('.')
    files = unsw_files(paths.data)
    Xtr_raw, ytr = load_csv(files["train"])  # binary labels
    Xte_raw, yte = load_csv(files["test"])   # binary labels

    pre = build_preprocessor(Xtr_raw)

    models = make_models(include_calibrated_rf=False)
    to_train = {}
    if "rf" in models:
        to_train["rf"] = models["rf"]
    if "xgb" in models and models["xgb"] is not None:
        to_train["xgb"] = models["xgb"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    trained_pipes: dict[str, Pipeline] = {}

    for name, est in to_train.items():
        print(f"Training {name} on UNSW train...")
        pipe = Pipeline([
            ("prep", pre),
            ("est", est),
        ])
        pipe.fit(Xtr_raw, ytr)
        trained_pipes[name] = pipe
        # quick eval on test
        try:
            proba = pipe.predict_proba(Xte_raw)
            pos_idx = 1 if proba.shape[1] > 1 else 0
            y_score = proba[:, pos_idx]
            print(f"{name} PR-AUC={pr_auc(yte, y_score):.4f} ROC-AUC={roc_auc(yte, y_score):.4f} F1@0.5={f1_at_threshold(yte, y_score, 0.5):.4f}")
        except Exception:
            pass
        out_path = args.out_dir / f"{name}.joblib"
        joblib.dump(pipe, out_path)
        print(f"Saved {name} pipeline to: {out_path}")

    missing = []
    if (args.out_dir / "rf.joblib").exists() is False:
        missing.append("rf.joblib")
    if (args.out_dir / "xgb.joblib").exists() is False:
        missing.append("xgb.joblib")
    if missing:
        print("Note: some expected outputs are missing:", ", ".join(missing))

    # Direct McNemar test (RF vs XGB) on UNSW test, if both models are available
    if "rf" in trained_pipes and "xgb" in trained_pipes:
        print()
        print("Running McNemar test (RF vs XGB) on UNSW test...")

        def _to_labels(model, X):
            # Prefer predict_proba for probabilistic pipelines
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                proba = np.asarray(proba)
                pos_idx = 1 if proba.shape[1] > 1 else 0
                scores = proba[:, pos_idx]
                return (scores >= 0.5).astype(int)
            if hasattr(model, "predict"):
                y_pred = model.predict(X)
                y_pred = np.asarray(y_pred)
                if y_pred.ndim == 2 and y_pred.shape[1] > 1:
                    return (y_pred[:, 1] >= 0.5).astype(int)
                if y_pred.ndim == 1 and y_pred.dtype.kind == "f":
                    return (y_pred >= 0.5).astype(int)
                return y_pred.astype(int)
            raise AttributeError("Model has neither predict_proba nor predict")

        y_rf = _to_labels(trained_pipes["rf"], Xte_raw)
        y_xgb = _to_labels(trained_pipes["xgb"], Xte_raw)
        res = mcnemar_test(yte, y_rf, y_xgb)

        print(f"McNemar test: RF vs XGB on UNSW test")
        print(f"b (RF wrong, XGB correct): {res['b']}")
        print(f"c (XGB wrong, RF correct): {res['c']}")
        print(f"p-value: {res['p_value']:.6g}")
        if res["p_value"] < 0.05:
            print("Result: difference is statistically significant (p < 0.05)")
        else:
            print("Result: no statistically significant difference (p >= 0.05)")


if __name__ == "__main__":
    main()
