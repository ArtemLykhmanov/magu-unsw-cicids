from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Dict
import json

import numpy as np

from sklearn.pipeline import Pipeline
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths, SEED  # type: ignore
from src.data_io import load_csv  # type: ignore
from src.preprocess import build_preprocessor  # type: ignore
from src.models import make_models  # type: ignore
from src.metrics_ext import pr_auc, roc_auc, f1_at_threshold  # type: ignore
from src.eval_utils import threshold_for_target_fpr  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train pipelines on a single CSV and save them for later reuse (no CV/test split here)."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root (where data/ and src/ live).")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to CSV dataset (absolute, or relative to <root>/data).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory where trained pipelines will be saved.",
    )
    parser.add_argument(
        "--models",
        default="rf,logreg,mlp,hgb,rf_cal,xgb",
        help="Comma-separated list of model keys from src.models.make_models() to include.",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help=(
            "Target false-positive rate used to pick a training-time threshold tau "
            "for each model (saved for later reuse during evaluation)."
        ),
    )
    args = parser.parse_args()

    paths = Paths.from_root(str(args.root))

    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = (paths.data / csv_path).resolve()

    outdir = args.outdir
    if not outdir.is_absolute():
        outdir = (paths.root / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Single-dataset model training (no automatic CV/test split) ===")
    print()
    print(f"Input CSV: {csv_path}")
    X, y = load_csv(csv_path)
    n_pos = int(np.asarray(y).sum())
    n_neg = int((1 - np.asarray(y)).sum())
    print(f"Full dataset shape: {X.shape}, positives: {n_pos}, negatives: {n_neg}")
    print()

    models_sel = [m.strip() for m in args.models.split(",") if m.strip()]
    all_models = make_models()

    pre = build_preprocessor(X)

    # Store training-based thresholds tau (per model) for the requested target FPR,
    # so that later evaluation scripts can reuse them on new traffic.
    tau_train: Dict[str, float] = {}

    for model_name, model in all_models.items():
        if models_sel and model_name not in models_sel:
            continue
        # Skip calibrated/meta variants in this train-only script
        if isinstance(model, tuple):
            print()
            print(f"Skipping calibrated/meta model variant '{model_name}' in train-only script.")
            continue

        print()
        print(f"Training pipeline for model: {model_name}")
        pipe = Pipeline([("pre", pre), ("est", model)])
        pipe.fit(X, y)

        # quick sanity-check metrics on the train data (for information only)
        try:
            proba = pipe.predict_proba(X)
            pos_idx = 1 if proba.shape[1] > 1 else 0
            y_score = proba[:, pos_idx]
            # training-time operating threshold for the requested target FPR
            tau = float(threshold_for_target_fpr(y, y_score, args.target_fpr))
            tau_train[model_name] = tau
            print(
                f"  Approx train-only metrics: "
                f"PR-AUC={pr_auc(y, y_score):.4f} ROC-AUC={roc_auc(y, y_score):.4f} "
                f"F1@0.5={f1_at_threshold(y, y_score, 0.5):.4f} "
                f"F1@tau={f1_at_threshold(y, y_score, tau):.4f} "
                f"(tau for FPR~{args.target_fpr} is {tau:.6f})"
            )
        except Exception:
            pass

        model_path = outdir / f"{model_name}.joblib"
        joblib.dump(pipe, model_path)
        print(f"Saved trained pipeline to: {model_path}")

    # Persist training-based thresholds (if any could be computed).
    if tau_train:
        tau_path = outdir / "tau_train.json"
        try:
            payload = {
                "target_fpr": float(args.target_fpr),
                "train_csv": str(csv_path),
                "thresholds": {k: float(v) for k, v in tau_train.items()},
            }
            with tau_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print()
            print(f"Saved training-based thresholds to: {tau_path}")
        except Exception as e:
            print()
            print(f"Warning: failed to save training-based thresholds: {e}")

    print()
    print("=== Training finished. Saved pipelines can now be reused for separate evaluation runs. ===")


if __name__ == "__main__":
    main()
