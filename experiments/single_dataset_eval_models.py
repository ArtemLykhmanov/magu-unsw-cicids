from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional
import warnings
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths, SEED  # type: ignore
from src.data_io import load_csv  # type: ignore
from src.metrics_ext import (  # type: ignore
    pr_auc,
    roc_auc,
    f1_at_threshold,
    expected_calibration_error,
    brier,
)
from src.eval_utils import threshold_for_target_fpr  # type: ignore
from experiments.eval_cross_many import drop_test_duplicates, write_leakage_report  # type: ignore

# Silence noisy library warnings in logs (e.g., unseen categories in OneHotEncoder).
warnings.filterwarnings(
    "ignore",
    message=(
        "Found unknown categories in columns .* during transform. "
        "These unknown categories will be encoded as all zeros"
    ),
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message='.*Parameters: { "use_label_encoder" } are not used.*',
    category=UserWarning,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained pipelines on a CSV, using an internal val/test split for metrics."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root (where data/ and src/ live).")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to CSV dataset for evaluation (absolute, or relative to <root>/data).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory that contains trained pipelines (.joblib); tables/ will be created inside for results.",
    )
    parser.add_argument(
        "--models",
        default="rf,logreg,mlp,hgb,rf_cal,xgb",
        help="Comma-separated list of model keys (filenames <name>.joblib) to evaluate.",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="Target false-positive rate used to pick tau on a validation split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of data to reserve for the final test split (0–1).",
    )
    args = parser.parse_args()

    paths = Paths.from_root(str(args.root))

    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = (paths.data / csv_path).resolve()

    models_dir = args.outdir
    if not models_dir.is_absolute():
        models_dir = (paths.root / models_dir).resolve()

    tables_dir = models_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Optionally load training-based thresholds tau (per model) and the training CSV path
    # if they were saved during the single_dataset_train_models.py run.
    tau_train: Dict[str, float] = {}
    train_csv_path: Optional[Path] = None
    tau_train_path = models_dir / "tau_train.json"
    if tau_train_path.exists():
        try:
            with tau_train_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            # training CSV used when models were fitted (if recorded)
            train_csv_str = payload.get("train_csv")
            if isinstance(train_csv_str, str) and train_csv_str:
                train_csv_path = Path(train_csv_str)
            # thresholds are tied to a specific target_fpr
            t_fpr = float(payload.get("target_fpr", args.target_fpr))
            thr = payload.get("thresholds", {})
            if abs(t_fpr - float(args.target_fpr)) < 1e-9 and isinstance(thr, dict):
                tau_train = {str(k): float(v) for k, v in thr.items()}
        except Exception:
            tau_train = {}
            train_csv_path = None

    if tau_train:
        print(f"Using training-based thresholds from: {tau_train_path}")
    else:
        print("No matching training-based thresholds found; tau will be chosen on the validation split.")

    print("=== Evaluation of pre-trained pipelines on single CSV ===")
    print()
    print(f"Eval CSV: {csv_path}")
    X, y = load_csv(csv_path)
    y_arr = np.asarray(y)
    n_pos = int(y_arr.sum())
    n_neg = int((1 - y_arr).sum())
    print(f"Eval dataset shape: {X.shape}, positives: {n_pos}, negatives: {n_neg}")
    print()

    # If we know which CSV was used for training, run a simple leakage/duplicate
    # check and remove exact duplicates of training rows from the evaluation set,
    # mirroring the UNSW CV+test pipeline.
    if train_csv_path is not None and train_csv_path.exists():
        try:
            print(f"Checking for data leakage between training CSV and evaluation CSV...")
            X_tr_leak, y_tr_leak = load_csv(train_csv_path)
            X_te_before = X.copy()
            X, y, n_removed, removed_idx = drop_test_duplicates(X_tr_leak, X, y)
            if n_removed > 0:
                print(f"Removed {n_removed} duplicate rows from evaluation CSV to avoid leakage.")
            else:
                print("No cross-duplicates between training and evaluation CSV detected.")
            # Persist audit report under the models directory (shared logs)
            write_leakage_report(models_dir, X_tr_leak, X_te_before, X, removed_idx)
            print()
        except Exception as e:
            print(f"Warning: leakage/duplicate check failed and was skipped: {e}")
            print()

    # If we have training-based thresholds, treat the whole CSV as test-only data.
    # Otherwise, keep the original val/test split and pick tau on the validation part.
    if tau_train:
        X_te, y_te = X, y
        X_val, y_val = None, None  # type: ignore[assignment]
        print("Using entire evaluation CSV as test set (no internal validation split).")
        print()
    else:
        X_val, X_te, y_val, y_te = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=SEED,
            stratify=y,
        )
        print(f"Validation split: {X_val.shape[0]} rows; test split: {X_te.shape[0]} rows")
        print()

    models_sel = [m.strip() for m in args.models.split(",") if m.strip()]

    rows_test: List[Dict[str, float]] = []

    for name in models_sel:
        model_path = models_dir / f"{name}.joblib"
        if not model_path.exists():
            print(f"Skipping model '{name}': file not found: {model_path}")
            continue
        print(f"Evaluating model '{name}' from: {model_path}")
        try:
            pipe = joblib.load(model_path)
        except Exception as e:
            print(f"  Failed to load model '{name}': {e}")
            continue

        try:
            if tau_train:
                proba_val = None  # type: ignore[assignment]
                proba_te = pipe.predict_proba(X_te)[:, 1]
            else:
                proba_val = pipe.predict_proba(X_val)[:, 1]  # type: ignore[arg-type]
                proba_te = pipe.predict_proba(X_te)[:, 1]
        except Exception:
            from scipy.special import expit  # type: ignore

            try:
                if tau_train:
                    proba_val = None  # type: ignore[assignment]
                    proba_te = expit(pipe.decision_function(X_te))
                else:
                    proba_val = expit(pipe.decision_function(X_val))  # type: ignore[arg-type]
                    proba_te = expit(pipe.decision_function(X_te))
            except Exception as e:
                print(f"  Failed to obtain probabilities for '{name}': {e}")
                continue

        if tau_train and name in tau_train:
            tau = float(tau_train[name])
            print(f"  Using training-based tau for FPR~{args.target_fpr}: {tau:.6f}")
        else:
            tau = threshold_for_target_fpr(y_val, proba_val, args.target_fpr)  # type: ignore[arg-type]
            print(f"  Selected tau for FPR~{args.target_fpr} on validation split: {tau:.6f}")
        pr = pr_auc(y_te, proba_te)
        roc = roc_auc(y_te, proba_te)
        f1_fixed = f1_at_threshold(y_te, proba_te, 0.5)
        f1_tau = f1_at_threshold(y_te, proba_te, tau)
        ece = expected_calibration_error(y_te, proba_te, 10)
        br = brier(y_te, proba_te)
        print(
            f"  Test metrics: PR-AUC={pr:.6f}, ROC-AUC={roc:.6f}, "
            f"F1@0.5={f1_fixed:.6f}, F1@tau={f1_tau:.6f}, ECE={ece:.6f}, Brier={br:.6f}"
        )

        rows_test.append(
            {
                "model": name,
                "pr_auc": pr,
                "roc_auc": roc,
                "f1@0.5": f1_fixed,
                "tau@FPR": tau,
                "f1@tau": f1_tau,
                "ece": ece,
                "brier": br,
            }
        )

    if rows_test:
        df = pd.DataFrame(rows_test)
        test_path = tables_dir / "table_single_eval_from_models.csv"
        df.to_csv(test_path, index=False)
        print()
        print(f"Saved evaluation table to: {test_path}")
    else:
        print("No models were successfully evaluated; no table written.")

    print()
    print("=== Evaluation finished. ===")


if __name__ == "__main__":
    main()
