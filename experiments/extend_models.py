from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Any
import json
import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths, SEED  # type: ignore
from src.data_io import load_csv  # type: ignore
from src.preprocess import build_preprocessor  # type: ignore
from src.models import make_models  # type: ignore
from src.metrics_ext import (  # type: ignore
    pr_auc,
    roc_auc,
    f1_at_threshold,
    expected_calibration_error,
    brier,
)
from src.eval_utils import threshold_for_target_fpr, save_manifest  # type: ignore
from src.feature_align import to_harmonized  # type: ignore


class _Tee:
    """Simple tee to log both to console and a file."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                continue
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                continue

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                continue


def _collect_base_train_csvs(base_outdir: Path) -> List[Path]:
    """Read base training CSV paths from tau_train.json if present.

    We support either a single string path or a list of paths under 'train_csv'.
    """
    tau_path = base_outdir / "tau_train.json"
    paths: List[Path] = []
    if not tau_path.exists():
        return paths
    try:
        with tau_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return paths

    train_csv = payload.get("train_csv")
    if isinstance(train_csv, str) and train_csv:
        paths.append(Path(train_csv))
    elif isinstance(train_csv, list):
        for p in train_csv:
            if isinstance(p, str) and p:
                paths.append(Path(p))
    return paths


def run_extend_models(
    paths: Paths,
    base_outdir: Path,
    train_csvs: List[Path],
    outdir: Path,
    models_sel: List[str],
    target_fpr: float,
) -> None:
    print("=== Extend pre-trained models with additional (possibly cross-dataset) data ===")
    print()

    base_outdir = base_outdir.resolve()
    print(f"Base models outdir: {base_outdir}")
    print(f"Extended models will be saved under: {outdir}")
    print()

    # Collect original training CSVs from the base tau_train.json (if available)
    base_train_paths = _collect_base_train_csvs(base_outdir)
    if base_train_paths:
        print("Base training CSVs (from tau_train.json):")
        for p in base_train_paths:
            print(f"  - {p}")
    else:
        print("No base training CSVs found in tau_train.json; extension will use only the new --train-csv files.")
    print()

    # Resolve new training CSVs
    train_paths: List[Path] = []
    for p in train_csvs:
        train_paths.append(p.resolve())

    if not train_paths and not base_train_paths:
        raise ValueError("No training CSVs provided or discovered; nothing to extend.")

    print("Additional training CSVs (including any fine-tune subsets materialized as CSV):")
    if train_paths:
        for p in train_paths:
            print(f"  - {p}")
    else:
        print("  (none)")
    print()

    # Load all training data (base + new + finetune subsets)
    Xtr_blocks: List[pd.DataFrame] = []
    ytr_blocks: List[pd.Series] = []
    train_sources: List[str] = []

    for p in base_train_paths:
        try:
            Xi, yi = load_csv(p)
        except Exception as e:
            print(f"Warning: failed to load base training CSV {p}: {e}")
            continue
        Xtr_blocks.append(Xi)
        ytr_blocks.append(yi)
        train_sources.append(str(p))

    for p in train_paths:
        try:
            Xi, yi = load_csv(p)
        except Exception as e:
            print(f"Warning: failed to load additional training CSV {p}: {e}")
            continue
        Xtr_blocks.append(Xi)
        ytr_blocks.append(yi)
        train_sources.append(str(p))

    if not Xtr_blocks:
        raise ValueError("Failed to load any training data; cannot extend models.")

    # Harmonize feature space across potentially different datasets
    Xtr_h_list: List[pd.DataFrame] = []
    for Xi in Xtr_blocks:
        Xh = to_harmonized(Xi)
        Xtr_h_list.append(Xh)

    Xtr = pd.concat(Xtr_h_list, axis=0, ignore_index=True)
    ytr = pd.concat(ytr_blocks, axis=0, ignore_index=True)

    n_pos = int(np.asarray(ytr).sum())
    n_neg = int((1 - np.asarray(ytr)).sum())
    print(f"Combined training shape (after harmonization): {Xtr.shape}, positives: {n_pos}, negatives: {n_neg}")
    print()

    pre = build_preprocessor(Xtr)

    selected_models = [m.strip() for m in models_sel if m.strip()] if models_sel else []
    all_models = make_models()

    outdir.mkdir(parents=True, exist_ok=True)

    tau_train: Dict[str, float] = {}

    for model_name, model in all_models.items():
        if selected_models and model_name not in selected_models:
            continue
        # Skip calibrated/meta variants; extend base models only for primary estimators
        if isinstance(model, tuple):
            print()
            print(f"Skipping calibrated/meta model variant '{model_name}' in extend-models script.")
            continue

        print()
        print(f"Extending model by retraining on combined data: {model_name}")
        pipe = Pipeline([("pre", pre), ("est", model)])
        pipe.fit(Xtr, ytr)

        # Approximate train-only metrics and training-time tau for reference
        try:
            proba = pipe.predict_proba(Xtr)
            pos_idx = 1 if proba.shape[1] > 1 else 0
            y_score = proba[:, pos_idx]
        except Exception:
            try:
                from scipy.special import expit  # type: ignore

                y_score = expit(pipe.decision_function(Xtr))
            except Exception:
                y_score = None  # type: ignore[assignment]

        if y_score is not None:
            tau = float(threshold_for_target_fpr(ytr, y_score, target_fpr))
            tau_train[model_name] = tau
            print(
                f"  Approx train-only metrics on harmonized features: "
                f"PR-AUC={pr_auc(ytr, y_score):.4f} ROC-AUC={roc_auc(ytr, y_score):.4f} "
                f"F1@0.5={f1_at_threshold(ytr, y_score, 0.5):.4f} "
                f"F1@tau={f1_at_threshold(ytr, y_score, tau):.4f} "
                f"(tau for FPR~{target_fpr} is {tau:.6f})"
            )
        else:
            print("  Warning: unable to compute probabilities on training data; tau will not be stored for this model.")

        model_path = outdir / f"{model_name}.joblib"
        joblib.dump(pipe, model_path)
        print(f"Saved extended model pipeline to: {model_path}")

    # Persist extended training thresholds and metadata
    if tau_train:
        tau_path = outdir / "tau_train.json"
        try:
            payload: Dict[str, Any] = {
                "target_fpr": float(target_fpr),
                "train_csv": train_sources,
                "base_outdir": str(base_outdir),
                "harmonized": True,
                "notes": "Extended training with combined (possibly cross-dataset) data based on existing models.",
                "thresholds": {k: float(v) for k, v in tau_train.items()},
            }
            with tau_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print()
            print(f"Saved extended training thresholds to: {tau_path}")
        except Exception as e:
            print()
            print(f"Warning: failed to save extended training thresholds: {e}")

    print()
    print("=== Extend-models run finished. Extended pipelines can now be used for further evaluation. ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extend existing pipelines with additional training data (optionally from a different dataset). "
            "Models are retrained on a harmonized feature space derived from base and new CSVs."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root (where data/ and src/ live).",
    )
    parser.add_argument(
        "--base-outdir",
        type=Path,
        required=True,
        help="Existing outdir that contains previously trained pipelines (.joblib) and tau_train.json.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        action="append",
        required=True,
        help="Path to additional training CSV (absolute, or relative to <root>/data). May be given multiple times.",
    )
    parser.add_argument(
        "--finetune-csv",
        type=Path,
        action="append",
        required=False,
        help=(
            "Optional CSV(s) whose rows can be partially used for fine-tuning. "
            "A fraction of rows from each file (see --finetune-frac) will be added to training data."
        ),
    )
    parser.add_argument(
        "--finetune-frac",
        type=float,
        default=0.0,
        help=(
            "Fraction of rows from each --finetune-csv to add to training data (0–1, or 0–100 interpreted as percent)."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory for extended models. If omitted, a new sibling directory of --base-outdir "
            "will be created automatically."
        ),
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
        help="Target false-positive rate used to pick a training-time threshold tau for each model.",
    )
    # For GUI compatibility; currently informational/ignored
    parser.add_argument(
        "--no-extra-plots",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-feature-importance",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    paths = Paths.from_root(str(args.root))

    base_outdir = args.base_outdir
    if not base_outdir.is_absolute():
        base_outdir = (paths.root / base_outdir).resolve()

    train_csvs: List[Path] = []
    for p in args.train_csv or []:
        if p.is_absolute():
            train_csvs.append(p.resolve())
        else:
            train_csvs.append((paths.data / p).resolve())

    finetune_csvs: List[Path] = []
    for p in args.finetune_csv or []:
        if p.is_absolute():
            finetune_csvs.append(p.resolve())
        else:
            finetune_csvs.append((paths.data / p).resolve())

    # Optional fine-tuning: convert fraction to [0, 1] and pre-select a subset of rows;
    # these subsets are materialized as temporary CSVs and treated as additional training CSVs.
    finetune_frac = float(args.finetune_frac)
    if finetune_frac > 1.0:
        finetune_frac = finetune_frac / 100.0
    finetune_frac = max(0.0, min(finetune_frac, 1.0))

    finetune_paths_effective: List[Path] = []
    if finetune_csvs and finetune_frac > 0.0:
        print("Preparing fine-tune data:")
        for p in finetune_csvs:
            try:
                Xi, yi = load_csv(p)
            except Exception as e:
                print(f"Warning: failed to load finetune CSV {p}: {e}")
                continue
            if Xi.shape[0] <= 0:
                print(f"  Skipping finetune CSV {p}: no rows.")
                continue
            # Sample a fraction of rows for fine-tuning and add them as extra training data
            X_ft, _, y_ft, _ = train_test_split(
                Xi,
                yi,
                train_size=finetune_frac,
                random_state=SEED,
                stratify=yi if yi.nunique() > 1 else None,
            )
            print(
                f"  Using {X_ft.shape[0]} rows ({finetune_frac:.3f} fraction) from finetune CSV {p} "
                "as additional training data."
            )
            # Persist this sampled subset as a temporary CSV to reuse the same loading path logic in run_extend_models
            tmp_name = f"_tmp_finetune_{p.stem}_{int(time.time())}.csv"
            tmp_path = base_outdir / tmp_name
            try:
                df_tmp = X_ft.copy()
                # load_csv will detect 'y' as the label column
                df_tmp["y"] = y_ft.values
                df_tmp.to_csv(tmp_path, index=False)
                finetune_paths_effective.append(tmp_path)
                print(f"  Materialized finetune subset as temporary CSV: {tmp_path}")
            except Exception as e:
                print(f"Warning: failed to materialize finetune subset for {p}: {e}")
        print()
    elif finetune_csvs and finetune_frac <= 0.0:
        print("Finetune CSVs were provided but finetune_frac <= 0; they will be ignored.")
        print()

    # Derive output directory
    outdir = args.outdir
    if outdir is not None:
        if not outdir.is_absolute():
            outdir = (paths.root / outdir).resolve()
    else:
        leaf = base_outdir.name or "out"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        new_leaf = f"{leaf}__extend__{stamp}"
        outdir = base_outdir.parent / new_leaf

    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run.log"

    save_manifest(outdir, vars(args))

    models_sel = [m.strip() for m in args.models.split(",") if m.strip()]

    orig_stdout = sys.stdout
    try:
        with log_path.open("w", encoding="utf-8") as f:
            sys.stdout = _Tee(orig_stdout, f)
            run_extend_models(
                paths=paths,
                base_outdir=base_outdir,
                train_csvs=train_csvs + finetune_paths_effective,
                outdir=outdir,
                models_sel=models_sel,
                target_fpr=args.target_fpr,
            )
    finally:
        sys.stdout = orig_stdout


if __name__ == "__main__":
    main()
