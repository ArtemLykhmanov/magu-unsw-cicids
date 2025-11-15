from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

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
from src.eval_utils import threshold_for_target_fpr  # type: ignore
from src.feature_align import to_harmonized  # type: ignore
from src.plots import (  # type: ignore
    pr_curve_overlay,
    roc_curve_overlay,
    reliability_diagram,
    f1_vs_threshold_curve,
)

from experiments.eval_cross_many import ProgressReporter, suspicious_feature_scan  # type: ignore

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class _Tee:
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


def _align_features(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Align features between train and test in an automatic, robust way.

    Strategy:
      1) Try 'harmonized' numeric feature set via to_harmonized().
         If it yields several usable (non-NaN) columns on both sides, use it.
      2) Otherwise, fall back to simple column intersection alignment.
    """
    # Try harmonized alignment first
    Xtr_h = to_harmonized(Xtr)
    Xte_h = to_harmonized(Xte)
    usable_cols = [
        c
        for c in Xtr_h.columns
        if not Xtr_h[c].isna().all() and c in Xte_h.columns and not Xte_h[c].isna().all()
    ]
    if len(usable_cols) >= 3:
        return Xtr_h[usable_cols].copy(), Xte_h[usable_cols].copy(), "harmonized"

    # Fallback: intersection of columns as-is
    common = sorted(set(Xtr.columns) & set(Xte.columns))
    if not common:
        raise ValueError("No common feature columns between train and test after alignment.")
    return Xtr[common].copy(), Xte[common].copy(), "intersect"


def run_cross_pair_generic(
    paths: Paths,
    train_csvs: List[Path],
    test_csvs: List[Path],
    outdir: Path,
    models_sel: List[str],
    target_fpr: float,
    gui_progress: bool,
    enable_plots: bool = True,
    enable_feature_importance: bool = True,
    finetune_frac: float = 0.0,
) -> None:
    print("=== Generic cross-dataset train->test experiment ===")
    print()

    # Resolve and load all train CSVs
    train_paths = [p.resolve() for p in train_csvs]
    test_paths = [p.resolve() for p in test_csvs]

    print("Train CSVs:")
    for p in train_paths:
        print(f"  - {p}")
    print("Test  CSVs:")
    for p in test_paths:
        print(f"  - {p}")

    Xtr_list: List[pd.DataFrame] = []
    ytr_list: List[pd.Series] = []
    for p in train_paths:
        Xi, yi = load_csv(p)
        Xtr_list.append(Xi)
        ytr_list.append(yi)
    Xtr_raw = pd.concat(Xtr_list, axis=0, ignore_index=True)
    ytr = pd.concat(ytr_list, axis=0, ignore_index=True)

    Xte_list: List[pd.DataFrame] = []
    yte_list: List[pd.Series] = []
    for p in test_paths:
        Xi, yi = load_csv(p)
        Xte_list.append(Xi)
        yte_list.append(yi)
    Xte_raw = pd.concat(Xte_list, axis=0, ignore_index=True)
    yte = pd.concat(yte_list, axis=0, ignore_index=True)
    print(f"Train shape (combined): {Xtr_raw.shape}, positives: {int(ytr.sum())}, negatives: {int((1 - ytr).sum())}")
    print(f"Test  shape (combined): {Xte_raw.shape}, positives: {int(yte.sum())}, negatives: {int((1 - yte).sum())}")
    print()

    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Suspicious-feature leakage scan and drop (single-feature AUC close to 1)
    suspicious_cols: List[str] = []
    try:
        df_susp = suspicious_feature_scan(
            Xtr_raw,
            ytr,
            outdir,
            prefix="cross_pair_train_test",
            Xte=Xte_raw,
            yte=yte,
        )
        suspicious_cols = [str(f) for f in df_susp.loc[df_susp["suspicious"] == True, "feature"].tolist()]
        from pathlib import Path as _P
        print(
            f"Saved suspicious-feature leakage report under "
            f"{_P(outdir) / 'tables' / 'table_cross_pair_train_test_suspicious_features.csv'}"
        )
        if suspicious_cols:
            print("Dropping suspicious potential leakage features:", ", ".join(suspicious_cols))
            Xtr_raw = Xtr_raw.drop(columns=suspicious_cols, errors="ignore")
            Xte_raw = Xte_raw.drop(columns=suspicious_cols, errors="ignore")
        print()
    except Exception:
        suspicious_cols = []

    # Automatic feature alignment between train and test
    Xtr_aligned, Xte_aligned, align_mode = _align_features(Xtr_raw, Xte_raw)
    print(f"Feature alignment mode: {align_mode}")
    print(f"Aligned train shape: {Xtr_aligned.shape}, aligned test shape: {Xte_aligned.shape}")
    print()

    # Optional fine-tuning on a fraction of test, excluded from final evaluation
    Xtr = Xtr_aligned
    Xte = Xte_aligned
    if finetune_frac and finetune_frac > 0.0:
        frac = float(finetune_frac)
        if frac > 1.0:
            frac = frac / 100.0
        frac = max(0.0, min(frac, 0.5))
        if frac > 0.0 and len(Xte) > 1:
            X_ft, X_eval, y_ft, y_eval = train_test_split(
                Xte,
                yte,
                test_size=1.0 - frac,
                random_state=SEED,
                stratify=yte if yte.nunique() > 1 else None,
            )
            print(
                f"Using a fraction of test for fine-tuning: {frac:.3f} "
                f"({X_ft.shape[0]} rows); remaining test rows: {X_eval.shape[0]}"
            )
            Xtr = pd.concat([Xtr_aligned, X_ft], axis=0, ignore_index=True)
            ytr = pd.concat([ytr, y_ft], axis=0, ignore_index=True)
            Xte = X_eval.reset_index(drop=True)
            yte = y_eval.reset_index(drop=True)
            print(f"Augmented train shape after fine-tune split: {Xtr.shape}")
            print()

    # Progress reporter
    all_models = [name for name in make_models().keys() if (not models_sel or name in models_sel)]
    total_steps = len(all_models) if all_models else 0
    progress = ProgressReporter(total_steps if (gui_progress and total_steps > 0) else None)

    rows: List[Dict[str, Any]] = []
    probas_by_model: Dict[str, np.ndarray] = {}
    for model_name, model in make_models().items():
        if all_models and model_name not in all_models:
            continue
        print()
        print(f"-> Model: {model_name} (train on train CSV, evaluate on test CSV)")
        pre = build_preprocessor(Xtr)

        # Use a validation split from train to pick tau for target FPR
        X_tr, X_va, y_tr, y_va = train_test_split(
            Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr
        )

        if isinstance(model, tuple) and model[1] == "calibrate":
            base = make_models(include_calibrated_rf=False)["rf"]
            pipe_base = Pipeline([("pre", pre), ("est", base)])
            pipe_base.fit(X_tr, y_tr)
            est_cal = CalibratedClassifierCV(pipe_base.named_steps["est"], cv="prefit", method="isotonic")
            est_cal.fit(pre.transform(X_va), y_va)
            proba_va = est_cal.predict_proba(pre.transform(X_va))[:, 1]
            proba_te = est_cal.predict_proba(pre.transform(Xte))[:, 1]
        else:
            pipe = Pipeline([("pre", pre), ("est", model)])
            pipe.fit(X_tr, y_tr)
            try:
                proba_va = pipe.predict_proba(X_va)[:, 1]
                proba_te = pipe.predict_proba(Xte)[:, 1]
            except Exception:
                from scipy.special import expit  # type: ignore

                proba_va = expit(pipe.decision_function(X_va))
                proba_te = expit(pipe.decision_function(Xte))

        tau = threshold_for_target_fpr(y_va, proba_va, target_fpr)
        pr = pr_auc(yte, proba_te)
        roc = roc_auc(yte, proba_te)
        f1_fixed = f1_at_threshold(yte, proba_te, 0.5)
        f1_tau = f1_at_threshold(yte, proba_te, tau)
        ece = expected_calibration_error(yte, proba_te, 10)
        br = brier(yte, proba_te)

        print(f"   Selected tau for FPR~{target_fpr}: {tau:.6f}")
        print(
            f"   Test metrics: PR-AUC={pr:.6f}, ROC-AUC={roc:.6f}, "
            f"F1@0.5={f1_fixed:.6f}, F1@tau={f1_tau:.6f}, ECE={ece:.6f}, Brier={br:.6f}"
        )

        rows.append(
            {
                "model": model_name,
                "align_mode": align_mode,
                "pr_auc": pr,
                "roc_auc": roc,
                "f1@0.5": f1_fixed,
                "tau@FPR": tau,
                "f1@tau": f1_tau,
                "ece": ece,
                "brier": br,
            }
        )
        probas_by_model[model_name] = proba_te
        if progress is not None:
            progress.step(f"cross_pair_test {model_name}")

    # Save metrics table
    df = pd.DataFrame(rows)
    out_path = tables_dir / "table_cross_pair_generic.csv"
    df.to_csv(out_path, index=False)
    print()
    print(f"Saved cross-dataset evaluation table to: {out_path}")
    print()

    # Optional diagnostic plots
    if enable_plots and probas_by_model:
        try:
            # Overlays across models
            pr_curve_overlay(yte.values, probas_by_model, str(figures_dir / "pr_curve_overlay.png"))
            roc_curve_overlay(yte.values, probas_by_model, str(figures_dir / "roc_curve_overlay.png"))
        except Exception:
            pass
        # Reliability and F1-vs-threshold for each model
        for mname, probs in probas_by_model.items():
            try:
                reliability_diagram(
                    yte.values,
                    probs,
                    10,
                    str(figures_dir / f"reliability_{mname}.png"),
                    strategy="quantile",
                )
                f1_vs_threshold_curve(
                    yte.values,
                    probs,
                    str(figures_dir / f"f1_vs_threshold_{mname}.png"),
                )
            except Exception:
                continue

    print("=== Cross-dataset experiment finished. You can now inspect the CSV tables in the output directory. ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic cross-dataset train->test experiment for two CSV files (automatic feature alignment + leakage scan)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root (where data/ and src/ live).",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        action="append",
        required=True,
        help="Path to training CSV (absolute, or relative to <root>/data). May be given multiple times.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        action="append",
        required=True,
        help="Path to test CSV (absolute, or relative to <root>/data). May be given multiple times.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out/cross_pair_generic"),
        help="Output directory for tables (absolute or relative to root).",
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
        help="Target false-positive rate used to pick tau on a validation split.",
    )
    parser.add_argument(
        "--gui-progress",
        action="store_true",
        help="Emit machine-readable progress markers for GUI.",
    )
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
    parser.add_argument(
        "--finetune-frac",
        type=float,
        default=0.0,
        help="Optional fraction of aligned test data to use for fine-tuning (0–1, or 0–100 for percent).",
    )
    args = parser.parse_args()

    paths = Paths.from_root(str(args.root))

    train_paths: List[Path] = []
    for p in args.train_csv:
        train_paths.append((paths.data / p).resolve() if not p.is_absolute() else p.resolve())

    test_paths: List[Path] = []
    for p in args.test_csv:
        test_paths.append((paths.data / p).resolve() if not p.is_absolute() else p.resolve())

    outdir = args.outdir
    if not outdir.is_absolute():
        outdir = (paths.root / outdir).resolve()

    models_sel = [m.strip() for m in args.models.split(",") if m.strip()]

    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run.log"

    orig_stdout = sys.stdout
    try:
        with log_path.open("w", encoding="utf-8") as f:
            sys.stdout = _Tee(orig_stdout, f)
            run_cross_pair_generic(
                paths=paths,
                train_csvs=train_paths,
                test_csvs=test_paths,
                outdir=outdir,
                models_sel=models_sel,
                target_fpr=args.target_fpr,
                gui_progress=args.gui_progress,
                enable_plots=not args.no_extra_plots,
                enable_feature_importance=not args.no_feature_importance,
                finetune_frac=args.finetune_frac,
            )
    finally:
        sys.stdout = orig_stdout


if __name__ == "__main__":
    main()
