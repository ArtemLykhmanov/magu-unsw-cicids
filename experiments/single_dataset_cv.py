from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def run_single_csv(
    paths: Paths,
    csv_path: Path,
    outdir: Path,
    models_sel: List[str],
    target_fpr: float,
    cv_folds: int,
    test_size: float,
    gui_progress: bool,
) -> None:
    print("=== Single-dataset CV + test experiment ===")
    print()

    csv_path = csv_path.resolve()
    print(f"Input CSV: {csv_path}")
    X, y = load_csv(csv_path)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    print(f"Full dataset shape: {X.shape}, positives: {n_pos}, negatives: {n_neg}")
    print()

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )
    print(f"Train shape: {Xtr.shape}, positives: {int(ytr.sum())}, negatives: {int((1 - ytr).sum())}")
    print(f"Test  shape: {Xte.shape}, positives: {int(yte.sum())}, negatives: {int((1 - yte).sum())}")
    print()

    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Leakage-style single-feature AUC scan (to detect features that almost encode the label)
    suspicious_cols: List[str] = []
    try:
        df_susp = suspicious_feature_scan(
            Xtr,
            ytr,
            outdir,
            prefix="single_dataset_train_test",
            Xte=Xte,
            yte=yte,
        )
        suspicious_cols = [str(f) for f in df_susp.loc[df_susp["suspicious"] == True, "feature"].tolist()]
        from pathlib import Path as _P
        print(
            f"Saved suspicious-feature leakage report under "
            f"{_P(outdir) / 'tables' / 'table_single_dataset_train_test_suspicious_features.csv'}"
        )
        if suspicious_cols:
            print("Dropping suspicious potential leakage features:", ", ".join(suspicious_cols))
            Xtr = Xtr.drop(columns=suspicious_cols, errors="ignore")
            Xte = Xte.drop(columns=suspicious_cols, errors="ignore")
        print()
    except Exception:
        suspicious_cols = []

    all_models = [name for name in make_models().keys() if (not models_sel or name in models_sel)]
    total_steps = len(all_models) * (cv_folds + 1)
    progress = ProgressReporter(total_steps if (gui_progress and total_steps > 0) else None)

    print(f"[1/2] Running {cv_folds}-fold cross-validation on the train split...")
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    rows_cv: List[Dict[str, Any]] = []

    for model_name, model in make_models().items():
        if all_models and model_name not in all_models:
            continue
        print()
        print(f"  -> Model: {model_name}")
        scores: List[float] = []
        f1s: List[float] = []
        fold_idx = 0
        for tr_idx, va_idx in kf.split(Xtr, ytr):
            fold_idx += 1
            X_tr, X_va = Xtr.iloc[tr_idx], Xtr.iloc[va_idx]
            y_tr, y_va = ytr.iloc[tr_idx], ytr.iloc[va_idx]
            print(f"     Fold {fold_idx}/{cv_folds}: train={X_tr.shape[0]} rows, val={X_va.shape[0]} rows")
            pre_local = build_preprocessor(X_tr)
            if isinstance(model, tuple) and model[1] == "calibrate":
                base = make_models(include_calibrated_rf=False)["rf"]
                clf = Pipeline([("pre", pre_local), ("est", base)])
                clf.fit(X_tr, y_tr)
                est_cal = CalibratedClassifierCV(clf.named_steps["est"], cv="prefit", method="isotonic")
                est_cal.fit(pre_local.transform(X_va), y_va)
                proba = est_cal.predict_proba(pre_local.transform(X_va))[:, 1]
            elif model_name == "mlp":
                clf = Pipeline([("pre", pre_local), ("est", model)])
                clf.fit(X_tr, y_tr)
                try:
                    est_cal = CalibratedClassifierCV(clf.named_steps["est"], cv="prefit", method="isotonic")
                    est_cal.fit(pre_local.transform(X_va), y_va)
                    proba = est_cal.predict_proba(pre_local.transform(X_va))[:, 1]
                except Exception:
                    proba = clf.predict_proba(X_va)[:, 1]
            else:
                pipe = Pipeline([("pre", pre_local), ("est", model)])
                pipe.fit(X_tr, y_tr)
                proba = pipe.predict_proba(X_va)[:, 1]
            fold_pr = pr_auc(y_va, proba)
            fold_f1 = f1_at_threshold(y_va, proba, 0.5)
            scores.append(fold_pr)
            f1s.append(fold_f1)
            print(f"       Metrics: PR-AUC={fold_pr:.6f}, F1@0.5={fold_f1:.6f}")
            progress.step(f"single_cv_fold {model_name} {fold_idx}")

        row_cv = {
            "model": model_name,
            "pr_auc_mean": float(np.mean(scores)),
            "pr_auc_std": float(np.std(scores)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        }
        rows_cv.append(row_cv)
        print(
            f"  CV summary for {model_name}: "
            f"PR-AUC={row_cv['pr_auc_mean']:.6f} ± {row_cv['pr_auc_std']:.6f}, "
            f"F1@0.5={row_cv['f1_mean']:.6f} ± {row_cv['f1_std']:.6f}"
        )

    df_cv = pd.DataFrame(rows_cv)
    cv_path = tables_dir / "table_single_cv.csv"
    df_cv.to_csv(cv_path, index=False)
    print()
    print(f"Saved CV summary table to: {cv_path}")
    print()

    print("[2/2] Training on full train split and evaluating on test split...")
    print(f"Target FPR for threshold selection: {target_fpr}")
    rows_test: List[Dict[str, Any]] = []

    for model_name, model in make_models().items():
        if all_models and model_name not in all_models:
            continue
        print()
        print(f"  -> Model: {model_name} (full-train + test evaluation)")
        pre_final = build_preprocessor(Xtr)

        if isinstance(model, tuple) and model[1] == "calibrate":
            base = make_models(include_calibrated_rf=False)["rf"]
            X_tr, X_va, y_tr, y_va = train_test_split(
                Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr
            )
            print("     Splitting train into train/validation for calibration (RF base).")
            clf = Pipeline([("pre", pre_final), ("est", base)])
            clf.fit(X_tr, y_tr)
            est_cal = CalibratedClassifierCV(clf.named_steps["est"], cv="prefit", method="isotonic")
            est_cal.fit(pre_final.transform(X_va), y_va)
            proba_te = est_cal.predict_proba(pre_final.transform(Xte))[:, 1]
            proba_va = est_cal.predict_proba(pre_final.transform(X_va))[:, 1]
        elif model_name == "mlp":
            X_tr, X_va, y_tr, y_va = train_test_split(
                Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr
            )
            print("     Splitting train into train/validation for possible MLP calibration.")
            pipe = Pipeline([("pre", pre_final), ("est", model)])
            pipe.fit(X_tr, y_tr)
            try:
                est_cal = CalibratedClassifierCV(pipe.named_steps["est"], cv="prefit", method="isotonic")
                est_cal.fit(pre_final.transform(X_va), y_va)
                proba_te = est_cal.predict_proba(pre_final.transform(Xte))[:, 1]
                proba_va = est_cal.predict_proba(pre_final.transform(X_va))[:, 1]
                print("     Calibration succeeded.")
            except Exception:
                proba_te = pipe.predict_proba(Xte)[:, 1]
                proba_va = pipe.predict_proba(X_va)[:, 1]
                print("     Calibration failed; using raw MLP probabilities.")
        else:
            pipe = Pipeline([("pre", pre_final), ("est", model)])
            pipe.fit(Xtr, ytr)
            X_tr, X_va, y_tr, y_va = train_test_split(
                Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr
            )
            print("     Using standard pipeline; validation split used only for threshold selection.")
            try:
                proba_te = pipe.predict_proba(Xte)[:, 1]
                proba_va = pipe.predict_proba(X_va)[:, 1]
            except Exception:
                from scipy.special import expit  # type: ignore

                proba_te = expit(pipe.decision_function(Xte))
                proba_va = expit(pipe.decision_function(X_va))

        tau = threshold_for_target_fpr(y_va, proba_va, target_fpr)
        pr = pr_auc(yte, proba_te)
        roc = roc_auc(yte, proba_te)
        f1_fixed = f1_at_threshold(yte, proba_te, 0.5)
        f1_tau = f1_at_threshold(yte, proba_te, tau)
        ece = expected_calibration_error(yte, proba_te, 10)
        br = brier(yte, proba_te)

        print(f"     Selected threshold tau for FPR~{target_fpr}: {tau:.6f}")
        print(
            f"     Test metrics: PR-AUC={pr:.6f}, ROC-AUC={roc:.6f}, "
            f"F1@0.5={f1_fixed:.6f}, F1@tau={f1_tau:.6f}, ECE={ece:.6f}, Brier={br:.6f}"
        )

        rows_test.append(
            {
                "model": model_name,
                "pr_auc": pr,
                "roc_auc": roc,
                "f1@0.5": f1_fixed,
                "tau@FPR": tau,
                "f1@tau": f1_tau,
                "ece": ece,
                "brier": br,
            }
        )
        progress.step(f"single_test {model_name}")

    df_test = pd.DataFrame(rows_test)
    test_path = tables_dir / "table_single_test.csv"
    df_test.to_csv(test_path, index=False)
    print()
    print(f"Saved test evaluation table to: {test_path}")
    print()

    try:
        df_sum = pd.merge(df_cv, df_test, on="model", how="outer", suffixes=("_cv", "_test"))
        sum_path = tables_dir / "table_single_summary.csv"
        df_sum.to_csv(sum_path, index=False)
        print(f"Saved combined CV+test summary table to: {sum_path}")
    except Exception:
        # Combined summary is optional; individual tables were already saved.
        print("Combined CV+test summary table skipped due to non-critical error.")

    print()
    print("=== Single-dataset experiment finished. You can now inspect the CSV tables in the output directory. ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic single-dataset CV + train/test experiment for a chosen CSV file."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root (where data/ and src/ live).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to CSV dataset (absolute, or relative to <root>/data).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out/single_dataset"),
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
        "--cv-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds to use on the train split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of data to reserve for the final test split (0–1).",
    )
    parser.add_argument(
        "--gui-progress",
        action="store_true",
        help="Emit machine-readable progress markers for GUI.",
    )
    # for GUI compatibility; currently ignored
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

    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = (paths.data / csv_path).resolve()

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
            run_single_csv(
                paths=paths,
                csv_path=csv_path,
                outdir=outdir,
                models_sel=models_sel,
                target_fpr=args.target_fpr,
                cv_folds=args.cv_folds,
                test_size=args.test_size,
                gui_progress=args.gui_progress,
            )
    finally:
        sys.stdout = orig_stdout


if __name__ == "__main__":
    main()
