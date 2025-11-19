from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Dict, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths, SEED  # type: ignore
from src.data_io import load_csv, unsw_files  # type: ignore
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
from src.plots import (  # type: ignore
    pr_curve_overlay,
    roc_curve_overlay,
    reliability_diagram,
    f1_vs_threshold_curve,
)

from experiments.eval_cross_many import (  # type: ignore
    check_data_leakage,
    drop_test_duplicates,
    suspicious_feature_scan,
)


warnings.filterwarnings("ignore")


class ProgressReporter:
    """Print simple machine-readable progress markers for the GUI."""

    def __init__(self, total_steps: int | None = None) -> None:
        self.total_steps = int(total_steps) if total_steps and total_steps > 0 else 0
        self.done = 0

    def step(self, label: str = "") -> None:
        if self.total_steps <= 0:
            return
        self.done += 1
        if self.done > self.total_steps:
            self.total_steps = self.done
        try:
            print(f"[GUI_PROGRESS] {self.done} {self.total_steps} {label}")
        except Exception:
            pass


def _format_models(models_sel: List[str]) -> str:
    return ", ".join(models_sel) if models_sel else "(all models)"


def _explain_f1_example(y_true, y_score, thr: float) -> None:
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    y_pred = (y_score_arr >= thr).astype(int)

    tp = int(((y_true_arr == 1) & (y_pred == 1)).sum())
    fp = int(((y_true_arr == 0) & (y_pred == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred == 0)).sum())
    tn = int(((y_true_arr == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("         Приклад обчислення F1@0.5 на цій валідації:")
    print(f"           TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"           Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.6f}")
    print(f"           Recall    = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.6f}")
    print("           F1        = 2 * P * R / (P + R)")
    print(f"                      = 2 * {precision:.6f} * {recall:.6f} / ({precision:.6f} + {recall:.6f}) = {f1:.6f}")
    print()


def _explain_tau_example(y_true, y_score, target_fpr: float) -> None:
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    neg_scores = y_score_arr[y_true_arr == 0]
    n_neg = int(len(neg_scores))
    if n_neg == 0:
        print("         Пояснення tau: у валідації немає негативних прикладів, тому поріг робиться максимально строгим (tau=1.0).")
        print()
        return

    order = np.argsort(-neg_scores)
    neg_sorted = neg_scores[order]
    k = int(np.floor(target_fpr * n_neg))

    print("         Як обчислюється поріг tau для заданого target FPR:")
    print(f"           Кількість негативних прикладів у валідації: N_neg = {n_neg}")
    print(f"           target_fpr = {target_fpr:.6f}")
    print(f"           Дозволяємо приблизно K = floor(target_fpr * N_neg) = floor({target_fpr:.6f} * {n_neg}) = {k} помилкових спрацювань (FP).")
    if k <= 0:
        print("           Оскільки K <= 0, ми ставимо tau трохи вище за максимум серед негативних score, щоб жоден негатив не проходив поріг.")
    else:
        approx_thr = float(neg_sorted[k - 1])
        print("           Сортуємо всі score негативного класу від найбільшого до найменшого")
        print("           і шукаємо таку межу, щоб приблизно перші K негативів могли перейти через поріг.")
        print(f"           Для цієї вибірки це означає, що tau близько до {approx_thr:.6f} (score K-го негативного).")
    print("           Далі ця tau використовується на тесті: все, що має score >= tau, вважаємо атакою.")
    print()


def _explain_calibration_example(y_true, y_prob, ece_value: float, brier_value: float, n_bins: int = 10) -> None:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    N = int(len(y_true_arr))

    print("         Пояснення Brier score та ECE:")
    print("           Нехай y_i належить {0,1} (справжній клас), а p_i — ймовірність атаки від моделі.")
    print("           Brier score = середнє квадратичне відхилення ймовірностей від міток:")
    print("             Brier = (1/N) * sum_i (p_i - y_i)^2")
    print(f"             Для цієї моделі Brier ~ {brier_value:.6f}: менше значення означає кращу калібровку.")
    print()

    print(f"           Expected Calibration Error (ECE) з {n_bins} бінів:")
    print("             1) Від 0 до 1 ділимо інтервал ймовірностей на рівні бін-и.")
    print("             2) Для кожного біну рахуємо:")
    print("                  · conf_bin = середній p_i у біні (середня впевненість),")
    print("                  · acc_bin  = частка y_i = 1 у цьому біні (фактична частота атак),")
    print("                  · w_bin    = частка об'єктів у цьому біні.")
    print("             3) ECE = sum_b w_bin * |acc_bin - conf_bin|.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    printed = 0
    max_to_print = 4
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        if printed < max_to_print or i == n_bins - 1:
            acc = float(y_true_arr[mask].mean())
            conf = float(y_prob_arr[mask].mean())
            w = cnt / max(N, 1)
            print(f"             Бін [{lo:.1f}, {hi:.1f}]: n={cnt}, w={w:.3f}, conf={conf:.3f}, acc={acc:.3f}")
            printed += 1

    print(f"             Підсумкове ECE для моделі = {ece_value:.6f} (0 — ідеальна калібровка).")
    print()


def _explain_f1_example(y_true, y_score, thr: float) -> None:  # type: ignore[override]
    """Clean re-definition: step-by-step F1 explanation without problematic Unicode symbols."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    y_pred = (y_score_arr >= thr).astype(int)

    tp = int(((y_true_arr == 1) & (y_pred == 1)).sum())
    fp = int(((y_true_arr == 0) & (y_pred == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred == 0)).sum())
    tn = int(((y_true_arr == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("         Приклад обчислення F1 на валідації (thr=0.5):")
    print(f"           TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"           Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.6f}")
    print(f"           Recall    = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.6f}")
    print("           F1        = 2 * P * R / (P + R)")
    print(
        f"                      = 2 * {precision:.6f} * {recall:.6f}"
        f" / ({precision:.6f} + {recall:.6f}) = {f1:.6f}"
    )
    print()


def _explain_tau_example(y_true, y_score, target_fpr: float) -> None:  # type: ignore[override]
    """Clean re-definition: explanation of tau for given target FPR using only ASCII/Cyrillic."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    neg_scores = y_score_arr[y_true_arr == 0]
    n_neg = int(len(neg_scores))
    if n_neg == 0:
        print("         Пояснення tau: у валідації немає негативних прикладів,")
        print("         тому ми не можемо оцінити FPR і беремо максимально строгий поріг (tau=1.0).")
        print()
        return

    order = np.argsort(-neg_scores)
    neg_sorted = neg_scores[order]
    k = int(np.floor(target_fpr * n_neg))

    print("         Як обчислюється поріг tau для заданого target FPR:")
    print(f"           Кількість негативних прикладів у валідації: N_neg = {n_neg}")
    print(f"           target_fpr = {target_fpr:.6f}")
    print("           Дозволяємо приблизно K = floor(target_fpr * N_neg) помилкових спрацювань (FP).")
    if k <= 0:
        print("           У цій вибірці K <= 0, тому беремо дуже строгий поріг,")
        print("           близький до максимального score серед негативних прикладів.")
    else:
        k = min(k, n_neg)
        approx_thr = float(neg_sorted[k - 1])
        print("           Сортуємо score негативного класу від більшого до меншого")
        print("           і беремо поріг біля score K-го негативного прикладу.")
        print(f"           Для цієї вибірки це дає tau приблизно {approx_thr:.6f}.")
    print("           Далі ця tau використовується на тесті: все, що має score >= tau, вважаємо атакою.")
    print()


def _explain_calibration_example(  # type: ignore[override]
    y_true, y_prob, ece_value: float, brier_value: float, n_bins: int = 10
) -> None:
    """Clean re-definition: explanation of Brier score and ECE, ASCII-friendly."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    N = int(len(y_true_arr))

    print("         Пояснення Brier score та ECE:")
    print("           Нехай y_i належить {0,1} (справжній клас), а p_i — ймовірність атаки від моделі.")
    print("           Brier score — це середнє квадратичне відхилення ймовірностей від міток:")
    print("             Brier = (1/N) * sum_i (p_i - y_i)^2")
    print(f"             Для цієї вибірки Brier ~ {brier_value:.6f} (чим ближче до 0, тим краще).")
    print()

    print(f"           Expected Calibration Error (ECE) для {n_bins} бінів за ймовірністю:")
    print("             1) Розбиваємо інтервал [0,1] на рівні біни за прогнозованою ймовірністю.")
    print("             2) Для кожного біна рахуємо:")
    print("                  conf_bin = середній p_i у біні (середня впевненість моделі),")
    print("                  acc_bin  = частку y_i = 1 у біні (справжня частота атак),")
    print("                  w_bin    = частку об'єктів у цьому біні.")
    print("             3) ECE = sum_b w_bin * |acc_bin - conf_bin|.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    printed = 0
    max_to_print = 4
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        if printed < max_to_print or i == n_bins - 1:
            acc = float(y_true_arr[mask].mean())
            conf = float(y_prob_arr[mask].mean())
            w = cnt / max(N, 1)
            print(f"             Бін [{lo:.1f}, {hi:.1f}]: n={cnt}, w={w:.3f}, conf={conf:.3f}, acc={acc:.3f}")
            printed += 1

    print(f"             Підсумковий ECE для цієї моделі = {ece_value:.6f} (0 — ідеально, більше = гірше).")
    print()


def run_unsw_cv_and_test_present(
    paths: Paths,
    outdir: Path,
    models_sel: List[str],
    target_fpr: float,
    cv_folds: int = 3,
    gui_progress: bool = False,
    extra_plots: bool = True,
    feature_importance: bool = True,
) -> None:
    print("=== UNSW CV + test experiment (presentation run) ===")
    print()

    print("Коротко про логіку експерименту:")
    print("  1) Завантажити офіційні train/test частини датасету UNSW-NB15.")
    print("  2) Перевірити, чи немає точних дублікатів рядків між train і test (leakage) і за потреби їх прибрати.")
    print("  3) На train виконати 3-fold cross-validation для кожної моделі:")
    print("       · розбиваємо train на 3 приблизно рівні частини;")
    print("       · у кожному фолді беремо 2 частини для тренування, 1 — для валідації;")
    print("       · повторюємо 3 рази так, щоб кожна частина побула валідаційною рівно один раз;")
    print("       · на кожному фолді рахуємо PR-AUC та F1@0.5 і потім усереднюємо по фолдах.")
    print("  4) Навчити моделі на повному train, підібрати поріг tau для заданого target FPR на окремій валідації,")
    print("       а потім оцінити якість на test (PR-AUC, ROC-AUC, F1@0.5, F1@tau, ECE, Brier).")
    print("  5) Зберегти окремі й зведені таблиці з результатами для подальшого аналізу.")
    print()

    files = unsw_files(paths.data)
    print("[1/5] Loading UNSW train/test datasets...")
    print(f"       Train CSV: {files['train']}")
    print(f"       Test  CSV: {files['test']}")
    Xtr, ytr = load_csv(files["train"])
    Xte, yte = load_csv(files["test"])
    print(f"       Train shape: {Xtr.shape}, positives: {int(ytr.sum())}, negatives: {int((1 - ytr).sum())}")
    print(f"       Test  shape: {Xte.shape}, positives: {int(yte.sum())}, negatives: {int((1 - yte).sum())}")
    print()

    print("[2/5] Checking for data leakage between train and test (exact duplicate rows)...")
    Xte_before = Xte.copy()
    removed_indices: List[int] = []
    try:
        check_data_leakage(Xtr, Xte)
        print("       Result: no exact-duplicate rows between train and test.")
    except ValueError as e:
        print(f"       Leakage detected: {e}")
        Xte, yte, n_removed, removed_indices = drop_test_duplicates(Xtr, Xte, yte)
        print(f"       Removed {n_removed} duplicate rows from test set to avoid optimistic bias.")
        print(f"       New test shape: {Xte.shape}")
    print()

    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    if extra_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)

    # Suspicious-feature leakage scan and drop
    suspicious_cols: List[str] = []
    try:
        df_susp = suspicious_feature_scan(Xtr, ytr, outdir, prefix="unsw_present_train_test", Xte=Xte, yte=yte)
        suspicious_cols = [str(f) for f in df_susp.loc[df_susp["suspicious"] == True, "feature"].tolist()]
        if suspicious_cols:
            print("       Dropping suspicious potential leakage features:", ", ".join(suspicious_cols))
            Xtr = Xtr.drop(columns=suspicious_cols, errors="ignore")
            Xte = Xte.drop(columns=suspicious_cols, errors="ignore")
    except Exception:
        suspicious_cols = []

    all_models = [name for name in make_models().keys() if (not models_sel or name in models_sel)]
    total_steps = len(all_models) * (cv_folds + 1)
    progress = ProgressReporter(total_steps if (gui_progress and total_steps > 0) else None)

    print("[3/5] Running cross-validation on the UNSW train set...")
    print(f"       Models: {_format_models(models_sel)}")
    print(f"       Random seed: {SEED}")
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    rows_cv: List[Dict[str, Any]] = []
    explanation_shown = False

    for model_name, model in make_models().items():
        if models_sel and model_name not in models_sel:
            continue
        print()
        print(f"    -> Model: {model_name}")
        scores: List[float] = []
        f1s: List[float] = []
        fold_idx = 0
        for tr_idx, va_idx in kf.split(Xtr, ytr):
            fold_idx += 1
            X_tr, X_va = Xtr.iloc[tr_idx], Xtr.iloc[va_idx]
            y_tr, y_va = ytr.iloc[tr_idx], ytr.iloc[va_idx]
            print(f"       Fold {fold_idx}/{cv_folds}: train={X_tr.shape[0]} rows, val={X_va.shape[0]} rows")
            pre_local = build_preprocessor(X_tr)
            if isinstance(model, tuple) and model[1] == "calibrate":
                base = make_models(include_calibrated_rf=False)["rf"]
                clf = Pipeline([("pre", pre_local), ("est", base)])
                print("         Using calibrated RandomForest (RF base + isotonic calibration).")
                clf.fit(X_tr, y_tr)
                est_cal = CalibratedClassifierCV(clf.named_steps["est"], cv="prefit", method="isotonic")
                est_cal.fit(pre_local.transform(X_va), y_va)
                proba = est_cal.predict_proba(pre_local.transform(X_va))[:, 1]
            elif model_name == "mlp":
                clf = Pipeline([("pre", pre_local), ("est", model)])
                print("         Using MLP; attempting isotonic calibration on validation split.")
                clf.fit(X_tr, y_tr)
                try:
                    est_cal = CalibratedClassifierCV(clf.named_steps["est"], cv="prefit", method="isotonic")
                    est_cal.fit(pre_local.transform(X_va), y_va)
                    proba = est_cal.predict_proba(pre_local.transform(X_va))[:, 1]
                    print("         Calibration succeeded.")
                except Exception:
                    proba = clf.predict_proba(X_va)[:, 1]
                    print("         Calibration failed; using raw MLP probabilities.")
            else:
                pipe = Pipeline([("pre", pre_local), ("est", model)])
                print("         Using standard pipeline (preprocessing + model).")
                pipe.fit(X_tr, y_tr)
                proba = pipe.predict_proba(X_va)[:, 1]
            fold_pr = pr_auc(y_va, proba)
            fold_f1 = f1_at_threshold(y_va, proba, 0.5)
            if not explanation_shown:
                _explain_f1_example(y_va, proba, thr=0.5)
                explanation_shown = True
            scores.append(fold_pr)
            f1s.append(fold_f1)
            print(f"         Fold metrics: PR-AUC={fold_pr:.6f}, F1@0.5={fold_f1:.6f}")
            progress.step(f"unsw_present_cv_fold {model_name} {fold_idx}")

        row_cv = {
            "model": model_name,
            "pr_auc_mean": float(np.mean(scores)),
            "pr_auc_std": float(np.std(scores)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        }
        rows_cv.append(row_cv)
        print(f"       CV summary for {model_name}: PR-AUC={row_cv['pr_auc_mean']:.6f} ± {row_cv['pr_auc_std']:.6f}, "
              f"F1@0.5={row_cv['f1_mean']:.6f} ± {row_cv['f1_std']:.6f}")

    df_cv = pd.DataFrame(rows_cv)
    cv_path = tables_dir / "table_unsw_cv_present.csv"
    df_cv.to_csv(cv_path, index=False)
    print()
    print(f"       Saved CV summary table to: {cv_path}")
    print()

    print("[4/5] Training on full UNSW train and evaluating on UNSW test...")
    print(f"       Target FPR for threshold selection: {target_fpr}")
    rows_test: List[Dict[str, Any]] = []
    proba_by_model: Dict[str, np.ndarray] = {}
    tau_explained = False
    calib_explained = False

    for model_name, model in make_models().items():
        if models_sel and model_name not in models_sel:
            continue
        print()
        print(f"    -> Model: {model_name} (full-train + test evaluation)")
        pre_final = build_preprocessor(Xtr)

        if isinstance(model, tuple) and model[1] == "calibrate":
            base = make_models(include_calibrated_rf=False)["rf"]
            X_tr, X_va, y_tr, y_va = train_test_split(
                Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr
            )
            print("         Splitting train into train/validation for calibration (RF base).")
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
            print("         Splitting train into train/validation for possible MLP calibration.")
            pipe = Pipeline([("pre", pre_final), ("est", model)])
            pipe.fit(X_tr, y_tr)
            try:
                est_cal = CalibratedClassifierCV(pipe.named_steps["est"], cv="prefit", method="isotonic")
                est_cal.fit(pre_final.transform(X_va), y_va)
                proba_te = est_cal.predict_proba(pre_final.transform(Xte))[:, 1]
                proba_va = est_cal.predict_proba(pre_final.transform(X_va))[:, 1]
                print("         Calibration succeeded.")
            except Exception:
                proba_te = pipe.predict_proba(Xte)[:, 1]
                proba_va = pipe.predict_proba(X_va)[:, 1]
                print("         Calibration failed; using raw MLP probabilities.")
        else:
            pipe = Pipeline([("pre", pre_final), ("est", model)])
            pipe.fit(Xtr, ytr)
            X_tr, X_va, y_tr, y_va = train_test_split(
                Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr
            )
            print("         Using standard pipeline; validation split used only for threshold selection.")
            try:
                proba_te = pipe.predict_proba(Xte)[:, 1]
                proba_va = pipe.predict_proba(X_va)[:, 1]
            except Exception:
                from scipy.special import expit
                proba_te = expit(pipe.decision_function(Xte))
                proba_va = expit(pipe.decision_function(X_va))

        # store test probabilities for later plotting
        proba_by_model[model_name] = proba_te

        tau = threshold_for_target_fpr(y_va, proba_va, target_fpr)
        if not tau_explained:
            _explain_tau_example(y_va, proba_va, target_fpr)
            tau_explained = True
        pr = pr_auc(yte, proba_te)
        roc = roc_auc(yte, proba_te)
        f1_fixed = f1_at_threshold(yte, proba_te, 0.5)
        f1_tau = f1_at_threshold(yte, proba_te, tau)
        ece = expected_calibration_error(yte, proba_te, 10)
        br = brier(yte, proba_te)

        print(f"         Selected threshold tau for FPR~{target_fpr}: {tau:.6f}")
        print(f"         Test metrics: PR-AUC={pr:.6f}, ROC-AUC={roc:.6f}, "
              f"F1@0.5={f1_fixed:.6f}, F1@tau={f1_tau:.6f}, ECE={ece:.6f}, Brier={br:.6f}")

        if not calib_explained:
            _explain_calibration_example(yte, proba_te, ece_value=ece, brier_value=br)
            calib_explained = True

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
        progress.step(f"unsw_present_test {model_name}")

    df_test = pd.DataFrame(rows_test)
    test_path = tables_dir / "table_unsw_test_present.csv"
    df_test.to_csv(test_path, index=False)
    print()
    print(f"       Saved test evaluation table to: {test_path}")
    print()

    # Optional diagnostic plots similar to eval_cross_many UNSW run
    if extra_plots and proba_by_model:
        try:
            best_model = max(rows_test, key=lambda r: r["pr_auc"])["model"]
            best_proba = proba_by_model.get(best_model)
            if best_proba is not None:
                reliability_diagram(
                    yte.values,
                    best_proba,
                    10,
                    str(figures_dir / f"reliability_{best_model}.png"),
                )
                f1_vs_threshold_curve(
                    yte.values,
                    best_proba,
                    str(figures_dir / f"f1_vs_threshold_{best_model}.png"),
                )
        except Exception:
            pass
        try:
            pr_curve_overlay(
                yte.values,
                {m: proba for m, proba in proba_by_model.items() if len(proba) == len(yte)},
                str(figures_dir / "pr_curve_overlay.png"),
            )
            roc_curve_overlay(
                yte.values,
                {m: proba for m, proba in proba_by_model.items() if len(proba) == len(yte)},
                str(figures_dir / "roc_curve_overlay.png"),
            )
        except Exception:
            pass

    try:
        df_sum = pd.merge(df_cv, df_test, on="model", how="outer", suffixes=("_cv", "_test"))
        sum_path = tables_dir / "table_unsw_summary_present.csv"
        df_sum.to_csv(sum_path, index=False)
        print(f"[5/5] Saved combined CV+test summary table to: {sum_path}")
    except Exception:
        # Combined summary is optional; CV and test tables are already saved.
        print("[5/5] Combined summary table skipped due to non-critical error.")

    print()
    print("=== Presentation run finished. You can now inspect the CSV tables in the output directory. ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Presentation-style UNSW CV + test run with step-by-step printed explanation."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root (where data/ and src/ live).")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out/unsw_present"),
        help="Output directory for tables.",
    )
    parser.add_argument(
        "--models",
        default="rf,hgb,mlp,xgb,rf_cal",
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
        help="Number of cross-validation folds to use on the UNSW train set.",
    )
    parser.add_argument(
        "--gui-progress",
        action="store_true",
        help="Emit machine-readable progress markers for GUI.",
    )
    # keep interface compatible with GUI flags used for other scripts
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
    outdir = args.outdir
    models_sel = [m.strip() for m in args.models.split(",") if m.strip()]

    run_unsw_cv_and_test_present(
        paths,
        outdir,
        models_sel,
        args.target_fpr,
        args.cv_folds,
        args.gui_progress,
        extra_plots=not args.no_extra_plots,
        feature_importance=not args.no_feature_importance,
    )


if __name__ == "__main__":
    main()
