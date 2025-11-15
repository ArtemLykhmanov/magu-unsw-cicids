from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import json

from src.config import Paths, SEED
from src.data_io import load_csv, unsw_files, cicids_day_files
from src.preprocess import build_preprocessor, infer_feature_types
from src.models import make_models
from src.metrics_ext import pr_auc, roc_auc, f1_at_threshold, expected_calibration_error, brier
from src.eval_utils import threshold_for_target_fpr, save_manifest, compute_shap_feature_importance
import unicodedata, re
from src.plots import (
    pr_curve_overlay,
    roc_curve_overlay,
    reliability_diagram,
    f1_vs_threshold_curve,
    bar_pr_auc_by_pair,
)


def check_data_leakage(X_a: pd.DataFrame, X_b: pd.DataFrame) -> None:
    a_hashes = pd.util.hash_pandas_object(X_a.apply(lambda r: tuple(r.values.tolist()), axis=1), index=False)
    b_hashes = pd.util.hash_pandas_object(X_b.apply(lambda r: tuple(r.values.tolist()), axis=1), index=False)
    inter = np.intersect1d(a_hashes.values, b_hashes.values)
    if inter.size > 0:
        raise ValueError(f"Found {len(inter)} identical rows between datasets!")


def drop_test_duplicates(X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, pd.Series, int, list[int]]:
    """Remove exact-duplicate rows in test that also appear in train (based on full row content).

    Returns: (X_test_clean, y_test_clean, n_removed, removed_indices)
    """
    a_hashes = pd.util.hash_pandas_object(X_train.apply(lambda r: tuple(r.values.tolist()), axis=1), index=False)
    b_hashes = pd.util.hash_pandas_object(X_test.apply(lambda r: tuple(r.values.tolist()), axis=1), index=False)
    train_set = set(a_hashes.values.tolist())
    mask_dup = b_hashes.map(lambda h: h in train_set)
    n_dup = int(mask_dup.sum()) if hasattr(mask_dup, 'sum') else 0
    removed_idx: list[int] = []
    if n_dup > 0:
        removed_idx = X_test.index[mask_dup].tolist()
        X_test = X_test.loc[~mask_dup].reset_index(drop=True)
        y_test = y_test.loc[~mask_dup].reset_index(drop=True)
    return X_test, y_test, n_dup, removed_idx


def write_leakage_report(outdir: Path, Xtr: pd.DataFrame, Xte_before: pd.DataFrame, Xte_after: pd.DataFrame,
                         removed_indices: list[int]) -> None:
    """Persist a leakage audit report under outdir/logs."""
    logs = outdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # within-set duplicates
    within_train = int(Xtr.duplicated().sum())
    within_test_before = int(Xte_before.duplicated().sum())
    within_test_after = int(Xte_after.duplicated().sum())
    # cross-set duplicates
    def cross_count(A: pd.DataFrame, B: pd.DataFrame) -> int:
        ah = pd.util.hash_pandas_object(A.apply(lambda r: tuple(r.values.tolist()), axis=1), index=False)
        bh = pd.util.hash_pandas_object(B.apply(lambda r: tuple(r.values.tolist()), axis=1), index=False)
        return int(np.intersect1d(ah.values, bh.values).size)
    cross_before = cross_count(Xtr, Xte_before)
    cross_after = cross_count(Xtr, Xte_after)
    report = {
        "within_train_duplicates": within_train,
        "within_test_duplicates_before": within_test_before,
        "within_test_duplicates_after": within_test_after,
        "cross_duplicates_before": cross_before,
        "cross_duplicates_after": cross_after,
        "removed_test_indices": removed_indices,
    }
    with open(logs / "leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _pre_feature_mapping(pre) -> tuple[list[str], list[str]]:
    out_names: list[str] = []
    base_map: list[str] = []

    def _sanitize(n: str) -> str:
        s = unicodedata.normalize('NFKD', str(n))
        s = ''.join(ch for ch in s if ch.isprintable())
        s = re.sub(r"\s+", "_", s.strip())
        s = re.sub(r"[^0-9A-Za-z_./-]", "_", s)
        s = re.sub(r"_+", "_", s)
        return s

    for name, trans, cols in pre.transformers_:
        if cols is None:
            continue
        cols = [ _sanitize(c) for c in list(cols) ]
        if name in ("num", "cat_high"):
            out_names.extend(cols)
            base_map.extend(cols)
        elif name == "cat_low":
            try:
                ohe = trans.named_steps.get("ohe")
            except Exception:
                ohe = None
            if ohe is not None:
                names = list(ohe.get_feature_names_out(cols))
                out_names.extend(names)
                cat_lists = getattr(ohe, "categories_", None)
                if cat_lists is not None:
                    for i, c in enumerate(cols):
                        cats = list(cat_lists[i]) if i < len(cat_lists) else []
                        drop_one = False
                        try:
                            drop_one = (ohe.drop == 'if_binary' and len(cats) >= 2) or (ohe.drop == 'first')
                        except Exception:
                            pass
                        n_out = max(1, len(cats) - (1 if drop_one else 0))
                        base_map.extend([_sanitize(c)] * n_out)
                else:
                    base_map.extend(cols)
            else:
                out_names.extend(cols)
                base_map.extend(cols)
        else:
            out_names.extend(cols)
            base_map.extend(cols)
    return out_names, base_map


def _extract_feature_importances(model_name: str, est, pre) -> list[dict]:
    names, base_map = _pre_feature_mapping(pre)
    imp = getattr(est, "feature_importances_", None)
    if imp is not None:
        imp = np.asarray(imp)
    else:
        coef = getattr(est, "coef_", None)
        if coef is None:
            return []
        coef = np.asarray(coef)
        imp = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef).ravel()
    if names and len(imp) != len(names):
        return []
    df = pd.DataFrame({"base": base_map, "imp": imp[: len(base_map)]})
    agg = df.groupby("base", sort=False)["imp"].sum()
    total = float(agg.sum())
    if total > 0:
        agg = agg / total
    return [{"model": model_name, "feature": k, "importance": float(v)} for k, v in agg.items()]


def _permutation_importances_agg(model_name: str, est, pre, X_val: pd.DataFrame, y_val: pd.Series,
                                 n_repeats: int = 3, random_state: int = SEED) -> list[dict]:
    """Compute permutation importances on transformed validation set and aggregate to base features."""
    try:
        X_t = pre.transform(X_val)
        names, base_map = _pre_feature_mapping(pre)
        res = permutation_importance(
            est,
            X_t,
            y_val,
            scoring="average_precision",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        imp = res.importances_mean
        if names and len(imp) != len(names):
            return []
        df = pd.DataFrame({"base": base_map, "imp": imp[: len(base_map)]})
        agg = df.groupby("base", sort=False)["imp"].sum()
        total = float(agg.sum())
        if total > 0:
            agg = agg / total
        return [{"model": model_name, "feature": k, "importance": float(v)} for k, v in agg.items()]
    except Exception:
        return []


def run_unsw_cv_and_test(paths: Paths, outdir: Path, models_sel, target_fpr: float):
    files = unsw_files(paths.data)
    Xtr, ytr = load_csv(files["train"])  # label conversion inside
    Xte, yte = load_csv(files["test"])   

    print("Checking for data leakage between train and test sets...")
    # Snapshot before-clean for report
    Xte_before = Xte.copy()
    try:
        check_data_leakage(Xtr, Xte)
        print("No data leakage detected!")
        removed = []
    except ValueError as e:
        print(str(e))
        Xte, yte, n_removed, removed = drop_test_duplicates(Xtr, Xte, yte)
        print(f"Removed {n_removed} duplicate rows from test to avoid leakage.")
    # Persist audit report
    write_leakage_report(outdir, Xtr, Xte_before, Xte, removed)

    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Cross-validation on train
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows_cv: list[dict] = []
    for model_name, model in make_models().items():
        if models_sel and model_name not in models_sel:
            continue
        scores = []
        f1s = []
        for tr_idx, va_idx in kf.split(Xtr, ytr):
            X_tr, X_va = Xtr.iloc[tr_idx], Xtr.iloc[va_idx]
            y_tr, y_va = ytr.iloc[tr_idx], ytr.iloc[va_idx]
            pre_local = build_preprocessor(X_tr)
            if isinstance(model, tuple) and model[1] == "calibrate":
                base = make_models(include_calibrated_rf=False)["rf"]
                clf = Pipeline([("pre", pre_local), ("est", base)])
                clf.fit(X_tr, y_tr)
                est_cal = CalibratedClassifierCV(clf.named_steps["est"], cv="prefit", method="isotonic")
                est_cal.fit(pre_local.transform(X_va), y_va)
                proba = est_cal.predict_proba(pre_local.transform(X_va))[:, 1]
            else:
                pipe = Pipeline([("pre", pre_local), ("est", model)])
                pipe.fit(X_tr, y_tr)
                proba = pipe.predict_proba(X_va)[:, 1]
            scores.append(pr_auc(y_va, proba))
            f1s.append(f1_at_threshold(y_va, proba, 0.5))
        rows_cv.append({
            "model": model_name,
            "pr_auc_mean": float(np.mean(scores)),
            "pr_auc_std": float(np.std(scores)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        })
    pd.DataFrame(rows_cv).to_csv(tables_dir / "table_unsw_cv.csv", index=False)

    # Train on full train, evaluate on test
    pr_by_model: dict[str, float] = {}
    proba_by_model: dict[str, np.ndarray] = {}
    fi_rows: list[dict] = []

    test_table_path = tables_dir / "table_unsw_test.csv"
    if test_table_path.exists():
        try:
            test_table_path.unlink()
        except Exception:
            pass

    for model_name, model in make_models().items():
        if models_sel and model_name not in models_sel:
            continue
        try:
            pre_final = build_preprocessor(Xtr)
            if isinstance(model, tuple) and model[1] == "calibrate":
                base = make_models(include_calibrated_rf=False)["rf"]
                X_tr, X_va, y_tr, y_va = train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr)
                clf = Pipeline([("pre", pre_final), ("est", base)])
                clf.fit(X_tr, y_tr)
                est_cal = CalibratedClassifierCV(clf.named_steps["est"], cv="prefit", method="isotonic")
                est_cal.fit(pre_final.transform(X_va), y_va)
                proba_te = est_cal.predict_proba(pre_final.transform(Xte))[:, 1]
                proba_va = est_cal.predict_proba(pre_final.transform(X_va))[:, 1]
                fi = _extract_feature_importances(model_name, clf.named_steps["est"], pre_final)
                if not fi:
                    fi = _permutation_importances_agg(model_name, clf.named_steps["est"], pre_final, X_va, y_va)
                fi_rows.extend(fi)
                # Optional SHAP for calibrated RF (use base estimator on transformed train subset)
                try:
                    Xi = pre_final.transform(Xtr)
                    compute_shap_feature_importance(clf.named_steps["est"], Xi, str(figures_dir / f"shap_{model_name}.png"))
                except Exception:
                    pass
            else:
                pipe = Pipeline([("pre", pre_final), ("est", model)])
                pipe.fit(Xtr, ytr)
                X_tr, X_va, y_tr, y_va = train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr)
                try:
                    proba_te = pipe.predict_proba(Xte)[:, 1]
                    proba_va = pipe.predict_proba(X_va)[:, 1]
                except Exception:
                    from scipy.special import expit
                    proba_te = expit(pipe.decision_function(Xte))
                    proba_va = expit(pipe.decision_function(X_va))
                fi = _extract_feature_importances(model_name, pipe.named_steps["est"], pre_final)
                if not fi:
                    fi = _permutation_importances_agg(model_name, pipe.named_steps["est"], pre_final, X_va, y_va)
                fi_rows.extend(fi)
                # Optional SHAP on transformed train subset
                try:
                    Xi = pre_final.transform(Xtr)
                    compute_shap_feature_importance(pipe.named_steps["est"], Xi, str(figures_dir / f"shap_{model_name}.png"))
                except Exception:
                    pass

            tau = threshold_for_target_fpr(y_va, proba_va, target_fpr)
            row = {
                "model": model_name,
                "pr_auc": pr_auc(yte, proba_te),
                "roc_auc": roc_auc(yte, proba_te),
                "f1@0.5": f1_at_threshold(yte, proba_te, 0.5),
                "tau@FPR": tau,
                "f1@tau": f1_at_threshold(yte, proba_te, tau),
                "ece": expected_calibration_error(yte, proba_te, 10),
                "brier": brier(yte, proba_te),
            }
            df_row = pd.DataFrame([row])
            if not test_table_path.exists():
                df_row.to_csv(test_table_path, index=False, mode="w")
            else:
                df_row.to_csv(test_table_path, index=False, mode="a", header=False)
            pr_by_model[model_name] = row["pr_auc"]
            proba_by_model[model_name] = proba_te
            print(f"Saved results for model: {model_name}")
        except Exception as e:
            print(f"Warning: failed to evaluate model {model_name}: {e}")

    try:
        if pr_by_model:
            best_model = max(pr_by_model.items(), key=lambda x: x[1])[0]
            best_proba = proba_by_model.get(best_model)
            if best_proba is not None:
                reliability_diagram(yte.values, best_proba, 10, str(figures_dir / f"reliability_{best_model}.png"))
                f1_vs_threshold_curve(yte.values, best_proba, str(figures_dir / f"f1_vs_threshold_{best_model}.png"))
            try:
                filtered_probas = {k: v for k, v in proba_by_model.items() if hasattr(v, "__len__") and len(v) == len(yte)}
                if filtered_probas:
                    pr_curve_overlay(yte.values, filtered_probas, str(figures_dir / "pr_curve_overlay.png"))
                    roc_curve_overlay(yte.values, filtered_probas, str(figures_dir / "roc_curve_overlay.png"))
            except Exception:
                pass
            try:
                top2 = sorted(pr_by_model.items(), key=lambda x: x[1], reverse=True)[:2]
                for mname, _ in top2:
                    probs = proba_by_model.get(mname)
                    if probs is not None:
                        reliability_diagram(yte.values, probs, 10, str(figures_dir / f"reliability_{mname}.png"))
                        f1_vs_threshold_curve(yte.values, probs, str(figures_dir / f"f1_vs_threshold_{mname}.png"))
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: failed to write diagnostic figures: {e}")

    try:
        if fi_rows:
            pd.DataFrame(fi_rows).to_csv(tables_dir / "table_unsw_feature_importance.csv", index=False)
    except Exception as e:
        print(f"Warning: failed to write feature importance table: {e}")

    try:
        cv_path = tables_dir / "table_unsw_cv.csv"
        test_path = tables_dir / "table_unsw_test.csv"
        if cv_path.exists() and test_path.exists():
            df_cv = pd.read_csv(cv_path)
            df_te = pd.read_csv(test_path)
            df_sum = pd.merge(df_cv, df_te, on="model", how="outer", suffixes=("_cv", "_test"))
            df_sum.to_csv(tables_dir / "table_unsw_summary.csv", index=False)
        # After saving fi_rows, also produce heatmaps and top-10 table
        fi_path = tables_dir / "table_unsw_feature_importance.csv"
        if fi_path.exists():
            try:
                df_fi = pd.read_csv(fi_path)
                # top-10 table
                rows_top = []
                for m, g in df_fi.groupby('model'):
                    g2 = g.sort_values('importance', ascending=False).head(10)
                    for r, rec in enumerate(g2.itertuples(index=False), start=1):
                        rows_top.append({'model': m, 'rank': r, 'feature': rec.feature, 'importance': rec.importance})
                pd.DataFrame(rows_top).to_csv(tables_dir / 'table_unsw_feature_top10.csv', index=False)
                # heatmaps
                from src.plots import feature_importance_heatmap, feature_importance_heatmap_overall
                feature_importance_heatmap(df_fi, str(figures_dir / 'feature_importance_heatmap.png'), top_n=20)
                feature_importance_heatmap_overall(df_fi, str(figures_dir / 'feature_importance_heatmap_overall.png'), top_n=20)
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: failed to write summary table: {e}")


def run_cicids_cross_day(paths: Paths, outdir: Path, models_sel, target_fpr: float, val_day: str | None):
    files = cicids_day_files(paths.data)
    val_key = val_day if val_day in files else "monday"
    X_val, y_val = load_csv(files[val_key])

    train_keys = ["tuesday", "wednesday"]
    X_tr_list, y_tr_list = [], []
    for k in train_keys:
        Xi, yi = load_csv(files[k])
        X_tr_list.append(Xi)
        y_tr_list.append(yi)
    Xtr = pd.concat(X_tr_list, axis=0, ignore_index=True)
    ytr = pd.concat(y_tr_list, axis=0, ignore_index=True)

    pairs = [
        ("Tue+Wed→Thu_web", files["thursday_web"]),
        ("Tue+Wed→Thu_inf", files["thursday_inf"]),
        ("Tue+Wed→Fri_port", files["friday_port"]),
        ("Tue+Wed→Fri_ddos", files["friday_ddos"]),
    ]

    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pr_map: dict[str, dict[str, float]] = {}
    for model_name, model in make_models().items():
        if models_sel and model_name not in models_sel:
            continue
        pr_map[model_name] = {}
        pre = build_preprocessor(Xtr)
        if isinstance(model, tuple) and model[1] == "calibrate":
            base = make_models(include_calibrated_rf=False)["rf"]
            pipe_base = Pipeline([("pre", pre), ("est", base)])
            X_tr, X_cal, y_tr, y_cal = train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr)
            pipe_base.fit(X_tr, y_tr)
            est_cal = CalibratedClassifierCV(pipe_base.named_steps["est"], cv="prefit", method="isotonic")
            est_cal.fit(pre.transform(X_cal), y_cal)
            proba_val = est_cal.predict_proba(pre.transform(X_val))[:, 1]
            tau = threshold_for_target_fpr(y_val, proba_val, target_fpr)
            predictor = lambda Xt: est_cal.predict_proba(pre.transform(Xt))[:, 1]
        else:
            pipe = Pipeline([("pre", pre), ("est", model)])
            pipe.fit(Xtr, ytr)
            proba_val = pipe.predict_proba(X_val)[:, 1]
            tau = threshold_for_target_fpr(y_val, proba_val, target_fpr)
            predictor = lambda Xt: pipe.predict_proba(Xt)[:, 1]

        for pair_name, test_path in pairs:
            Xte, yte = load_csv(test_path)
            proba_te = predictor(Xte)
            row = {
                "pair": pair_name,
                "model": model_name,
                "pr_auc": pr_auc(yte, proba_te),
                "f1@0.5": f1_at_threshold(yte, proba_te, 0.5),
                "tau@FPR": tau,
                "f1@tau": f1_at_threshold(yte, proba_te, tau),
            }
            rows.append(row)
            pr_map[model_name][pair_name] = row["pr_auc"]
    pd.DataFrame(rows).to_csv(tables_dir / "table_cicids_cross_day.csv", index=False)

    pairs_names = [p[0] for p in pairs]
    models_names = [m for m in make_models().keys() if (not models_sel or m in models_sel)]
    pr_values = {pair: {m: pr_map.get(m, {}).get(pair, np.nan) for m in models_names} for pair in pairs_names}
    bar_pr_auc_by_pair(pairs_names, models_names, pr_values, str(figures_dir / "fig_pr_auc_by_pair.png"))


def run_cross_dataset(paths: Paths, outdir: Path, models_sel, target_fpr: float):
    cic = cicids_day_files(paths.data)
    unsw = unsw_files(paths.data)
    Xtr_list, ytr_list = [], []
    for k in ["tuesday", "wednesday"]:
        Xi, yi = load_csv(cic[k])
        Xtr_list.append(Xi)
        ytr_list.append(yi)
    Xtr = pd.concat(Xtr_list, axis=0, ignore_index=True)
    ytr = pd.concat(ytr_list, axis=0, ignore_index=True)
    Xte, yte = load_csv(unsw["test"])  # different schema

    all_cols = sorted(set(Xtr.columns) | set(Xte.columns))
    Xtr_ = Xtr.reindex(columns=all_cols, fill_value=0)
    Xte_ = Xte.reindex(columns=all_cols, fill_value=0)

    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, model in make_models().items():
        if models_sel and model_name not in models_sel:
            continue
        num_cols, _ = infer_feature_types(Xtr_)
        if num_cols:
            Xtr_[num_cols] = Xtr_[num_cols].apply(pd.to_numeric, errors="coerce")
            Xte_[num_cols] = Xte_[num_cols].apply(pd.to_numeric, errors="coerce")

        pre = build_preprocessor(Xtr_)
        pipe = Pipeline([("pre", pre), ("est", model)])
        pipe.fit(Xtr_, ytr)
        proba_te = pipe.predict_proba(Xte_)[:, 1]
        tau = threshold_for_target_fpr(ytr, pipe.predict_proba(Xtr_)[:, 1], target_fpr)
        row = {
            "train→test": "CIC(Tue+Wed)→UNSW(test)",
            "model": model_name,
            "pr_auc": pr_auc(yte, proba_te),
            "f1@0.5": f1_at_threshold(yte, proba_te, 0.5),
            "f1@tau": f1_at_threshold(yte, proba_te, tau),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables_dir / "table_cross_dataset.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, choices=['unsw_cv_test','cicids_cross_day','cross_dataset'])
    ap.add_argument('--root', default='.')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--models', default='rf,logreg,mlp,hgb,rf_cal,xgb')
    ap.add_argument('--target-fpr', type=float, default=0.01)
    ap.add_argument('--val', default='monday', help='validation set for cicids_cross_day (monday|tuesday|...)')
    args = ap.parse_args()

    paths = Paths.from_root(args.root)
    outdir = Path(args.outdir)
    models_sel = [m.strip() for m in args.models.split(',') if m.strip()]
    save_manifest(outdir, vars(args))

    if args.scenario == 'unsw_cv_test':
        run_unsw_cv_and_test(paths, outdir, models_sel, args.target_fpr)
    elif args.scenario == 'cicids_cross_day':
        run_cicids_cross_day(paths, outdir, models_sel, args.target_fpr, args.val)
    elif args.scenario == 'cross_dataset':
        run_cross_dataset(paths, outdir, models_sel, args.target_fpr)


if __name__ == '__main__':
    main()
