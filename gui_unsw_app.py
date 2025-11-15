import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import csv
import json
import re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_io import detect_label_columns, to_binary_labels, drop_non_feature_columns


@dataclass
class Scenario:
    label: str
    build_cmd: Callable[[str, float, List[str], int, str | None], List[str]]
    default_outdir: str
    default_target_fpr: float
    default_cv_folds: int
    default_models: List[str]
    supports_cv_folds: bool = True
    supports_models: bool = True
    requires_dataset: bool = False
    default_extra_plots: bool = True
    default_feature_importance: bool = True


def _make_scenarios(project_root: Path) -> Dict[str, Scenario]:
    root_str = str(project_root)

    def single_csv_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        models_arg = ",".join(models_sel) if models_sel else ""
        csv_arg = dataset_path or ""
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "single_dataset_cv.py"),
            "--root",
            root_str,
            "--csv",
            csv_arg,
            "--outdir",
            outdir,
            "--models",
            models_arg,
            "--target-fpr",
            str(target_fpr),
            "--cv-folds",
            str(cv_folds),
            "--gui-progress",
        ]

    def unsw_standard_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        models_arg = ",".join(models_sel) if models_sel else ""
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "eval_cross_many.py"),
            "--scenario",
            "unsw_cv_test",
            "--root",
            root_str,
            "--target-fpr",
            str(target_fpr),
            "--outdir",
            outdir,
            "--models",
            models_arg,
            "--cv-folds",
            str(cv_folds),
            "--gui-progress",
        ]

    def unsw_present_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        models_arg = ",".join(models_sel) if models_sel else ""
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "present_unsw_cv.py"),
            "--root",
            root_str,
            "--outdir",
            outdir,
            "--models",
            models_arg,
            "--target-fpr",
            str(target_fpr),
            "--cv-folds",
            str(cv_folds),
            "--gui-progress",
        ]

    def unsw_repro_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        models_arg = ",".join(models_sel) if models_sel else ""
        compare_model = models_sel[0] if models_sel else "rf"
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "eval_cross_many.py"),
            "--scenario",
            "unsw_cv_test",
            "--root",
            root_str,
            "--target-fpr",
            str(target_fpr),
            "--outdir",
            outdir,
            "--models",
            models_arg,
            "--cv-folds",
            str(cv_folds),
            "--gui-progress",
            "--compare-to",
            "out/unsw",
            "--compare-model",
            compare_model,
        ]

    def cicids_nocal_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        models_arg = ",".join(models_sel) if models_sel else ""
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "eval_cross_many.py"),
            "--scenario",
            "cicids_cross_day",
            "--root",
            root_str,
            "--target-fpr",
            str(target_fpr),
            "--outdir",
            outdir,
            "--models",
            models_arg,
            "--val",
            "monday",
            "--pairs",
            "thursday_web,friday_ddos",
            "--gui-progress",
        ]

    def cicids_nocal_repro_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        models_arg = ",".join(models_sel) if models_sel else ""
        compare_model = models_sel[0] if models_sel else "rf"
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "eval_cross_many.py"),
            "--scenario",
            "cicids_cross_day",
            "--root",
            root_str,
            "--target-fpr",
            str(target_fpr),
            "--outdir",
            outdir,
            "--models",
            models_arg,
            "--val",
            "monday",
            "--pairs",
            "friday_ddos",
            "--subsample",
            "50000",
            "--gui-progress",
            "--compare-to",
            "out/cicids_nocal",
            "--compare-model",
            compare_model,
        ]

    def cicids_smooth_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        models_arg = ",".join(models_sel) if models_sel else ""
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "eval_cross_many.py"),
            "--scenario",
            "cicids_cross_day",
            "--root",
            root_str,
            "--target-fpr",
            str(target_fpr),
            "--outdir",
            outdir,
            "--models",
            models_arg,
            "--val",
            "monday",
            "--pairs",
            "thursday_web,friday_ddos",
            "--calibrate-all",
            "--gui-progress",
        ]

    def cicids_smooth_repro_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        models_arg = ",".join(models_sel) if models_sel else ""
        compare_model = models_sel[0] if models_sel else "rf"
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "eval_cross_many.py"),
            "--scenario",
            "cicids_cross_day",
            "--root",
            root_str,
            "--target-fpr",
            str(target_fpr),
            "--outdir",
            outdir,
            "--models",
            models_arg,
            "--val",
            "monday",
            "--pairs",
            "friday_ddos",
            "--calibrate-all",
            "--subsample",
            "50000",
            "--gui-progress",
            "--compare-to",
            "out/cicids_smooth",
            "--compare-model",
            compare_model,
        ]

    def unsw_mcnemar_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "train_unsw_models_for_mcnemar.py"),
            "--out-dir",
            outdir,
        ]

    def unsw_mcnemar_repro_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "train_unsw_models_for_mcnemar.py"),
            "--out-dir",
            outdir,
        ]

    def unsw_shap_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "shap_stability_unsw.py"),
            "--n-splits",
            "5",
            "--sample-size",
            "800",
            "--out-dir",
            outdir,
        ]

    def unsw_shap_repro_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "shap_stability_unsw.py"),
            "--n-splits",
            "3",
            "--sample-size",
            "200",
            "--out-dir",
            outdir,
        ]

    def unsw_ablation_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "feature_selection_ablation.py"),
            "--top-n",
            "20",
            "10",
            "--out-root",
            outdir,
        ]

    def xdom_harm_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        models_arg = ",".join(models_sel) if models_sel else ""
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "eval_cross_many.py"),
            "--scenario",
            "cross_dataset",
            "--root",
            root_str,
            "--target-fpr",
            str(target_fpr),
            "--outdir",
            outdir,
            "--models",
            models_arg,
            "--align",
            "harmonized",
            "--gui-progress",
        ]

    def xdom_harm_repro_cmd(outdir: str, target_fpr: float, models_sel: List[str], cv_folds: int, dataset_path: str | None) -> List[str]:
        models_arg = ",".join(models_sel) if models_sel else ""
        compare_model = models_sel[0] if models_sel else "rf"
        return [
            sys.executable,
            "-u",
            str(project_root / "experiments" / "eval_cross_many.py"),
            "--scenario",
            "cross_dataset",
            "--root",
            root_str,
            "--target-fpr",
            str(target_fpr),
            "--outdir",
            outdir,
            "--models",
            models_arg,
            "--align",
            "harmonized",
            "--gui-progress",
            "--compare-to",
            "out/xdom_harm",
            "--compare-model",
            compare_model,
        ]

    return {
        "Single CSV (generic)":
            Scenario(
                label="Single CSV (generic CV + test split)",
                build_cmd=single_csv_cmd,
                default_outdir="out/custom_result",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=["rf", "logreg", "mlp", "hgb", "rf_cal", "xgb"],
                requires_dataset=True,
            ),
        "Cross CSV (train->test)":
            Scenario(
                label="Cross CSV (train->test)",
                build_cmd=single_csv_cmd,  # not used; command built in GUI
                default_outdir="out/custom_result",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=["rf", "logreg", "mlp", "hgb", "rf_cal", "xgb"],
                requires_dataset=True,
            ),
        "Cross CSV (train->test + finetune)":
            Scenario(
                label="Cross CSV (train->test + finetune)",
                build_cmd=single_csv_cmd,  # not used; command built in GUI
                default_outdir="out/custom_result",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=["rf", "logreg", "mlp", "hgb", "rf_cal", "xgb"],
                requires_dataset=True,
            ),
        "UNSW CV + test (standard)":
            Scenario(
                label="UNSW CV + test (standard)",
                build_cmd=unsw_standard_cmd,
                default_outdir="out/unsw",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=["rf", "mlp", "hgb", "rf_cal", "xgb"],
            ),
        "UNSW CV + test (RF repro-check)":
            Scenario(
                label="UNSW CV + test (RF repro-check)",
                build_cmd=unsw_repro_cmd,
                default_outdir="out_check/unsw_cv_test",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=["rf"],
            ),
        "UNSW CV + test (presentation)":
            Scenario(
                label="UNSW CV + test (presentation)",
                build_cmd=unsw_present_cmd,
                default_outdir="out/unsw_present",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=["rf", "mlp", "hgb", "rf_cal", "xgb"],
            ),
        "CICIDS cross-day (no calibration)":
            Scenario(
                label="CICIDS cross-day (no calibration)",
                build_cmd=cicids_nocal_cmd,
                default_outdir="out/cicids_nocal",
                default_target_fpr=0.001,
                default_cv_folds=3,
                default_models=["rf", "rf_cal", "logreg", "hgb", "xgb"],
                supports_cv_folds=False,
            ),
        "CICIDS cross-day (no calib, RF repro-check)":
            Scenario(
                label="CICIDS cross-day (no calib, RF repro-check)",
                build_cmd=cicids_nocal_repro_cmd,
                default_outdir="out_check/cicids_nocal",
                default_target_fpr=0.001,
                default_cv_folds=3,
                default_models=["rf"],
                supports_cv_folds=False,
            ),
        "CICIDS cross-day (calibrated)":
            Scenario(
                label="CICIDS cross-day (calibrated)",
                build_cmd=cicids_smooth_cmd,
                default_outdir="out/cicids_smooth",
                default_target_fpr=0.001,
                default_cv_folds=3,
                default_models=["rf", "rf_cal", "logreg", "hgb", "xgb"],
                supports_cv_folds=False,
            ),
        "CICIDS cross-day (calib, repro-check)":
            Scenario(
                label="CICIDS cross-day (calib, repro-check)",
                build_cmd=cicids_smooth_repro_cmd,
                default_outdir="out_check/cicids_smooth",
                default_target_fpr=0.001,
                default_cv_folds=3,
                default_models=["rf", "logreg"],
                supports_cv_folds=False,
            ),
        "Cross-dataset CIC(Tue+Wed)->UNSW (harmonized)":
            Scenario(
                label="Cross-dataset CIC(Tue+Wed)->UNSW (harmonized)",
                build_cmd=xdom_harm_cmd,
                default_outdir="out/xdom_harm",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=["rf", "logreg", "mlp", "hgb", "xgb"],
                supports_cv_folds=False,
            ),
        "Cross-dataset CIC(Tue+Wed)->UNSW (harmonized, RF repro-check)":
            Scenario(
                label="Cross-dataset CIC(Tue+Wed)->UNSW (harmonized, RF repro-check)",
                build_cmd=xdom_harm_repro_cmd,
                default_outdir="out_check/xdom_harm",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=["rf"],
                supports_cv_folds=False,
            ),
        "UNSW models for McNemar":
            Scenario(
                label="UNSW models for McNemar",
                build_cmd=unsw_mcnemar_cmd,
                default_outdir="out/models",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=[],
                supports_cv_folds=False,
                supports_models=False,
            ),
        "UNSW models for McNemar (repro-check)":
            Scenario(
                label="UNSW models for McNemar (repro-check)",
                build_cmd=unsw_mcnemar_repro_cmd,
                default_outdir="out_check/models",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=[],
                supports_cv_folds=False,
                supports_models=False,
            ),
        "UNSW SHAP stability (RF)":
            Scenario(
                label="UNSW SHAP stability (RF)",
                build_cmd=unsw_shap_cmd,
                default_outdir="out/unsw_shap",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=[],
                supports_cv_folds=False,
                supports_models=False,
            ),
        "UNSW SHAP stability (repro-check)":
            Scenario(
                label="UNSW SHAP stability (repro-check)",
                build_cmd=unsw_shap_repro_cmd,
                default_outdir="out_check/unsw_shap",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=[],
                supports_cv_folds=False,
                supports_models=False,
            ),
        "UNSW feature selection ablation (RF)":
            Scenario(
                label="UNSW feature selection ablation (RF)",
                build_cmd=unsw_ablation_cmd,
                default_outdir="out/feature_ablation",
                default_target_fpr=0.01,
                default_cv_folds=3,
                default_models=[],
                supports_cv_folds=False,
                supports_models=False,
            ),
    }


class App(tk.Tk):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.title("UNSW experiments GUI")
        self.geometry("900x600")

        self.project_root = project_root
        self.scenarios = _make_scenarios(project_root)
        self.scenario_groups: Dict[str, List[str]] = {
            "Preconfigured": [
                "UNSW CV + test",
                "CICIDS cross-day",
                "Cross-dataset CIC->UNSW",
                "UNSW models for McNemar",
                "UNSW SHAP stability",
                "UNSW feature ablation",
            ],
            "Custom / dataset-based": [
                "Single CSV (generic)",
                "Cross CSV (train->test)",
                "Cross CSV (train->test + finetune)",
            ],
        }
        # Mapping from family name + variant name to scenario key
        self.scenario_variants: Dict[str, Dict[str, str]] = {
            "UNSW CV + test": {
                "Standard": "UNSW CV + test (standard)",
                "RF repro-check": "UNSW CV + test (RF repro-check)",
                "Presentation": "UNSW CV + test (presentation)",
            },
            "CICIDS cross-day": {
                "No calibration": "CICIDS cross-day (no calibration)",
                "No calib, RF repro-check": "CICIDS cross-day (no calib, RF repro-check)",
                "Calibrated": "CICIDS cross-day (calibrated)",
                "Calib, repro-check": "CICIDS cross-day (calib, repro-check)",
            },
            "Cross-dataset CIC->UNSW": {
                "Harmonized": "Cross-dataset CIC(Tue+Wed)->UNSW (harmonized)",
                "Harmonized, RF repro-check": "Cross-dataset CIC(Tue+Wed)->UNSW (harmonized, RF repro-check)",
            },
            "UNSW models for McNemar": {
                "Full (RF+XGB)": "UNSW models for McNemar",
                "Repro-check": "UNSW models for McNemar (repro-check)",
            },
            "UNSW SHAP stability": {
                "Full": "UNSW SHAP stability (RF)",
                "Repro-check": "UNSW SHAP stability (repro-check)",
            },
            "UNSW feature ablation": {
                "RF ablation": "UNSW feature selection ablation (RF)",
            },
            "Single CSV (generic)": {
                "Default": "Single CSV (generic)",
            },
            "Cross CSV (train->test)": {
                "Default": "Cross CSV (train->test)",
            },
            "Cross CSV (train->test + finetune)": {
                "Default": "Cross CSV (train->test + finetune)",
            },
        }

        self._current_process: subprocess.Popen | None = None
        self._runner_thread: threading.Thread | None = None
        self._start_time: float | None = None

        self.model_keys: List[str] = ["rf", "rf_cal", "logreg", "hgb", "mlp", "xgb"]
        self.model_vars: Dict[str, tk.BooleanVar] = {}
        self.model_checkbuttons: Dict[str, tk.Checkbutton] = {}

        self.extra_plots_var = tk.BooleanVar(value=True)
        self.feature_importance_var = tk.BooleanVar(value=True)

        self.train_csv_paths: List[str] = []
        self.test_csv_paths: List[str] = []
        self.finetune_var = tk.StringVar()

        self.results_csv_var = tk.StringVar()
        self.results_text: scrolledtext.ScrolledText | None = None

        self.profiles_path = self.project_root / "profiles.json"

        self._build_widgets()

    def _build_widgets(self) -> None:
        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        # Scenario group selection
        tk.Label(top_frame, text="Scenario group:").grid(row=0, column=0, sticky="w")
        self.group_var = tk.StringVar()
        group_names = list(self.scenario_groups.keys())
        if group_names:
            self.group_var.set(group_names[0])
        self.group_menu = tk.OptionMenu(top_frame, self.group_var, *group_names, command=self._on_group_change)
        self.group_menu.grid(row=0, column=1, sticky="w", padx=5)

        # Scenario selection within group
        tk.Label(top_frame, text="Scenario:").grid(row=1, column=0, sticky="w")
        self.scenario_family_var = tk.StringVar()
        self.scenario_menu = tk.OptionMenu(top_frame, self.scenario_family_var, "")
        self.scenario_menu.grid(row=1, column=1, sticky="w", padx=5)

        # Scenario variant (right of Scenario)
        tk.Label(top_frame, text="Variant:").grid(row=1, column=2, sticky="w")
        self.variant_var = tk.StringVar()
        self.variant_menu = tk.OptionMenu(top_frame, self.variant_var, "")
        self.variant_menu.grid(row=1, column=3, sticky="w", padx=5)

        # Target FPR
        tk.Label(top_frame, text="Target FPR:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.target_fpr_var = tk.StringVar()
        self.target_fpr_entry = tk.Entry(top_frame, textvariable=self.target_fpr_var, width=10)
        self.target_fpr_entry.grid(row=2, column=1, sticky="w", padx=5, pady=(5, 0))

        # CV folds
        tk.Label(top_frame, text="CV folds:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.cv_folds_var = tk.StringVar()
        self.cv_folds_entry = tk.Spinbox(top_frame, from_=2, to=10, width=5, textvariable=self.cv_folds_var)
        self.cv_folds_entry.grid(row=3, column=1, sticky="w", padx=5, pady=(5, 0))

        # Finetune percentage for Cross CSV finetune scenario
        tk.Label(top_frame, text="Finetune % (test):").grid(row=3, column=2, sticky="w", pady=(5, 0))
        self.finetune_entry = tk.Spinbox(top_frame, from_=0, to=50, width=5, textvariable=self.finetune_var)
        self.finetune_entry.grid(row=3, column=3, sticky="w", padx=5, pady=(5, 0))

        # Outdir
        tk.Label(top_frame, text="Output directory:").grid(row=4, column=0, sticky="w", pady=(5, 0))
        self.outdir_var = tk.StringVar()
        self.outdir_entry = tk.Entry(top_frame, textvariable=self.outdir_var, width=40)
        self.outdir_entry.grid(row=4, column=1, sticky="w", padx=5, pady=(5, 0))
        browse_btn = tk.Button(top_frame, text="Browse...", command=self._browse_outdir)
        browse_btn.grid(row=4, column=2, sticky="w", padx=5, pady=(5, 0))

        # Models checklist
        tk.Label(top_frame, text="Models:").grid(row=5, column=0, sticky="nw", pady=(5, 0))
        models_frame = tk.Frame(top_frame)
        models_frame.grid(row=5, column=1, columnspan=2, sticky="w", padx=5, pady=(5, 0))

        for idx, key in enumerate(self.model_keys):
            var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(models_frame, text=key, variable=var)
            cb.grid(row=0, column=idx, sticky="w", padx=(0, 5))
            self.model_vars[key] = var
            self.model_checkbuttons[key] = cb

        # Dataset CSVs for custom scenarios
        tk.Label(top_frame, text="Train CSV:").grid(row=6, column=0, sticky="w", pady=(5, 0))
        self.dataset_var = tk.StringVar()
        self.dataset_entry = tk.Entry(top_frame, textvariable=self.dataset_var, width=30)
        self.dataset_entry.grid(row=6, column=1, sticky="w", padx=5, pady=(5, 0))
        self.dataset_browse_btn = tk.Button(top_frame, text="Browse...", command=self._browse_dataset)
        self.dataset_browse_btn.grid(row=6, column=2, sticky="w", padx=5, pady=(5, 0))
        self.train_submit_btn = tk.Button(top_frame, text="Submit", command=self._on_train_submit)
        self.train_submit_btn.grid(row=6, column=3, sticky="w", padx=5, pady=(5, 0))

        tk.Label(top_frame, text="Test CSV:").grid(row=6, column=4, sticky="w", pady=(5, 0))
        self.test_dataset_var = tk.StringVar()
        self.test_dataset_entry = tk.Entry(top_frame, textvariable=self.test_dataset_var, width=30)
        self.test_dataset_entry.grid(row=6, column=5, sticky="w", padx=5, pady=(5, 0))
        self.test_dataset_browse_btn = tk.Button(top_frame, text="Browse...", command=self._browse_test_dataset)
        self.test_dataset_browse_btn.grid(row=6, column=6, sticky="w", padx=5, pady=(5, 0))
        self.test_submit_btn = tk.Button(top_frame, text="Submit", command=self._on_test_submit)
        self.test_submit_btn.grid(row=6, column=7, sticky="w", padx=5, pady=(5, 0))

        # Extra outputs options
        extras_frame = tk.Frame(top_frame)
        extras_frame.grid(row=7, column=1, columnspan=2, sticky="w", padx=5, pady=(5, 0))
        tk.Checkbutton(
            extras_frame,
            text="Extra plots (PR/ROC, calibration, overlays)",
            variable=self.extra_plots_var,
        ).pack(anchor="w")
        tk.Checkbutton(
            extras_frame,
            text="Feature-importance outputs (tables, heatmaps)",
            variable=self.feature_importance_var,
        ).pack(anchor="w")

        # Run / stop buttons
        buttons_frame = tk.Frame(self)
        buttons_frame.pack(fill=tk.X, padx=10)

        self.run_btn = tk.Button(buttons_frame, text="Run", command=self._on_run_clicked)
        self.run_btn.pack(side=tk.LEFT)

        self.stop_btn = tk.Button(buttons_frame, text="Stop", command=self._on_stop_clicked, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.save_cfg_btn = tk.Button(buttons_frame, text="Save config", command=self._on_save_config)
        self.save_cfg_btn.pack(side=tk.LEFT, padx=5)

        self.load_cfg_btn = tk.Button(buttons_frame, text="Load config", command=self._on_load_config)
        self.load_cfg_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready.")
        status_label = tk.Label(buttons_frame, textvariable=self.status_var, anchor="w")
        status_label.pack(side=tk.LEFT, padx=10)

        self.elapsed_var = tk.StringVar()
        self.elapsed_var.set("Elapsed: 0.0 s")
        elapsed_label = tk.Label(buttons_frame, textvariable=self.elapsed_var, anchor="w")
        elapsed_label.pack(side=tk.LEFT, padx=10)

        self.progress = ttk.Progressbar(buttons_frame, mode="indeterminate", length=150)
        self.progress.pack(side=tk.RIGHT, padx=10)

        copy_btn = tk.Button(buttons_frame, text="Copy log", command=self._copy_log_to_clipboard)
        copy_btn.pack(side=tk.RIGHT, padx=5)

        # Bottom notebook: Log + Results
        bottom_notebook = ttk.Notebook(self)
        bottom_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Log output tab
        log_frame = tk.Frame(bottom_notebook)
        bottom_notebook.add(log_frame, text="Log")

        tk.Label(log_frame, text="Log output:").pack(anchor="w")
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Enable Ctrl+C copy from selection explicitly
        self.log_text.bind("<Control-c>", self._copy_selection_to_clipboard)

        # Results tab
        results_frame = tk.Frame(bottom_notebook)
        bottom_notebook.add(results_frame, text="Results")

        top_res = tk.Frame(results_frame)
        top_res.pack(fill=tk.X, pady=(0, 5))

        tk.Label(top_res, text="Results CSV:").grid(row=0, column=0, sticky="w")
        res_entry = tk.Entry(top_res, textvariable=self.results_csv_var, width=50)
        res_entry.grid(row=0, column=1, sticky="w", padx=5)
        tk.Button(top_res, text="Browse...", command=self._browse_results_csv).grid(
            row=0, column=2, sticky="w", padx=5
        )
        tk.Button(top_res, text="Load", command=self._load_results_csv).grid(
            row=0, column=3, sticky="w", padx=5
        )

        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.NONE)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        self._update_scenario_menu()
        self._update_variant_menu()
        self._apply_scenario_defaults()

    def _on_group_change(self, *_: object) -> None:
        self._update_scenario_menu()
        self._update_variant_menu()
        self._apply_scenario_defaults()

    def _update_scenario_menu(self) -> None:
        group = self.group_var.get()
        names = self.scenario_groups.get(group, list(self.scenarios.keys()))
        menu = self.scenario_menu["menu"]
        menu.delete(0, "end")
        if not names:
            self.scenario_family_var.set("")
            return
        for name in names:
            menu.add_command(label=name, command=lambda v=name: self._set_family(v))
        self.scenario_family_var.set(names[0])

    def _update_variant_menu(self) -> None:
        group = self.group_var.get()
        family = self.scenario_family_var.get()
        menu = self.variant_menu["menu"]
        menu.delete(0, "end")
        if group == "Preconfigured":
            variants = list(self.scenario_variants.get(family, {}).keys())
        else:
            variants = ["Default"]
        if not variants:
            self.variant_var.set("")
            return
        for v in variants:
            menu.add_command(label=v, command=lambda val=v: self._set_variant(val))
        self.variant_var.set(variants[0])

    def _set_family(self, value: str) -> None:
        self.scenario_family_var.set(value)
        self._update_variant_menu()
        self._on_scenario_change()

    def _set_variant(self, value: str) -> None:
        self.variant_var.set(value)
        self._on_scenario_change()

    def _on_scenario_change(self, *_: object) -> None:
        self._apply_scenario_defaults()

    def _apply_scenario_defaults(self) -> None:
        group = self.group_var.get()
        if group == "Preconfigured":
            family = self.scenario_family_var.get()
            variant = self.variant_var.get()
            key = self.scenario_variants.get(family, {}).get(variant)
        else:
            key = self.scenario_family_var.get()
        scenario = self.scenarios.get(key)
        if not scenario:
            return
        self.target_fpr_var.set(str(scenario.default_target_fpr))
        self.outdir_var.set(scenario.default_outdir)
        self.cv_folds_var.set(str(scenario.default_cv_folds))

        for key, var in self.model_vars.items():
            var.set(key in scenario.default_models)

        # Extra outputs defaults per scenario
        self.extra_plots_var.set(scenario.default_extra_plots)
        self.feature_importance_var.set(scenario.default_feature_importance)

        state_cv = tk.NORMAL if scenario.supports_cv_folds else tk.DISABLED
        self.cv_folds_entry.config(state=state_cv)

        # Finetune field: enabled only for Cross CSV finetune scenario
        if scenario.label.startswith("Cross CSV (train->test + finetune)"):
            self.finetune_entry.config(state=tk.NORMAL)
            # default 10% if empty
            if not self.finetune_var.get().strip():
                self.finetune_var.set("10")
        else:
            self.finetune_entry.config(state=tk.DISABLED)
            self.finetune_var.set("")

        state_models = tk.NORMAL if scenario.supports_models else tk.DISABLED
        for cb in self.model_checkbuttons.values():
            cb.config(state=state_models)

        if scenario.requires_dataset:
            # Single-CSV: only train field is used; test is disabled
            if scenario.label.startswith("Single CSV"):
                self.dataset_entry.config(state=tk.NORMAL)
                self.dataset_browse_btn.config(state=tk.NORMAL)
                self.train_submit_btn.config(state=tk.NORMAL)
                self.test_dataset_entry.config(state=tk.DISABLED)
                self.test_dataset_browse_btn.config(state=tk.DISABLED)
                self.test_submit_btn.config(state=tk.DISABLED)
            elif scenario.label.startswith("Cross CSV"):
                # Cross CSV: both train and test fields enabled
                self.dataset_entry.config(state=tk.NORMAL)
                self.dataset_browse_btn.config(state=tk.NORMAL)
                self.train_submit_btn.config(state=tk.NORMAL)
                self.test_dataset_entry.config(state=tk.NORMAL)
                self.test_dataset_browse_btn.config(state=tk.NORMAL)
                self.test_submit_btn.config(state=tk.NORMAL)
        else:
            self.dataset_entry.config(state=tk.DISABLED)
            self.dataset_browse_btn.config(state=tk.DISABLED)
            self.train_submit_btn.config(state=tk.DISABLED)
            self.test_dataset_entry.config(state=tk.DISABLED)
            self.test_dataset_browse_btn.config(state=tk.DISABLED)
            self.test_submit_btn.config(state=tk.DISABLED)
        self.dataset_var.set("")
        self.test_dataset_var.set("")
        self.train_csv_paths = []
        self.test_csv_paths = []

    def _browse_outdir(self) -> None:
        initial_dir = self.project_root
        current = self.outdir_var.get().strip()
        if current:
            candidate = (self.project_root / current).resolve()
            if candidate.exists():
                initial_dir = candidate
        selected = filedialog.askdirectory(initialdir=str(initial_dir), title="Select output directory")
        if selected:
            try:
                rel = Path(selected).resolve().relative_to(self.project_root.resolve())
                self.outdir_var.set(str(rel))
            except ValueError:
                self.outdir_var.set(selected)

    def _browse_dataset(self) -> None:
        data_root = (self.project_root / "data").resolve()
        initial_dir = data_root if data_root.exists() else self.project_root
        selected = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            title="Select train dataset CSV (inside data/)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            try:
                rel = Path(selected).resolve().relative_to(data_root)
                self.dataset_var.set(str(rel))
            except ValueError:
                self.dataset_var.set(selected)

    def _browse_test_dataset(self) -> None:
        data_root = (self.project_root / "data").resolve()
        initial_dir = data_root if data_root.exists() else self.project_root
        selected = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            title="Select test dataset CSV (inside data/)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            try:
                rel = Path(selected).resolve().relative_to(data_root)
                self.test_dataset_var.set(str(rel))
            except ValueError:
                self.test_dataset_var.set(selected)

    def _on_train_submit(self) -> None:
        path = self.dataset_var.get().strip()
        if not path:
            messagebox.showerror("Error", "Train CSV path is empty.")
            return
        if path not in self.train_csv_paths:
            self.train_csv_paths.append(path)
        self.status_var.set(f"Stored train CSVs ({len(self.train_csv_paths)}): {path}")
        self._validate_dataset(path, role="train")

    def _on_test_submit(self) -> None:
        path = self.test_dataset_var.get().strip()
        if not path:
            messagebox.showerror("Error", "Test CSV path is empty.")
            return
        if path not in self.test_csv_paths:
            self.test_csv_paths.append(path)
        self.status_var.set(f"Stored test CSVs ({len(self.test_csv_paths)}): {path}")
        self._validate_dataset(path, role="test")

    def _collect_current_config(self) -> Dict[str, object]:
        # Ensure current entries are included in path lists
        train_paths = list(self.train_csv_paths)
        train_entry = self.dataset_var.get().strip()
        if train_entry and train_entry not in train_paths:
            train_paths.append(train_entry)

        test_paths = list(self.test_csv_paths)
        test_entry = self.test_dataset_var.get().strip()
        if test_entry and test_entry not in test_paths:
            test_paths.append(test_entry)

        selected_models = [key for key, var in self.model_vars.items() if var.get()]

        cfg: Dict[str, object] = {
            "group": self.group_var.get(),
            "scenario_family": self.scenario_family_var.get(),
            "variant": self.variant_var.get(),
            "target_fpr": self.target_fpr_var.get(),
            "cv_folds": self.cv_folds_var.get(),
            "finetune_pct": self.finetune_var.get(),
            "outdir": self.outdir_var.get(),
            "models": selected_models,
            "extra_plots": bool(self.extra_plots_var.get()),
            "feature_importance": bool(self.feature_importance_var.get()),
            "train_csvs": train_paths,
            "test_csvs": test_paths,
        }
        return cfg

    def _apply_config(self, cfg: Dict[str, object]) -> None:
        group = str(cfg.get("group", ""))
        if group and group in self.scenario_groups:
            self.group_var.set(group)
            self._update_scenario_menu()

        family = str(cfg.get("scenario_family", ""))
        if family and family in self.scenario_groups.get(group, []):
            self.scenario_family_var.set(family)
            self._update_variant_menu()

        variant = str(cfg.get("variant", ""))
        if group == "Preconfigured":
            variants = self.scenario_variants.get(family, {})
            if variant and variant in variants:
                self.variant_var.set(variant)
        self._apply_scenario_defaults()

        tf = cfg.get("target_fpr")
        if tf is not None:
            self.target_fpr_var.set(str(tf))
        cf = cfg.get("cv_folds")
        if cf is not None:
            self.cv_folds_var.set(str(cf))
        ft = cfg.get("finetune_pct")
        if ft is not None and str(ft) != "":
            self.finetune_var.set(str(ft))

        outdir = cfg.get("outdir")
        if outdir is not None:
            self.outdir_var.set(str(outdir))

        models = cfg.get("models")
        if isinstance(models, list):
            model_set = {str(m) for m in models}
            for key, var in self.model_vars.items():
                var.set(key in model_set)

        ep = cfg.get("extra_plots")
        if ep is not None:
            self.extra_plots_var.set(bool(ep))
        fi = cfg.get("feature_importance")
        if fi is not None:
            self.feature_importance_var.set(bool(fi))

        train_csvs = cfg.get("train_csvs")
        if isinstance(train_csvs, list):
            self.train_csv_paths = [str(p) for p in train_csvs]
            if self.train_csv_paths:
                self.dataset_var.set(self.train_csv_paths[-1])

        test_csvs = cfg.get("test_csvs")
        if isinstance(test_csvs, list):
            self.test_csv_paths = [str(p) for p in test_csvs]
            if self.test_csv_paths:
                self.test_dataset_var.set(self.test_csv_paths[-1])

        self.status_var.set("Config loaded; adjust if needed and Run.")

    def _resolve_data_csv(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        data_root = (self.project_root / "data").resolve()
        candidate = (data_root / p).resolve()
        if candidate.exists():
            return candidate
        return (self.project_root / p).resolve()

    def _single_feature_auc(self, s: pd.Series, y: pd.Series) -> float:
        try:
            if s.dtype == "O" or str(s.dtype).startswith("category"):
                tmp = pd.DataFrame({"s": s.astype(str), "y": y.values})
                rates = tmp.groupby("s", dropna=False)["y"].mean()
                x = s.astype(str).map(rates).astype(float)
            else:
                x = pd.to_numeric(s, errors="coerce")
            x = x.replace([np.inf, -np.inf], np.nan)
            if x.isna().all():
                return float("nan")
            xm = x.fillna(x.median())
            if xm.nunique() <= 1 or y.nunique() <= 1:
                return float("nan")
            return float(roc_auc_score(y, xm))
        except Exception:
            return float("nan")

    def _validate_dataset(self, path_str: str, role: str) -> None:
        try:
            path = self._resolve_data_csv(path_str)
            if not path.exists():
                messagebox.showwarning("Dataset validation", f"{role.capitalize()} CSV not found:\n{path}")
                return
            df = pd.read_csv(path, encoding="latin1", low_memory=False)
            lbl_col, atac_col = detect_label_columns(df)
            label_info = None
            try:
                y = to_binary_labels(df, lbl_col, atac_col)
                n = int(len(y))
                pos = int(y.sum())
                neg = n - pos
            except Exception:
                y = None
                n = len(df)
                pos = neg = 0

            warnings_list: List[str] = []
            if y is None:
                warnings_list.append("label column not detected; dataset cannot be used for supervised training as-is.")
            else:
                if n < 1000:
                    warnings_list.append("dataset is small (< 1000 rows).")
                if pos == 0 or neg == 0:
                    warnings_list.append("only one class present (all positives or all negatives).")
                frac_pos = pos / max(n, 1)
                if frac_pos < 0.01 or frac_pos > 0.99:
                    warnings_list.append("strong class imbalance (positives <1% or >99%).")

            suspicious_cols: List[tuple[str, float]] = []
            if y is not None and y.nunique() > 1:
                # Derive feature matrix similar to load_csv behavior
                X = df.copy()
                for c in [lbl_col, atac_col]:
                    if c and c in X.columns:
                        X = X.drop(columns=[c])
                X = drop_non_feature_columns(X)
                # To keep it light, optionally subsample
                if len(X) > 50000:
                    X_sample, _, y_sample, _ = train_test_split(
                        X, y, train_size=50000, random_state=42, stratify=y
                    )
                else:
                    X_sample, y_sample = X, y
                for col in X_sample.columns:
                    auc = self._single_feature_auc(X_sample[col], y_sample)
                    if isinstance(auc, float) and not np.isnan(auc) and auc >= 0.995:
                        suspicious_cols.append((str(col), auc))

            # Build summary text
            lines: List[str] = []
            lines.append(f"Dataset summary ({role}, {path}):")
            lines.append(f"  Rows: {n}")
            if y is not None:
                lines.append(f"  Positives: {pos} ({pos / max(n,1):.3f}), negatives: {neg} ({neg / max(n,1):.3f})")
            else:
                lines.append("  Label: not detected.")
            if lbl_col or atac_col:
                lines.append(f"  Detected label columns: label={lbl_col}, attack_cat={atac_col}")
            if suspicious_cols:
                lines.append(f"  Suspicious features (AUC≥0.995): {len(suspicious_cols)}")
                for name, auc in suspicious_cols[:5]:
                    lines.append(f"    - {name}: AUC={auc:.6f}")
            else:
                lines.append("  Suspicious features (AUC≥0.995): none detected (on quick scan).")
            if warnings_list:
                lines.append("  Warnings:")
                for w in warnings_list:
                    lines.append(f"    - {w}")

            text = "\n".join(lines)
            self._append_log(text + "\n\n")
            messagebox.showinfo("Dataset summary", text)
        except Exception as e:
            messagebox.showerror("Dataset validation error", f"Failed to inspect dataset:\n{e}")

    def _browse_results_csv(self) -> None:
        initial_dir = (self.project_root / "out").resolve()
        selected = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            title="Select results CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            try:
                rel = Path(selected).resolve().relative_to(self.project_root.resolve())
                self.results_csv_var.set(str(rel))
            except ValueError:
                self.results_csv_var.set(selected)

    def _load_results_csv(self) -> None:
        path_str = self.results_csv_var.get().strip()
        if not path_str:
            messagebox.showerror("Error", "Results CSV path is empty.")
            return
        path = Path(path_str)
        if not path.is_absolute():
            path = self.project_root / path
        if not path.exists():
            messagebox.showerror("Error", f"Results CSV not found:\n{path}")
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i > 200:
                        break
            if not rows:
                text = "(empty file)"
            else:
                # simple column-width formatting
                cols = max(len(r) for r in rows)
                widths = [0] * cols
                for r in rows:
                    for j, cell in enumerate(r):
                        widths[j] = max(widths[j], len(cell))
                lines = []
                for r in rows:
                    padded = [
                        (r[j] if j < len(r) else "").ljust(widths[j] + 1)
                        for j in range(cols)
                    ]
                    lines.append("".join(padded))
                text = "\n".join(lines)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n{e}")
            return

        if self.results_text is not None:
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, text)

    def _on_run_clicked(self) -> None:
        if self._current_process is not None:
            messagebox.showinfo("Already running", "An experiment is already running.")
            return

        group = self.group_var.get()
        if group == "Preconfigured":
            family = self.scenario_family_var.get()
            variant = self.variant_var.get()
            key = self.scenario_variants.get(family, {}).get(variant)
        else:
            key = self.scenario_family_var.get()
        scenario = self.scenarios.get(key)
        if not scenario:
            messagebox.showerror("Error", "Unknown scenario selection.")
            return

        try:
            target_fpr = float(self.target_fpr_var.get())
        except ValueError:
            messagebox.showerror("Error", "Target FPR must be a number.")
            return

        if scenario.supports_cv_folds:
            try:
                cv_folds = int(self.cv_folds_var.get())
            except ValueError:
                messagebox.showerror("Error", "CV folds must be an integer.")
                return
            if cv_folds < 2:
                messagebox.showerror("Error", "CV folds must be at least 2.")
                return
        else:
            cv_folds = scenario.default_cv_folds

        outdir = self.outdir_var.get().strip()
        if not outdir:
            messagebox.showerror("Error", "Output directory must not be empty.")
            return

        dataset_path: str | None
        if scenario.requires_dataset:
            dataset_path = self.dataset_var.get().strip()
            if not dataset_path:
                messagebox.showerror("Error", "Please select a dataset CSV (inside data/).")
                return
        else:
            dataset_path = None

        selected_models = [key for key, var in self.model_vars.items() if var.get()]
        if scenario.supports_models and not selected_models:
            messagebox.showerror("Error", "Please select at least one model.")
            return
        if not scenario.supports_models:
            selected_models = scenario.default_models

        # Build run-specific output directory label when user kept default
        base_outdir = outdir
        label_for_outdir = scenario.label
        dataset_labels: List[str] = []
        if group == "Custom / dataset-based":
            if scenario.label.startswith("Single CSV"):
                if dataset_path:
                    dataset_labels = [dataset_path]
            elif scenario.label.startswith("Cross CSV"):
                if self.train_csv_paths:
                    dataset_labels.append(self.train_csv_paths[0])
                if self.test_csv_paths:
                    dataset_labels.append(self.test_csv_paths[0])
        if outdir == scenario.default_outdir:
            outdir = self._make_run_outdir(
                base_outdir=base_outdir,
                scenario_label=label_for_outdir,
                models=selected_models,
                dataset_labels=dataset_labels,
            )

        # Special handling for cross-dataset custom scenario
        if scenario.label.startswith("Cross CSV"):
            if not self.train_csv_paths or not self.test_csv_paths:
                messagebox.showerror("Error", "Please submit both train and test CSV paths.")
                return
            cmd = [
                sys.executable,
                "-u",
                str(self.project_root / "experiments" / "cross_pair_generic.py"),
                "--root",
                str(self.project_root),
                "--outdir",
                outdir,
                "--models",
                ",".join(selected_models),
                "--target-fpr",
                str(target_fpr),
                "--gui-progress",
            ]
            for p in self.train_csv_paths:
                cmd.extend(["--train-csv", p])
            for p in self.test_csv_paths:
                cmd.extend(["--test-csv", p])
            # Fine-tune fraction from finetune field for finetune scenario
            if scenario.label.startswith("Cross CSV (train->test + finetune)"):
                try:
                    ft_percent = float(self.finetune_var.get())
                except ValueError:
                    ft_percent = 10.0
                cmd.extend(["--finetune-frac", str(ft_percent / 100.0)])
            if not self.extra_plots_var.get():
                cmd.append("--no-extra-plots")
            if not self.feature_importance_var.get():
                cmd.append("--no-feature-importance")
        else:
            cmd = scenario.build_cmd(outdir, target_fpr, selected_models, cv_folds, dataset_path)

        if not self.extra_plots_var.get():
            cmd.append("--no-extra-plots")
        if not self.feature_importance_var.get():
            cmd.append("--no-feature-importance")

        self.log_text.delete("1.0", tk.END)
        self._append_log(f"Running: {' '.join(cmd)}\n\n")

        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Running...")

        self._start_time = time.time()
        self._update_elapsed()
        self.progress.config(mode="indeterminate")
        self.progress.start(100)

        def runner() -> None:
            try:
                self._current_process = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as e:  # pragma: no cover - GUI error path
                self._schedule_on_main_thread(
                    lambda: self._on_process_finished(
                        exit_code=None,
                        error_message=f"Failed to start process: {e}",
                    )
                )
                return

            assert self._current_process.stdout is not None
            for line in self._current_process.stdout:
                self._schedule_on_main_thread(lambda l=line: self._handle_process_output_line(l))

            exit_code = self._current_process.wait()
            self._schedule_on_main_thread(lambda: self._on_process_finished(exit_code=exit_code, error_message=None))

        self._runner_thread = threading.Thread(target=runner, daemon=True)
        self._runner_thread.start()

    def _on_stop_clicked(self) -> None:
        if self._current_process is None:
            return
        try:
            self._current_process.terminate()
        except Exception:
            pass
        self.status_var.set("Stopping...")

    def _on_process_finished(self, exit_code: int | None, error_message: str | None) -> None:
        if error_message is not None:
            self._append_log(f"\n{error_message}\n")
            messagebox.showerror("Error", error_message)
        elif exit_code is not None:
            self._append_log(f"\nProcess finished with exit code {exit_code}.\n")
            if exit_code == 0:
                messagebox.showinfo("Done", "Experiment finished successfully.")
            else:
                messagebox.showwarning("Finished with errors", f"Experiment finished with exit code {exit_code}.")

        self._current_process = None
        self._start_time = None
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Ready.")
        self.progress.stop()
        self.progress["value"] = 0

    def _copy_log_to_clipboard(self) -> None:
        try:
            text = self.log_text.get("1.0", tk.END)
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_var.set("Log copied to clipboard.")
        except Exception:
            messagebox.showerror("Error", "Failed to copy log to clipboard.")

    def _handle_process_output_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped.startswith("[GUI_PROGRESS]"):
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    done = int(parts[1])
                    total = int(parts[2])
                    fraction = done / max(total, 1)
                except ValueError:
                    return
                self.progress.stop()
                self.progress.config(mode="determinate", maximum=1000)
                self.progress["value"] = int(1000 * fraction)
                self.status_var.set(f"Running... {fraction * 100:.1f}%")
            return
        self._append_log(line)

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _schedule_on_main_thread(self, func) -> None:
        self.after(0, func)

    def _update_elapsed(self) -> None:
        if self._start_time is None:
            return
        elapsed = time.time() - self._start_time
        self.elapsed_var.set(f"Elapsed: {elapsed:.1f} s")
        self.after(500, self._update_elapsed)

    def _copy_selection_to_clipboard(self, event: tk.Event | None = None) -> str:
        try:
            text = self.log_text.get("sel.first", "sel.last")
        except tk.TclError:
            return "break"
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_var.set("Selected text copied to clipboard.")
        except Exception:
            messagebox.showerror("Error", "Failed to copy selection to clipboard.")
        return "break"

    def _on_save_config(self) -> None:
        name = simpledialog.askstring("Save config", "Profile name:")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        cfg = self._collect_current_config()
        profiles: Dict[str, object] = {}
        if self.profiles_path.exists():
            try:
                with self.profiles_path.open("r", encoding="utf-8") as f:
                    profiles = json.load(f)
            except Exception:
                profiles = {}
        profiles[name] = cfg
        try:
            with self.profiles_path.open("w", encoding="utf-8") as f:
                json.dump(profiles, f, indent=2)
            self.status_var.set(f"Saved config profile '{name}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profile:\n{e}")

    def _on_load_config(self) -> None:
        if not self.profiles_path.exists():
            messagebox.showerror("Error", f"No profiles file found at:\n{self.profiles_path}")
            return
        try:
            with self.profiles_path.open("r", encoding="utf-8") as f:
                profiles = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read profiles file:\n{e}")
            return
        if not isinstance(profiles, dict) or not profiles:
            messagebox.showerror("Error", "No saved profiles found.")
            return
        existing_names = sorted(profiles.keys())
        hint = ", ".join(existing_names)
        name = simpledialog.askstring("Load config", f"Profile name:\nAvailable: {hint}")
        if not name:
            return
        name = name.strip()
        if name not in profiles:
            messagebox.showerror("Error", f"Profile '{name}' not found.\nAvailable: {hint}")
            return
        cfg = profiles[name]
        if not isinstance(cfg, dict):
            messagebox.showerror("Error", f"Profile '{name}' has invalid format.")
            return
        try:
            self._apply_config(cfg)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply profile '{name}':\n{e}")

    def _slug(self, text: str, max_len: int = 40) -> str:
        s = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text).strip())
        s = s.strip("_.")
        if not s:
            s = "run"
        if len(s) > max_len:
            s = s[:max_len]
        return s

    def _make_run_outdir(
        self,
        base_outdir: str,
        scenario_label: str,
        models: List[str],
        dataset_labels: List[str],
    ) -> str:
        base_path = Path(base_outdir)
        leaf = base_path.name or "out"

        scen = self._slug(scenario_label, max_len=40)
        model_part = ""
        if models:
            model_part = self._slug("-".join(models), max_len=30)
        data_part = ""
        if dataset_labels:
            parts = []
            for p in dataset_labels:
                stem = Path(p).stem
                parts.append(self._slug(stem, max_len=30))
            data_part = self._slug("__".join(parts), max_len=60)

        stamp = time.strftime("%Y%m%d_%H%M%S")

        # For custom_result we want out/custom_result/<run_name>/...
        if leaf == "custom_result":
            name_parts = [scen]
            if data_part:
                name_parts.append(data_part)
            if model_part:
                name_parts.append(model_part)
            name_parts.append(stamp)
            run_name = "__".join(name_parts)
            return str((base_path / run_name).as_posix())

        # For other outdirs, keep flat naming: out/<leaf__...>
        root = base_path.parent if base_path.parent != Path("") else Path(".")
        name_parts = [leaf, scen]
        if data_part:
            name_parts.append(data_part)
        if model_part:
            name_parts.append(model_part)
        name_parts.append(stamp)
        new_leaf = "__".join(name_parts)

        return str((root / new_leaf).as_posix())


def main() -> None:
    project_root = Path(__file__).resolve().parent
    app = App(project_root)
    app.mainloop()


if __name__ == "__main__":
    main()
