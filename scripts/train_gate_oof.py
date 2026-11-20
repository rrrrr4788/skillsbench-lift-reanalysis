#!/usr/bin/env python3
"""
Module S19: SkillGate Model Training, Out-of-Fold Predictions, ROC-AUC, and Feature Importances
(`scripts/train_gate_oof.py`)

Independently trains the FastRandomForest models from upstream provenance tables:
  - `S02/normalized_trials.csv` (reconstructs deterministic raw-trial stream arrival order)
  - `S02/paired_cells.csv` (1,421 paired cells ground-truth two-arm rewards)
  - `S05/cell_features_1421.csv` (17D static hygiene, lexical alignment, and competence prior features)
  - `S12/method_config.json` (versioned hyperparameters, CV fold counts, and RNG seeds)

Outputs (`artifacts/provenance/S19/`):
  - `oof_predictions_1421.csv`: per-cell fold assignments and OOF predicted gain / negative-transfer risk
  - `feature_importances_17d.csv`: 10-fold averaged normalized split-gain importances across all 17 features
  - `cv_fold_membership_summary.csv`: fold sizes and strict zero-leakage verification across Cell-CV, TaskGroup-CV, and LOCO
  - `cv_order_sensitivity.json`: comparison of canonical stream order vs. alphabetical cell_id order
  - `metrics.json` & `run_manifest.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from provenance_common import ModuleRecorder


FEATURE_NAMES = [
    "wo_rew",
    "is_openhands",
    "n_skills",
    "prose_tokens",
    "code_tokens",
    "ref_tokens",
    "script_tokens",
    "n_ref_files",
    "n_script_files",
    "bundle_to_inst_ratio",
    "code_ratio",
    "ref_ratio",
    "has_synthetic_stub",
    "unmatched_pdf_forms",
    "unmatched_xlsx",
    "lex_recall",
    "lex_prec",
]

FEATURE_LABELS_PRETTY = {
    "wo_rew": "Baseline Competence Prior (wo_rew)",
    "bundle_to_inst_ratio": "Skill-to-Prompt Size Ratio",
    "prose_tokens": "Skill Prose Tokens",
    "lex_prec": "Prompt-Skill Lexical Precision",
    "lex_recall": "Prompt-Skill Lexical Recall",
    "code_ratio": "Executable Code Density",
    "code_tokens": "Skill Code Block Tokens",
    "ref_tokens": "Reference Doc Tokens",
    "ref_ratio": "Reference Doc Ratio",
    "n_ref_files": "Auxiliary Reference Files",
    "script_tokens": "Helper Script Tokens",
    "n_skills": "Attached Skill Count",
    "is_openhands": "Harness Catalog Dilution Flag",
    "n_script_files": "Helper Script Count",
    "has_synthetic_stub": "Synthetic Template Stub Flag",
    "unmatched_pdf_forms": "Unmatched PDF-Forms Module",
    "unmatched_xlsx": "Unmatched XLSX Recalc Module",
}


class FastTreeRegressor:
    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 8, max_features: int | None = None, rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastTreeRegressor":
        self.tree = self._build(X, y, 0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int):
        n, m = X.shape
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or np.var(y) < 1e-8:
            return ("leaf", float(np.mean(y)), 0.0)
        k = self.max_features or max(1, int(np.sqrt(m)))
        feats = self.rng.choice(m, size=min(k, m), replace=False)
        best_gain = -1.0
        best_split = None
        base_var = np.var(y) * n
        for f in feats:
            vals = X[:, f]
            qs = np.unique(np.percentile(vals, [15, 30, 50, 70, 85]))
            for thr in qs:
                left = vals <= thr
                n_l = int(np.sum(left))
                n_r = n - n_l
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue
                gain = base_var - (np.var(y[left]) * n_l + np.var(y[~left]) * n_r)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (int(f), float(thr), left, float(gain))
        if best_split is None:
            return ("leaf", float(np.mean(y)), 0.0)
        f, thr, left, gain = best_split
        return (
            "node",
            f,
            thr,
            gain,
            self._build(X[left], y[left], depth + 1),
            self._build(X[~left], y[~left], depth + 1),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X), dtype=float)
        self._pred_vec(self.tree, X, np.arange(len(X)), out)
        return out

    def _pred_vec(self, node, X: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
        if len(idx) == 0:
            return
        if node[0] == "leaf":
            out[idx] = node[1]
            return
        _, f, thr, _, l_child, r_child = node
        mask = X[idx, f] <= thr
        self._pred_vec(l_child, X, idx[mask], out)
        self._pred_vec(r_child, X, idx[~mask], out)

    def accumulate_importance(self, node, imp: np.ndarray) -> None:
        if node[0] == "leaf":
            return
        _, f, _, gain, l_child, r_child = node
        imp[f] += gain
        self.accumulate_importance(l_child, imp)
        self.accumulate_importance(r_child, imp)


class FastRandomForest:
    def __init__(self, n_estimators: int = 25, max_depth: int = 5, min_samples_leaf: int = 8, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rng = np.random.default_rng(seed)
        self.trees: List[FastTreeRegressor] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastRandomForest":
        self.trees = []
        n, m = X.shape
        imp = np.zeros(m, dtype=float)
        for _ in range(self.n_estimators):
            idx = self.rng.choice(n, size=n, replace=True)
            t = FastTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max(2, int(m * 0.65)),
                rng=self.rng,
            )
            t.fit(X[idx], y[idx])
            t.accumulate_importance(t.tree, imp)
            self.trees.append(t)
        if np.sum(imp) > 0:
            self.feature_importances_ = imp / np.sum(imp)
        else:
            self.feature_importances_ = np.ones(m) / m
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros((self.n_estimators, len(X)), dtype=float)
        for i, t in enumerate(self.trees):
            preds[i] = t.predict(X)
        return np.mean(preds, axis=0)


def compute_roc_auc(y_true_bin: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true_bin == 1]
    neg = scores[y_true_bin == 0]
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def reconstruct_stream_cell_order(trials_csv_path: Path) -> List[str]:
    """
    Reconstructs the deterministic insertion order of paired `(config_id, task)` cells
    from raw trial arrival streams (`selected_trials.csv` v1.1 rows 1..6590 followed by
    `paper_v1_manifest.json` v1.0 rows 1..2830).
    """
    v11_rows: List[Tuple[int, str, str, str, bool]] = []
    v10_rows: List[Tuple[int, str, str, str, bool]] = []
    with open(trials_csv_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["flag_reward_valid"] != "True":
                continue
            seq_num = int(r["source_row_number"])
            cfg = r["config_id"]
            task = r["task"]
            mode = r["normalized_mode"]
            in_repo = (r["flag_task_in_repo"] == "True")
            if r["version"] == "v1.1":
                v11_rows.append((seq_num, cfg, task, mode, in_repo))
            else:
                v10_rows.append((seq_num, cfg, task, mode, in_repo))

    v11_rows.sort(key=lambda x: x[0])
    v10_rows.sort(key=lambda x: x[0])

    seen_keys: "OrderedDict[Tuple[str, str, str], bool]" = OrderedDict()
    for _, cfg, task, mode, in_repo in v11_rows + v10_rows:
        key = (cfg, task, mode)
        if key not in seen_keys:
            seen_keys[key] = in_repo

    ordered_cell_ids: List[str] = []
    for (cfg, task, mode), in_repo in seen_keys.items():
        if not in_repo:
            continue
        if mode == "with" and (cfg, task, "without") in seen_keys:
            ordered_cell_ids.append(f"{cfg}::{task}")
    return ordered_cell_ids


def run_cell_10fold(
    X: np.ndarray,
    y_delta: np.ndarray,
    y_neg: np.ndarray,
    cv_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    n = len(X)
    n_folds = int(cv_cfg["cv_folds_cell"])
    rng_cv = np.random.default_rng(int(cv_cfg["seed_cv_split"]))
    perm = rng_cv.permutation(n)
    folds = np.array_split(perm, n_folds)

    pred_delta_cv = np.zeros(n, dtype=float)
    pred_neg_cv = np.zeros(n, dtype=float)
    pred_delta_static_cv = np.zeros(n, dtype=float)
    pred_delta_prior_only_cv = np.zeros(n, dtype=float)
    pred_neg_prior_only_cv = np.zeros(n, dtype=float)
    feature_imp_accum = np.zeros(X.shape[1], dtype=float)
    cell_fold_id = np.full(n, -1, dtype=int)

    for fold_num, fold_idx in enumerate(folds):
        cell_fold_id[fold_idx] = fold_num
        train_mask = np.ones(n, dtype=bool)
        train_mask[fold_idx] = False

        rf_d = FastRandomForest(
            n_estimators=int(cv_cfg["full_rf_n_estimators"]),
            max_depth=int(cv_cfg["full_rf_max_depth"]),
            min_samples_leaf=int(cv_cfg["full_rf_min_samples_leaf"]),
            seed=int(cv_cfg["seed_rf_delta"]),
        )
        rf_d.fit(X[train_mask], y_delta[train_mask])
        pred_delta_cv[fold_idx] = rf_d.predict(X[fold_idx])

        rf_n = FastRandomForest(
            n_estimators=int(cv_cfg["full_rf_n_estimators"]),
            max_depth=int(cv_cfg["full_rf_max_depth"]),
            min_samples_leaf=int(cv_cfg["full_rf_min_samples_leaf"]),
            seed=int(cv_cfg["seed_rf_neg"]),
        )
        rf_n.fit(X[train_mask], y_neg[train_mask])
        pred_neg_cv[fold_idx] = rf_n.predict(X[fold_idx])
        feature_imp_accum += rf_n.feature_importances_ / float(n_folds)

        rf_s = FastRandomForest(
            n_estimators=int(cv_cfg["ablation_rf_n_estimators"]),
            max_depth=int(cv_cfg["ablation_rf_max_depth"]),
            min_samples_leaf=int(cv_cfg["ablation_rf_min_samples_leaf"]),
            seed=int(cv_cfg["seed_rf_static"]),
        )
        rf_s.fit(X[train_mask, 2:], y_delta[train_mask])
        pred_delta_static_cv[fold_idx] = rf_s.predict(X[fold_idx, 2:])

        rf_p = FastRandomForest(
            n_estimators=int(cv_cfg["ablation_rf_n_estimators"]),
            max_depth=int(cv_cfg["ablation_rf_max_depth"]),
            min_samples_leaf=int(cv_cfg["ablation_rf_min_samples_leaf"]),
            seed=int(cv_cfg["seed_rf_prior_delta"]),
        )
        rf_p.fit(X[train_mask, :2], y_delta[train_mask])
        pred_delta_prior_only_cv[fold_idx] = rf_p.predict(X[fold_idx, :2])

        rf_pn = FastRandomForest(
            n_estimators=int(cv_cfg["ablation_rf_n_estimators"]),
            max_depth=int(cv_cfg["ablation_rf_max_depth"]),
            min_samples_leaf=int(cv_cfg["ablation_rf_min_samples_leaf"]),
            seed=int(cv_cfg["seed_rf_prior_neg"]),
        )
        rf_pn.fit(X[train_mask, :2], y_neg[train_mask])
        pred_neg_prior_only_cv[fold_idx] = rf_pn.predict(X[fold_idx, :2])

    return {
        "cell_fold_id": cell_fold_id,
        "folds": folds,
        "pred_delta_cv": pred_delta_cv,
        "pred_neg_cv": pred_neg_cv,
        "pred_delta_static_cv": pred_delta_static_cv,
        "pred_delta_prior_only_cv": pred_delta_prior_only_cv,
        "pred_neg_prior_only_cv": pred_neg_prior_only_cv,
        "feature_imp_accum": feature_imp_accum,
        "auc_full": compute_roc_auc(y_neg, pred_neg_cv),
        "auc_prior_only": compute_roc_auc(y_neg, pred_neg_prior_only_cv),
    }


def main() -> None:
    import time
    repo_root = SCRIPT_DIR.parent
    prov_tables = repo_root / "data" / "provenance_tables"
    run_stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    parser = argparse.ArgumentParser(description="S19: Train SkillGate OOF models and compute ROC-AUCs & importances.")
    parser.add_argument(
        "--s02-trials",
        type=Path,
        default=prov_tables / "S02" / "normalized_trials.csv",
    )
    parser.add_argument(
        "--s02-cells",
        type=Path,
        default=prov_tables / "S02" / "paired_cells.csv",
    )
    parser.add_argument(
        "--s05-features",
        type=Path,
        default=prov_tables / "S05" / "cell_features_1421.csv",
    )
    parser.add_argument(
        "--s12-config",
        type=Path,
        default=prov_tables / "S12" / "method_config.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "artifacts" / "runs" / f"skillgate_oof_{run_stamp}",
    )
    args = parser.parse_args()

    method_cfg = json.loads(args.s12_config.read_text(encoding="utf-8"))
    cv_cfg = method_cfg["skillgate_models_and_cv"]
    neg_tol = -1e-6

    s19_config = {
        "feature_names": FEATURE_NAMES,
        "negative_transfer_delta_threshold": neg_tol,
        "skillgate_models_and_cv": cv_cfg,
        "tree_split_quantiles_pct": [15, 30, 50, 70, 85],
        "rf_max_features_fraction": 0.65,
        "row_ordering_convention": "raw_trial_stream_v1.1_then_v1.0",
    }

    rec = ModuleRecorder(
        module_id="S19",
        script_path=Path(__file__),
        out_dir=args.out_dir,
        config=s19_config,
        cli_args={k: str(v) for k, v in vars(args).items()},
    )
    rec.register_input_file("S02.normalized_trials.csv", args.s02_trials)
    rec.register_input_file("S02.paired_cells.csv", args.s02_cells)
    rec.register_input_file("S05.cell_features_1421.csv", args.s05_features)
    rec.register_input_file("S12.method_config.json", args.s12_config)

    # Load S02 paired cells and S05 cell features
    s02_by_id: Dict[str, Dict[str, str]] = {}
    with open(args.s02_cells, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            s02_by_id[r["cell_id"]] = r

    s05_rows_alpha: List[Dict[str, str]] = []
    s05_by_id: Dict[str, Dict[str, str]] = {}
    with open(args.s05_features, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            s05_rows_alpha.append(r)
            s05_by_id[r["cell_id"]] = r

    stream_cell_ids = reconstruct_stream_cell_order(args.s02_trials)

    rec.check_conservation(
        "cell_bijection_s02_s05_stream",
        "Stream-reconstructed cell_ids form an exact bijection with S02 and S05 (1,421 cells)",
        len(stream_cell_ids) == len(s02_by_id) == len(s05_by_id) and set(stream_cell_ids) == set(s02_by_id) == set(s05_by_id),
        True,
    )

    max_rew_diff = max(
        max(
            abs(float(s05_by_id[cid]["wo_rew"]) - float(s02_by_id[cid]["wo_mean_unrounded"])),
            abs(float(s05_by_id[cid]["w_rew"]) - float(s02_by_id[cid]["w_mean_unrounded"])),
            abs(float(s05_by_id[cid]["delta"]) - float(s02_by_id[cid]["delta_unrounded"])),
        )
        for cid in stream_cell_ids
    )
    rec.check_conservation(
        "s05_s02_reward_alignment",
        "Maximum abs difference in wo_rew/w_rew/delta between S05 and S02 <= 1e-12",
        max_rew_diff,
        0.0,
        tol=1e-12,
    )

    # Build canonical feature matrix X and targets in stream order
    ordered_rows = [s05_by_id[cid] for cid in stream_cell_ids]
    X = np.array([[float(r[col]) for col in FEATURE_NAMES] for r in ordered_rows], dtype=float)
    y_delta = np.array([float(r["delta"]) for r in ordered_rows], dtype=float)
    y_neg = np.array([1.0 if float(r["delta"]) < neg_tol else 0.0 for r in ordered_rows], dtype=float)
    task_arr = [r["task"] for r in ordered_rows]
    cfg_arr = [r["cfg"] for r in ordered_rows]

    n_cells = len(X)
    rec.check_conservation("feature_matrix_dim", "Feature matrix has 17 columns", int(X.shape[1]), 17)

    # 1. 10-Fold Cell-Level CV + Feature Ablations
    cell_cv_res = run_cell_10fold(X, y_delta, y_neg, cv_cfg)
    cell_fold_id = cell_cv_res["cell_fold_id"]
    rec.check_conservation(
        "all_cells_assigned_cell_cv_fold",
        "Every cell receives exactly one OOF fold assignment in 0..9",
        bool(np.all((cell_fold_id >= 0) & (cell_fold_id < int(cv_cfg["cv_folds_cell"])))),
        True,
    )

    # 2. Task-Grouped 10-Fold CV (zero task leakage across train/test)
    unique_tasks_sorted = sorted(set(task_arr))
    n_tg_folds = int(cv_cfg["cv_folds_task_group"])
    rng_tg = np.random.default_rng(int(cv_cfg["seed_cv_split"]))
    task_perm = rng_tg.permutation(len(unique_tasks_sorted))
    task_folds = np.array_split(task_perm, n_tg_folds)

    pred_delta_taskgroup = np.zeros(n_cells, dtype=float)
    pred_neg_taskgroup = np.zeros(n_cells, dtype=float)
    taskgroup_fold_id = np.full(n_cells, -1, dtype=int)
    task_leakage_violations = 0

    for tg_fold_num, t_fold in enumerate(task_folds):
        heldout_tasks = {unique_tasks_sorted[i] for i in t_fold}
        test_mask = np.array([t in heldout_tasks for t in task_arr], dtype=bool)
        train_mask = ~test_mask
        train_tasks = {task_arr[i] for i in np.where(train_mask)[0]}
        if heldout_tasks & train_tasks:
            task_leakage_violations += 1
        taskgroup_fold_id[test_mask] = tg_fold_num

        rf_d = FastRandomForest(
            n_estimators=int(cv_cfg["full_rf_n_estimators"]),
            max_depth=int(cv_cfg["full_rf_max_depth"]),
            min_samples_leaf=int(cv_cfg["full_rf_min_samples_leaf"]),
            seed=int(cv_cfg["seed_rf_delta"]),
        )
        rf_d.fit(X[train_mask], y_delta[train_mask])
        pred_delta_taskgroup[test_mask] = rf_d.predict(X[test_mask])

        rf_n = FastRandomForest(
            n_estimators=int(cv_cfg["full_rf_n_estimators"]),
            max_depth=int(cv_cfg["full_rf_max_depth"]),
            min_samples_leaf=int(cv_cfg["full_rf_min_samples_leaf"]),
            seed=int(cv_cfg["seed_rf_neg"]),
        )
        rf_n.fit(X[train_mask], y_neg[train_mask])
        pred_neg_taskgroup[test_mask] = rf_n.predict(X[test_mask])

    rec.check_conservation(
        "zero_task_leakage_in_taskgroup_cv",
        "Zero overlap between train tasks and held-out tasks in Task-Grouped 10-Fold CV",
        task_leakage_violations,
        0,
    )
    auc_taskgroup = compute_roc_auc(y_neg, pred_neg_taskgroup)

    # 3. Leave-One-Config-Out (LOCO) across all 28 model-harness configurations
    unique_cfgs = sorted(set(cfg_arr))
    rec.check_conservation(
        "loco_configs_count_check",
        "Number of distinct model-harness configurations equals S12 loco_configs_count (28)",
        len(unique_cfgs),
        int(cv_cfg["loco_configs_count"]),
    )

    pred_delta_loco = np.zeros(n_cells, dtype=float)
    pred_neg_loco = np.zeros(n_cells, dtype=float)
    loco_fold_id = np.full(n_cells, -1, dtype=int)
    cfg_leakage_violations = 0

    for cfg_idx, ucfg in enumerate(unique_cfgs):
        test_mask = np.array([c == ucfg for c in cfg_arr], dtype=bool)
        train_mask = ~test_mask
        train_cfgs = {cfg_arr[i] for i in np.where(train_mask)[0]}
        if ucfg in train_cfgs:
            cfg_leakage_violations += 1
        loco_fold_id[test_mask] = cfg_idx

        rf_d = FastRandomForest(
            n_estimators=int(cv_cfg["loco_rf_n_estimators"]),
            max_depth=int(cv_cfg["loco_rf_max_depth"]),
            min_samples_leaf=int(cv_cfg["loco_rf_min_samples_leaf"]),
            seed=int(cv_cfg["seed_rf_delta"]),
        )
        rf_d.fit(X[train_mask], y_delta[train_mask])
        pred_delta_loco[test_mask] = rf_d.predict(X[test_mask])

        rf_n = FastRandomForest(
            n_estimators=int(cv_cfg["loco_rf_n_estimators"]),
            max_depth=int(cv_cfg["loco_rf_max_depth"]),
            min_samples_leaf=int(cv_cfg["loco_rf_min_samples_leaf"]),
            seed=int(cv_cfg["seed_rf_neg"]),
        )
        rf_n.fit(X[train_mask], y_neg[train_mask])
        pred_neg_loco[test_mask] = rf_n.predict(X[test_mask])

    rec.check_conservation(
        "zero_config_leakage_in_loco",
        "Zero overlap between train configs and held-out config in 28-fold LOCO",
        cfg_leakage_violations,
        0,
    )
    auc_loco = compute_roc_auc(y_neg, pred_neg_loco)

    # 4. Sensitivity check: alphabetical cell_id row order
    X_alpha = np.array([[float(r[col]) for col in FEATURE_NAMES] for r in s05_rows_alpha], dtype=float)
    y_delta_alpha = np.array([float(r["delta"]) for r in s05_rows_alpha], dtype=float)
    y_neg_alpha = np.array([1.0 if float(r["delta"]) < neg_tol else 0.0 for r in s05_rows_alpha], dtype=float)
    alpha_cv_res = run_cell_10fold(X_alpha, y_delta_alpha, y_neg_alpha, cv_cfg)

    # Check feature importance normalization conservation
    imp_vec = cell_cv_res["feature_imp_accum"]
    rec.check_conservation(
        "feature_importance_sum_to_one",
        "10-fold averaged normalized split-gain feature importances sum to 1.0",
        float(np.sum(imp_vec)),
        1.0,
        tol=1e-12,
    )

    # Write member-level CSVs
    oof_rows: List[Dict[str, Any]] = []
    for idx, r in enumerate(ordered_rows):
        oof_rows.append(
            {
                "pipeline_row_idx": idx,
                "cell_id": r["cell_id"],
                "cfg": r["cfg"],
                "version": r["version"],
                "harness": r["harness"],
                "model": r["model"],
                "task": r["task"],
                "wo_rew": repr(float(r["wo_rew"])),
                "w_rew": repr(float(r["w_rew"])),
                "delta": repr(float(r["delta"])),
                "y_neg": int(y_neg[idx]),
                "skill_total_tokens": r["skill_total_tokens"],
                "cell_cv_fold": int(cell_fold_id[idx]),
                "taskgroup_cv_fold": int(taskgroup_fold_id[idx]),
                "loco_fold": int(loco_fold_id[idx]),
                "pred_delta_cv": repr(float(cell_cv_res["pred_delta_cv"][idx])),
                "pred_neg_cv": repr(float(cell_cv_res["pred_neg_cv"][idx])),
                "pred_delta_static_cv": repr(float(cell_cv_res["pred_delta_static_cv"][idx])),
                "pred_delta_prior_only_cv": repr(float(cell_cv_res["pred_delta_prior_only_cv"][idx])),
                "pred_neg_prior_only_cv": repr(float(cell_cv_res["pred_neg_prior_only_cv"][idx])),
                "pred_delta_taskgroup": repr(float(pred_delta_taskgroup[idx])),
                "pred_neg_taskgroup": repr(float(pred_neg_taskgroup[idx])),
                "pred_delta_loco": repr(float(pred_delta_loco[idx])),
                "pred_neg_loco": repr(float(pred_neg_loco[idx])),
            }
        )
    rec.write_csv("oof_predictions_1421.csv", oof_rows)

    imp_pairs = sorted(enumerate(FEATURE_NAMES), key=lambda kv: float(imp_vec[kv[0]]), reverse=True)
    imp_rows: List[Dict[str, Any]] = []
    for rank, (f_idx, fname) in enumerate(imp_pairs, start=1):
        frac = float(imp_vec[f_idx])
        pct = frac * 100.0
        grp = "prior_competence" if fname == "wo_rew" else ("prior_harness" if fname == "is_openhands" else "static_15d")
        imp_rows.append(
            {
                "rank": rank,
                "feature_index": f_idx,
                "feature_name": fname,
                "pretty_label": FEATURE_LABELS_PRETTY[fname],
                "feature_group": grp,
                "importance_fraction": f"{frac:.10f}",
                "importance_pct_unrounded": f"{pct:.8f}",
                "importance_pct_2dec": f"{pct:.2f}",
            }
        )
    rec.write_csv("feature_importances_17d.csv", imp_rows)

    fold_summary_rows: List[Dict[str, Any]] = []
    for k in range(int(cv_cfg["cv_folds_cell"])):
        m = (cell_fold_id == k)
        fold_summary_rows.append(
            {
                "scheme": "cell_10fold",
                "fold_id": k,
                "n_test_cells": int(np.sum(m)),
                "n_test_neg_cells": int(np.sum(y_neg[m] == 1)),
                "n_distinct_tasks": len({task_arr[i] for i in np.where(m)[0]}),
                "n_distinct_configs": len({cfg_arr[i] for i in np.where(m)[0]}),
                "group_disjoint_verified": False,
            }
        )
    for k in range(n_tg_folds):
        m = (taskgroup_fold_id == k)
        fold_summary_rows.append(
            {
                "scheme": "taskgroup_10fold",
                "fold_id": k,
                "n_test_cells": int(np.sum(m)),
                "n_test_neg_cells": int(np.sum(y_neg[m] == 1)),
                "n_distinct_tasks": len({task_arr[i] for i in np.where(m)[0]}),
                "n_distinct_configs": len({cfg_arr[i] for i in np.where(m)[0]}),
                "group_disjoint_verified": True,
            }
        )
    for k, ucfg in enumerate(unique_cfgs):
        m = (loco_fold_id == k)
        fold_summary_rows.append(
            {
                "scheme": f"loco_28cfg ({ucfg})",
                "fold_id": k,
                "n_test_cells": int(np.sum(m)),
                "n_test_neg_cells": int(np.sum(y_neg[m] == 1)),
                "n_distinct_tasks": len({task_arr[i] for i in np.where(m)[0]}),
                "n_distinct_configs": 1,
                "group_disjoint_verified": True,
            }
        )
    rec.write_csv("cv_fold_membership_summary.csv", fold_summary_rows)

    rec.write_json(
        "cv_order_sensitivity.json",
        {
            "canonical_stream_order": {
                "auc_full_10fold": cell_cv_res["auc_full"],
                "auc_prior_only_10fold": cell_cv_res["auc_prior_only"],
                "auc_taskgroup_10fold": auc_taskgroup,
                "auc_loco_28cfg": auc_loco,
            },
            "alphabetical_cell_id_order": {
                "auc_full_10fold": alpha_cv_res["auc_full"],
                "auc_prior_only_10fold": alpha_cv_res["auc_prior_only"],
            },
            "note": "TaskGroup-10Fold and LOCO-28Cfg partition by sorted unique tasks/configs; Cell-10Fold permutes 1,421 rows with seed=42.",
        },
    )

    # Register metrics
    auc_full = cell_cv_res["auc_full"]
    auc_prior = cell_cv_res["auc_prior_only"]
    rec.add_metric(
        "roc_auc_full_10fold",
        auc_full,
        "roc_auc",
        "1,421 paired cells (10-fold cell CV, FastRandomForest 17D)",
        1421,
        "Exact Mann-Whitney ROC-AUC of OOF pred_neg_cv vs y_neg (delta < -1e-6)",
        "oof_predictions_1421.csv",
        ".4f",
    )
    rec.add_metric(
        "roc_auc_prior_only_10fold",
        auc_prior,
        "roc_auc",
        "1,421 paired cells (10-fold cell CV, Prior-Only 2D: wo_rew + is_openhands)",
        1421,
        "Exact Mann-Whitney ROC-AUC of OOF pred_neg_prior_only_cv vs y_neg",
        "oof_predictions_1421.csv",
        ".4f",
    )
    rec.add_metric(
        "roc_auc_taskgroup_10fold",
        auc_taskgroup,
        "roc_auc",
        "1,421 paired cells across 83 evaluated tasks (Task-Grouped 10-Fold CV, zero task leakage)",
        1421,
        "Exact Mann-Whitney ROC-AUC of OOF pred_neg_taskgroup vs y_neg",
        "oof_predictions_1421.csv",
        ".4f",
    )
    rec.add_metric(
        "roc_auc_loco_28cfg",
        auc_loco,
        "roc_auc",
        "1,421 paired cells across 28 configs (Leave-One-Config-Out CV)",
        1421,
        "Exact Mann-Whitney ROC-AUC of OOF pred_neg_loco vs y_neg",
        "oof_predictions_1421.csv",
        ".4f",
    )
    rec.add_metric(
        "n_static_features",
        15,
        "features",
        "Static skill structure (10) + hygiene flags (3) + lexical alignment (2)",
        17,
        "Count of static task/skill features in 17D SkillGate vector (X[:, 2:])",
        "feature_importances_17d.csv",
        "d",
    )
    rec.add_metric(
        "n_evaluated_tasks_in_cv",
        len(unique_tasks_sorted),
        "tasks",
        "Repository tasks with >=1 paired cell in S02",
        87,
        "len(sorted(set(task_arr)))",
        "cv_fold_membership_summary.csv",
        "d",
    )

    for idx, fname in enumerate(FEATURE_NAMES):
        pct_val = float(imp_vec[idx] * 100.0)
        rec.add_metric(
            f"imp_pct.{fname}",
            pct_val,
            "pct",
            f"10-fold averaged normalized split-gain importance of {fname} in neg-risk RF",
            "100% total split-gain across 17 features",
            f"mean_folds(rf_n.feature_importances_[{idx}]) * 100",
            "feature_importances_17d.csv",
            ".2f",
        )

    static_15d_sum_pct = float(np.sum(imp_vec[2:]) * 100.0)
    non_wo_16d_sum_pct = float(np.sum(imp_vec[1:]) * 100.0)
    rec.add_metric(
        "imp_pct_static_15d_sum",
        static_15d_sum_pct,
        "pct",
        "Sum of normalized split-gain importances across all 15 static features (cols 2..16)",
        "100% total split-gain across 17 features",
        "sum(imp_vec[2:]) * 100",
        "feature_importances_17d.csv",
        ".2f",
    )
    rec.add_metric(
        "imp_pct_non_wo_rew_16d_sum",
        non_wo_16d_sum_pct,
        "pct",
        "Sum of normalized split-gain importances across all 16 non-wo_rew features (cols 1..16)",
        "100% total split-gain across 17 features",
        "sum(imp_vec[1:]) * 100 (100% - wo_rew%)",
        "feature_importances_17d.csv",
        ".2f",
    )

    metrics_path, manifest_path = rec.finalize()
    print(f"[S19] PASS | AUC_10Fold={auc_full:.4f}, AUC_PriorOnly={auc_prior:.4f}, AUC_TaskGroup={auc_taskgroup:.4f}, AUC_LOCO={auc_loco:.4f}")
    print(f"[S19] Importances: wo_rew={imp_vec[0]*100:.2f}%, is_openhands={imp_vec[1]*100:.2f}%, static_15d={static_15d_sum_pct:.2f}%, non_wo_16d={non_wo_16d_sum_pct:.2f}%")
    print(f"[S19] Wrote {metrics_path} and {manifest_path}")


if __name__ == "__main__":
    main()
