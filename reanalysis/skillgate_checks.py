#!/usr/bin/env python3
"""Reanalysis of the SkillsBench paired-trial snapshot (version 2).

Uses only the repository's own tables (data/provenance_tables) and its own
model code (scripts/train_gate_oof.py: FastRandomForest, run_cell_10fold).

Usage:
  python skillgate_checks.py REPO_DIR OUT_DIR PART [N]

PART is one of:
  base  - replicate the reported numbers; descriptive tables; stratified comparisons with task-bootstrap intervals
  exact - exact expected number of cells with a negative / positive difference under within-cell label permutation
  tasks - task-level permutation tests with N permutations (two-sided and one-sided, add-one Monte Carlo p-values)
  seeds - AUC and replay gain across N different CV / forest seeds
  null  - N within-cell label permutations; every permutation recomputes rewards, labels and refits the forests
  split - N random splits for the forest policies (2,000 for model-free ones): decisions use the decision trials only,
          scores use the held-out trial of each condition

Each part writes OUT_DIR/<part>.json; tasks and split also write CSV record files.
"""
import csv
import gzip
import hashlib
import itertools
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
PART = sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 0

sys.path.insert(0, str(REPO / "scripts"))
import train_gate_oof as tg  # noqa: E402  (the paper's own model code)

PT = REPO / "data" / "provenance_tables"
NEG_TOL = -1e-6
POS_TOL = 1e-6
SEEDS = {"null": 20261112, "split": 20261113, "tasks": 20261114, "bootstrap": 42}
INPUT_FILES = {
    "normalized_trials": PT / "S02" / "normalized_trials.csv",
    "paired_cells": PT / "S02" / "paired_cells.csv",
    "cell_features": PT / "S05" / "cell_features_1421.csv",
    "task_partition": PT / "S18" / "task_partition_87.csv",
    "method_config": PT / "S12" / "method_config.json",
    "model_code": REPO / "scripts" / "train_gate_oof.py",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------- load data
trials = read_csv(INPUT_FILES["normalized_trials"])
rew_by_id = {r["raw_row_id"]: float(r["reward_float"]) for r in trials if r["flag_reward_valid"] == "True"}
cells = {c["cell_id"]: c for c in read_csv(INPUT_FILES["paired_cells"])}
feats = {r["cell_id"]: r for r in read_csv(INPUT_FILES["cell_features"])}
strata = {r["task"]: r["sub_stratum"] for r in read_csv(INPUT_FILES["task_partition"])}
method_cfg = json.loads(INPUT_FILES["method_config"].read_text(encoding="utf-8"))
CV = method_cfg["skillgate_models_and_cv"]
POL = method_cfg["gating_policy_thresholds"]
P0_CUT = float(POL["level_cutoff_p0_threshold"])

order = tg.reconstruct_stream_cell_order(INPUT_FILES["normalized_trials"])
assert set(order) == set(cells) == set(feats), "cell id sets differ"
n_cells = len(order)

wo_ids_all = [cells[cid]["wo_trial_ids"].split(";") for cid in order]
w_ids_all = [cells[cid]["w_trial_ids"].split(";") for cid in order]
wo_tr = [np.array([rew_by_id[t] for t in ids]) for ids in wo_ids_all]
w_tr = [np.array([rew_by_id[t] for t in ids]) for ids in w_ids_all]
WO = np.array([a.mean() for a in wo_tr])
W = np.array([a.mean() for a in w_tr])
XBASE = np.array([[float(feats[cid][c]) for c in tg.FEATURE_NAMES] for cid in order], dtype=float)
TASKS = [feats[cid]["task"] for cid in order]
CFGS = [feats[cid]["cfg"] for cid in order]
VERS = [feats[cid]["version"] for cid in order]
assert np.max(np.abs(WO - XBASE[:, 0])) < 1e-9, "wo_rew recomputed from trials differs from feature table"
assert np.max(np.abs((W - WO) - np.array([float(feats[c]["delta"]) for c in order]))) < 1e-9
FLAGGED = {"oracle_conflict_3", "senior_stub_4", "license_dir_2"}


def labels(wo, w):
    d = w - wo
    return d, (d < NEG_TOL).astype(float)


def auc(y, s):
    y = np.asarray(y)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return tg.compute_roc_auc(y, np.asarray(s, dtype=float))


def rf_kwargs(cv, kind):
    return dict(n_estimators=int(cv[kind + "_rf_n_estimators"]), max_depth=int(cv[kind + "_rf_max_depth"]),
                min_samples_leaf=int(cv[kind + "_rf_min_samples_leaf"]))


def group_oof(X, y_delta, y_neg, task_arr, cv, cols, kind, seed_d, seed_n):
    """Task-grouped 10-fold out-of-fold predictions, same procedure as the paper's script."""
    uniq = sorted(set(task_arr))
    rng = np.random.default_rng(int(cv["seed_cv_split"]))
    folds = np.array_split(rng.permutation(len(uniq)), int(cv["cv_folds_task_group"]))
    Xs = X[:, cols]
    pd_ = np.zeros(len(X))
    pn_ = np.zeros(len(X))
    kw = rf_kwargs(cv, kind)
    for fold in folds:
        held = {uniq[i] for i in fold}
        test = np.array([t in held for t in task_arr], dtype=bool)
        train = ~test
        pd_[test] = tg.FastRandomForest(seed=int(seed_d), **kw).fit(Xs[train], y_delta[train]).predict(Xs[test])
        pn_[test] = tg.FastRandomForest(seed=int(seed_n), **kw).fit(Xs[train], y_neg[train]).predict(Xs[test])
    return pd_, pn_


def cell_oof(X, y_delta, y_neg, cv, cols, kind, seed_d, seed_n):
    """Cell-level 10-fold out-of-fold predictions with the same split rule as tg.run_cell_10fold.

    Every cell is predicted by forests that were fit on the other nine folds only.
    """
    n = len(X)
    rng = np.random.default_rng(int(cv["seed_cv_split"]))
    folds = np.array_split(rng.permutation(n), int(cv["cv_folds_cell"]))
    assert sorted(np.concatenate(folds).tolist()) == list(range(n)), "folds must partition the cells"
    Xs = X[:, cols]
    pd_ = np.zeros(n)
    pn_ = np.zeros(n)
    kw = rf_kwargs(cv, kind)
    for fold in folds:
        train = np.ones(n, dtype=bool)
        train[fold] = False
        pd_[fold] = tg.FastRandomForest(seed=int(seed_d), **kw).fit(Xs[train], y_delta[train]).predict(Xs[fold])
        pn_[fold] = tg.FastRandomForest(seed=int(seed_n), **kw).fit(Xs[train], y_neg[train]).predict(Xs[fold])
    return pd_, pn_


def fit_all(X, y_delta, y_neg, task_arr, cv):
    """Everything the paper fits (via its own run_cell_10fold) plus baseline-only forests under task-grouped CV."""
    res = tg.run_cell_10fold(X, y_delta, y_neg, cv)
    pred = {
        "pd": res["pred_delta_cv"], "pn": res["pred_neg_cv"],
        "pd_static": res["pred_delta_static_cv"],
        "pd_prior": res["pred_delta_prior_only_cv"], "pn_prior": res["pred_neg_prior_only_cv"],
    }
    allc = list(range(X.shape[1]))
    pred["pd_tg"], pred["pn_tg"] = group_oof(X, y_delta, y_neg, task_arr, cv, allc, "full",
                                             cv["seed_rf_delta"], cv["seed_rf_neg"])
    pred["pd_prior_tg"], pred["pn_prior_tg"] = group_oof(X, y_delta, y_neg, task_arr, cv, [0, 1], "ablation",
                                                        cv["seed_rf_prior_delta"], cv["seed_rf_prior_neg"])
    return pred


def fit_fast(X, y_delta, y_neg, cv, task_arr=None):
    """Full and baseline-only forests under cell CV (and task-grouped CV if task_arr is given)."""
    allc = list(range(X.shape[1]))
    pred = {}
    pred["pd"], pred["pn"] = cell_oof(X, y_delta, y_neg, cv, allc, "full", cv["seed_rf_delta"], cv["seed_rf_neg"])
    pred["pd_prior"], pred["pn_prior"] = cell_oof(X, y_delta, y_neg, cv, [0, 1], "ablation",
                                                  cv["seed_rf_prior_delta"], cv["seed_rf_prior_neg"])
    if task_arr is not None:
        pred["pd_tg"], pred["pn_tg"] = group_oof(X, y_delta, y_neg, task_arr, cv, allc, "full",
                                                 cv["seed_rf_delta"], cv["seed_rf_neg"])
        pred["pd_prior_tg"], pred["pn_prior_tg"] = group_oof(X, y_delta, y_neg, task_arr, cv, [0, 1], "ablation",
                                                            cv["seed_rf_prior_delta"], cv["seed_rf_prior_neg"])
    return pred


def check_fast_matches_paper():
    yd, yn = labels(WO, W)
    pred = fit_fast(XBASE, yd, yn, CV)
    a_full, a_prior = auc(yn, pred["pn"]), auc(yn, pred["pn_prior"])
    assert abs(a_full - 0.8782279713405298) < 1e-9 and abs(a_prior - 0.8373513620810253) < 1e-9, (a_full, a_prior)
    return {"auc_full": a_full, "auc_prior": a_prior}


def decide(pred, wo, y_delta):
    """Load decisions. Thresholds come unchanged from the repository's method_config.json."""
    tg_ra, tr_ra = float(POL["risk_averse_tau_gain"]), float(POL["risk_averse_tau_risk"])
    out = {
        "always_without": np.zeros(len(wo), dtype=bool),
        "level_cutoff_080": wo < P0_CUT,
        "oracle_best_of_two": y_delta > 0.0,
    }
    if "pd" in pred:
        out["skillgate_maxgain"] = pred["pd"] > float(POL["max_gain_tau_gain"])
        out["skillgate_riskaverse"] = (pred["pd"] > tg_ra) & (pred["pn"] < tr_ra)
    if "pd_prior" in pred:
        out["prior_only_rf"] = (pred["pd_prior"] > tg_ra) & (pred["pn_prior"] < tr_ra)
    if "pd_static" in pred:
        out["zero_prior_static_rf"] = pred["pd_static"] > float(POL["zero_prior_static_tau_gain"])
    if "pd_tg" in pred:
        out["skillgate_taskgroup_cv"] = (pred["pd_tg"] > tg_ra) & (pred["pn_tg"] < tr_ra)
    if "pd_prior_tg" in pred:
        out["prior_only_rf_taskgroup"] = (pred["pd_prior_tg"] > tg_ra) & (pred["pn_prior_tg"] < tr_ra)
    return out


def gain_pp(load, wo_eval, w_eval):
    return float(np.mean(np.where(load, w_eval, wo_eval) - w_eval) * 100.0)


def auc_block(pred, y_neg, wo):
    pos = wo > 0
    out = {"trivial_wo_gt0": auc(y_neg, pos.astype(float)), "trivial_wo_value": auc(y_neg, wo)}
    names = {"pn": "full_cellcv", "pn_prior": "prior_cellcv", "pn_tg": "full_taskgroup", "pn_prior_tg": "prior_taskgroup"}
    for key, name in names.items():
        if key in pred:
            out[name] = auc(y_neg, pred[key])
            if pos.sum() > 10:
                out["wo>0_only_" + name] = auc(y_neg[pos], pred[key][pos])
    if pos.sum() > 10:
        out["wo>0_only_trivial_wo_value"] = auc(y_neg[pos], wo[pos])
    return out


def summarize(vals):
    a = np.array(vals, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return None
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "sd": sd, "mc_se_of_mean": sd / float(np.sqrt(len(a))),
            "p2.5": float(np.percentile(a, 2.5)), "p97.5": float(np.percentile(a, 97.5)),
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def task_index(task_arr):
    uniq = sorted(set(task_arr))
    pos = {t: i for i, t in enumerate(uniq)}
    return uniq, np.array([pos[t] for t in task_arr])


def ratio_ci(num_cell, den_cell, task_arr, scale=100.0, n_boot=10000, seed=None):
    """Estimate sum(num)/sum(den) and a 95% percentile interval from resampling whole tasks.

    A resampled task brings all of its cells. Nothing is refit inside the bootstrap, so the interval describes
    sampling of tasks for the decisions that were actually made; it does not include refitting variability.
    """
    uniq, ti = task_index(task_arr)
    num_t = np.bincount(ti, weights=np.asarray(num_cell, dtype=float), minlength=len(uniq))
    den_t = np.bincount(ti, weights=np.asarray(den_cell, dtype=float), minlength=len(uniq))
    rng = np.random.default_rng(SEEDS["bootstrap"] if seed is None else seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    num_b = num_t[idx].sum(axis=1)
    den_b = den_t[idx].sum(axis=1)
    ok = den_b > 0
    g = num_b[ok] / den_b[ok] * scale
    return {"estimate": float(num_t.sum() / den_t.sum() * scale),
            "ci95": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
            "resamples_used": int(ok.sum()), "tasks": len(uniq)}


def cluster_ci(per_cell_diff, task_arr):
    r = ratio_ci(per_cell_diff, np.ones(len(per_cell_diff)), task_arr)
    return r["ci95"]


def bh(ps, q):
    ps = np.array(ps)
    o = np.argsort(ps)
    m = len(ps)
    ok = ps[o] <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.where(ok)[0]) + 1) if ok.any() else 0
    keep = np.zeros(m, dtype=bool)
    keep[o[:k]] = True
    return keep


result = {
    "part": PART, "n_cells": n_cells, "n": N, "seeds": SEEDS,
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
    "inputs_sha256": {k: sha256(p) for k, p in INPUT_FILES.items()},
    "thresholds_from_method_config": {k: POL[k] for k in sorted(POL) if not isinstance(POL[k], (list, dict))},
}
t0 = time.time()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------- base
if PART == "base":
    yd, yn = labels(WO, W)
    neg = yd < NEG_TOL
    posd = yd > POS_TOL
    result["sign_counts"] = {"pos": int(posd.sum()), "zero": int((np.abs(yd) <= POS_TOL).sum()), "neg": int(neg.sum())}
    result["trials_per_arm"] = {f"{a}|{b}": c for (a, b), c in
                                sorted(Counter((len(x), len(y)) for x, y in zip(wo_tr, w_tr)).items())}
    raw_cfg = defaultdict(set)
    for r in trials:
        raw_cfg[r["version"]].add((r["config_id"], r["harness"]))
    result["accounting"] = {
        "raw_rows": len(trials), "raw_rows_by_export": dict(Counter(r["version"] for r in trials)),
        "rows_by_exclusion_reason": dict(Counter(r["exclusion_reason"] for r in trials)),
        "paired_trials": int(sum(r["included_in_standard"] == "True" for r in trials)),
        "configs_in_raw_export": {v: len(s) for v, s in raw_cfg.items()},
        "openhands_configs_in_raw_export": int(sum(h == "openhands" for _, h in raw_cfg.get("v1.1", set()))),
        "configs_with_paired_cells": len(set(CFGS)), "tasks_with_paired_cells": len(set(TASKS)),
        "repository_tasks": len(strata), "cells_by_export": dict(Counter(VERS)),
        "max_trials_per_condition": int(max(max(len(x), len(y)) for x, y in zip(wo_tr, w_tr)))}
    result["cells_three_trials_both"] = int(sum(len(x) == 3 and len(y) == 3 for x, y in zip(wo_tr, w_tr)))
    result["cells_single_trial_in_a_condition"] = int(sum(len(x) == 1 or len(y) == 1 for x, y in zip(wo_tr, w_tr)))
    allr = np.concatenate(wo_tr + w_tr)
    result["trial_rewards"] = {"n": int(len(allr)), "zero": int((allr == 0).sum()), "one": int((allr == 1).sum()),
                               "partial": int(((allr > 0) & (allr < 1)).sum())}
    result["mean_reward_pct"] = {"without": float(WO.mean() * 100), "with": float(W.mean() * 100),
                                 "lift_pp": float(yd.mean() * 100)}
    result["mean_delta_pp"] = {"cells_with_loss": float(yd[neg].mean() * 100), "cells_with_gain": float(yd[posd].mean() * 100),
                               "cells_wo_ge_0.8": float(yd[WO >= P0_CUT].mean() * 100), "n_cells_wo_ge_0.8": int((WO >= P0_CUT).sum())}

    pred = fit_all(XBASE, yd, yn, TASKS, CV)
    result["auc"] = auc_block(pred, yn, WO)
    dec = decide(pred, WO, yd)
    result["replay_as_in_paper"] = {}
    for k, m in dec.items():
        diff = np.where(m, W, WO) - W
        result["replay_as_in_paper"][k] = {
            "gain_pp": float(diff.mean() * 100), "task_cluster_ci95": cluster_ci(diff, TASKS),
            "neg_left": int((m & neg).sum()), "neg_removed_pct": float(100 * (1 - (m & neg).sum() / neg.sum())),
            "pos_kept": int((m & posd).sum()), "load_rate": float(m.mean())}

    bins = [("wo=0", WO == 0), ("0<wo<1", (WO > 0) & (WO < 1)), ("wo=1", WO == 1)]
    result["by_wo_level"] = {name: {"cells": int(m.sum()), "neg": int((m & neg).sum()), "pos": int((m & posd).sum()),
                                    "zero": int((m & ~neg & ~posd).sum())} for name, m in bins}
    result["cells_wo_gt0"] = {"cells": int((WO > 0).sum()), "neg": int(((WO > 0) & neg).sum()),
                              "neg_rate_pct": float(100 * ((WO > 0) & neg).sum() / (WO > 0).sum())}

    # by export
    vers = np.array(VERS)
    result["by_export"] = {}
    for v in sorted(set(VERS)):
        m = vers == v
        result["by_export"][v] = {"cells": int(m.sum()), "baseline_pct": float(WO[m].mean() * 100),
                                  "with_pct": float(W[m].mean() * 100), "lift_pp": float(yd[m].mean() * 100),
                                  "cells_wo_eq_1_pct": float(100 * (WO[m] == 1).mean()),
                                  "cells_wo_eq_0_pct": float(100 * (WO[m] == 0).mean()),
                                  "loss_cells": int((m & neg).sum()), "loss_cells_pct": float(100 * (m & neg).sum() / m.sum()),
                                  "configs": len({CFGS[i] for i in np.where(m)[0]})}

    # Historical five-task candidate partition retained for reproduction only.
    # Superseded for Section III by section3_revision.py; not five confirmed cases.
    st = np.array([strata[t] for t in TASKS])
    v5 = st == "undisclosed_verifier_5"
    result["verifier_requirement_tasks"] = {
        "cells": int(v5.sum()), "cells_pct": float(100 * v5.mean()),
        "share_of_summed_lift_pct": float(100 * yd[v5].sum() / yd.sum()),
        "baseline_pct": float(WO[v5].mean() * 100), "with_pct": float(W[v5].mean() * 100), "lift_pp": float(yd[v5].mean() * 100),
        "lift_without_them_pp": float(yd[~v5].mean() * 100), "loss_cells": int((v5 & neg).sum()),
        "per_task": {t: {"cells": int((np.array(TASKS) == t).sum()),
                         "baseline_pct": float(WO[np.array(TASKS) == t].mean() * 100),
                         "with_pct": float(W[np.array(TASKS) == t].mean() * 100),
                         "lift_pp": float(yd[np.array(TASKS) == t].mean() * 100)}
                     for t in sorted({TASKS[i] for i in np.where(v5)[0]})}}

    # structural features versus losses
    flagged = np.isin(st, list(FLAGGED))
    rest = ~flagged
    wpos = WO > 0
    feat = {
        "flagged_tasks": len({TASKS[i] for i in np.where(flagged)[0]}), "other_tasks": len({TASKS[i] for i in np.where(rest)[0]}),
        "all_cells": {"flagged": [int(flagged.sum()), int((flagged & neg).sum())], "other": [int(rest.sum()), int((rest & neg).sum())],
                      "flagged_loss_pct": float(100 * (flagged & neg).sum() / flagged.sum()),
                      "other_loss_pct": float(100 * (rest & neg).sum() / rest.sum()),
                      "flagged_baseline_pct": float(WO[flagged].mean() * 100), "other_baseline_pct": float(WO[rest].mean() * 100)},
        "cells_wo_gt0": {"flagged": [int((flagged & wpos).sum()), int((flagged & wpos & neg).sum())],
                         "other": [int((rest & wpos).sum()), int((rest & wpos & neg).sum())],
                         "flagged_loss_pct": float(100 * (flagged & wpos & neg).sum() / (flagged & wpos).sum()),
                         "other_loss_pct": float(100 * (rest & wpos & neg).sum() / (rest & wpos).sum())},
    }
    num = den = 0.0
    for name, m in [("0<wo<=0.5", (WO > 0) & (WO <= 0.5)), ("0.5<wo<1", (WO > 0.5) & (WO < 1)), ("wo=1", WO == 1)]:
        a = int((m & flagged & neg).sum()); b = int((m & flagged & ~neg).sum())
        c = int((m & rest & neg).sum()); d = int((m & rest & ~neg).sum())
        tot = a + b + c + d
        if tot:
            num += a * d / tot; den += b * c / tot
    feat["mantel_haenszel_odds_ratio_wo_gt0"] = num / den
    # stratified task bootstrap for the difference in loss rate among cells with a positive baseline
    uniq, ti = task_index(TASKS)
    negc = np.bincount(ti, weights=(wpos & neg).astype(float), minlength=len(uniq))
    cnt = np.bincount(ti, weights=wpos.astype(float), minlength=len(uniq))
    is_fl = np.array([strata[t] in FLAGGED for t in uniq])
    fidx, ridx = np.where(is_fl)[0], np.where(~is_fl)[0]
    rng = np.random.default_rng(SEEDS["bootstrap"])
    bf = rng.integers(0, len(fidx), size=(10000, len(fidx)))
    br = rng.integers(0, len(ridx), size=(10000, len(ridx)))
    with np.errstate(divide="ignore", invalid="ignore"):
        dboot = (negc[fidx][bf].sum(1) / cnt[fidx][bf].sum(1) - negc[ridx][br].sum(1) / cnt[ridx][br].sum(1)) * 100
    dboot = dboot[np.isfinite(dboot)]
    feat["loss_rate_difference_wo_gt0_pp"] = {
        "estimate": feat["cells_wo_gt0"]["flagged_loss_pct"] - feat["cells_wo_gt0"]["other_loss_pct"],
        "ci95_stratified_task_bootstrap": [float(np.percentile(dboot, 2.5)), float(np.percentile(dboot, 97.5))]}
    neg_tasks = sorted({t for t in uniq if yd[np.array(TASKS) == t].mean() < 0})
    feat["negative_lift_tasks"] = len(neg_tasks)
    feat["negative_lift_tasks_without_feature"] = int(sum(strata[t] not in FLAGGED for t in neg_tasks))
    result["structural_features"] = feat

    # time-outs over all included trials of the export that records them
    inc = [r for r in trials if r["included_in_standard"] == "True"]
    to = defaultdict(lambda: [0, 0])
    per_task = defaultdict(lambda: np.zeros(4))
    for r in inc:
        to[(r["version"], r["normalized_mode"])][0] += 1
        to[(r["version"], r["normalized_mode"])][1] += r["is_timeout"] == "True"
        if r["version"] == "v1.1":
            j = 0 if r["normalized_mode"] == "with" else 2
            per_task[r["task"]][j] += 1
            per_task[r["task"]][j + 1] += r["is_timeout"] == "True"
    result["timeouts_[n,timeouts]"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(to.items())}
    arr = np.array([per_task[t] for t in sorted(per_task)])
    rng = np.random.default_rng(SEEDS["bootstrap"])
    bi = rng.integers(0, len(arr), size=(10000, len(arr)))
    s = arr[bi].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = (s[:, 1] / s[:, 0]) / (s[:, 3] / s[:, 2])
    rb = rb[np.isfinite(rb)]
    tot = arr.sum(axis=0)
    result["timeouts_v1.1"] = {"with_pct": float(100 * tot[1] / tot[0]), "without_pct": float(100 * tot[3] / tot[2]),
                               "ratio": float((tot[1] / tot[0]) / (tot[3] / tot[2])),
                               "ratio_ci95_task_bootstrap": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
                               "with": [int(tot[0]), int(tot[1])], "without": [int(tot[2]), int(tot[3])]}

# --------------------------------------------------------------------- exact
elif PART == "exact":
    p_neg = np.zeros(n_cells)
    p_pos = np.zeros(n_cells)
    for i in range(n_cells):
        pool = np.concatenate([wo_tr[i], w_tr[i]])
        k = len(wo_tr[i])
        n = len(pool)
        tot = c_neg = c_pos = 0
        for comb in itertools.combinations(range(n), k):
            mask = np.zeros(n, dtype=bool)
            mask[list(comb)] = True
            d = pool[~mask].mean() - pool[mask].mean()
            tot += 1
            c_neg += d < NEG_TOL
            c_pos += d > POS_TOL
        p_neg[i] = c_neg / tot
        p_pos[i] = c_pos / tot

    def poisson_binomial(p):
        pmf = np.array([1.0])
        for q in p:
            pmf = np.concatenate([pmf * (1 - q), [0.0]]) + np.concatenate([[0.0], pmf * q])
        cdf = np.cumsum(pmf)
        return {"mean": float(p.sum()), "sd": float(np.sqrt((p * (1 - p)).sum())),
                "q2.5": int(np.searchsorted(cdf, 0.025)), "q97.5": int(np.searchsorted(cdf, 0.975))}

    yd, _ = labels(WO, W)
    pb_neg = poisson_binomial(p_neg)
    pmf_tail = None
    result["expected_cells_under_within_cell_label_permutation"] = {
        "negative": pb_neg, "positive": poisson_binomial(p_pos),
        "observed_negative": int((yd < NEG_TOL).sum()), "observed_positive": int((yd > POS_TOL).sum()),
        "cells_where_negative_is_impossible_under_permutation": int((p_neg == 0).sum())}

# --------------------------------------------------------------------- tasks
elif PART == "tasks":
    rng = np.random.default_rng(SEEDS["tasks"])
    B = N
    uniq, ti = task_index(TASKS)
    vers = np.array(VERS)
    obs_cell = W - WO
    cnt_all = np.bincount(ti, minlength=len(uniq)).astype(float)
    sum_null = np.zeros((len(uniq), B))
    sum_null_v11 = np.zeros((len(uniq), B))
    nneg = np.zeros(B)
    npos = np.zeros(B)
    for i in range(n_cells):
        p = np.concatenate([wo_tr[i], w_tr[i]])
        k = len(wo_tr[i])
        v = p[np.argsort(rng.random((B, len(p))), axis=1)]
        d = v[:, k:].mean(axis=1) - v[:, :k].mean(axis=1)
        sum_null[ti[i]] += d
        if vers[i] == "v1.1":
            sum_null_v11[ti[i]] += d
        nneg += d < NEG_TOL
        npos += d > POS_TOL
    rows = []
    for j, t in enumerate(uniq):
        m = ti == j
        obs = float(obs_cell[m].mean())
        nul = sum_null[j] / cnt_all[j]
        row = {"task": t, "stratum": strata[t], "cells": int(m.sum()), "lift_pp": obs * 100.0,
               "p_two_sided": float((1 + (np.abs(nul) >= abs(obs) - 1e-9).sum()) / (B + 1)),
               "p_lower": float((1 + (nul <= obs + 1e-9).sum()) / (B + 1)),
               "p_upper": float((1 + (nul >= obs - 1e-9).sum()) / (B + 1))}
        for v in ("v1.0", "v1.1"):
            mv = m & (vers == v)
            row["cells_" + v] = int(mv.sum())
            row["lift_pp_" + v] = float(obs_cell[mv].mean() * 100.0) if mv.any() else None
        mv = m & (vers == "v1.1")
        if mv.any():
            o11 = float(obs_cell[mv].mean())
            n11 = sum_null_v11[j] / mv.sum()
            row["p_two_sided_v1.1"] = float((1 + (np.abs(n11) >= abs(o11) - 1e-9).sum()) / (B + 1))
            row["p_lower_v1.1"] = float((1 + (n11 <= o11 + 1e-9).sum()) / (B + 1))
        rows.append(row)
    for key, flag in (("p_two_sided", "sig_two_sided_fdr10"), ("p_lower", "sig_lower_fdr10"), ("p_upper", "sig_upper_fdr10")):
        keep = bh([r[key] for r in rows], 0.10)
        for r, k_ in zip(rows, keep):
            r[flag] = bool(k_)
    rows.sort(key=lambda r: r["lift_pp"])
    negs = [r for r in rows if r["lift_pp"] < 0]
    both = [r for r in negs if r["cells_v1.0"] and r["cells_v1.1"]]
    result["task_tests"] = {
        "permutations": B, "tasks": len(rows),
        "family": "one p-value per task for all 83 tasks; Benjamini-Hochberg at 10% applied to the 83 two-sided p-values",
        "p_value_definition": "(1 + number of permutations at least as extreme) / (B + 1)",
        "mc_se_at_p_0.05": float(np.sqrt(0.05 * 0.95 / B)),
        "significant_two_sided_fdr10": int(sum(r["sig_two_sided_fdr10"] for r in rows)),
        "significant_two_sided_fdr10_positive": int(sum(r["sig_two_sided_fdr10"] and r["lift_pp"] > 0 for r in rows)),
        "significant_two_sided_fdr10_negative": int(sum(r["sig_two_sided_fdr10"] and r["lift_pp"] < 0 for r in rows)),
        "two_sided_p_below_0.05": int(sum(r["p_two_sided"] < 0.05 for r in rows)),
        "two_sided_p_below_0.05_positive": int(sum(r["p_two_sided"] < 0.05 and r["lift_pp"] > 0 for r in rows)),
        "two_sided_p_below_0.05_negative": int(sum(r["p_two_sided"] < 0.05 and r["lift_pp"] < 0 for r in rows)),
        "tasks_lift_negative": len(negs), "tasks_lift_zero": int(sum(abs(r["lift_pp"]) < 1e-9 for r in rows)),
        "tasks_lift_positive": int(sum(r["lift_pp"] > 1e-9 for r in rows)),
        "negative_lift_smallest_two_sided_p": min(r["p_two_sided"] for r in negs),
        "negative_lift_smallest_one_sided_p": min(r["p_lower"] for r in negs),
        "one_sided_lower_family_fdr10": int(sum(r["sig_lower_fdr10"] for r in rows)),
        "one_sided_upper_family_fdr10": int(sum(r["sig_upper_fdr10"] for r in rows)),
        "one_sided_upper_p_below_0.05": int(sum(r["p_upper"] < 0.05 for r in rows)),
        "negative_lift_tasks_in_both_exports": len(both),
        "negative_lift_tasks_with_opposite_signs": int(sum(r["lift_pp_v1.0"] * r["lift_pp_v1.1"] < 0 for r in both)),
    }
    result["all_tasks"] = rows
    result["cell_counts_under_no_effect_null_mc"] = {"neg": summarize(nneg), "pos": summarize(npos)}
    cols = ["task", "stratum", "cells", "lift_pp", "p_two_sided", "p_lower", "p_upper", "sig_two_sided_fdr10",
            "sig_lower_fdr10", "sig_upper_fdr10", "cells_v1.0", "lift_pp_v1.0", "cells_v1.1", "lift_pp_v1.1",
            "p_two_sided_v1.1", "p_lower_v1.1"]
    with open(OUT_DIR / "task_tests.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

# --------------------------------------------------------------------- seeds
elif PART == "seeds":
    yd, yn = labels(WO, W)
    acc = defaultdict(list)
    for s in range(N):
        cv = dict(CV)
        for k in ("seed_cv_split", "seed_rf_delta", "seed_rf_neg", "seed_rf_static", "seed_rf_prior_delta", "seed_rf_prior_neg"):
            cv[k] = int(CV[k]) + 7919 * (s + 1)
        pred = fit_fast(XBASE, yd, yn, cv, TASKS)
        ab = auc_block(pred, yn, WO)
        for k, v in ab.items():
            acc["auc." + k].append(v)
        for k, m in decide(pred, WO, yd).items():
            acc["gain." + k].append(gain_pp(m, WO, W))
            acc["negleft." + k].append(float((m & (yd < NEG_TOL)).sum()))
        acc["aucdiff.full_minus_prior_cellcv"].append(ab["full_cellcv"] - ab["prior_cellcv"])
        acc["aucdiff.full_minus_prior_taskgroup"].append(ab["full_taskgroup"] - ab["prior_taskgroup"])
        acc["aucdiff.wo>0_full_minus_prior_taskgroup"].append(ab["wo>0_only_full_taskgroup"] - ab["wo>0_only_prior_taskgroup"])
        acc["aucdiff.full_cellcv_minus_trivial_wo_gt0"].append(ab["full_cellcv"] - ab["trivial_wo_gt0"])
    result["over_seeds"] = {k: summarize(v) for k, v in sorted(acc.items())}

# ---------------------------------------------------------------------- null
elif PART == "null":
    result["fast_fitter_check"] = check_fast_matches_paper()
    rng = np.random.default_rng(SEEDS["null"])
    pooled = [np.concatenate([a, b]) for a, b in zip(wo_tr, w_tr)]
    nwo = [len(a) for a in wo_tr]

    def permute():
        """Reassign condition labels within every cell; the multiset of rewards and the trial counts are kept."""
        wo_s = np.empty(n_cells); w_s = np.empty(n_cells)
        for i, p in enumerate(pooled):
            q = rng.permutation(p)
            wo_s[i] = q[:nwo[i]].mean(); w_s[i] = q[nwo[i]:].mean()
        return wo_s, w_s

    acc = defaultdict(list)
    per_perm = []
    for b in range(N):
        wo_s, w_s = permute()
        yd, yn = labels(wo_s, w_s)          # labels recomputed from the permuted rewards
        X = XBASE.copy(); X[:, 0] = wo_s    # the baseline feature recomputed from the permuted rewards
        pred = fit_fast(X, yd, yn, CV)      # forests refit from scratch
        row = {"permutation": b, "cells_negative": int((yd < NEG_TOL).sum()), "cells_positive": int((yd > POS_TOL).sum())}
        for k, v in auc_block(pred, yn, wo_s).items():
            acc["auc." + k].append(v); row["auc." + k] = v
        negs = float((yd < NEG_TOL).sum())
        for k, m in decide(pred, wo_s, yd).items():
            g = gain_pp(m, wo_s, w_s)
            rm = 100.0 * (1.0 - float((m & (yd < NEG_TOL)).sum()) / max(1.0, negs))
            acc["gain_as_in_paper." + k].append(g); acc["neg_removed_pct." + k].append(rm)
            row["gain." + k] = g; row["neg_removed_pct." + k] = rm
        per_perm.append(row)
    result["under_no_effect_null_models"] = {k: summarize(v) for k, v in sorted(acc.items())}
    result["per_permutation"] = per_perm
    acc2 = defaultdict(list)
    for b in range(2000):
        wo_s, w_s = permute()
        yd, yn = labels(wo_s, w_s)
        acc2["count.neg"].append(float((yd < NEG_TOL).sum())); acc2["count.pos"].append(float((yd > POS_TOL).sum()))
        acc2["auc.trivial_wo_gt0"].append(auc(yn, (wo_s > 0).astype(float))); acc2["auc.trivial_wo_value"].append(auc(yn, wo_s))
        acc2["gain_as_in_paper.level_cutoff_080"].append(gain_pp(wo_s < P0_CUT, wo_s, w_s))
        acc2["gain_as_in_paper.oracle_best_of_two"].append(gain_pp(yd > 0, wo_s, w_s))
        acc2["neg_removed_pct.level_cutoff_080"].append(100.0 * (1.0 - float(((wo_s < P0_CUT) & (yd < NEG_TOL)).sum()) / max(1.0, float((yd < NEG_TOL).sum()))))
    result["under_no_effect_null_model_free_2000"] = {k: summarize(v) for k, v in sorted(acc2.items())}

# --------------------------------------------------------------------- split
elif PART == "split":
    result["fast_fitter_check"] = check_fast_matches_paper()
    rng = np.random.default_rng(SEEDS["split"])
    elig = np.array([len(a) >= 2 and len(b) >= 2 for a, b in zip(wo_tr, w_tr)])
    idx = np.where(elig)[0]
    ne = len(idx)
    tasks_e = [TASKS[i] for i in idx]
    result["eligibility"] = {"rule": "at least two trials in each condition, so that one can be held out",
                             "eligible_cells": int(ne), "excluded_cells": int(n_cells - ne)}
    yd_f, yn_f = labels(WO[idx], W[idx])
    pred_f = fit_fast(XBASE[idx], yd_f, yn_f, CV)
    result["all_trials_replay_as_in_paper"] = {k: gain_pp(m, WO[idx], W[idx]) for k, m in decide(pred_f, WO[idx], yd_f).items()}
    result["all_trials_auc"] = auc_block(pred_f, yn_f, WO[idx])
    result["all_trials_sign_counts"] = {"neg": int((yd_f < NEG_TOL).sum()), "pos": int((yd_f > POS_TOL).sum())}

    def draw():
        """Hold out one trial per condition in every eligible cell; return its index and both sets of means."""
        ha = np.empty(ne, dtype=int); hc = np.empty(ne, dtype=int)
        woA = np.empty(ne); wA = np.empty(ne); woB = np.empty(ne); wB = np.empty(ne)
        for j, i in enumerate(idx):
            a, c = wo_tr[i], w_tr[i]
            ha[j] = rng.integers(len(a)); hc[j] = rng.integers(len(c))
            woB[j] = a[ha[j]]; wB[j] = c[hc[j]]
            woA[j] = (a.sum() - a[ha[j]]) / (len(a) - 1); wA[j] = (c.sum() - c[hc[j]]) / (len(c) - 1)
        return ha, hc, woA, wA, woB, wB

    def decide_from_decision_trials(woA, wA):
        """All a policy may use: decision-trial rewards and the static features. Held-out rewards are not an argument."""
        ydA, ynA = labels(woA, wA)
        X = XBASE[idx].copy(); X[:, 0] = woA
        pred = fit_fast(X, ydA, ynA, CV)
        return decide(pred, woA, ydA), pred, ydA, ynA

    checks = {"decision_and_heldout_trial_ids_disjoint": True, "decision_feature_recomputed_from_ids_matches": True,
              "refit_is_deterministic": None}
    acc = defaultdict(list)
    cell_sum = defaultdict(lambda: np.zeros(ne))
    per_split = []
    rec_path = OUT_DIR / "split_records_forest_policies.csv.gz"
    rec_cols = ["split", "cell_id", "task", "heldout_without_trial_id", "heldout_with_trial_id",
                "decision_without_trial_ids", "decision_with_trial_ids", "r0_decision", "r1_decision",
                "r0_heldout", "r1_heldout", "pred_delta", "pred_loss_probability",
                "load_cutoff", "load_rf_max_gain", "load_rf_risk_averse", "load_rf_baseline_only", "load_best_of_two"]
    with gzip.open(rec_path, "wt", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(rec_cols)
        for b in range(N):
            ha, hc, woA, wA, woB, wB = draw()
            dec, pred, ydA, ynA = decide_from_decision_trials(woA, wA)
            ydB, ynB = labels(woB, wB)
            if b == 0:
                dec2, _, _, _ = decide_from_decision_trials(woA, wA)
                checks["refit_is_deterministic"] = bool(all(np.array_equal(dec[k], dec2[k]) for k in dec))
            for j, i in enumerate(idx):
                d_wo = [t for q, t in enumerate(wo_ids_all[i]) if q != ha[j]]
                d_w = [t for q, t in enumerate(w_ids_all[i]) if q != hc[j]]
                h_wo, h_w = wo_ids_all[i][ha[j]], w_ids_all[i][hc[j]]
                if h_wo in d_wo or h_w in d_w or len(d_wo) + 1 != len(wo_ids_all[i]) or len(d_w) + 1 != len(w_ids_all[i]):
                    checks["decision_and_heldout_trial_ids_disjoint"] = False
                if abs(np.mean([rew_by_id[t] for t in d_wo]) - woA[j]) > 1e-12 or abs(np.mean([rew_by_id[t] for t in d_w]) - wA[j]) > 1e-12:
                    checks["decision_feature_recomputed_from_ids_matches"] = False
                wr.writerow([b, order[i], TASKS[i], h_wo, h_w, ";".join(d_wo), ";".join(d_w),
                             repr(float(woA[j])), repr(float(wA[j])), repr(float(woB[j])), repr(float(wB[j])),
                             repr(float(pred["pd"][j])), repr(float(pred["pn"][j])),
                             int(dec["level_cutoff_080"][j]), int(dec["skillgate_maxgain"][j]),
                             int(dec["skillgate_riskaverse"][j]), int(dec["prior_only_rf"][j]), int(dec["oracle_best_of_two"][j])])
            negB = float((ydB < NEG_TOL).sum())
            row = {"split": b}
            for k, m in dec.items():
                gh, gn = gain_pp(m, woB, wB), gain_pp(m, woA, wA)
                acc["heldout_gain." + k].append(gh); acc["decision_trial_replay_gain." + k].append(gn)
                row["heldout_gain." + k] = gh; row["decision_trial_replay_gain." + k] = gn
                cell_sum[k] += np.where(m, wB, woB) - wB
                acc["heldout_neg_removed_pct." + k].append(100.0 * (1.0 - float((m & (ydB < NEG_TOL)).sum()) / max(1.0, negB)))
            pos = woA > 0
            for name, sc in (("full_cellcv", pred["pn"]), ("prior_cellcv", pred["pn_prior"]),
                             ("trivial_woA_value", woA), ("trivial_woA_gt0", pos.astype(float))):
                sc = np.asarray(sc, dtype=float)
                acc["heldout_auc." + name].append(auc(ynB, sc))
                acc["decision_auc." + name].append(auc(ynA, sc))
                acc["heldout_auc_woA>0." + name].append(auc(ynB[pos], sc[pos]))
            per_split.append(row)
    result["leakage_checks"] = checks
    assert checks["decision_and_heldout_trial_ids_disjoint"] and checks["decision_feature_recomputed_from_ids_matches"]
    assert checks["refit_is_deterministic"] is not False
    result["forest_policies_over_splits"] = {k: summarize(v) for k, v in sorted(acc.items())}
    result["forest_policies_per_split"] = per_split
    result["forest_policies_heldout_gain"] = {}
    for k, v in cell_sum.items():
        r = ratio_ci(v / max(1, N), np.ones(ne), tasks_e)
        result["forest_policies_heldout_gain"][k] = {"mean_pp": r["estimate"], "ci95_task_bootstrap": r["ci95"],
                                                     "mc_se_over_splits": summarize(acc["heldout_gain." + k])["mc_se_of_mean"]}
    result["interval_definition"] = ("per-cell held-out gain is averaged over the random splits first; whole tasks are then "
                                     "resampled 10,000 times with all their cells; forests are not refit inside the bootstrap")
    result["records_file"] = rec_path.name

    NF = 2000
    acc2 = defaultdict(list)
    cs2 = defaultdict(lambda: np.zeros(ne))
    g = {name: {k: np.zeros(ne) for k in ("n", "dA", "dB", "negB", "posB")} for name in ("loss_on_decision_trials", "baseline_ge_0.8", "all_eligible")}
    for b in range(NF):
        ha, hc, woA, wA, woB, wB = draw()
        ydA, ynA = labels(woA, wA)
        ydB, ynB = labels(woB, wB)
        pols = {"always_without": np.zeros(ne, dtype=bool), "level_cutoff_080": woA < P0_CUT,
                "oracle_best_of_two": ydA > 0.0}
        for k, m in pols.items():
            acc2["heldout_gain." + k].append(gain_pp(m, woB, wB))
            acc2["decision_trial_replay_gain." + k].append(gain_pp(m, woA, wA))
            cs2[k] += np.where(m, wB, woB) - wB
        for name, sel in (("loss_on_decision_trials", ydA < NEG_TOL), ("baseline_ge_0.8", woA >= P0_CUT),
                          ("all_eligible", np.ones(ne, dtype=bool))):
            a = g[name]
            a["n"] += sel; a["dA"] += np.where(sel, ydA, 0.0); a["dB"] += np.where(sel, ydB, 0.0)
            a["negB"] += sel & (ydB < NEG_TOL); a["posB"] += sel & (ydB > POS_TOL)
        acc2["heldout_auc.trivial_woA_value"].append(auc(ynB, woA))
    result["model_free_over_2000_splits"] = {k: summarize(v) for k, v in sorted(acc2.items())}
    result["model_free_heldout_gain"] = {}
    for k, v in cs2.items():
        r = ratio_ci(v / NF, np.ones(ne), tasks_e)
        result["model_free_heldout_gain"][k] = {"mean_pp": r["estimate"], "ci95_task_bootstrap": r["ci95"],
                                                "mc_se_over_splits": summarize(acc2["heldout_gain." + k])["mc_se_of_mean"]}
    sel_out = {}
    for name, a in g.items():
        sel_out[name] = {
            "cells_per_split": float(a["n"].sum() / NF),
            "decision_trial_delta_pp": ratio_ci(a["dA"], a["n"], tasks_e),
            "heldout_delta_pp": ratio_ci(a["dB"], a["n"], tasks_e),
            "heldout_share_negative_pct": ratio_ci(a["negB"], a["n"], tasks_e),
            "heldout_share_positive_pct": ratio_ci(a["posB"], a["n"], tasks_e)}
    result["same_set_decision_vs_heldout"] = sel_out
else:
    raise SystemExit("unknown part")

result["seconds"] = round(time.time() - t0, 1)
(OUT_DIR / (PART + ".json")).write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
print("done", PART, result["seconds"], "s")
