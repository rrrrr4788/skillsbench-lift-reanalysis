#!/usr/bin/env python3
"""Check that the numbers printed in the paper are the ones in the result files.

Usage: python verify_paper_numbers.py RESULTS_DIR MAIN_TEX
From the repository root:
    python -B reanalysis/verify_paper_numbers.py reanalysis/results paper/main_full.tex

Every expected string is formatted here from the JSON written by skillgate_checks.py, from the reviewed
three-task subset, and from the provenance tables, and is then searched for in the LaTeX source. Nothing
is copied from the paper into this file by hand, so a number that was typed wrongly in the paper, or that
changed after a rerun, shows up as MISSING.

The expected strings follow the v3 manuscript (timeouts scored as 0). paper/main.tex (8 pages) is
paper/main_full.tex with some passages switched off, so every string is present in both sources.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
# Fail closed: a manuscript check never bypasses primary-input identity checks.
subprocess.run([sys.executable, "-B", str(ROOT / "reanalysis/check_inputs.py"), str(ROOT)], check=True)
RES = Path(sys.argv[1])
TEX = Path(sys.argv[2]).read_text(encoding="utf-8")
sys.path.insert(0, str(ROOT / "reanalysis"))


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


base, exact, tasks, seeds, null, split = (load(n + ".json") for n in ("base", "exact", "tasks", "seeds", "null", "split"))
checks = []


def want(label, text):
    checks.append((label, text))


def c(n):
    return format(n, ",")


def sg(x, digits):
    """Signed number as it stands in running text and table cells: +2.92, or a math minus and 0.67."""
    s = "%+.*f" % (digits, x)
    return "$-$" + s[1:] if s.startswith("-") else s


# ---- Table I and Section II
a = base["accounting"]
r = a["rows_by_exclusion_reason"]
tr = base["trial_rewards"]
be = base["by_export"]
m = base["mean_reward_pct"]
sc = base["sign_counts"]
want("T1 raw rows", "Raw run rows of the two exports & %s \\\\" % c(a["raw_rows"]))
want("T1 one condition", "Only one condition in the cell & %d \\\\" % r["UNPAIRED_MODE_IN_CELL"])
want("T1 task not in repository", "Task not in the audited repository & %d \\\\" % r["TASK_NOT_IN_REPO"])
want("T1 paired runs", "Paired runs included in the analysis & %s \\\\" % c(a["paired_trials"]))
want("T1 unscored timeouts scored 0", "Timed out without a score, scored 0 & %s \\\\" % c(a["rescored_unscored_timeouts_in_paired_cells"]))
want("T1 score 0 / 1 / partial", "Score exactly 0 / exactly 1 / partial & %s / %s / %d \\\\" % (c(tr["zero"]), c(tr["one"]), tr["partial"]))
want("T1 configurations", "Configurations (first / second batch export) & %d (%d / %d) \\\\" % (
    a["configs_with_paired_cells"], be["v1.0"]["configs"], be["v1.1"]["configs"]))
want("T1 tasks with cells", "Tasks with at least one paired cell & %d of %d \\\\" % (a["tasks_with_paired_cells"], a["repository_tasks"]))
want("T1 cells by export", "Paired cells (first / second batch export) & %s (%d / %s) \\\\" % (
    c(base["n_cells"]), be["v1.0"]["cells"], c(be["v1.1"]["cells"])))
want("T1 cells 3+3", "Three runs under both conditions & %s \\\\" % c(base["cells_three_trials_both"]))
want("T1 cells single run", "Only one run under at least one condition & %d \\\\" % base["cells_single_trial_in_a_condition"])
want("T1 mean score", "Average score without / with skill & %.2f\\%% / %.2f\\%% \\\\" % (m["without"], m["with"]))
want("T1 sign counts", "& %d / %d / %d \\\\" % (sc["pos"], sc["zero"], sc["neg"]))
want("abstract trials and cells", "reanalyze %s public trials across %s paired" % (c(a["paired_trials"]), c(base["n_cells"])))
want("first export rows", "%s rows across %d configurations of Claude Code" % (
    c(a["raw_rows_by_export"]["v1.0"]), a["configs_in_raw_export"]["v1.0"]))
want("second export rows", "adds %s rows across %d configurations, %d on OpenHands" % (
    c(a["raw_rows_by_export"]["v1.1"]), a["configs_in_raw_export"]["v1.1"], a["openhands_configs_in_raw_export"]))
want("rescored unscored timeouts in repo", "%s timeout runs without grader scores are scored as 0" % c(a["rescored_unscored_timeouts_in_repo"]))
want("cells with a single run (text)", "%d cells have only one run" % base["cells_single_trial_in_a_condition"])
want("share of 0/1 scores", "%.0f\\%% of the runs in our analysis scored exactly 0 or 1" % (100.0 * (tr["zero"] + tr["one"]) / a["paired_trials"]))
want("tasks without paired cells", "leaves four repository tasks without paired cells")
assert a["repository_tasks"] - a["tasks_with_paired_cells"] == 4
want("share of raw rows without score", "%.0f\\%% of raw rows lack finite grader scores" % (100.0 * a["raw_rows_without_score"] / a["raw_rows"]))

# ---- sign of the difference by baseline
lv = base["by_wo_level"]
for key, name in (("wo=0", "Exactly 0"), ("0<wo<1", "Between 0 and 1"), ("wo=1", "Exactly 1")):
    b = lv[key]
    want("sign table " + name, "%s & %d & %d & %d & %d \\\\" % (name, b["cells"], b["neg"], b["zero"], b["pos"]))
want("sign table total", "All cells & %s & %d & %d & %d \\\\" % (c(base["n_cells"]), sc["neg"], sc["zero"], sc["pos"]))
share0 = 100.0 * lv["wo=0"]["cells"] / base["n_cells"]
want("share that cannot show a loss (abstract)", "%.0f\\%% of cells have zero baseline" % share0)
want("share that cannot show a loss (V-D)", "excludes the %.0f\\%% of zero-baseline cells" % share0)
want("cells with all no-skill trials failing", "In the %d cells where all no-skill trials fail" % lv["wo=0"]["cells"])
want("cells with all no-skill trials passing", "in the %d cells where all no-skill trials pass" % lv["wo=1"]["cells"])
g = base["cells_wo_gt0"]
want("cells with positive baseline", "only in the %d cells with positive baseline---within this subset, %.0f\\%% (%d/%d)" % (
    g["cells"], g["neg_rate_pct"], sc["neg"], g["cells"]))
want("loss cells overall", "%.1f\\%% of cells (%d/%s)" % (100.0 * sc["neg"] / base["n_cells"], sc["neg"], c(base["n_cells"])))

# ---- exports
want("baseline by export", "(%.1f\\%% vs.\\ %.1f\\%%)" % (be["v1.1"]["baseline_pct"], be["v1.0"]["baseline_pct"]))
want("ceiling cells by export", "(%.1f\\%% vs.\\ %.1f\\%%)" % (be["v1.1"]["cells_wo_eq_1_pct"], be["v1.0"]["cells_wo_eq_1_pct"]))
want("lift by export", "(%.1f vs.\\ %.1f\\,pp)" % (be["v1.1"]["lift_pp"], be["v1.0"]["lift_pp"]))
want("loss cells by export", "(%.1f\\%% vs.\\ %.1f\\%%)" % (be["v1.1"]["loss_cells_pct"], be["v1.0"]["loss_cells_pct"]))

# ---- reviewed three-task subset (recomputed from the trial records by the repository's own script)
from section3_revision import compute  # noqa: E402
v = compute()["current_three"]
for t, x in v["per_task"].items():
    want("task row " + t, "\\task{%s} & %d & %.1f / %.1f & $%+.1f$" % (t, x["cells"], x["baseline_pct"], x["with_pct"], x["lift_pp"]))
want("three tasks total", "Total for three tasks & %d & %.1f / %.1f & $%+.1f$" % (v["cells"], v["baseline_pct"], v["with_pct"], v["lift_pp"]))
want("three tasks, share of cells", "represent %.0f\\%% of all paired cells (%d/%s)" % (v["cells_pct"], v["cells"], c(base["n_cells"])))
want("three tasks, share of lift", "descriptively account for %.0f\\%% of summed lift" % v["share_of_summed_lift_pct"])
want("lift without the three", "from %.1f to %.1f\\,pp" % (m["lift_pp"], v["lift_without_them_pp"]))
want("abstract lifts", "averaging %.1f\\,pp; excluding them reduces the mean lift from %.1f to %.1f\\,pp" % (v["lift_pp"], m["lift_pp"], v["lift_without_them_pp"]))
want("discussion share of lift", "contribute roughly %.0f\\%% of summed lift" % v["share_of_summed_lift_pct"])
assert v["loss_cells"] == 0, "the text says these tasks contain no loss cells"
want("three tasks contain no loss cells", "contain no loss cells")

# ---- controlled grader runs (artifacts/controlled_mechanism_verification.json)
cm = json.loads((ROOT / "artifacts" / "controlled_mechanism_verification.json").read_text(encoding="utf-8"))
eq = cm["earthquake-plate-calculation"]
want("earthquake distances", "Spherical great-circle distance (%skm), Web Mercator distance (%skm), and EPSG:4087 distance (%skm)" % (
    format(eq["method_B_exact_spherical_great_circle"]["distance_km"], ",.2f"),
    format(eq["method_C_epsg3857_web_mercator"]["distance_km"], ",.2f"),
    format(eq["method_A_epsg4087_skill_secret"]["distance_km"], ",.2f")))
assert eq["method_A_epsg4087_skill_secret"]["passed_tests"] == eq["method_A_epsg4087_skill_secret"]["total_tests"] == 8
assert eq["method_B_exact_spherical_great_circle"]["official_verifier_reward"] == 0.0
assert eq["method_C_epsg3857_web_mercator"]["official_verifier_reward"] == 0.0
assert len({eq[k]["farthest_id_found"] for k in ("method_A_epsg4087_skill_secret", "method_B_exact_spherical_great_circle",
                                                 "method_C_epsg3857_web_mercator")}) == 1
want("earthquake assertions", "only EPSG:4087 passed all eight assertions, and the other two failed the distance check and scored 0")
want("earthquake tolerance", "only within %.2f\\,km of the reference value" % eq["verifier_tolerance_km"])
want("earthquake tolerance in parts per million", "(about %.1f parts per million)" % (1e6 * eq["verifier_tolerance_km"] / eq["verifier_target_km"]))
pr = cm["protein-expression-analysis"]
assert pr["formulas_identical_across_conditions"] is True
want("protein formula cells", "identical formulas in all %d formula cells" % pr["n_total_formula_cells_verified"])
assert (pr["condition_A_uncached_formulas"]["passed_tests"], pr["condition_A_uncached_formulas"]["total_tests"]) == (3, 6)
assert (pr["condition_B_cached_formulas"]["passed_tests"], pr["condition_B_cached_formulas"]["total_tests"]) == (6, 6)
want("protein tests", "The version without cached values passed three of the six tests, and the version with cached values passed all six")
xb = cm["exam-block-sequencing"]
want("exam-block model size", "(%d blocks, %d windows, and %s distinct block triples), only the assignment variables and window variables need %s binary variables and %s linearization constraints" % (
    xb["n_blocks"], xb["n_triple_windows"], c(xb["n_csv_ordered_triplets_distinct_blocks"]),
    c(xb["distinct_block_total_xw_binary_vars"]), c(xb["distinct_block_triplet_linearization_constraints_4x"])))
want("mars parameter grid", "%d-point parameter grid" % cm["mars-clouds-clustering"]["hyperparameter_grid_size"])

# ---- task tests
t = tasks["task_tests"]
want("permutations", "Using %s permutations" % c(t["permutations"]))
want("observable tasks", "across all %d observable tasks, %d tasks show significant gains and none shows a significant loss" % (
    a["tasks_with_paired_cells"], t["significant_two_sided_fdr10_positive"]))
want("significant tasks (Section IV)", "A total of %d tasks reach significance" % t["significant_two_sided_fdr10_positive"])
assert t["significant_two_sided_fdr10_negative"] == 0 and t["significant_two_sided_fdr10"] == t["significant_two_sided_fdr10_positive"]
want("smallest p among negative tasks", "among all %d negative tasks is %.2f (one-sided %.3f)" % (
    t["tasks_lift_negative"], t["negative_lift_smallest_two_sided_p"], t["negative_lift_smallest_one_sided_p"]))
want("negative-lift tasks (intro of Section IV)", "and %d tasks show net negative lift" % t["tasks_lift_negative"])
rows = {r_["task"]: r_ for r_ in tasks["all_tasks"]}
most_neg = min(rows, key=lambda k: rows[k]["lift_pp"])
assert most_neg == "radar-vital-signs"
ex = rows[most_neg]
cells = list(csv.DictReader(open(ROOT / "data/provenance_tables/S02/paired_cells.csv", encoding="utf-8", newline="")))
exd = [float(x["delta_unrounded"]) for x in cells if x["task"] == most_neg]
want("most negative task overall", "(mean $%+.1f$\\,pp across %d cells), contains only %d loss cells, %d gain cells, and %d ties ($p=%.2f$)" % (
    ex["lift_pp"], ex["cells"], sum(d < -1e-9 for d in exd), sum(d > 1e-9 for d in exd), sum(abs(d) <= 1e-9 for d in exd), ex["p_two_sided"]))
# the two task-attached means, recomputed from the cell table
for task in ("dapt-intrusion-detection", "llm-prefix-cache-replay"):
    selected = [x for x in cells if x["task"] == task]
    if not selected:
        raise ValueError("Missing task: " + task)
    lo_mean, hi_mean = [100 * mean(float(x[k]) for x in selected) for k in ("wo_mean_unrounded", "w_mean_unrounded")]
    low_text = "0" if lo_mean == 0 else "%.1f" % lo_mean
    want("task-attached means: " + task, "\\task{%s} (%s to %.1f\\%%)" % (task, low_text, hi_mean))
# "the next two highest-lift tasks": largest task lifts outside the three retained tasks
others = sorted((row["lift_pp"], name) for name, row in rows.items() if name not in v["per_task"])
assert {others[-1][1], others[-2][1]} == {"llm-prefix-cache-replay", "dapt-intrusion-detection"}
r2r = rows["r2r-mpc-control"]
want("r2r by export", "\\task{r2r-mpc-control} moves from $%+.1f$\\,pp (%d cells) in the first export to $%+.1f$\\,pp (%d cells) in the second" % (
    r2r["lift_pp_v1.0"], r2r["cells_v1.0"], r2r["lift_pp_v1.1"], r2r["cells_v1.1"]))
v11_most_neg = min((r_ for r_ in rows.values() if r_["cells_v1.1"] > 0), key=lambda r_: r_["lift_pp_v1.1"])
assert v11_most_neg["task"] == "multilingual-video-dubbing"
want("most negative task in v1.1", "\\task{%s} ($%+.1f$\\,pp across %d cells)---reaches only an uncorrected two-sided $p=%.2f$" % (
    v11_most_neg["task"], v11_most_neg["lift_pp_v1.1"], v11_most_neg["cells_v1.1"], v11_most_neg["p_two_sided_v1.1"]))
want("sign changes", "Among the %d overall negative-lift tasks observed in both exports, %d flip sign" % (
    t["negative_lift_tasks_in_both_exports"], t["negative_lift_tasks_with_opposite_signs"]))

# ---- Section III numbers that come from the provenance tables
P = ROOT / "data" / "provenance_tables"


def table(rel):
    return list(csv.DictReader(open(P / rel, encoding="utf-8", newline="")))


# the three tasks whose skill points to another algorithm
three_alg = ("exam-block-sequencing", "mars-clouds-clustering", "r2r-mpc-control")
assert not any(rows[k]["sig_two_sided_fdr10"] for k in three_alg), "the text says none of the three is significant"
assert sum((rows[k]["lift_pp_v1.0"] or 0) > 0 for k in three_alg) == 2, "the text says two of them had a positive lift in the first export"
want("three algorithm tasks", "none of the three tasks shows a statistically significant loss in pooled tests, and two show positive lift in the first export")

# placeholder and license directories (S08)
dirs = table("S08/stub_and_license_directories.csv")
stub = [x for x in dirs if x["subcategory"] == "senior_stub_dir"]
lic = [x for x in dirs if x["subcategory"] == "license_dir"]
assert len(stub) == 4 and len(lic) == 2 and "senior-data-engineer" in [x["skill_dir"] for x in stub]
want("placeholder directories", "these four directories contain %d files totaling approximately %.0f KiB" % (
    sum(int(x["n_files"]) for x in stub), sum(int(x["total_bytes"]) for x in stub) / 1024))
assert sum(int(x["n_literal_marker_files"]) for x in stub) == 9
assert sum(int(x["n_literal_marker_files"]) > 0 for x in stub) == 2
want("placeholder files", "Nine files across two of the directories contain literal code-generator placeholders")
stub_lifts = [rows[x["task"]]["lift_pp"] for x in stub]
min_stub = 0.0 if abs(min(stub_lifts)) < 1e-9 else min(stub_lifts)
lic_lifts = sorted((rows[x["task"]]["lift_pp"] for x in lic), reverse=True)
want("lifts of the placeholder and license tasks", "the four placeholder tasks have lifts between $%.1f$ and $%+.1f$\\,pp, the two license tasks improve by $%+.1f$ and $%+.1f$\\,pp" % (
    min_stub, max(stub_lifts), lic_lifts[0], lic_lifts[1]))

# built-in skills in the OpenHands prompts (S14)
oh = table("S14/openhands_190_trajectories.csv")
oh_with = [x for x in oh if x["mode"] == "with"]
oh_without = [x for x in oh if x["mode"] == "without"]
with_builtin = [x for x in oh_with if int(x["builtin_skills"]) == 48]
task_only = [x for x in oh_with if int(x["builtin_skills"]) == 0 and int(x["task_skills"]) > 0]
assert len(with_builtin) + len(task_only) == len(oh_with)
want("OpenHands trajectories", "first-turn system prompts of %d OpenHands trajectories" % len(oh))
want("with-skill prompts listing built-in skills", "Out of the %d with-skill trials, %d prompts listed 48 built-in OpenHands skills before %d to %d task-specific skills" % (
    len(oh_with), len(with_builtin), min(int(x["task_skills"]) for x in with_builtin), max(int(x["task_skills"]) for x in with_builtin)))
builtin_chars = {x["catalog_text_chars"] for x in oh_without if int(x["builtin_skills"]) == 48}
assert len(builtin_chars) == 1
want("length of the built-in directory text", "is %s characters long" % c(int(builtin_chars.pop())))
shares = [float(x["task_skill_count_share_pct"]) for x in with_builtin]
want("share of task-specific entries", "only %.0f\\%%-%.0f\\%% of the directory entries" % (min(shares), max(shares)))
want("no-skill prompts listing built-in skills", "appeared in %d of the %d no-skill trials" % (
    sum(int(x["builtin_skills"]) == 48 for x in oh_without), len(oh_without)))
want("with-skill prompts without built-in skills", "Among the %d with-skill trials, %d listed these built-in skills and the remaining %d listed only task-specific skills" % (
    len(oh_with), len(with_builtin), len(task_only)))

# leaderboard comparison (S13)
lb = {x["model"]: x for x in table("S13/table3_official_harness_comparison.csv")}
for model, label in (("GPT-5.5", "GPT-5.5"), ("Opus 4.7", "Claude Opus 4.7"), ("Gemini 3.1 Pro", "Gemini 3.1 Pro")):
    x = lb[model]
    want("leaderboard row " + label, "%s & %s / %s ($+%s$) & %s / %s ($+%s$) \\\\" % (
        label, x["native_without_pct"], x["native_with_pct"], x["native_lift_col_pp"],
        x["openhands_without_pct"], x["openhands_with_pct"], x["openhands_lift_col_pp"]))
lbm = lb["Three-Model Arithmetic Mean"]
want("leaderboard mean lifts", "with a mean lift of %s\\,pp for native CLI and %s\\,pp for OpenHands" % (
    lbm["native_lift_col_pp"], lbm["openhands_lift_col_pp"]))
assert sum(float(lb[k]["openhands_without_pct"]) > float(lb[k]["native_without_pct"]) for k in ("GPT-5.5", "Opus 4.7", "Gemini 3.1 Pro")) == 1
want("one model with a higher baseline under OpenHands", "since one model already has a higher no-skill baseline under OpenHands")

# timing of the two distance computations (S23)
from statistics import median  # noqa: E402
timed = [x for x in table("S23/mars_live_timing_trials.csv") if x["phase"] == "timed"]
assert len(timed) == 7
t_skill = median(float(x["skill_callable_sec"]) for x in timed)
t_ref = median(float(x["oracle_numpy_sec"]) for x in timed)
grid = cm["mars-clouds-clustering"]["hyperparameter_grid_size"]
want("timing of the distance callback", "Over the seven timed benchmarks in this task's dataset, a single round with the Python callback has a median time of %.2f seconds and that of the reference solution's vectorized distance computation is %.3f seconds" % (t_skill, t_ref))
want("timing extrapolated to the grid", "distance evaluation alone extrapolates to %.1f minutes compared with %.0f seconds" % (grid * t_skill / 60, grid * t_ref))

# upstream review records (S15)
s15 = json.loads((P / "S15" / "metrics.json").read_text(encoding="utf-8"))["metrics"]
want("upstream review records", "maintainer audit records of %s trials, and %d of them are marked as anomalous skill loading or no-skill leakage, and %d as suspected score gaming or grader leakage" % (
    c(s15["S15.pr7_reviewed_rows"]["value"]), s15["S15.no_skill_leakage_runs"]["value"], s15["S15.verifier_leakage_total_runs"]["value"]))

# ---- small calculations stated in the text
import math  # noqa: E402
want("one failure among three trials", "produces a %.0f\\,pp apparent drop" % (100 / 3))
want("three passes at 0.7", "$0.7\\times0.7\\times0.7=%.2f$" % 0.7 ** 3)
want("gain from noise in the example", "creating about $%+.0f$\\,pp of gain" % (100 * 0.7 ** 3 * (1 - 0.7)))
# planning numbers: independent pass/fail trials at a pass rate of 0.5, 18 configurations,
# one-sided 5% test, 80% power
se3, se10 = math.sqrt(2 * 0.25 / (18 * 3)), math.sqrt(2 * 0.25 / (18 * 10))
z = 1.6448536269514722 + 0.8416212335729143
want("standard error", "a task-lift standard error near %.0f\\,pp" % (100 * se3))
want("planning: detectable loss, three trials", "true loss exceeding roughly %.0f\\,pp for 80\\%% power" % (100 * z * se3))
want("planning: detectable loss, ten trials", "uncorrected threshold to about %.0f\\,pp" % (100 * z * se10))

# ---- exact expectation under the no-effect null
e = exact["expected_cells_under_within_cell_label_permutation"]
want("expected losses under the null", "yields %.1f expected loss cells (central 95\\%% range %d--%d) and %.1f expected gain cells" % (
    e["negative"]["mean"], e["negative"]["q2.5"], e["negative"]["q97.5"], e["positive"]["mean"]))
want("observed losses and gains", "versus %d loss and %d gain cells" % (sc["neg"], sc["pos"]))
want("reference again", "fall below %.1f" % e["negative"]["mean"])

# ---- structural features
f = base["structural_features"]
want("flagged and other tasks", "these %d flagged tasks show a higher loss rate than the other %d tasks" % (
    f["flagged_tasks"], f["other_tasks"]))
want("loss rate, flagged vs other", "(%.1f\\%% vs.\\ %.1f\\%%)" % (f["all_cells"]["flagged_loss_pct"], f["all_cells"]["other_loss_pct"]))
want("baseline, flagged vs other", "(%.1f\\%% vs.\\ %.1f\\%%)" % (f["all_cells"]["flagged_baseline_pct"], f["all_cells"]["other_baseline_pct"]))
cf = f["cells_wo_gt0"]
want("loss rate with positive baseline", "%.1f\\%% (%d/%d) vs.~%.1f\\%% (%d/%d)" % (
    cf["flagged_loss_pct"], cf["flagged"][1], cf["flagged"][0], cf["other_loss_pct"], cf["other"][1], cf["other"][0]))
want("Mantel-Haenszel", "odds ratio of %.2f" % f["mantel_haenszel_odds_ratio_wo_gt0"])
d = f["loss_rate_difference_wo_gt0_pp"]
want("rate difference", "rate difference of $%+.1f$\\,pp (95\\%% interval $[%+.1f,%+.1f]$)" % (
    d["estimate"], d["ci95_stratified_task_bootstrap"][0], d["ci95_stratified_task_bootstrap"][1]))
want("negative tasks without feature", "%d of the %d negative-lift tasks have none of these three structural features" % (
    f["negative_lift_tasks_without_feature"], t["tasks_lift_negative"]))

# ---- time-outs
to = base["timeouts_v1.1"]
want("time-outs", "paired with-skill trials time out at %.1f\\%% (%d/%s) versus %.1f\\%% (%d/%s) for no-skill trials, a ratio of %.2f (95\\%% task-bootstrap interval %.2f--%.2f)" % (
    to["with_pct"], to["with"][1], c(to["with"][0]), to["without_pct"], to["without"][1], c(to["without"][0]),
    to["ratio"], to["ratio_ci95_task_bootstrap"][0], to["ratio_ci95_task_bootstrap"][1]))
tol = base["timeouts_v1.1_loss_cells"]
want("time-outs in loss cells", "ratio rises to %.1f (%.1f\\%% vs.\\ %.1f\\%%)" % (tol["ratio"], tol["with_pct"], tol["without_pct"]))
tot_task = base["timeouts_v1.1_paired_by_task"]
want("exam-block paired time-outs", "\\task{exam-block-sequencing} (%d/%d vs.\\ %d/%d)" % (
    tot_task["exam-block-sequencing"]["with"][1], tot_task["exam-block-sequencing"]["with"][0],
    tot_task["exam-block-sequencing"]["without"][1], tot_task["exam-block-sequencing"]["without"][0]))
want("mars-clouds paired time-outs", "\\task{mars-clouds-clustering} (%d/%d vs.\\ %d/%d)" % (
    tot_task["mars-clouds-clustering"]["with"][1], tot_task["mars-clouds-clustering"]["with"][0],
    tot_task["mars-clouds-clustering"]["without"][1], tot_task["mars-clouds-clustering"]["without"][0]))

# ---- gate table
rp = base["replay_as_in_paper"]
nm = null["under_no_effect_null_models"]
nf = null["under_no_effect_null_model_free_2000"]
hm = split["forest_policies_heldout_gain"]
hf = split["model_free_heldout_gain"]
fs = split["forest_policies_over_splits"]
ms = split["model_free_over_2000_splits"]


def ci2(ci):
    return "(%s, %s)" % (sg(ci[0], 2), sg(ci[1], 2))


names = [("level_cutoff_080", "Baseline cutoff"), ("skillgate_maxgain", "RF gate, max gain"),
         ("skillgate_riskaverse", "RF gate, risk averse"), ("prior_only_rf", "RF gate, baseline only"),
         ("oracle_best_of_two", "Best of two")]
null_gain, null_removed, held_upper = {}, {}, {}
for key, label in names:
    free = ("gain_as_in_paper." + key) in nf
    ng = (nf if free else nm)["gain_as_in_paper." + key]["mean"]
    nr = (nf if free and ("neg_removed_pct." + key) in nf else nm)["neg_removed_pct." + key]["mean"] if key != "oracle_best_of_two" else 100.0
    held = hf[key] if key in hf else hm[key]
    dec = (ms if key in hf else fs)["decision_trial_replay_gain." + key]["mean"]
    null_gain[key], null_removed[key], held_upper[key] = ng, nr, held["ci95_task_bootstrap"][1]
    want("gate table: " + label, "%s & %s %s & %.0f\\%% & %s & %.0f\\%% & %s & %s %s \\\\" % (
        label, sg(rp[key]["gain_pp"], 2), ci2(rp[key]["task_cluster_ci95"]), rp[key]["neg_removed_pct"], sg(ng, 1), nr,
        sg(dec, 1), sg(held["mean_pp"], 2), ci2(held["ci95_task_bootstrap"])))
three = ("level_cutoff_080", "skillgate_maxgain", "skillgate_riskaverse")
want("replay gains (text)", "by %.1f--%.1f\\,pp" % (min(rp[k]["gain_pp"] for k in three), max(rp[k]["gain_pp"] for k in three)))
want("replay gains (abstract)", "gain %.1f--%.1f\\,pp in offline replay" % (min(rp[k]["gain_pp"] for k in three), max(rp[k]["gain_pp"] for k in three)))
want("replay removal (text)", "skips %.0f\\%% of apparent-loss cells" % rp["skillgate_riskaverse"]["neg_removed_pct"])
want("permuted gains (text)", "gain %.1f--%.1f\\,pp and" % (min(null_gain[k] for k in three), max(null_gain[k] for k in three)))
want("permuted gains (abstract)", "yet still gain %.1f--%.1f\\,pp after permuting condition labels" % (min(null_gain[k] for k in three), max(null_gain[k] for k in three)))
want("permuted removal (text)", "skip %.0f\\%% of shuffled ``loss'' cells" % null_removed["skillgate_riskaverse"])
want("held-out upper bounds (text)", "upper bounds below 1\\,pp")
assert all(held_upper[k] < 1.0 for k in three)
assert all(held[1]["ci95_task_bootstrap"][0] < 0 < held[1]["ci95_task_bootstrap"][1] for held in
           ((k, hf[k] if k in hf else hm[k]) for k in three)), "the text says these intervals cross zero"
bo = hf["oracle_best_of_two"] if "oracle_best_of_two" in hf else hm["oracle_best_of_two"]
pr_rf = hf["prior_only_rf"] if "prior_only_rf" in hf else hm["prior_only_rf"]
want("best of two on held-out trials", "(Best of two) loses %.2f\\,pp (95\\%% CI $[%.2f,%.2f]$)" % (
    -bo["mean_pp"], bo["ci95_task_bootstrap"][0], bo["ci95_task_bootstrap"][1]))
want("baseline-only RF on held-out trials", "baseline-only forest likewise incurs a significant held-out loss ($%+.2f$\\,pp, $[%.2f,%.2f]$)" % (
    pr_rf["mean_pp"], pr_rf["ci95_task_bootstrap"][0], pr_rf["ci95_task_bootstrap"][1]))
want("eligible cells", "leaves %s eligible cells and excludes %d" % (c(split["eligibility"]["eligible_cells"]), split["eligibility"]["excluded_cells"]))
want("eligible cells (Section IV)", "the %s cells with at least two trials per condition" % c(split["eligibility"]["eligible_cells"]))
want("number of forest splits (protocol)", "over random splits (%d for forests, 2,000 for other rules)" % split["n"])
want("number of forest splits (threats)", "over repeated splits (%d for forests, 2,000 for other rules)" % split["n"])
want("number of forest permutations", "Forest policies average %d permutations" % null["n"])
want("excluded cells (threats)", "excludes %d low-trial-count cells" % split["eligibility"]["excluded_cells"])

# ---- same-set comparison on held-out trials
ss = split["same_set_decision_vs_heldout"]
hi = ss["baseline_ge_0.8"]
want("high-baseline cells", "about %.0f per split) have mean $\\Delta=%+.1f$\\,pp on decision trials, but rebound to $%+.1f$\\,pp on held-out trial pairs (95\\%% interval $[%+.1f,%+.1f]$)" % (
    hi["cells_per_split"], hi["decision_trial_delta_pp"]["estimate"], hi["heldout_delta_pp"]["estimate"],
    hi["heldout_delta_pp"]["ci95"][0], hi["heldout_delta_pp"]["ci95"][1]))
lo = ss["loss_on_decision_trials"]
want("cells with a loss on decision trials", "about %.0f per split) average $%+.1f$\\,pp on decision trials, but $%+.1f$\\,pp on held-out trial pairs (95\\%% interval $[%+.1f,%+.1f]$), with %.0f\\%% showing losses and %.0f\\%% gains" % (
    lo["cells_per_split"], lo["decision_trial_delta_pp"]["estimate"], lo["heldout_delta_pp"]["estimate"],
    lo["heldout_delta_pp"]["ci95"][0], lo["heldout_delta_pp"]["ci95"][1],
    lo["heldout_share_negative_pct"]["estimate"], lo["heldout_share_positive_pct"]["estimate"]))
want("ratio sentence", "the ratio $%+.1f$ to $%+.1f$\\,pp" % (lo["heldout_delta_pp"]["estimate"], lo["decision_trial_delta_pp"]["estimate"]))

# ---- AUC
au = base["auc"]
sd = seeds["over_seeds"]


def pm(k):
    return "$%.3f\\pm%.3f$" % (sd[k]["mean"], sd[k]["sd"])


want("AUC headline", "ROC-AUC of %.3f" % au["full_cellcv"])
want("AUC indicator (text)", "already yields AUC %.3f" % au["trivial_wo_gt0"])
want("AUC indicator (table)", "Indicator $r_{0}>0$ & %.3f & 0.500" % au["trivial_wo_gt0"])
want("AUC value of baseline", "Numeric $r_{0}$ & %.3f & %.3f" % (au["trivial_wo_value"], au["wo>0_only_trivial_wo_value"]))
want("AUC forests, cell folds, baseline", "Random forest (baseline + harness) & %s & %s" % (pm("auc.prior_cellcv"), pm("auc.wo>0_only_prior_cellcv")))
want("AUC forests, cell folds, all", "Random forest (all 17 features) & %s & %s" % (pm("auc.full_cellcv"), pm("auc.wo>0_only_full_cellcv")))
want("AUC forests, task folds, baseline", "Random forest (baseline + harness) & %s & %s" % (pm("auc.prior_taskgroup"), pm("auc.wo>0_only_prior_taskgroup")))
want("AUC forests, task folds, all", "Random forest (all 17 features) & %s & %s" % (pm("auc.full_taskgroup"), pm("auc.wo>0_only_full_taskgroup")))
want("AUC within positive baseline (text)", "reaches %.2f under cell-level cross-validation but drops to %.2f when holding out entire tasks" % (
    sd["auc.wo>0_only_full_cellcv"]["mean"], sd["auc.wo>0_only_full_taskgroup"]["mean"]))
want("AUC difference between the two forests", "trailed the baseline-and-harness forest by $%.3f\\pm%.3f$ AUC across all cells and by $%.3f\\pm%.3f$ among cells with $r_{0}>0$" % (
    -sd["aucdiff.full_minus_prior_taskgroup"]["mean"], sd["aucdiff.full_minus_prior_taskgroup"]["sd"],
    -sd["aucdiff.wo>0_full_minus_prior_taskgroup"]["mean"], sd["aucdiff.wo>0_full_minus_prior_taskgroup"]["sd"]))
assert seeds["n"] == 8
want("number of seeds (text)", "paired differences across eight random seeds")
want("number of seeds (table note)", "standard deviation across %d random seeds" % seeds["n"])
want("AUC on permuted labels", "still achieves AUC %.3f on label-shuffled data" % nm["auc.full_cellcv"]["mean"])
want("AUC against held-out sign", "it scores %.2f, only slightly ahead of the %.2f" % (
    fs["heldout_auc.full_cellcv"]["mean"], fs["heldout_auc.trivial_woA_gt0"]["mean"]))

missing = 0
for label, text in checks:
    ok = text in TEX
    missing += not ok
    print("%-8s %-50s %s" % ("ok" if ok else "MISSING", label, "" if ok else text))
print("\n%d checks, %d missing" % (len(checks), missing))
sys.exit(1 if missing else 0)
