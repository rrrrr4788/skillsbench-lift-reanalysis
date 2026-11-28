#!/usr/bin/env python3
"""Draw the two figures of the paper from the check results (version 2).

Usage: python make_figures.py RESULTS_DIR OUT_DIR
Reads base.json, tasks.json, null.json and split.json written by skillgate_checks.py.
No number is typed in by hand: every bar and whisker comes from those files.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType, no Type 3 fonts
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 7.5
matplotlib.rcParams["axes.linewidth"] = 0.6
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

INK = "#222222"
GREY = "#b8b8b8"
BLUE = "#2a6fb0"
ORANGE = "#d9742b"
GREEN = "#3a8f5c"


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ figure 1
tasks = sorted(load("tasks.json")["all_tasks"], key=lambda r: r["lift_pp"])
retained = set(load("section3_revision.json")["classification"])
assert retained <= {r["task"] for r in tasks}
x = list(range(len(tasks)))
colors = []
for r in tasks:
    sig_gain = r["sig_two_sided_fdr10"] and r["lift_pp"] > 1e-9
    if r["task"] in retained:
        assert sig_gain, "the retained cases are expected to be significant"
        colors.append(ORANGE)
    elif sig_gain:
        colors.append(BLUE)
    else:
        colors.append(GREY)

fig, ax = plt.subplots(figsize=(7.1, 2.05))
ax.bar(x, [r["lift_pp"] for r in tasks], color=colors, width=0.82, linewidth=0)
ax.axhline(0, color=INK, linewidth=0.6)
ax.set_xlim(-1, len(tasks))
ax.set_ylim(-24, 104)
ax.set_ylabel("Task-level skill lift (pp)")
ax.set_xlabel("83 tasks with paired cells, sorted by lift")
ax.set_xticks([0, 20, 40, 60, 80])
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#dddddd", linewidth=0.4)
ax.set_axisbelow(True)
n_neg = sum(r["lift_pp"] < -1e-9 for r in tasks)
n_sig_gain = sum(r["sig_two_sided_fdr10"] and r["lift_pp"] > 1e-9 for r in tasks)
n_sig_loss = sum(r["sig_two_sided_fdr10"] and r["lift_pp"] < -1e-9 for r in tasks)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, BLUE, GREY)]
labels = [
    "Retained instruction–evaluation gaps (%d tasks)" % len(retained),
    "Other tasks with a significant gain (FDR 10%)",
    "Not significant",
]
ax.legend(handles, labels, loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0.2)
note = "%d tasks below zero; none significant" % n_neg if n_sig_loss == 0 else "%d tasks below zero" % n_neg
ax.annotate(note, xy=(8, -6.5), xytext=(18.5, -13.5), ha="left", va="center", fontsize=7,
            arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.5))
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_task_lift.pdf")
plt.close(fig)
print("fig 1: %d tasks, %d below zero, %d significant gains, %d significant losses" % (len(tasks), n_neg, n_sig_gain, n_sig_loss))

# ------------------------------------------------------------------ figure 2
base = load("base.json")["replay_as_in_paper"]
null = load("null.json")
split = load("split.json")
nm = null["under_no_effect_null_models"]
nf = null["under_no_effect_null_model_free_2000"]
hm = split["forest_policies_heldout_gain"]
hf = split["model_free_heldout_gain"]

order = [
    ("level_cutoff_080", "Baseline\ncutoff"),
    ("skillgate_maxgain", "RF gate,\nmax gain"),
    ("skillgate_riskaverse", "RF gate,\nrisk averse"),
    ("prior_only_rf", "RF gate,\nbaseline\nonly"),
    ("oracle_best_of_two", "Best of\ntwo"),
]


def replay_stat(key):
    s = base[key]
    return s["gain_pp"], s["task_cluster_ci95"][0], s["task_cluster_ci95"][1]


def null_stat(key):
    """Mean and central 95% of the permutation outcomes (not a confidence interval of the mean)."""
    src = nf if ("gain_as_in_paper." + key) in nf else nm
    s = src["gain_as_in_paper." + key]
    return s["mean"], s["p2.5"], s["p97.5"]


def held_stat(key):
    s = hf[key] if key in hf else hm[key]     # model-free policies use the 2,000-split estimate
    return s["mean_pp"], s["ci95_task_bootstrap"][0], s["ci95_task_bootstrap"][1]


fig, ax = plt.subplots(figsize=(3.45, 2.25))
w = 0.26
for i, (key, _) in enumerate(order):
    series = [(replay_stat(key), BLUE), (null_stat(key), GREY), (held_stat(key), GREEN)]
    for j, ((m, lo, hi), col) in enumerate(series):
        xpos = i + (j - 1) * w
        ax.bar(xpos, m, width=w * 0.92, color=col, linewidth=0)
        ax.plot([xpos, xpos], [lo, hi], color=INK, linewidth=0.7)
ax.axhline(0, color=INK, linewidth=0.6)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([lab for _, lab in order], fontsize=6.5)
ax.set_ylabel("Gain over always loading (pp)")
ax.set_ylim(-4, 11.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#dddddd", linewidth=0.4)
ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, GREY, GREEN)]
ax.legend(handles, ["Replay on the logged trials", "Same replay, labels permuted (no skill effect)",
                    "Scored on a held-out trial"], loc="upper left", frameon=False,
          handlelength=1.2, borderaxespad=0.1, fontsize=6.5)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_gate_eval.pdf")
plt.close(fig)
for key, _ in order:
    print("fig 2 %-22s replay %+5.2f [%+.2f, %+.2f] | null %+5.2f [%+.2f, %+.2f] | held-out %+5.2f [%+.2f, %+.2f]" %
          ((key,) + replay_stat(key) + null_stat(key) + held_stat(key)))
