"""Recompute the reviewed Section III subset from cells and independently from trials.

Run from any directory: python reanalysis/section3_revision.py
Historical inputs/results are never overwritten.
"""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RETAINED = {
    "earthquake-plate-calculation": "controlled verifier comparison",
    "protein-expression-analysis": "controlled verifier comparison",
    "flood-risk-analysis": "source-level operational-definition gap",
}
UNCONFIRMED = ["weighted-gdp-calc", "sec-financial-report"]


def compute():
    paths = [ROOT / "data/provenance_tables/S02" / n for n in
             ("paired_cells.csv", "normalized_trials.csv")]
    def read(p):
        with p.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    cells, trials = map(read, paths)
    rewards = {r["raw_row_id"]: float(r["reward_float"]) for r in trials
               if r["flag_reward_valid"] == "True"}
    for c in cells:
        wo = mean(rewards[i] for i in c["wo_trial_ids"].split(";"))
        wi = mean(rewards[i] for i in c["w_trial_ids"].split(";"))
        for k, v in [("wo_mean_unrounded", wo), ("w_mean_unrounded", wi),
                     ("delta_unrounded", wi - wo)]:
            assert abs(float(c[k]) - v) < 1e-9, (c["cell_id"], k)
    def summary(names):
        selected = [c for c in cells if c["task"] in names]
        others = [c for c in cells if c["task"] not in names]
        def avg(rows, key):
            return mean(float(c[key]) for c in rows) * 100
        return dict(tasks=len(names), cells=len(selected),
                    baseline_pct=avg(selected, "wo_mean_unrounded"),
                    with_pct=avg(selected, "w_mean_unrounded"),
                    lift_pp=avg(selected, "delta_unrounded"),
                    cells_pct=100 * len(selected) / len(cells),
                    share_of_summed_lift_pct=100 * sum(float(c["delta_unrounded"]) for c in selected) / sum(float(c["delta_unrounded"]) for c in cells),
                    lift_without_them_pp=avg(others, "delta_unrounded"),
                    loss_cells=sum(float(c["delta_unrounded"]) < -1e-6 for c in selected))
    old, current = summary(set(RETAINED) | set(UNCONFIRMED)), summary(RETAINED)
    current["per_task"] = {t: summary({t}) for t in RETAINED}
    return dict(input_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                weighting="equal weight per paired configuration-task cell; not equal weight per task",
                classification=RETAINED, unconfirmed=UNCONFIRMED,
                historical_five=old, current_three=current,
                change={k: current[k] - old[k] for k in old},
                all_cells=len(cells), all_lift_pp=mean(float(c["delta_unrounded"]) for c in cells)*100,
                independently_checked_cells=len(cells))


if __name__ == "__main__":
    result = compute()
    dest = ROOT / "reanalysis/results/section3_revision.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["current_three"], indent=2))
