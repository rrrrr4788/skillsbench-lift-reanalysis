"""Compare two output directories of skillgate_checks.py.

Usage:
    python compare_runs.py OUT_A OUT_B

For every result JSON present in both directories the script lists the leaves
that differ. Leaves that describe the run rather than the result (input hashes,
timings) are counted separately. It also compares task_tests.csv and, for the
held-out evaluation, the per-cell records of the splits both runs contain.

If the two runs used a different number of repetitions (the N argument), values
aggregated over repetitions are not comparable: the script then compares the
per-repetition entries both runs contain and the values that do not depend on N,
and says which sections it skipped. In the split part the 2,000 model-free
splits continue the random stream after the N forest splits, so they also
depend on N.

Exit status is 0 when every compared result leaf is identical and 1 otherwise.
"""
import csv
import gzip
import json
import sys
from pathlib import Path

RUN_KEYS = ("/inputs_sha256/", "/seconds")
# top-level sections of null.json and split.json that do not depend on the number of repetitions
N_INDEPENDENT = {"part", "n_cells", "seeds", "thresholds_from_method_config", "environment",
                 "fast_fitter_check", "eligibility", "all_trials_replay_as_in_paper", "all_trials_auc",
                 "all_trials_sign_counts", "leakage_checks", "interval_definition", "records_file",
                 "observed_replay_as_in_paper", "observed_auc"}


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k in sorted(obj):
            yield from leaves(obj[k], f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def is_run_leaf(path):
    return any(k in path + "/" for k in RUN_KEYS)


def compare_json(a_path, b_path):
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    skipped = []
    if a.get("n") != b.get("n"):
        keep_a, keep_b = {}, {}
        for k in sorted(set(a) & set(b)):
            if k in N_INDEPENDENT:
                keep_a[k], keep_b[k] = a[k], b[k]
            elif isinstance(a[k], list) and isinstance(b[k], list):
                m = min(len(a[k]), len(b[k]))
                keep_a[k], keep_b[k] = a[k][:m], b[k][:m]
            elif k not in ("n", "inputs_sha256", "seconds"):
                skipped.append(k)
        note = f"N differs ({a.get('n')} vs {b.get('n')}); sections not comparable and skipped: {', '.join(skipped)}"
        a, b = keep_a, keep_b
    else:
        note = ""
    la, lb = dict(leaves(a)), dict(leaves(b))
    diff_result, diff_run, same = [], [], 0
    for p in sorted(set(la) & set(lb)):
        if is_run_leaf(p):
            if la[p] != lb[p]:
                diff_run.append(p)
        elif la[p] == lb[p]:
            same += 1
        else:
            diff_result.append((p, la[p], lb[p]))
    only = sorted(set(la) ^ set(lb))
    return same, diff_result, diff_run, only, note


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    out_a, out_b = Path(sys.argv[1]), Path(sys.argv[2])
    bad = 0
    for name in ("base.json", "exact.json", "tasks.json", "seeds.json", "null.json", "split.json"):
        pa, pb = out_a / name, out_b / name
        if not (pa.exists() and pb.exists()):
            print(f"{name}: not in both directories, skipped")
            continue
        same, diff_result, diff_run, only, note = compare_json(pa, pb)
        print(f"{name}: {same} identical result leaves, {len(diff_result)} differing result leaves, "
              f"{len(diff_run)} differing run-description leaves (input hashes, timings), "
              f"{len(only)} leaves present in one run only")
        if note:
            print(f"    {note}")
        for p, va, vb in diff_result[:40]:
            print(f"    RESULT DIFF {p}: {va!r} vs {vb!r}")
        if len(diff_result) > 40:
            print(f"    ... {len(diff_result) - 40} more")
        for p in only[:12]:
            print(f"    present in one run only: {p}")
        bad += len(diff_result) + len(only)

    name = "task_tests.csv"
    pa, pb = out_a / name, out_b / name
    if pa.exists() and pb.exists():
        ra, rb = read_rows(pa), read_rows(pb)
        ndiff = sum(x != y for x, y in zip(ra, rb)) + abs(len(ra) - len(rb))
        print(f"{name}: {len(ra)} and {len(rb)} rows, {ndiff} differing rows")
        bad += ndiff

    name = "split_records_forest_policies.csv.gz"
    pa, pb = out_a / name, out_b / name
    if pa.exists() and pb.exists():
        ra, rb = read_rows(pa), read_rows(pb)
        shared = {r["split"] for r in ra} & {r["split"] for r in rb}
        sa = [r for r in ra if r["split"] in shared]
        sb = [r for r in rb if r["split"] in shared]
        ndiff = sum(x != y for x, y in zip(sa, sb)) + abs(len(sa) - len(sb))
        print(f"{name}: {len(shared)} shared splits, {len(sa)} and {len(sb)} rows in them, {ndiff} differing rows")
        bad += ndiff

    print("RESULT:", "identical" if bad == 0 else f"{bad} differences")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
