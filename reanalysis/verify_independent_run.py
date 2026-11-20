"""Compare the required base/exact/tasks/split rerun with frozen results.

Usage: python verify_independent_run.py RECORDED RERUN REPORT_JSON
Exact comparisons are reported separately from numerical tolerance (1e-10).
A split-1 rerun covers the shared split only, not the frozen split-60 aggregates.
"""
import json
import math
import sys
from pathlib import Path

from compare_runs import compare_json, read_rows


def close(a, b):
    return (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)
            and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10))


def compare(a, b):
    result = {"tolerance": {"absolute": 1e-10, "relative": 1e-10}, "parts": {}, "tables": {}}
    failed = False
    for name in ("base", "exact", "tasks", "split"):
        pa, pb = a / (name + ".json"), b / (name + ".json")
        if not pa.exists() or not pb.exists():
            raise FileNotFoundError(f"Required result missing: {name}")
        same, differences, run, only, note = compare_json(pa, pb)
        environment, floating, substantive = [], [], []
        for path, va, vb in differences:
            entry = {"path": path, "recorded": va, "rerun": vb}
            if path.startswith("/environment/"):
                environment.append(entry)
            elif close(va, vb):
                floating.append(entry)
            else:
                substantive.append(entry)
        result["parts"][name] = dict(exactly_equal_leaves=same,
            environment_differences=environment, within_tolerance=floating,
            substantive_differences=substantive, run_description_differences=run,
            unmatched_leaves=only, skipped_scope=note)
        failed |= bool(substantive or only)
    for filename in ("task_tests.csv", "split_records_forest_policies.csv.gz"):
        ra, rb = read_rows(a / filename), read_rows(b / filename)
        if filename.endswith(".gz"):
            shared = {r["split"] for r in ra} & {r["split"] for r in rb}
            if not shared:
                raise ValueError("No shared split records")
            ra = [r for r in ra if r["split"] in shared]
            rb = [r for r in rb if r["split"] in shared]
        differences, tolerant = [], 0
        for i, (x, y) in enumerate(zip(ra, rb)):
            for k in set(x) | set(y):
                if x.get(k) == y.get(k):
                    continue
                try:
                    if close(float(x[k]), float(y[k])):
                        tolerant += 1
                        continue
                except (ValueError, KeyError):
                    pass
                differences.append({"row": i, "column": k, "recorded": x.get(k), "rerun": y.get(k)})
        failed |= bool(differences or len(ra) != len(rb))
        result["tables"][filename] = dict(recorded_rows=len(ra), rerun_rows=len(rb),
            within_tolerance_fields=tolerant, substantive_differences=differences)
    result["passed_with_stated_scope_and_tolerance"] = not failed
    return result


if __name__ == "__main__":
    report = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": report["passed_with_stated_scope_and_tolerance"],
                      "parts": {k: {"exact": v["exactly_equal_leaves"],
                                    "tolerant": len(v["within_tolerance"]),
                                    "substantive": len(v["substantive_differences"])}
                                for k, v in report["parts"].items()},
                      "tables": report["tables"]}, indent=2))
    sys.exit(0 if report["passed_with_stated_scope_and_tolerance"] else 1)
