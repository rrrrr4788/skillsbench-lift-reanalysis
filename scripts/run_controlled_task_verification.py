#!/usr/bin/env python3
"""
run_controlled_task_verification.py
Executes controlled mechanism verifications directly on the real SkillsBench task input files
and official verifier scripts:
  1. protein-expression-analysis:
     - Real input: tasks/protein-expression-analysis/environment/data/protein_expression.xlsx (117,651 B)
     - Official verifier: tasks/protein-expression-analysis/verifier/test_outputs.py (all 6 pytest assertions)
     - Single-factor control: Dynamically maps Control (Data!D:H) and Treated (Data!I:M) columns by sample name.
       Both Condition A and Condition B write the exact same 143 valid Excel formulas:
       * Step 1 (C11:L20): 83 non-null INDEX-MATCH formulas (with 17 missing observations left blank/None)
       * Step 2 (B24:K27): 40 ROUND(AVERAGE(...), 3) and single-observation-safe ROUND(IF(COUNT(...)>1, STDEV(...), 0), 3) formulas
       * Step 3 (C32:D41): 20 ROUND(Log2FC, 3) and ROUND(2^D{r}, 3) formulas
       Condition A saves via openpyxl with empty <v></v> cache tags (3/6 tests pass -> Reward 0.0).
       Condition B populates the XML <v> cached numeric evaluations on the exact same 143 formula cells,
       verified both by cache roundtrip (cache_roundtrip_abs_diff = 0.0) and by an independent formula
       parser/evaluator (max_formula_vs_cache_abs_diff = 0.0; 6/6 tests pass -> Reward 1.0).
  2. earthquake-plate-calculation:
     - Real inputs: tasks/earthquake-plate-calculation/environment/earthquakes_2024.json (1,504 global -> 107 Pacific plate quakes),
       PB2002_plates.json, and PB2002_boundaries.json (1,392 PA boundary segments)
     - Official verifier: tasks/earthquake-plate-calculation/verifier/test_outputs.py (TestEarthquakeTask, all 8 unittest methods)
     - End-to-end full search over all 107 Pacific plate earthquakes & 1,392 PA boundary segments under:
       * Method A: EPSG:4087 projection (provided in geospatial-analysis/SKILL.md Line 20) -> 3,878.27 km (8/8 tests pass, Reward 1.0)
       * Method B: Exact spherical great-circle point-to-segment distance (R=6371 km) -> 3,552.98 km (7/8 tests pass, test_distance_within_tolerance fails by 325.29 km, Reward 0.0)
       * Method C: Standard EPSG:3857 Web Mercator -> 4,152.56 km (7/8 tests pass, test_distance_within_tolerance fails by 274.29 km, Reward 0.0)
  3. mars-clouds-clustering:
     - Real inputs: tasks/mars-clouds-clustering/environment/data/citsci_train.csv (46.2 MB, 369 unique base images, 25,856 rows)
       and expert_train.csv (3.1 MB)
     - Measures weighted 2D Euclidean distance matrix computation d(a,b) = sqrt((w*dx)^2 + ((2-w)*dy)^2) on [x, y]
       comparing oracle/solve.py's 3D NumPy broadcast vs. custom-distance-metrics skill's Python callable metric across all 369 images,
       plus linear extrapolation across the 847-point hyperparameter grid (distance computation only).
  4. exam-block-sequencing:
     - Real inputs: tasks/exam-block-sequencing/environment/input/instance.json, pair_counts.csv, triplet_counts.csv,
       and tasks/exam-block-sequencing/verifier/oracle_metrics.json.
     - Analytical variable and constraint accounting for the x + w ordered-window MIP subproblem on the real 24-block instance
       (both for the 11,130 distinct-block triplets [245,436 binary vars, 979,440 constraints] and all 12,415 CSV rows
       [273,706 binary vars, 1,092,520 constraints], with 4 linearization inequalities per window variable) vs. official Oracle (0 MIP vars).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import types
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CANONICAL_ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DEFAULT_DATA_DIR = PROJECT_DIR / "external_data" / "skillsbench_audit"


def _check_safe_write_dir(target_dir: Path, allow_overwrite_canonical: bool) -> None:
    resolved = target_dir.resolve()
    for s_name in ("provenance", "provenance_v2", "provenance_v3", "provenance_v4", "release_v1_0_0_rc1"):
        s_dir = (CANONICAL_ARTIFACTS_DIR / s_name).resolve()
        if resolved == s_dir or s_dir in resolved.parents:
            raise ValueError(f"Refusing to write inside sealed directory: {s_dir}")
    if not allow_overwrite_canonical and resolved == CANONICAL_ARTIFACTS_DIR.resolve():
        raise ValueError(
            "Refusing to write directly into canonical artifacts/ without --allow-overwrite-canonical. "
            "Use the default isolated --out-dir under artifacts/runs/."
        )


def run_controlled_verification(
    *,
    data_dir: Path,
    out_dir: Path,
    work_dir: Path,
    run_id: str,
    mars_trials: int = 7,
    task_audit_csv: Path | None = None,
) -> dict:
    if "pytest" not in sys.modules:
        sys.modules["pytest"] = types.SimpleNamespace(
            fixture=lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    TASKS_DIR = data_dir / "skillsbench_repo" / "tasks"
    TASK_AUDIT_CSV = task_audit_csv or (CANONICAL_ARTIFACTS_DIR / "skillsbench_87_tasks_forensic_audit.csv")

    task_audit_df = pd.read_csv(TASK_AUDIT_CSV).set_index("task")

    def get_task_lifts(task_id):
        row = task_audit_df.loc[task_id]
        v11_raw = row["v11_delta_mean"]
        v11_val = "N/A" if (pd.isna(v11_raw) or str(v11_raw).strip().upper() == "N/A") else round(float(v11_raw) * 100.0, 2)
        return {
            "all_configs_wo_pct": round(float(row["wo_mean"]) * 100.0, 2),
            "all_configs_w_pct": round(float(row["w_mean"]) * 100.0, 2),
            "all_configs_delta_pp": round(float(row["delta_mean"]) * 100.0, 2),
            "v11_delta_pp": v11_val,
        }

    results = {}

    # ==============================================================================
    # 1. PROTEIN-EXPRESSION-ANALYSIS: Single-Factor Controlled Test of Formula Caching
    # ==============================================================================
    print("[1/4] Running single-factor formula caching verification on real protein_expression.xlsx...")
    prot_task_dir = TASKS_DIR / "protein-expression-analysis"
    prot_src_xlsx = prot_task_dir / "environment" / "data" / "protein_expression.xlsx"
    prot_ver_py = prot_task_dir / "verifier" / "test_outputs.py"
    prot_cond_a = work_dir / "protein_expression_conditionA_formulas.xlsx"
    prot_cond_b = work_dir / "protein_expression_conditionB_evaluated.xlsx"

    # Build workbook with valid Excel formulas in BOTH Condition A and Condition B
    shutil.copy2(prot_src_xlsx, prot_cond_a)
    wb_src = openpyxl.load_workbook(prot_src_xlsx, data_only=True)
    task_ws_val = wb_src["Task"]
    data_ws_val = wb_src["Data"]

    data_max_row = data_ws_val.max_row
    data_max_col = data_ws_val.max_column
    data_max_col_let = openpyxl.utils.get_column_letter(data_max_col)

    # Map sample names in Data sheet row 1 to their exact column index and column letter
    data_sample_to_col = {}
    for col in range(1, data_max_col + 1):
        s_name = data_ws_val.cell(row=1, column=col).value
        if s_name is not None:
            data_sample_to_col[s_name] = (col, openpyxl.utils.get_column_letter(col))

    target_proteins = [task_ws_val.cell(row=r, column=1).value for r in range(11, 21)]
    target_samples = [task_ws_val.cell(row=10, column=c).value for c in range(3, 13)]

    control_samples = []
    treated_samples = []
    control_data_cols = []
    treated_data_cols = []
    for t_col in range(3, 13):
        s_name = task_ws_val.cell(row=10, column=t_col).value
        grp = task_ws_val.cell(row=9, column=t_col).value
        if s_name in data_sample_to_col:
            d_col_idx, d_col_let = data_sample_to_col[s_name]
            if grp == "Control":
                control_samples.append(s_name)
                control_data_cols.append((d_col_idx, d_col_let))
            elif grp == "Treated":
                treated_samples.append(s_name)
                treated_data_cols.append((d_col_idx, d_col_let))

    control_data_cols.sort(key=lambda x: x[0])
    treated_data_cols.sort(key=lambda x: x[0])
    # Verify contiguous ranges in Data sheet (D:H for Control, I:M for Treated)
    assert [c[0] for c in control_data_cols] == list(range(control_data_cols[0][0], control_data_cols[-1][0] + 1))
    assert [c[0] for c in treated_data_cols] == list(range(treated_data_cols[0][0], treated_data_cols[-1][0] + 1))
    ctrl_start_let, ctrl_end_let = control_data_cols[0][1], control_data_cols[-1][1]
    treat_start_let, treat_end_let = treated_data_cols[0][1], treated_data_cols[-1][1]
    assert f"{ctrl_start_let}:{ctrl_end_let}" == "D:H", f"Expected D:H for Control, got {ctrl_start_let}:{ctrl_end_let}"
    assert f"{treat_start_let}:{treat_end_let}" == "I:M", f"Expected I:M for Treated, got {treat_start_let}:{treat_end_let}"

    def lookup_val(p_id, s_name):
        p_row = None
        for r in range(2, data_max_row + 1):
            if data_ws_val.cell(row=r, column=1).value == p_id:
                p_row = r
                break
        if p_row is None or s_name not in data_sample_to_col:
            return None
        return data_ws_val.cell(row=p_row, column=data_sample_to_col[s_name][0]).value

    wb_a = openpyxl.load_workbook(prot_cond_a)
    ws_a = wb_a["Task"]
    cached_cell_map = {}  # cell ref (e.g. "C11") -> exact numeric evaluation of its formula <f>
    n_step1_non_null = 0

    # Step 1: C11:L20 INDEX-MATCH formulas on all 83 non-missing observations (leaving the 17 missing observations blank/None)
    for r_offset, prot_id in enumerate(target_proteins):
        r_idx = 11 + r_offset
        for c_offset, s_name in enumerate(target_samples):
            c_idx = 3 + c_offset
            col_let = openpyxl.utils.get_column_letter(c_idx)
            cell_ref = f"{col_let}{r_idx}"
            v = lookup_val(prot_id, s_name)
            if v is not None:
                ws_a[cell_ref] = (
                    f"=INDEX(Data!$A$1:${data_max_col_let}${data_max_row}, "
                    f"MATCH($A{r_idx}, Data!$A$1:$A${data_max_row}, 0), "
                    f"MATCH({col_let}$10, Data!$A$1:${data_max_col_let}$1, 0))"
                )
                cached_cell_map[cell_ref] = float(v)
                n_step1_non_null += 1
            else:
                ws_a[cell_ref] = None

    # Step 2: B24:K27 statistics formulas referencing Data!D{p_row}:H{p_row} (Control) and Data!I{p_row}:M{p_row} (Treated)
    # & Step 3: C32:D41 FC and Log2FC formulas (Col C = FC =ROUND(2^D{fc_row}, 3), Col D = Log2FC =ROUND({col_let}26-{col_let}24, 3))
    import statistics
    for p_offset, prot_id in enumerate(target_proteins):
        col_idx = 2 + p_offset  # B..K
        col_let = openpyxl.utils.get_column_letter(col_idx)
        p_row = None
        for r in range(2, data_max_row + 1):
            if data_ws_val.cell(row=r, column=1).value == prot_id:
                p_row = r
                break
        c_vals = [
            float(data_ws_val.cell(row=p_row, column=c_i).value)
            for c_i, _ in control_data_cols
            if data_ws_val.cell(row=p_row, column=c_i).value is not None
        ]
        t_vals = [
            float(data_ws_val.cell(row=p_row, column=c_i).value)
            for c_i, _ in treated_data_cols
            if data_ws_val.cell(row=p_row, column=c_i).value is not None
        ]
        c_mean = round(statistics.mean(c_vals), 3) if c_vals else 0.0
        c_std = round(statistics.stdev(c_vals), 3) if len(c_vals) > 1 else 0.0
        t_mean = round(statistics.mean(t_vals), 3) if t_vals else 0.0
        t_std = round(statistics.stdev(t_vals), 3) if len(t_vals) > 1 else 0.0

        ctrl_rng = f"Data!{ctrl_start_let}{p_row}:{ctrl_end_let}{p_row}"
        treat_rng = f"Data!{treat_start_let}{p_row}:{treat_end_let}{p_row}"

        ws_a[f"{col_let}24"] = f"=ROUND(AVERAGE({ctrl_rng}), 3)"
        ws_a[f"{col_let}25"] = f"=ROUND(IF(COUNT({ctrl_rng})>1, STDEV({ctrl_rng}), 0), 3)"
        ws_a[f"{col_let}26"] = f"=ROUND(AVERAGE({treat_rng}), 3)"
        ws_a[f"{col_let}27"] = f"=ROUND(IF(COUNT({treat_rng})>1, STDEV({treat_rng}), 0), 3)"
        cached_cell_map[f"{col_let}24"] = c_mean
        cached_cell_map[f"{col_let}25"] = c_std
        cached_cell_map[f"{col_let}26"] = t_mean
        cached_cell_map[f"{col_let}27"] = t_std

        # Step 3: Row 32 + p_offset
        fc_row = 32 + p_offset
        gene = task_ws_val.cell(row=11 + p_offset, column=2).value
        ws_a.cell(row=fc_row, column=1, value=prot_id)
        ws_a.cell(row=fc_row, column=2, value=gene)
        # Column D (col 4) is Log2FC (=ROUND({col_let}26-{col_let}24, 3)); Column C (col 3) is FC (=ROUND(2^D{fc_row}, 3))
        ws_a[f"D{fc_row}"] = f"=ROUND({col_let}26-{col_let}24, 3)"
        ws_a[f"C{fc_row}"] = f"=ROUND(2^D{fc_row}, 3)"
        log2fc = round(cached_cell_map[f"{col_let}26"] - cached_cell_map[f"{col_let}24"], 3)
        fc = round(2.0 ** log2fc, 3)
        cached_cell_map[f"D{fc_row}"] = log2fc
        cached_cell_map[f"C{fc_row}"] = fc

    wb_a.save(prot_cond_a)

    # Create Condition B by taking Condition A's exact zip archive and adding <v>cached_value</v>
    # to each formula cell <c r="..."><f>...</f></c> in xl/worksheets/sheet1.xml (Task sheet),
    # preserving 100% identical formula strings <f> while populating the cached evaluation <v>.
    with zipfile.ZipFile(prot_cond_a, "r") as zin, zipfile.ZipFile(prot_cond_b, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                xml_txt = data.decode("utf-8")
                def inject_cache(match):
                    full_c = match.group(0)
                    cell_ref = match.group(1)
                    if cell_ref in cached_cell_map and "<f>" in full_c:
                        val_str = repr(cached_cell_map[cell_ref])
                        if "<v></v>" in full_c:
                            return full_c.replace("<v></v>", f"<v>{val_str}</v>")
                        elif "<v>" not in full_c:
                            return full_c.replace("</f>", f"</f><v>{val_str}</v>")
                    return full_c
                xml_txt = re.sub(r'<c r="([A-Z]+\d+)"[^>]*>.*?</c>', inject_cache, xml_txt, flags=re.S)
                data = xml_txt.encode("utf-8")
            zout.writestr(item, data)

    # Verify that BOTH Condition A and Condition B contain identical formulas under data_only=False,
    # AND independently parse and evaluate every <f> formula string in Condition B from its referenced
    # sheet cells (Data and Task) to verify that the XML <v> cached values match independent evaluation.
    wb_a_f = openpyxl.load_workbook(prot_cond_a, data_only=False)["Task"]
    wb_b_f = openpyxl.load_workbook(prot_cond_b, data_only=False)["Task"]
    wb_b_v = openpyxl.load_workbook(prot_cond_b, data_only=True)["Task"]
    formulas_identical = all(
        wb_a_f.cell(r, c).value == wb_b_f.cell(r, c).value
        for r in range(1, 45) for c in range(1, 14)
    )
    assert formulas_identical, "Formulas must be 100% identical between Condition A and Condition B!"

    def evaluate_formula_independently(f_str, task_ws_v, data_ws_v):
        """Independently parses and evaluates the Excel formula string f_str against sheet cell values."""
        s = f_str.strip()
        # 1. Step 1 INDEX-MATCH: =INDEX(Data!$A$1:$BA$201, MATCH($A11, Data!$A$1:$A$201, 0), MATCH(C$10, Data!$A$1:$BA$1, 0))
        m_idx = re.match(
            r"^=INDEX\(Data!\$([A-Z]+)\$(\d+):\$([A-Z]+)\$(\d+),\s*"
            r"MATCH\(\$([A-Z]+)(\d+),\s*Data!\$([A-Z]+)\$(\d+):\$([A-Z]+)\$(\d+),\s*0\),\s*"
            r"MATCH\(([A-Z]+)\$(\d+),\s*Data!\$([A-Z]+)\$(\d+):\$([A-Z]+)\$(\d+),\s*0\)\)$",
            s,
        )
        if m_idx:
            c_start = openpyxl.utils.column_index_from_string(m_idx.group(1))
            r_start = int(m_idx.group(2))
            c_end = openpyxl.utils.column_index_from_string(m_idx.group(3))
            r_end = int(m_idx.group(4))
            key_row_ref = f"{m_idx.group(5)}{m_idx.group(6)}"
            key_col_ref = f"{m_idx.group(11)}{m_idx.group(12)}"
            target_row_key = task_ws_v[key_row_ref].value
            target_col_key = task_ws_v[key_col_ref].value
            matched_r = next(
                r for r in range(r_start, r_end + 1)
                if data_ws_v.cell(row=r, column=c_start).value == target_row_key
            )
            matched_c = next(
                c for c in range(c_start, c_end + 1)
                if data_ws_v.cell(row=r_start, column=c).value == target_col_key
            )
            raw_v = data_ws_v.cell(row=matched_r, column=matched_c).value
            return None if raw_v is None else float(raw_v)

        # 2. Step 2 AVERAGE: =ROUND(AVERAGE(Data!D15:H15), 3)
        m_avg = re.match(r"^=ROUND\(AVERAGE\(Data!([A-Z]+)(\d+):([A-Z]+)(\d+)\),\s*(\d+)\)$", s)
        if m_avg:
            c1 = openpyxl.utils.column_index_from_string(m_avg.group(1))
            r1 = int(m_avg.group(2))
            c2 = openpyxl.utils.column_index_from_string(m_avg.group(3))
            r2 = int(m_avg.group(4))
            ndig = int(m_avg.group(5))
            vals = [
                float(data_ws_v.cell(row=r, column=c).value)
                for r in range(r1, r2 + 1)
                for c in range(c1, c2 + 1)
                if data_ws_v.cell(row=r, column=c).value is not None
            ]
            return round(statistics.mean(vals), ndig)

        # 3. Step 2 STDEV with COUNT guard: =ROUND(IF(COUNT(Data!D15:H15)>1, STDEV(Data!D15:H15), 0), 3)
        m_std = re.match(
            r"^=ROUND\(IF\(COUNT\(Data!([A-Z]+)(\d+):([A-Z]+)(\d+)\)>1,\s*"
            r"STDEV\(Data!([A-Z]+)(\d+):([A-Z]+)(\d+)\),\s*0\),\s*(\d+)\)$",
            s,
        )
        if m_std:
            c1 = openpyxl.utils.column_index_from_string(m_std.group(1))
            r1 = int(m_std.group(2))
            c2 = openpyxl.utils.column_index_from_string(m_std.group(3))
            r2 = int(m_std.group(4))
            ndig = int(m_std.group(9))
            vals = [
                float(data_ws_v.cell(row=r, column=c).value)
                for r in range(r1, r2 + 1)
                for c in range(c1, c2 + 1)
                if data_ws_v.cell(row=r, column=c).value is not None
            ]
            return round(statistics.stdev(vals), ndig) if len(vals) > 1 else 0.0

        # 4. Step 3 Log2FC difference: =ROUND(B26-B24, 3)
        m_sub = re.match(r"^=ROUND\(([A-Z]+\d+)-([A-Z]+\d+),\s*(\d+)\)$", s)
        if m_sub:
            v1 = float(task_ws_v[m_sub.group(1)].value)
            v2 = float(task_ws_v[m_sub.group(2)].value)
            ndig = int(m_sub.group(3))
            return round(v1 - v2, ndig)

        # 5. Step 3 FC power: =ROUND(2^D32, 3)
        m_pow = re.match(r"^=ROUND\(2\^([A-Z]+\d+),\s*(\d+)\)$", s)
        if m_pow:
            exp_v = float(task_ws_v[m_pow.group(1)].value)
            ndig = int(m_pow.group(2))
            return round(2.0 ** exp_v, ndig)

        raise ValueError(f"Unrecognized formula syntax in independent evaluator: {s}")

    formula_cache_checked = 0
    cache_roundtrip_abs_diff = 0.0
    independent_formula_eval_vs_cache_abs_diff = 0.0

    # Order cells so Step 1 & Step 2 are evaluated before Step 3 (which references Step 2 and Col D)
    all_formula_coords = []
    for r in range(1, 45):
        for c in range(1, 14):
            f_val = wb_b_f.cell(r, c).value
            if isinstance(f_val, str) and f_val.startswith("="):
                all_formula_coords.append((r, c, f"{openpyxl.utils.get_column_letter(c)}{r}"))
    # Sort so row<=27 first, then column D (col 4) in rows 32..41 before column C (col 3)
    all_formula_coords.sort(key=lambda x: (0 if x[0] <= 27 else (1 if x[1] == 4 else 2), x[0], x[1]))

    for r, c, cell_ref in all_formula_coords:
        f_str = wb_b_f[cell_ref].value
        v_val = wb_b_v[cell_ref].value
        assert isinstance(f_str, str) and f_str.startswith("="), f"Missing formula in {cell_ref}"
        assert isinstance(v_val, (int, float)), f"Missing numeric cache in {cell_ref}: {v_val}"
        # (a) Roundtrip check against cached_cell_map
        rt_diff = abs(float(v_val) - float(cached_cell_map[cell_ref]))
        cache_roundtrip_abs_diff = max(cache_roundtrip_abs_diff, rt_diff)
        assert rt_diff <= 1e-12, f"Cache roundtrip mismatch at {cell_ref}: {v_val} vs {cached_cell_map[cell_ref]}"
        # (b) Independent formula parser & evaluator check against sheet values
        indep_val = evaluate_formula_independently(f_str, wb_b_v, data_ws_val)
        indep_diff = abs(float(v_val) - float(indep_val))
        independent_formula_eval_vs_cache_abs_diff = max(independent_formula_eval_vs_cache_abs_diff, indep_diff)
        assert indep_diff <= 1e-12, f"Independent formula evaluation mismatch at {cell_ref}: cache={v_val} vs eval={indep_val}"
        formula_cache_checked += 1

    assert formula_cache_checked == 143, f"Expected 143 formula cells, got {formula_cache_checked}"
    max_formula_cache_diff = independent_formula_eval_vs_cache_abs_diff

    # Run official verifier test_outputs.py assertions against Condition A and Condition B
    ver_code = prot_ver_py.read_text(encoding="utf-8")
    test_names = [
        "test_file_exists",
        "test_sheets_exist",
        "test_step1_expression_values_filled",
        "test_step2_statistics_calculated",
        "test_step3_fold_changes_calculated",
        "test_values_are_numeric",
    ]

    def run_protein_verifier(xlsx_path):
        ns = {"__file__": str(prot_ver_py)}
        patched_code = ver_code.replace('"/root/protein_expression.xlsx"', repr(str(xlsx_path)))
        exec(compile(patched_code, str(prot_ver_py), "exec"), ns)
        wb_loaded = openpyxl.load_workbook(xlsx_path, data_only=True)
        passed = []
        failed = {}
        for tname in test_names:
            try:
                fn = ns[tname]
                if fn.__code__.co_argcount > 0:
                    fn(wb_loaded)
                else:
                    fn()
                passed.append(tname)
            except Exception as e:
                failed[tname] = str(e)
        ws_check = wb_loaded["Task"]
        non_null_step1 = sum(1 for r in range(11, 21) for c in range(3, 13) if ws_check.cell(r, c).value is not None)
        non_null_step2 = sum(1 for r in range(24, 28) for c in range(2, 12) if isinstance(ws_check.cell(r, c).value, (int, float)))
        non_null_fc = sum(1 for r in range(32, 42) if isinstance(ws_check.cell(r, 3).value, (int, float)))
        non_null_log2fc = sum(1 for r in range(32, 42) if isinstance(ws_check.cell(r, 4).value, (int, float)))
        wb_form = openpyxl.load_workbook(xlsx_path, data_only=False)["Task"]
        formula_step1 = sum(1 for r in range(11, 21) for c in range(3, 13) if str(wb_form.cell(r, c).value or "").startswith("="))
        return {
            "passed_tests": len(passed),
            "total_tests": len(test_names),
            "failed_tests": failed,
            "formula_cells_step1_data_only_false": formula_step1,
            "non_null_step1_data_only_true": non_null_step1,
            "non_null_step2_data_only_true": non_null_step2,
            "non_null_fc_colC_data_only_true": non_null_fc,
            "non_null_log2fc_colD_data_only_true": non_null_log2fc,
            "official_verifier_reward": 1.0 if len(passed) == len(test_names) else 0.0,
        }

    prot_res_a = run_protein_verifier(prot_cond_a)
    prot_res_b = run_protein_verifier(prot_cond_b)
    print(f"  -> Formulas identical (data_only=False): {formulas_identical} (Step1={prot_res_a['formula_cells_step1_data_only_false']}/100 formula cells, {n_step1_non_null}/100 non-null lookups)")
    print(f"  -> Formula vs. Cache verification ({formula_cache_checked}/143 cells): independent_eval_diff={independent_formula_eval_vs_cache_abs_diff}, roundtrip_diff={cache_roundtrip_abs_diff}")
    print(f"  -> Condition A (Uncached formulas): {prot_res_a['passed_tests']}/{prot_res_a['total_tests']} passed, Reward={prot_res_a['official_verifier_reward']}")
    print(f"  -> Condition B (Cached formulas):   {prot_res_b['passed_tests']}/{prot_res_b['total_tests']} passed, Reward={prot_res_b['official_verifier_reward']}")

    results["protein-expression-analysis"] = {
        "verification_type": "Part_A_End_to_End_Official_Verifier_Single_Factor_Formula_Cache",
        "real_input_file": "tasks/protein-expression-analysis/environment/data/protein_expression.xlsx",
        "real_input_bytes": os.path.getsize(prot_src_xlsx),
        "control_sample_columns_in_data": f"{ctrl_start_let}:{ctrl_end_let}",
        "treated_sample_columns_in_data": f"{treat_start_let}:{treat_end_let}",
        "official_verifier": "tasks/protein-expression-analysis/verifier/test_outputs.py",
        "formulas_identical_across_conditions": formulas_identical,
        "n_total_formula_cells_verified": formula_cache_checked,
        "cache_roundtrip_abs_diff": cache_roundtrip_abs_diff,
        "independent_formula_eval_vs_cache_abs_diff": independent_formula_eval_vs_cache_abs_diff,
        "max_formula_vs_cache_abs_diff": max_formula_cache_diff,
        "n_populated_step1_cells": n_step1_non_null,
        "condition_A_uncached_formulas": prot_res_a,
        "condition_B_cached_formulas": prot_res_b,
        "historical_benchmark": get_task_lifts("protein-expression-analysis"),
    }

    # ==============================================================================
    # 2. EARTHQUAKE-PLATE-CALCULATION: Full End-to-End Search + Official Verifier
    # ==============================================================================
    print("[2/4] Running full end-to-end earthquake search & official verifier on real GeoJSONs...")
    from datetime import datetime, timezone

    eq_task_dir = TASKS_DIR / "earthquake-plate-calculation"
    eq_json_path = eq_task_dir / "environment" / "earthquakes_2024.json"
    plates_json_path = eq_task_dir / "environment" / "PB2002_plates.json"
    pb_json_path = eq_task_dir / "environment" / "PB2002_boundaries.json"
    eq_ver_path = eq_task_dir / "verifier" / "test_outputs.py"

    eq_raw = json.loads(eq_json_path.read_text(encoding="utf-8"))
    plates_raw = json.loads(plates_json_path.read_text(encoding="utf-8"))
    pb_raw = json.loads(pb_json_path.read_text(encoding="utf-8"))

    # Extract Pacific Plate polygon rings from PB2002_plates.json (PlateName == "Pacific")
    pacific_rings = []
    for feat in plates_raw["features"]:
        if feat.get("properties", {}).get("PlateName") == "Pacific":
            geom = feat.get("geometry", {})
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])
            polys = [coords] if gtype == "Polygon" else (coords if gtype == "MultiPolygon" else [])
            for poly in polys:
                if poly:
                    pacific_rings.append(np.asarray(poly[0], dtype=float))

    def point_in_ring(lon, lat, ring):
        x, y = ring[:, 0], ring[:, 1]
        x1, y1 = x[:-1], y[:-1]
        x2, y2 = x[1:], y[1:]
        cond = ((y1 > lat) != (y2 > lat)) & (lon < (x2 - x1) * (lat - y1) / np.where(y2 - y1 == 0, 1e-16, y2 - y1) + x1)
        return bool(np.sum(cond) % 2 == 1)

    def is_in_pacific_plate(lon, lat):
        return any(point_in_ring(lon, lat, ring) for ring in pacific_rings)

    # Extract all valid earthquakes and filter to Pacific Plate following oracle/solution.py
    all_eq_count = len(eq_raw["features"])
    eq_records = []
    for feat in eq_raw["features"]:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        if not geom or geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2 or coords[0] is None or coords[1] is None:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        if not is_in_pacific_plate(lon, lat):
            continue
        eq_id = feat.get("id") or props.get("id") or props.get("code")
        mag = props.get("mag")
        place = props.get("place")
        t_ms = props.get("time")
        if eq_id is None or mag is None:
            continue
        time_iso = datetime.fromtimestamp(float(t_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        eq_records.append({
            "id": str(eq_id),
            "place": str(place),
            "time": time_iso,
            "mag": float(mag),
            "lon": lon,
            "lat": lat,
        })

    # Extract all Pacific boundary line segments ("PA" in Name) from PB2002_boundaries.json following oracle/solution.py Line 84
    seg_starts = []
    seg_ends = []
    for feat in pb_raw["features"]:
        name = str(feat.get("properties", {}).get("Name", ""))
        if "PA" not in name:
            continue
        geom = feat.get("geometry", {})
        if not geom:
            continue
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        lines = [coords] if gtype == "LineString" else (coords if gtype == "MultiLineString" else [])
        for ln in lines:
            arr = np.asarray(ln, dtype=float)
            if len(arr) >= 2:
                seg_starts.append(arr[:-1, :2])
                seg_ends.append(arr[1:, :2])
    seg_A = np.vstack(seg_starts)  # (M, 2) [lon, lat]
    seg_B = np.vstack(seg_ends)    # (M, 2) [lon, lat]

    R_WGS84_KM = 6378.137
    R_SPHERE_KM = 6371.0

    def proj_epsg4087(lon_deg, lat_deg):
        return np.radians(lon_deg) * R_WGS84_KM, np.radians(lat_deg) * R_WGS84_KM

    def proj_epsg3857(lon_deg, lat_deg):
        lat_c = np.clip(lat_deg, -85.05112878, 85.05112878)
        x = np.radians(lon_deg) * R_WGS84_KM
        y = np.log(np.tan(np.pi / 4.0 + np.radians(lat_c) / 2.0)) * R_WGS84_KM
        return x, y

    def min_planar_dists_km(eq_x, eq_y, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay
        len2 = dx * dx + dy * dy
        len2 = np.where(len2 < 1e-16, 1e-16, len2)
        out = np.zeros(len(eq_x), dtype=float)
        for i in range(len(eq_x)):
            px, py = eq_x[i], eq_y[i]
            t = np.clip(((px - ax) * dx + (py - ay) * dy) / len2, 0.0, 1.0)
            cx = ax + t * dx
            cy = ay + t * dy
            out[i] = float(np.sqrt(np.min((px - cx) ** 2 + (py - cy) ** 2)))
        return out

    eq_lons = np.array([r["lon"] for r in eq_records], dtype=float)
    eq_lats = np.array([r["lat"] for r in eq_records], dtype=float)

    # Method A: Full search over all Pacific earthquakes in EPSG:4087 (disclosed only in geospatial-analysis/SKILL.md Line 20)
    eqx_4087, eqy_4087 = proj_epsg4087(eq_lons, eq_lats)
    ax_4087, ay_4087 = proj_epsg4087(seg_A[:, 0], seg_A[:, 1])
    bx_4087, by_4087 = proj_epsg4087(seg_B[:, 0], seg_B[:, 1])
    dists_4087_km = min_planar_dists_km(eqx_4087, eqy_4087, ax_4087, ay_4087, bx_4087, by_4087)
    idx_max_4087 = int(np.argmax(dists_4087_km))
    row_4087 = eq_records[idx_max_4087]
    dist_4087_val = round(float(dists_4087_km[idx_max_4087]), 2)

    # Method C: Full search over all Pacific earthquakes in EPSG:3857 (Standard Web Mercator)
    eqx_3857, eqy_3857 = proj_epsg3857(eq_lons, eq_lats)
    ax_3857, ay_3857 = proj_epsg3857(seg_A[:, 0], seg_A[:, 1])
    bx_3857, by_3857 = proj_epsg3857(seg_B[:, 0], seg_B[:, 1])
    dists_3857_km = min_planar_dists_km(eqx_3857, eqy_3857, ax_3857, ay_3857, bx_3857, by_3857)
    idx_max_3857 = int(np.argmax(dists_3857_km))
    row_3857 = eq_records[idx_max_3857]
    dist_3857_val = round(float(dists_3857_km[idx_max_3857]), 2)

    # Method B: Full search over all Pacific earthquakes using spherical great-circle point-to-segment distance (R = 6371.0 km)
    def lonlat_to_unit_xyz(lon_deg, lat_deg):
        lon = np.radians(lon_deg)
        lat = np.radians(lat_deg)
        clat = np.cos(lat)
        return np.stack([clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)], axis=-1)

    A_xyz = lonlat_to_unit_xyz(seg_A[:, 0], seg_A[:, 1])
    B_xyz = lonlat_to_unit_xyz(seg_B[:, 0], seg_B[:, 1])
    AB_vec = B_xyz - A_xyz
    AB_len2 = np.sum(AB_vec * AB_vec, axis=1)
    AB_len2 = np.where(AB_len2 < 1e-16, 1e-16, AB_len2)
    P_xyz = lonlat_to_unit_xyz(eq_lons, eq_lats)

    min_spherical_dists_km = np.zeros(len(eq_records), dtype=float)
    for i in range(len(eq_records)):
        p = P_xyz[i]
        t = np.clip(np.sum((p[None, :] - A_xyz) * AB_vec, axis=1) / AB_len2, 0.0, 1.0)
        C_xyz = A_xyz + t[:, None] * AB_vec
        C_norm = np.linalg.norm(C_xyz, axis=1, keepdims=True)
        C_unit = C_xyz / np.where(C_norm < 1e-15, 1.0, C_norm)
        dots = np.clip(np.sum(C_unit * p[None, :], axis=1), -1.0, 1.0)
        min_spherical_dists_km[i] = float(np.min(np.arccos(dots)) * R_SPHERE_KM)

    idx_max_sph = int(np.argmax(min_spherical_dists_km))
    row_sph = eq_records[idx_max_sph]
    dist_sph_val = round(float(min_spherical_dists_km[idx_max_sph]), 2)

    # Run all 8 official unittest assertions from tasks/earthquake-plate-calculation/verifier/test_outputs.py
    eq_ver_code = eq_ver_path.read_text(encoding="utf-8")
    eq_test_names = [
        "test_output_file_and_structure",
        "test_earthquake_id_correct",
        "test_place_correct",
        "test_time_correct",
        "test_magnitude_within_tolerance",
        "test_latitude_within_tolerance",
        "test_longitude_within_tolerance",
        "test_distance_within_tolerance",
    ]

    def run_eq_official_verifier(row_obj, dist_val, tmp_json_name):
        out_path = work_dir / tmp_json_name
        payload = {
            "id": str(row_obj["id"]),
            "place": str(row_obj["place"]),
            "time": str(row_obj["time"]),
            "magnitude": float(row_obj["mag"]),
            "latitude": float(row_obj["lat"]),
            "longitude": float(row_obj["lon"]),
            "distance_km": float(dist_val),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ns = {}
        patched = eq_ver_code.replace('paths = ["/root/answer.json", "answer.json"]', f'paths = [{repr(str(out_path))}]')
        exec(compile(patched, str(eq_ver_path), "exec"), ns)
        test_cls = ns["TestEarthquakeTask"]
        test_cls.setUpClass()
        inst = test_cls()
        passed = []
        failed = {}
        for tname in eq_test_names:
            try:
                getattr(inst, tname)()
                passed.append(tname)
            except Exception as e:
                failed[tname] = str(e)
        return {
            "farthest_id_found": str(row_obj["id"]),
            "distance_km": float(dist_val),
            "abs_error_vs_verifier_3878_27_km": round(abs(float(dist_val) - 3878.27), 2),
            "tolerance_multiple": round(abs(float(dist_val) - 3878.27) / 0.01, 1),
            "passed_tests": len(passed),
            "total_tests": len(eq_test_names),
            "failed_tests": failed,
            "official_verifier_reward": 1.0 if len(passed) == len(eq_test_names) else 0.0,
        }

    eq_res_4087 = run_eq_official_verifier(row_4087, dist_4087_val, "farthest_band_epsg4087.json")
    eq_res_sph = run_eq_official_verifier(row_sph, dist_sph_val, "farthest_band_spherical.json")
    eq_res_3857 = run_eq_official_verifier(row_3857, dist_3857_val, "farthest_band_epsg3857.json")

    print(f"  -> Full search ({all_eq_count} global -> {len(eq_records)} Pacific plate earthquakes x {len(seg_A)} PA boundary segments) found farthest ID: {eq_res_4087['farthest_id_found']}")
    print(f"  -> Method A (EPSG:4087): {eq_res_4087['distance_km']:.2f} km | {eq_res_4087['passed_tests']}/8 tests pass | Reward={eq_res_4087['official_verifier_reward']}")
    print(f"  -> Method B (Spherical Great-Circle): {eq_res_sph['distance_km']:.2f} km | {eq_res_sph['passed_tests']}/8 tests pass | Reward={eq_res_sph['official_verifier_reward']}")
    print(f"  -> Method C (EPSG:3857): {eq_res_3857['distance_km']:.2f} km | {eq_res_3857['passed_tests']}/8 tests pass | Reward={eq_res_3857['official_verifier_reward']}")

    results["earthquake-plate-calculation"] = {
        "verification_type": "Part_A_End_to_End_Search_and_Official_Verifier_Projection_Sensitivity",
        "real_input_files": [
            "tasks/earthquake-plate-calculation/environment/earthquakes_2024.json",
            "tasks/earthquake-plate-calculation/environment/PB2002_plates.json",
            "tasks/earthquake-plate-calculation/environment/PB2002_boundaries.json",
        ],
        "n_global_earthquakes": all_eq_count,
        "n_pacific_plate_earthquakes_searched": len(eq_records),
        "n_pa_boundary_segments_searched": len(seg_A),
        "official_verifier": "tasks/earthquake-plate-calculation/verifier/test_outputs.py",
        "verifier_target_km": 3878.27,
        "verifier_tolerance_km": 0.01,
        "method_A_epsg4087_skill_secret": eq_res_4087,
        "method_B_exact_spherical_great_circle": eq_res_sph,
        "method_C_epsg3857_web_mercator": eq_res_3857,
        "historical_benchmark": get_task_lifts("earthquake-plate-calculation"),
    }

    # ==============================================================================
    # 3. MARS-CLOUDS-CLUSTERING: Real Dataset Weighted 2D Euclidean Distance Timing
    # ==============================================================================
    print("[3/4] Running weighted 2D Euclidean distance benchmark on real citsci_train.csv (46.2 MB)...")
    mars_dir = TASKS_DIR / "mars-clouds-clustering"
    citsci_csv = mars_dir / "environment" / "data" / "citsci_train.csv"
    expert_csv = mars_dir / "environment" / "data" / "expert_train.csv"

    df_cit = pd.read_csv(citsci_csv)
    df_exp = pd.read_csv(expert_csv)

    # Follow oracle/solve.py Lines 19-23 and 138-142 exactly:
    # Group by unique base images in expert_train.csv['file_rad'] (369 images),
    # and compute weighted 2D Euclidean distance d(a, b) = sqrt((w * dx)^2 + ((2 - w) * dy)^2) on [x, y]
    shape_weight = 1.5
    weights_2d = np.array([shape_weight, 2.0 - shape_weight], dtype=float)
    unique_images = df_exp["file_rad"].unique()
    grouped_pts = [
        df_cit[df_cit["file_rad"] == img][["x", "y"]].to_numpy(dtype=float)
        for img in unique_images
    ]
    non_empty_pts = [pts for pts in grouped_pts if len(pts) > 0]

    # Exact Python closure pattern taught in custom-distance-metrics/SKILL.md Lines 30-36 & 74-79
    def create_weighted_distance(weight_x, weight_y):
        def distance(a, b):
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return math.sqrt((weight_x * dx) ** 2 + (weight_y * dy) ** 2)
        return distance

    skill_distance_fn = create_weighted_distance(shape_weight, 2.0 - shape_weight)

    mars_trial_rows = []
    oracle_mats = []
    skill_mats = []
    for trial_idx in range(1, max(1, int(mars_trials)) + 1):
        t0 = time.perf_counter()
        cur_ora = []
        for pts in non_empty_pts:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            weighted_diff = diff * weights_2d
            dist_mat = np.sqrt((weighted_diff ** 2).sum(axis=2))
            cur_ora.append(dist_mat)
        t_ora_i = time.perf_counter() - t0

        t0 = time.perf_counter()
        cur_sk = []
        for pts in non_empty_pts:
            dist_mat = squareform(pdist(pts, metric=skill_distance_fn)) if len(pts) > 1 else np.zeros((len(pts), len(pts)))
            cur_sk.append(dist_mat)
        t_sk_i = time.perf_counter() - t0

        if trial_idx == 1:
            oracle_mats = cur_ora
            skill_mats = cur_sk
        mars_trial_rows.append({
            "run_id": run_id,
            "trial_index": trial_idx,
            "oracle_sec": round(t_ora_i, 6),
            "skill_sec": round(t_sk_i, 6),
            "pairwise_slowdown_x": round(t_sk_i / max(1e-9, t_ora_i), 4),
        })

    mars_csv_out = out_dir / "mars_live_timing_trials.csv"
    with open(mars_csv_out, "w", encoding="utf-8", newline="") as _mf:
        _mw = csv.DictWriter(_mf, fieldnames=["run_id", "trial_index", "oracle_sec", "skill_sec", "pairwise_slowdown_x"])
        _mw.writeheader()
        _mw.writerows(mars_trial_rows)

    t_oracle_single = float(np.median([r["oracle_sec"] for r in mars_trial_rows]))
    t_skill_single = float(np.median([r["skill_sec"] for r in mars_trial_rows]))
    pairwise_median_ratio = float(np.median([r["pairwise_slowdown_x"] for r in mars_trial_rows]))
    max_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(oracle_mats, skill_mats))
    speedup_ratio = t_skill_single / max(1e-9, t_oracle_single)
    n_grid_points = 7 * 11 * 11  # 847 hyperparameter combinations in oracle/solve.py Lines 146-150

    print(f"  -> {len(unique_images)} real images ({len(df_cit)} annotations): Oracle 3D NumPy={t_oracle_single:.4f}s vs. Skill Python callable={t_skill_single:.4f}s ({speedup_ratio:.1f}x slower, max diff={max_diff:.2e})")
    print(f"  -> Linear extrapolation across {n_grid_points} grid points (distance matrix only): {t_oracle_single*n_grid_points:.1f}s vs. {t_skill_single*n_grid_points:.1f}s ({t_skill_single*n_grid_points/60.0:.1f} min)")

    results["mars-clouds-clustering"] = {
        "verification_type": "Part_B_Real_Dataset_Distance_Computation_Benchmark_and_Grid_Extrapolation",
        "real_input_files": [
            "tasks/mars-clouds-clustering/environment/data/citsci_train.csv",
            "tasks/mars-clouds-clustering/environment/data/expert_train.csv",
        ],
        "citsci_bytes": os.path.getsize(citsci_csv),
        "n_unique_images": len(unique_images),
        "n_non_empty_images": len(non_empty_pts),
        "n_citsci_annotations": len(df_cit),
        "n_expert_annotations": len(df_exp),
        "distance_metric": "Weighted 2D Euclidean distance d(a,b) = sqrt((w*dx)^2 + ((2-w)*dy)^2) on [x, y] (oracle/solve.py Lines 19-23 vs. custom-distance-metrics/SKILL.md Lines 30-36)",
        "oracle_numpy_single_pass_sec": round(t_oracle_single, 4),
        "skill_callable_single_pass_sec": round(t_skill_single, 4),
        "single_pass_slowdown_factor": round(speedup_ratio, 1),
        "max_numerical_difference": float(f"{max_diff:.2e}"),
        "hyperparameter_grid_size": n_grid_points,
        "extrapolated_847_grid_oracle_distance_sec": round(t_oracle_single * n_grid_points, 1),
        "extrapolated_847_grid_skill_distance_sec": round(t_skill_single * n_grid_points, 1),
        "extrapolated_847_grid_skill_distance_min": round(t_skill_single * n_grid_points / 60.0, 1),
        "extrapolation_note": "Linear extrapolation of pairwise distance matrix evaluation across the 847 (7x11x11) grid combinations; excludes DBSCAN clustering and post-processing overhead.",
        "historical_benchmark": get_task_lifts("mars-clouds-clustering"),
    }

    # ==============================================================================
    # 4. EXAM-BLOCK-SEQUENCING: Exact Analytical MIP Subproblem Scale Accounting
    # ==============================================================================
    print("[4/4] Computing exact analytical MIP variable & constraint counts on real exam-block-sequencing inputs...")
    exam_dir = TASKS_DIR / "exam-block-sequencing"
    inst_json = json.loads((exam_dir / "environment" / "data" / "instance.json").read_text(encoding="utf-8"))
    pair_rows = list(csv.DictReader(open(exam_dir / "environment" / "data" / "pair_counts.csv", encoding="utf-8")))
    triplet_rows = list(csv.DictReader(open(exam_dir / "environment" / "data" / "triplet_counts.csv", encoding="utf-8")))
    oracle_metrics = json.loads((exam_dir / "verifier" / "oracle_metrics.json").read_text(encoding="utf-8"))

    n_blocks = len(inst_json["all_blocks"])
    n_slots = len(inst_json["all_blocks"])
    n_triple_windows = len(inst_json["triple_day_start"]) + len(inst_json["triple_24_start"])  # 8 + 14 = 22 ordered triple-window starts

    nonzero_pairs = [r for r in pair_rows if float(r.get("count", 0)) > 0]
    nonzero_triplets_all = [r for r in triplet_rows if float(r.get("count", 0)) > 0]
    # Filter to triplets with 3 distinct blocks (i != j != k)
    distinct_block_triplets = [
        r for r in nonzero_triplets_all
        if len({r["block_i"], r["block_j"], r["block_k"]}) == 3
    ]

    n_assign_vars = n_blocks * n_slots  # 24 * 24 = 576 binary assignment variables x[b,t]
    # Ordered triplet-window variables w[t,i,j,k] across 22 window starts (no *6 factor since CSV is already ordered!)
    n_win_vars_distinct = len(distinct_block_triplets) * n_triple_windows  # 11,130 * 22 = 244,860
    n_total_vars_distinct = n_assign_vars + n_win_vars_distinct            # 245,436
    n_lin_constrs_distinct = 4 * n_win_vars_distinct                       # 4 * 244,860 = 979,440

    n_win_vars_all_csv = len(nonzero_triplets_all) * n_triple_windows      # 12,415 * 22 = 273,130
    n_total_vars_all_csv = n_assign_vars + n_win_vars_all_csv              # 273,706
    n_lin_constrs_all_csv = 4 * n_win_vars_all_csv                         # 4 * 273,130 = 1,092,520

    print(f"  -> Real instance: {n_blocks} blocks, {n_slots} slots, {len(nonzero_pairs)} ordered pairs, {len(nonzero_triplets_all)} CSV triplets ({len(distinct_block_triplets)} distinct-block triplets)")
    print(f"  -> Distinct-block x+w MIP subproblem: {n_total_vars_distinct:,} binary vars ({n_assign_vars} x + {n_win_vars_distinct:,} w) & {n_lin_constrs_distinct:,} triplet linearization constraints (4 per w)")
    print(f"  -> All-CSV-row x+w MIP subproblem:    {n_total_vars_all_csv:,} binary vars ({n_assign_vars} x + {n_win_vars_all_csv:,} w) & {n_lin_constrs_all_csv:,} triplet linearization constraints")
    print(f"  -> Official oracle_metrics.json: solver_status={oracle_metrics['solver_status']}, objective={oracle_metrics['objective']}, MIP vars=0")

    results["exam-block-sequencing"] = {
        "verification_type": "Part_B_Real_Instance_Analytical_MIP_Subproblem_Scale_and_Oracle_Comparison",
        "real_input_files": [
            "tasks/exam-block-sequencing/environment/data/instance.json",
            "tasks/exam-block-sequencing/environment/data/pair_counts.csv",
            "tasks/exam-block-sequencing/environment/data/triplet_counts.csv",
            "tasks/exam-block-sequencing/verifier/oracle_metrics.json",
        ],
        "n_blocks": n_blocks,
        "n_slots": n_slots,
        "n_triple_windows": n_triple_windows,
        "n_nonzero_ordered_pairs": len(nonzero_pairs),
        "n_csv_ordered_triplets_total": len(nonzero_triplets_all),
        "n_csv_ordered_triplets_distinct_blocks": len(distinct_block_triplets),
        "assignment_binary_vars_x": n_assign_vars,
        "distinct_block_triplet_window_vars_w": n_win_vars_distinct,
        "distinct_block_total_xw_binary_vars": n_total_vars_distinct,
        "distinct_block_triplet_linearization_constraints_4x": n_lin_constrs_distinct,
        "all_csv_triplet_window_vars_w": n_win_vars_all_csv,
        "all_csv_total_xw_binary_vars": n_total_vars_all_csv,
        "all_csv_triplet_linearization_constraints_4x": n_lin_constrs_all_csv,
        "official_oracle_solver_status": oracle_metrics["solver_status"],
        "official_oracle_best_bound": oracle_metrics["solver_best_bound"],
        "official_oracle_objective": oracle_metrics["objective"],
        "official_oracle_mip_vars": 0,
        "historical_benchmark": get_task_lifts("exam-block-sequencing"),
    }


    results["mars-clouds-clustering"]["run_id"] = run_id
    results["mars-clouds-clustering"]["n_logged_trials"] = len(mars_trial_rows)
    results["mars-clouds-clustering"]["pairwise_median_slowdown_factor"] = round(pairwise_median_ratio, 1)
    results["mars-clouds-clustering"]["trials_csv"] = str(mars_csv_out.relative_to(out_dir).as_posix())

    out_json = out_dir / "controlled_mechanism_verification.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    meta_json = out_dir / "run_metadata.json"
    meta_json.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "script": "scripts/run_controlled_task_verification.py",
                "out_dir": str(out_dir),
                "work_dir": str(work_dir),
                "mars_trials": len(mars_trial_rows),
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "openpyxl": openpyxl.__version__,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved {out_json} (intermediate files isolated in {work_dir})")
    return results


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    default_run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    default_data = Path(os.environ.get("SKILLSBENCH_AUDIT_DIR", str(DEFAULT_DATA_DIR)))

    parser = argparse.ArgumentParser(
        description="Run controlled mechanism verifications in an isolated output/work directory."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--run-id", type=str, default=default_run_id)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Isolated output directory (default: artifacts/runs/controlled_<run_id>).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Isolated directory for generated .xlsx and intermediate .json files (default: <out-dir>/work).",
    )
    parser.add_argument(
        "--mars-trials",
        type=int,
        default=7,
        help="Number of logged timing trials on mars-clouds-clustering (default: 7).",
    )
    parser.add_argument(
        "--allow-overwrite-canonical",
        action="store_true",
        help="Explicitly allow writing into canonical artifacts/ directory.",
    )
    args = parser.parse_args(argv)

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (CANONICAL_ARTIFACTS_DIR / "runs" / f"controlled_{args.run_id}").resolve()
    )
    work_dir = args.work_dir.resolve() if args.work_dir is not None else (out_dir / "work").resolve()

    _check_safe_write_dir(out_dir, args.allow_overwrite_canonical)
    _check_safe_write_dir(work_dir, args.allow_overwrite_canonical)

    run_controlled_verification(
        data_dir=args.data_dir.resolve(),
        out_dir=out_dir,
        work_dir=work_dir,
        run_id=args.run_id,
        mars_trials=args.mars_trials,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
