#!/usr/bin/env python3
"""
Shared Provenance & Audit Infrastructure (`scripts/provenance_common.py`)

Enforces the unified provenance contract across all modular scripts (A01, S01–S24):
  1. Explicit CLI input/output paths; raw inputs opened strictly read-only and verified via pre/post SHA256.
  2. Separation of unrounded `value` from presentation `formatted_value` / `format_spec`.
  3. Standardized `metrics.json` and `run_manifest.json` schemas with `metric_id`, `value`,
     `formatted_value`, `format_spec`, `unit`, `scope`, `denominator`, `rule`, `evidence_file`,
     `source_hashes`, `script_version`, `config_hash`, and `run_id`.
  4. Conservation and invariant check logging without using target paper numbers to filter data.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_file_bytes_and_newlines(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    size_b = len(raw)
    stat_b = path.stat().st_size
    if size_b != stat_b:
        raise RuntimeError(f"Byte size mismatch on {path}: read_bytes={size_b} vs stat={stat_b}")

    sha = sha256_bytes(raw)
    mtime_utc = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).isoformat()

    if size_b == 0:
        line_fmt = "EMPTY"
        crlf_count = 0
        lf_only_count = 0
    elif b"\x00" in raw[:8192]:
        line_fmt = "BINARY"
        crlf_count = 0
        lf_only_count = 0
    else:
        crlf_count = raw.count(b"\r\n")
        total_lf = raw.count(b"\n")
        lf_only_count = total_lf - crlf_count
        if crlf_count > 0 and lf_only_count == 0:
            line_fmt = "CRLF"
        elif lf_only_count > 0 and crlf_count == 0:
            line_fmt = "LF"
        elif crlf_count > 0 and lf_only_count > 0:
            line_fmt = "MIXED"
        else:
            line_fmt = "SINGLE_LINE_NO_EOL"

    return {
        "path": str(path),
        "bytes": size_b,
        "sha256": sha,
        "mtime_utc": mtime_utc,
        "line_ending_format": line_fmt,
        "crlf_count": crlf_count,
        "lf_only_count": lf_only_count,
    }


inspect_file_bytes = inspect_file_bytes_and_newlines


def get_script_version(script_path: Path) -> Dict[str, str]:
    script_sha = sha256_file(script_path)
    git_commit = "unknown"
    try:
        res = subprocess.run(
            ["git", "-C", str(script_path.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            git_commit = res.stdout.strip()
    except Exception:
        pass
    return {
        "script_name": script_path.name,
        "script_sha256": script_sha,
        "git_commit": git_commit,
        "version_tag": f"{git_commit[:12]}:{script_sha[:12]}",
    }


def compute_config_hash(config: Dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(canonical.encode("utf-8"))


def format_metric_value(value: Any, format_spec: str) -> str:
    if value is None or value == "NA" or value == "N/A":
        return "N/A"
    if not format_spec:
        return str(value)
    return format(value, format_spec)


DEFAULT_PROVENANCE_TIERS: Dict[str, str] = {
    "A01": "LOCAL_SNAPSHOT_VERIFIED",
    "A02": "LOCAL_SNAPSHOT_VERIFIED",
    "S01": "RAW_REPO_DETERMINISTIC_COMPUTATION",
    "S02": "FROZEN_HISTORICAL_SNAPSHOT_COMPUTATION",
    "S03": "FROZEN_HISTORICAL_SNAPSHOT_COMPUTATION",
    "S04": "FROZEN_HISTORICAL_SNAPSHOT_COMPUTATION",
    "S05": "FROZEN_HISTORICAL_SNAPSHOT_COMPUTATION",
    "S06": "FROZEN_SNAPSHOT_PLUS_AUDITED_TAXONOMY_COMPUTATION",
    "S07": "RAW_REPO_DETERMINISTIC_COMPUTATION",
    "S08": "RAW_REPO_DETERMINISTIC_COMPUTATION",
    "S09": "FROZEN_HISTORICAL_SNAPSHOT_COMPUTATION",
    "S10": "RAW_REPO_DETERMINISTIC_COMPUTATION",
    "S11": "FROZEN_SNAPSHOT_PLUS_AUDITED_TAXONOMY_COMPUTATION",
    "S12": "EXECUTED_CONFIG_AND_AUDITED_DECLARATIONS",
    "S13": "FROZEN_HISTORICAL_SNAPSHOT_COMPUTATION",
    "S14": "FROZEN_HISTORICAL_SNAPSHOT_COMPUTATION",
    "S15": "FROZEN_SNAPSHOT_PLUS_AUDITED_TAXONOMY_COMPUTATION",
    "S16": "FROZEN_SNAPSHOT_PLUS_AUDITED_TAXONOMY_COMPUTATION",
    "S17": "FROZEN_SNAPSHOT_PLUS_AUDITED_TAXONOMY_COMPUTATION",
    "S18": "FROZEN_SNAPSHOT_PLUS_AUDITED_TAXONOMY_COMPUTATION",
    "S19": "FROZEN_HISTORICAL_SNAPSHOT_COMPUTATION",
    "S20": "FROZEN_HISTORICAL_SNAPSHOT_COMPUTATION",
    "S21": "RAW_REPO_DETERMINISTIC_COMPUTATION",
    "S22": "RAW_REPO_DETERMINISTIC_COMPUTATION",
    "S23": "LIVE_BENCHMARK_LOGGED_BATCH",
    "S24": "RAW_REPO_DETERMINISTIC_COMPUTATION",
    "R01": "REGISTERED_UNVERIFIED_EXTERNAL_LITERATURE",
}


class ModuleRecorder:
    def __init__(
        self,
        module_id: str,
        script_path: Path,
        out_dir: Path,
        config: Dict[str, Any],
        cli_args: Dict[str, Any],
        provenance_tier: Optional[str] = None,
    ):
        self.module_id = module_id
        self.script_path = script_path.resolve()
        self.out_dir = out_dir.resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.config_hash = compute_config_hash(config)
        self.cli_args = cli_args
        self.provenance_tier = provenance_tier or DEFAULT_PROVENANCE_TIERS.get(module_id, "DETERMINISTIC_COMPUTATION")
        self.script_version = get_script_version(self.script_path)
        now_utc = dt.datetime.now(dt.timezone.utc)
        self.timestamp_utc = now_utc.isoformat()
        ts_compact = now_utc.strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{module_id}-{ts_compact}-{self.script_version['script_sha256'][:8]}-{self.config_hash[:8]}"

        self.input_hashes_pre: Dict[str, Dict[str, Any]] = {}
        self.source_hashes_short: Dict[str, str] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.conservation_checks: List[Dict[str, Any]] = []
        self.output_files: List[Dict[str, Any]] = []

    def register_input_file(self, label: str, path: Path) -> Dict[str, Any]:
        p = path.resolve()
        if not p.exists():
            raise FileNotFoundError(f"[{self.module_id}] Missing required input file: {p}")
        info = inspect_file_bytes_and_newlines(p)
        self.input_hashes_pre[label] = info
        self.source_hashes_short[label] = info["sha256"]
        return info

    def register_input_dir_summary(self, label: str, dir_path: Path, file_count: int, total_bytes: int, tree_sha256: str):
        p = dir_path.resolve()
        self.input_hashes_pre[label] = {
            "path": str(p),
            "file_count": file_count,
            "bytes": total_bytes,
            "sha256": tree_sha256,
        }
        self.source_hashes_short[label] = tree_sha256

    def add_metric(
        self,
        metric_id: str,
        value: Any,
        unit: str,
        scope: str,
        denominator: Any,
        rule: str,
        evidence_file: str,
        format_spec: str = "",
        formatted_override: Optional[str] = None,
        notes: str = "",
    ) -> Any:
        full_id = metric_id if metric_id.startswith(f"{self.module_id}.") else f"{self.module_id}.{metric_id}"
        fmt_val = formatted_override if formatted_override is not None else format_metric_value(value, format_spec)
        ev_slash = str(evidence_file).replace("\\", "/")
        ev_norm = Path(ev_slash).name if ("artifacts/provenance/" in ev_slash or "artifacts/provenance_v2/" in ev_slash) else str(evidence_file)
        self.metrics[full_id] = {
            "metric_id": full_id,
            "module_id": self.module_id,
            "value": value,
            "formatted_value": fmt_val,
            "format_spec": format_spec,
            "unit": unit,
            "scope": scope,
            "denominator": denominator,
            "rule": rule,
            "evidence_file": ev_norm,
            "notes": notes,
            "source_hashes": dict(self.source_hashes_short),
            "script_version": self.script_version["version_tag"],
            "config_hash": self.config_hash,
            "run_id": self.run_id,
        }
        return value

    def check_conservation(self, check_id: str, description: str, lhs: Any, rhs: Any, tol: float = 0.0):
        if isinstance(lhs, float) or isinstance(rhs, float):
            passed = abs(float(lhs) - float(rhs)) <= tol
        else:
            passed = (lhs == rhs)
        rec = {
            "check_id": check_id,
            "description": description,
            "lhs": lhs,
            "rhs": rhs,
            "tolerance": tol,
            "passed": bool(passed),
        }
        self.conservation_checks.append(rec)
        if not passed:
            raise AssertionError(
                f"[{self.module_id}] Conservation check FAILED ({check_id}): {description} | LHS={lhs} vs RHS={rhs}"
            )

    def write_csv(self, filename: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> Path:
        out_path = self.out_dir / filename
        if fieldnames is None:
            fieldnames = list(rows[0].keys()) if rows else []
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        info = inspect_file_bytes_and_newlines(out_path)
        info["filename"] = filename
        info["row_count"] = len(rows)
        self.output_files.append(info)
        return out_path

    def write_json(self, filename: str, payload: Any) -> Path:
        out_path = self.out_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        info = inspect_file_bytes_and_newlines(out_path)
        info["filename"] = filename
        self.output_files.append(info)
        return out_path

    def finalize(self) -> Tuple[Path, Path]:
        # Verify all registered input files remained strictly untouched (read-only guarantee)
        for label, pre in self.input_hashes_pre.items():
            p = Path(pre["path"])
            if p.is_file():
                post_sha = sha256_file(p)
                if post_sha != pre["sha256"]:
                    raise RuntimeError(f"[{self.module_id}] Read-only violation! Input file modified: {p}")

        metrics_path = self.write_json(
            "metrics.json",
            {
                "module_id": self.module_id,
                "run_id": self.run_id,
                "provenance_tier": self.provenance_tier,
                "script_version": self.script_version,
                "config": self.config,
                "config_hash": self.config_hash,
                "metrics_count": len(self.metrics),
                "metrics": self.metrics,
            },
        )

        manifest_path = self.out_dir / "run_manifest.json"
        all_passed = all(c["passed"] for c in self.conservation_checks)
        manifest_payload = {
            "module_id": self.module_id,
            "run_id": self.run_id,
            "provenance_tier": self.provenance_tier,
            "status": "PASS" if all_passed else "FAIL",
            "metrics_count": len(self.metrics),
            "timestamp_utc": self.timestamp_utc,
            "script_path": str(self.script_path),
            "script_version": self.script_version,
            "cli_args": self.cli_args,
            "config": self.config,
            "config_hash": self.config_hash,
            "inputs": self.input_hashes_pre,
            "inputs_read_only_verified": True,
            "conservation_checks": self.conservation_checks,
            "all_conservation_checks_passed": all_passed,
            "outputs": self.output_files,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2, ensure_ascii=False)
        return metrics_path, manifest_path


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> Path:
    out_path = path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_json(path: Path, payload: Any) -> Path:
    out_path = path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


class ProvenanceRecorder(ModuleRecorder):
    """Upper-camel-case adapter over ModuleRecorder used across Batch 2 provenance scripts."""

    def __init__(
        self,
        group_id: str,
        script_path: Path,
        out_dir: Path,
        config_dict: Dict[str, Any],
        cli_args: Optional[Dict[str, Any]] = None,
        provenance_tier: Optional[str] = None,
    ):
        super().__init__(
            module_id=group_id,
            script_path=script_path,
            out_dir=out_dir,
            config=config_dict,
            cli_args=cli_args or {"out_dir": str(out_dir.resolve())},
            provenance_tier=provenance_tier,
        )

    def RegisterInput(self, label: str, path: Path) -> Dict[str, Any]:
        return self.register_input_file(label, path)

    def RegisterOutput(self, path: Path, row_count: Optional[int] = None) -> Dict[str, Any]:
        p = path.resolve()
        info = inspect_file_bytes_and_newlines(p)
        info["filename"] = p.name
        if row_count is not None:
            info["row_count"] = row_count
        self.output_files.append(info)
        return info

    def CheckConservation(
        self,
        name: str,
        lhs: Any,
        rhs: Any,
        details: str = "",
        tolerance: float = 0.0,
    ) -> None:
        self.check_conservation(
            check_id=name,
            description=details or name,
            lhs=lhs,
            rhs=rhs,
            tol=tolerance,
        )

    def AddMetric(
        self,
        metric_id: str,
        value: Any,
        formatted_value: str,
        unit: str,
        scope: str,
        denominator: Any,
        evidence_file: str,
        rule: str,
        format_spec: str = "",
        notes: str = "",
    ) -> Any:
        return self.add_metric(
            metric_id=metric_id,
            value=value,
            unit=unit,
            scope=scope,
            denominator=denominator,
            rule=rule,
            evidence_file=evidence_file,
            format_spec=format_spec,
            formatted_override=str(formatted_value),
            notes=notes,
        )

    def Finalize(self) -> Dict[str, Any]:
        _, manifest_path = self.finalize()
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

