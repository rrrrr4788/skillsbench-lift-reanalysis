"""Check the analysis inputs in a checkout against the files committed in the paper's repository.

Usage:
    python check_inputs.py REPO_DIR
    python check_inputs.py REPO_DIR --write     (maintainers: rebuild repo_snapshot/SOURCE.json)

REPO_DIR is a checkout of the repository, or the repo_snapshot directory of this
package. For each of the seven files the analysis reads or imports, the script
prints the SHA-256 of the file on disk, and the SHA-256 and git blob id after
line endings are normalized to LF, and compares them with repo_snapshot/SOURCE.json.

A Windows checkout with core.autocrlf=true stores the same text with CRLF line
endings, so its on-disk hash differs from the committed one while the normalized
hash matches. Python reads both forms identically; the results do not change.

Exit status is 0 when every file has the committed content and 1 otherwise.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "repo_snapshot" / "SOURCE.json"

FILES = {
    "scripts/train_gate_oof.py": "model_code",
    "scripts/provenance_common.py": "imported by the model code",
    "data/provenance_tables/S02/normalized_trials.csv": "normalized_trials",
    "data/provenance_tables/S02/paired_cells.csv": "paired_cells",
    "data/provenance_tables/S05/cell_features_1421.csv": "cell_features",
    "data/provenance_tables/S12/method_config.json": "method_config",
    "data/provenance_tables/S18/task_partition_87.csv": "task_partition",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def describe(path):
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return {
        "bytes_on_disk": len(raw),
        "sha256_on_disk": sha256(raw),
        "has_crlf": b"\r\n" in raw,
        "bytes_lf": len(lf),
        "sha256_lf": sha256(lf),
        "git_blob_lf": git_blob(lf),
        "sha256_crlf": sha256(lf.replace(b"\n", b"\r\n")),
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    repo = Path(sys.argv[1])
    write = "--write" in sys.argv[2:]
    found = {rel: describe(repo / rel) for rel in FILES}

    if write:
        old = json.loads(SOURCE.read_text(encoding="utf-8")) if SOURCE.exists() else {}
        out = {k: v for k, v in old.items() if k != "files"}
        out["files"] = {
            rel: {"role": FILES[rel], "git_blob": d["git_blob_lf"], "bytes": d["bytes_lf"],
                  "sha256_as_committed": d["sha256_lf"], "sha256_crlf_checkout": d["sha256_crlf"]}
            for rel, d in found.items()}
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        SOURCE.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
        print("wrote", SOURCE)
        return

    expected = json.loads(SOURCE.read_text(encoding="utf-8"))
    print("repository:", expected.get("repository"), " commit:", expected.get("commit"))
    bad = 0
    for rel, d in found.items():
        e = expected["files"][rel]
        if d["sha256_on_disk"] == e["sha256_as_committed"]:
            status = "identical to the committed file"
        elif d["sha256_lf"] == e["sha256_as_committed"]:
            status = "same content as committed, with CRLF line endings"
        else:
            status = "DIFFERENT CONTENT from the committed file"
            bad += 1
        print(f"{rel}\n    {status}")
        print(f"    on disk    sha256 {d['sha256_on_disk']}  ({d['bytes_on_disk']} bytes)")
        print(f"    LF form    sha256 {d['sha256_lf']}  git blob {d['git_blob_lf']}")
        print(f"    committed  sha256 {e['sha256_as_committed']}  git blob {e['git_blob']}")
    print("RESULT:", "all seven files have the committed content" if bad == 0 else f"{bad} file(s) differ in content")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
