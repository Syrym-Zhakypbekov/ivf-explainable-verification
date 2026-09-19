from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ARTIFACT_MANIFEST.json"

RELEASE_PATHS = [
    ROOT / ".python-version",
    ROOT / "ARTIFACT.md",
    ROOT / "CITATION.cff",
    ROOT / "README.md",
    ROOT / "RELEASE_NOTES_v2.0.0.md",
    ROOT / "WHERE_EVERYTHING_IS.md",
    ROOT / "requirements-lock.txt",
    ROOT / "requirements.txt",
    ROOT / "scripts" / "verify_release.py",
    ROOT / "scripts" / "build_release_archive.py",
]
RELEASE_DIRS = [
    ROOT / "results" / "v2",
    ROOT / "results" / "v2pd",
    ROOT / "figures" / "v2",
]


def close(actual, expected, tolerance=5e-4):
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, got {actual}")


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        average = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = average
        i = j
    return ranks


def roc_auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise AssertionError("AUROC needs both classes")
    ranks = average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_v2():
    directory = ROOT / "results" / "v2"
    rows = read_csv(directory / "stress2_all.csv")
    labels = [int(row["defective"]) for row in rows]
    v_scores = [1.0 - float(row["V"]) for row in rows]
    f1_scores = [1.0 - float(row["macro_f1"]) for row in rows]
    auc_v = roc_auc(labels, v_scores)
    auc_f1 = roc_auc(labels, f1_scores)

    assert len(rows) == 348, len(rows)
    assert sum(labels) == 336, sum(labels)
    close(auc_v, 0.891)
    close(auc_f1, 0.720)

    summary = json.loads((directory / "stress2_summary.json").read_text(encoding="utf-8"))
    assert summary["конфигураций"] == 348
    close(float(summary["AUROC_V"]), auc_v)
    close(float(summary["AUROC_MacroF1"]), auc_f1)
    close(float(summary["разница_AUROC"]), 0.171)
    close(float(summary["CI95_разницы_кластерный"][0]), 0.0229, 1e-7)
    close(float(summary["CI95_разницы_кластерный"][1]), 0.3287, 1e-7)

    ablation = {row["вариант"]: row for row in read_csv(directory / "stress2_ablation.csv")}
    close(float(ablation["полная V"]["AUROC(все)"]), 0.891)
    close(float(ablation["без R"]["AUROC(все)"]), 0.921)
    close(float(ablation["арифм. среднее"]["AUROC(все)"]), 0.909)
    print(f"v2: n={len(rows)} AUROC(V)={auc_v:.3f} AUROC(Macro-F1)={auc_f1:.3f}")


def verify_v2pd():
    directory = ROOT / "results" / "v2pd"
    summary = json.loads((directory / "stress2_summary.json").read_text(encoding="utf-8"))
    assert summary["конфигураций"] == 348
    close(float(summary["AUROC_V"]), 0.8994, 1e-7)
    close(float(summary["AUROC_MacroF1"]), 0.7147, 1e-7)
    close(float(summary["разница_AUROC"]), 0.1848, 1e-7)
    meta = json.loads((directory / "final_meta.json").read_text(encoding="utf-8"))
    assert (meta["n_train"], meta["n_calib"], meta["n_test"]) == (1259, 896, 901)
    print("v2pd: n=348 AUROC(V)=0.899 AUROC(Macro-F1)=0.715")


def release_files():
    files = list(RELEASE_PATHS)
    for directory in RELEASE_DIRS:
        files.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest():
    payload = {
        "artifact": "ivf-explainable-verification-v2",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in release_files()
        ],
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {MANIFEST.name}: {len(payload['files'])} files")


def verify_manifest():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in payload["files"]}
    actual = {path.relative_to(ROOT).as_posix(): path for path in release_files()}
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise AssertionError(f"manifest paths differ: missing={missing}, extra={extra}")
    for name, path in actual.items():
        if sha256(path) != expected[name]["sha256"]:
            raise AssertionError(f"hash mismatch: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    verify_v2()
    verify_v2pd()
    if args.write_manifest:
        write_manifest()
    else:
        verify_manifest()
        print("OK ✓ release artifact matches the manuscript")


if __name__ == "__main__":
    main()
