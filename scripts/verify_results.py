# Copyright 2026 Sergio Sánchez de la Fuente
# SPDX-License-Identifier: Apache-2.0
"""Check data integrity and reproduce the stored cutoff, not merely the year."""
from pathlib import Path
import hashlib
import json
import math
import argparse

ROOT = Path(__file__).resolve().parents[1]


def compare(actual, expected, location="root"):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"Different keys at {location}")
        for key in expected:
            compare(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Different lengths at {location}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{location}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"Different numerical result at {location}: {actual} vs {expected}")
    elif actual != expected:
        raise ValueError(f"Different result at {location}: {actual!r} vs {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-raw", action="store_true", help="Require every original export and verify its hash.")
    args = parser.parse_args()
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    verified_exports = 0
    for record in provenance["raw_exports"]:
        export = ROOT / record["file"]
        if not export.exists():
            if args.require_raw:
                raise ValueError(f"Original export missing: {record['file']}")
            continue
        if hashlib.sha256(export.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Original export changed: {record['file']}")
        verified_exports += 1
    actual = json.loads((ROOT / "bibliometrics/analysis/results/cutoff/summary.json").read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "bibliometrics/reference/cutoff_summary.json").read_text(encoding="utf-8"))
    for key in ("scope", "decision", "leave_one_source_out", "source_only", "classifier_agreement"):
        compare(actual[key], expected[key], key)
    print(f"PASS: complete cutoff calculations, database sensitivity, and classifier agreement. Original export hashes checked: {verified_exports}/{len(provenance['raw_exports'])}.")
    if verified_exports < len(provenance["raw_exports"]):
        print("NOTE: restricted database exports absent; verified cached-output workflow only.")


if __name__ == "__main__":
    main()
