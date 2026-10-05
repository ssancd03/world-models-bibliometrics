# Copyright 2026 Sergio Sánchez de la Fuente
# SPDX-License-Identifier: Apache-2.0
"""Recompute cutoff evidence and figures, with optional raw-data rebuild."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "bibliometrics/analysis/scripts"


def run(path, *arguments):
    subprocess.run([sys.executable, str(path), *arguments], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-raw", action="store_true", help="Rebuild corpus and lexical labels from original exports.")
    args = parser.parse_args()
    if args.from_raw:
        run(SCRIPTS / "prepare_exports.py")
        run(SCRIPTS / "corpus.py")
        run(SCRIPTS / "classify_b.py", "--reuse-existing")
        run(SCRIPTS / "classify_c.py", "--reuse-existing")
    run(SCRIPTS / "select_cutoff.py")
    run(ROOT / "scripts/make_cutoff_figure.py")
    run(ROOT / "scripts/verify_results.py", *(["--require-raw"] if args.from_raw else []))


if __name__ == "__main__":
    main()
