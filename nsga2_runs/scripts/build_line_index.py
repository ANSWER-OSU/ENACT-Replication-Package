#!/usr/bin/env python3
"""
Build a fast index from (file,line) -> list of test indices.

Why:
- line_to_tests.csv is large but it's the best direction for change-aware coverage.
- We only need "who covers this changed line?" quickly for many PRs.

Input:
- line_to_tests.csv with columns: file,line,n_tests,tests

Output:
- line_index.npz: compressed arrays representing a sparse mapping
- tests.txt: mapping from test_index -> test_name
- line_keys.txt: mapping from line_key_index -> "relpath:line"

Data structure:
- We assign each unique line_key an integer ID.
- For each line_key, we store the list of test IDs covering it in CSR-like format:
    offsets[line_id]..offsets[line_id+1] is a slice in test_ids[]
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def normalize_to_lammps_relpath(path: str) -> str:
    """Convert absolute build paths into repo-relative paths like 'src/foo.cpp'."""
    if path is None:
        return ""
    p = str(path).replace("\\", "/")
    marker = "/lammps/"
    if marker in p:
        return p.split(marker, 1)[1].lstrip("/")
    if p.startswith(("src/", ".github/", "doc/")):
        return p
    return p  # fallback: keep as-is


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line-to-tests", required=True)
    ap.add_argument("--out-prefix", required=True, help="prefix for outputs, e.g. .../line_index")
    ap.add_argument("--keep-prefix", default="src/")
    args = ap.parse_args()

    # Build test name -> test_id
    test2id: Dict[str, int] = {}
    linekey2tests: Dict[Tuple[str, int], List[int]] = defaultdict(list)

    def get_test_id(name: str) -> int:
        if name not in test2id:
            test2id[name] = len(test2id)
        return test2id[name]

    with open(args.line_to_tests, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rel = normalize_to_lammps_relpath(row.get("file", ""))
            if args.keep_prefix and not rel.startswith(args.keep_prefix):
                continue

            try:
                ln = int(row.get("line", "0"))
            except Exception:
                continue

            tests = row.get("tests", "")
            if not tests or not str(tests).strip():
                continue

            # CSV has tests field quoted with commas inside
            # We split by comma and strip.
            tnames = [t.strip() for t in str(tests).split(",") if t.strip()]
            if not tnames:
                continue

            key = (rel, ln)
            for tn in tnames:
                linekey2tests[key].append(get_test_id(tn))

    # Freeze order of line keys
    line_keys = list(linekey2tests.keys())
    line_keys.sort()

    # CSR build
    offsets = np.zeros(len(line_keys) + 1, dtype=np.int64)
    all_test_ids: List[int] = []

    for i, lk in enumerate(line_keys):
        tids = linekey2tests[lk]
        # De-dup (some rows can have duplicates)
        tids = sorted(set(tids))
        all_test_ids.extend(tids)
        offsets[i + 1] = offsets[i] + len(tids)

    test_ids = np.array(all_test_ids, dtype=np.int32)

    out_npz = args.out_prefix + ".npz"
    np.savez_compressed(out_npz, offsets=offsets, test_ids=test_ids)

    # Write mappings for human/debug
    out_tests = args.out_prefix + "_tests.txt"
    out_lines = args.out_prefix + "_lines.txt"

    id2test = [""] * len(test2id)
    for name, i in test2id.items():
        id2test[i] = name

    with open(out_tests, "w") as f:
        for t in id2test:
            f.write(t + "\n")

    with open(out_lines, "w") as f:
        for rel, ln in line_keys:
            f.write(f"{rel}:{ln}\n")

    print("Wrote:", out_npz)
    print("Wrote:", out_tests)
    print("Wrote:", out_lines)
    print("Unique tests:", len(id2test))
    print("Unique src lines:", len(line_keys))
    print("Total line->test edges:", len(test_ids))


if __name__ == "__main__":
    main()