#!/usr/bin/env python3
"""
Run NSGA-II for every PR run in unittest_linux_runs_with_pr_changes.csv.

Goal
- For each run (code change Δ), find Pareto-optimal test subsets balancing:
  - minimize energy (J)
  - maximize change-aware coverage over modified src lines

Key idea (fast change-aware coverage)
- Parse modified_files_and_statements -> changed lines SΔ (repo-relative paths)
- Convert changed lines into IDs using a prebuilt line index:
    line_index_lines.txt:  "src/foo/bar.cpp:123"  (canonical keys)
    line_index.npz:        CSR offsets + adjacency test_ids per line_id
- Build per-test bitsets over ONLY the changed lines for this run
- Evaluate any subset by OR-unioning selected tests' bitsets and dividing by |SΔ|

Inputs
- --runs-csv: PR runs with modified files/lines
- --line-index-prefix: prefix to line_index files (npz + _lines.txt + _tests.txt)
- --energy-csv: per-test energy measurements (codecarbon_energy_pertest.csv)

Outputs
- out/pareto/{run_id}.csv : pareto solutions for that run
- out/summary.csv         : one row per run with key pareto points
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize


# ------------------------------------------------------------
# Path + change parsing
# ------------------------------------------------------------

def normalize_to_lammps_relpath(path: str) -> str:
    """
    Convert paths from PR / coverage contexts into canonical repo-relative form.

    Canonical must match line_index_lines.txt entries, e.g.:
      'src/AMOEBA/angle_amoeba.cpp'

    Handles:
    - absolute paths containing '/lammps/...'
    - prefixed paths like 'lammps/src/...'
    - './src/...'
    - windows backslashes
    """
    if path is None:
        return ""
    p = str(path).strip().replace("\\", "/")

    # strip quotes if present
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]

    # remove leading ./ repeatedly
    while p.startswith("./"):
        p = p[2:]

    # remove leading /
    p = p.lstrip("/")

    # if absolute-like includes /lammps/, take suffix after it
    marker = "/lammps/"
    if marker in p:
        p = p.split(marker, 1)[1].lstrip("/")

    # if still prefixed with lammps/
    if p.startswith("lammps/"):
        p = p[len("lammps/"):]

    return p


def parse_modified_files_and_statements(s: str) -> Dict[str, Set[int]]:
    """
    Parse strings like:
      'src/a.cpp#L1-L10|L50;src/b.h#L3'

    Returns:
      dict[file_relpath -> set(line_numbers)]
    """
    out: Dict[str, Set[int]] = {}
    if not isinstance(s, str) or not s.strip():
        return out

    parts = s.split(";")
    for part in parts:
        part = part.strip()
        if not part or "#" not in part:
            continue

        file_part, line_part = part.split("#", 1)
        rel = normalize_to_lammps_relpath(file_part.strip())
        if not rel:
            continue

        tokens = [t.strip() for t in line_part.split("|") if t.strip()]
        for tok in tokens:
            m1 = re.fullmatch(r"L(\d+)", tok)
            m2 = re.fullmatch(r"L(\d+)-L(\d+)", tok)
            if m1:
                out.setdefault(rel, set()).add(int(m1.group(1)))
            elif m2:
                a = int(m2.group(1))
                b = int(m2.group(2))
                if b < a:
                    a, b = b, a
                out.setdefault(rel, set()).update(range(a, b + 1))

    return out


# ------------------------------------------------------------
# Line -> tests index (CSR)
# ------------------------------------------------------------

@dataclass
class LineIndex:
    """
    CSR-like index for:
      line_id -> list of test_ids that cover that line

    offsets: size n_lines + 1
    test_ids: size n_edges (flattened adjacency)
    test_names: list of tests (ids are indices into this list)
    line_map: (relpath, line) -> line_id
    """
    offsets: np.ndarray
    test_ids: np.ndarray
    test_names: List[str]
    line_map: Dict[Tuple[str, int], int]


def load_line_index(prefix: str) -> LineIndex:
    """
    Load:
      {prefix}.npz            containing arrays: offsets, test_ids
      {prefix}_tests.txt      one test name per line
      {prefix}_lines.txt      one line key per line: 'src/foo.cpp:123'
    """
    npz = np.load(prefix + ".npz")
    offsets = npz["offsets"]
    test_ids = npz["test_ids"]

    with open(prefix + "_tests.txt", "r") as f:
        test_names = [ln.strip() for ln in f if ln.strip()]

    line_map: Dict[Tuple[str, int], int] = {}
    with open(prefix + "_lines.txt", "r") as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            rel, lno = ln.rsplit(":", 1)
            rel = normalize_to_lammps_relpath(rel)  # just in case
            line_map[(rel, int(lno))] = i

    return LineIndex(offsets=offsets, test_ids=test_ids, test_names=test_names, line_map=line_map)


# ------------------------------------------------------------
# Energy alignment
# ------------------------------------------------------------

def load_energy(energy_csv: str) -> pd.DataFrame:
    """
    Load per-test energy measurements.

    Keeps only rc==0 if present.
    Returns at least columns: test, energy_j, (optional) wall_s
    """
    df = pd.read_csv(energy_csv)
    if "rc" in df.columns:
        df = df[df["rc"] == 0].copy()

    keep = ["test", "energy_j"]
    if "wall_s" in df.columns:
        keep.append("wall_s")
    return df[keep].copy()


def align_tests(index_tests: List[str], energy_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align energy (and wall time if present) to the ordering of line_index test_names.

    Missing energy => NaN (will be penalized in objective).
    """
    e_map = dict(zip(energy_df["test"].astype(str), energy_df["energy_j"].astype(float)))

    wall_map = {}
    if "wall_s" in energy_df.columns:
        wall_map = dict(zip(energy_df["test"].astype(str), energy_df["wall_s"]))

    energy = np.full(len(index_tests), np.nan, dtype=np.float64)
    wall_s = np.full(len(index_tests), np.nan, dtype=np.float64)

    for i, t in enumerate(index_tests):
        if t in e_map:
            energy[i] = e_map[t]
        if t in wall_map:
            try:
                wall_s[i] = float(wall_map[t])
            except Exception:
                pass

    return energy, wall_s


# ------------------------------------------------------------
# Change lines -> bitsets
# ------------------------------------------------------------

def changed_lines_to_ids(
    changed: Dict[str, Set[int]],
    line_map: Dict[Tuple[str, int], int],
    keep_prefix: str = "src/",
) -> List[int]:
    """
    Convert changed lines {file->set(lines)} into existing line IDs in the index.

    keep_prefix="src/" means ignore non-src files (docs, github workflows, etc).
    """
    ids: List[int] = []
    for f, lines in changed.items():
        f = normalize_to_lammps_relpath(f)
        if keep_prefix and not f.startswith(keep_prefix):
            continue
        for ln in lines:
            key = (f, int(ln))
            if key in line_map:
                ids.append(line_map[key])
    return sorted(set(ids))


def build_test_bitsets_for_changed_lines(changed_line_ids: List[int], idx: LineIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-test bitsets over ONLY the changed lines.

    We define local changed-line positions j=0..m-1.
    For each changed line, mark which tests cover it in bitset.

    Returns:
      test_bits: uint64 array (n_tests, n_words)
      coverable_mask: bool array (m,)  True if this changed line is covered by >=1 test
    """
    n_tests = len(idx.test_names)
    m = len(changed_line_ids)
    if m == 0:
        return np.zeros((n_tests, 0), dtype=np.uint64), np.zeros((0,), dtype=bool)

    n_words = (m + 63) // 64
    test_bits = np.zeros((n_tests, n_words), dtype=np.uint64)
    coverable_mask = np.zeros((m,), dtype=bool)

    pos = {lid: j for j, lid in enumerate(changed_line_ids)}

    for lid in changed_line_ids:
        j = pos[lid]
        off0 = int(idx.offsets[lid])
        off1 = int(idx.offsets[lid + 1])
        if off1 <= off0:
            continue

        coverable_mask[j] = True
        word = j // 64
        bit = j % 64
        mask = np.uint64(1) << np.uint64(bit)

        for tid in idx.test_ids[off0:off1]:
            test_bits[int(tid), word] |= mask

    return test_bits, coverable_mask


def restrict_to_coverable_lines(test_bits: np.ndarray, coverable_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Reduce the representation so we count coverage EXACTLY over coverable changed lines only.

    Implementation approach:
    - Expand coverable_mask into explicit indices
    - Rebuild a new bitset matrix with only those columns.
    """
    m = int(np.sum(coverable_mask))
    if m == 0:
        return np.zeros((test_bits.shape[0], 0), dtype=np.uint64), 0

    keep = np.where(coverable_mask)[0].tolist()
    n_tests = test_bits.shape[0]

    n_words = (m + 63) // 64
    out = np.zeros((n_tests, n_words), dtype=np.uint64)

    for new_j, old_j in enumerate(keep):
        old_word = old_j // 64
        old_bit = old_j % 64
        old_mask = np.uint64(1) << np.uint64(old_bit)

        new_word = new_j // 64
        new_bit = new_j % 64
        new_mask = np.uint64(1) << np.uint64(new_bit)

        # tests that cover old_j get new_j
        hits = (test_bits[:, old_word] & old_mask) != 0
        out[hits, new_word] |= new_mask

    return out, m


# ------------------------------------------------------------
# NSGA-II problem
# ------------------------------------------------------------

class EnergyCoverageProblem(Problem):
    """
    Binary subset selection:
      X[i] = 1 means select test i

    Objectives:
      f1 = total_energy (minimize)
      f2 = -change_coverage (minimize)  <=> maximize coverage

    Coverage definition:
      coverage(X) = (# coverable changed lines covered by any selected test) / ( # coverable changed lines )
    """

    def __init__(
        self,
        energy_j: np.ndarray,
        test_bits: np.ndarray,
        m_coverable: int,
        penalize_missing_energy: float = 1e12,
    ):
        self.energy_j = energy_j
        self.test_bits = test_bits
        self.m_coverable = int(m_coverable)
        self.penalty = float(penalize_missing_energy)
        super().__init__(n_var=len(energy_j), n_obj=2, xl=0, xu=1, type_var=bool)

    def _evaluate(self, X, out, *args, **kwargs):
        # Energy: sum with penalty for missing energy
        e = np.where(np.isnan(self.energy_j), self.penalty, self.energy_j)
        energy = (X * e).sum(axis=1)

        # Coverage: OR-union bitsets across selected tests
        pop = X.shape[0]
        cov = np.zeros((pop,), dtype=np.float64)

        if self.m_coverable > 0 and self.test_bits.shape[1] > 0:
            for i in range(pop):
                sel = X[i].astype(bool)
                if not np.any(sel):
                    cov[i] = 0.0
                    continue
                union = np.bitwise_or.reduce(self.test_bits[sel], axis=0)
                covered = int(np.unpackbits(union.view(np.uint8)).sum())
                cov[i] = covered / float(self.m_coverable)

        out["F"] = np.column_stack([energy, -cov])


# ------------------------------------------------------------
# One run -> run NSGA-II and write pareto
# ------------------------------------------------------------

def run_nsga2_for_run(
    run_id: int,
    pr_number: Optional[int],
    changed_str: str,
    idx: LineIndex,
    energy_j: np.ndarray,
    out_dir: str,
    keep_prefix: str = "src/",
    pop: int = 80,
    gens: int = 120,
    seed: int = 1,
) -> Dict:
    """
    Execute the full pipeline for one PR run:
    - parse changed lines
    - map to indexed line IDs
    - build per-test bitsets for these changed lines
    - restrict to coverable changed lines
    - run NSGA-II and save pareto CSV
    - compute summary stats

    Returns a summary dict for summary.csv
    """
    changed = parse_modified_files_and_statements(changed_str)
    changed_ids = changed_lines_to_ids(changed, idx.line_map, keep_prefix=keep_prefix)

    if len(changed_ids) == 0:
        return {
            "run_id": run_id,
            "pr_number": pr_number,
            "changed_lines_mapped": 0,
            "coverable_changed_lines": 0,
            "max_change_cov": 0.0,
            "pareto_points": 0,
            "status": "skip_no_changed_lines",
        }

    test_bits, coverable_mask = build_test_bitsets_for_changed_lines(changed_ids, idx)
    test_bits2, m_coverable = restrict_to_coverable_lines(test_bits, coverable_mask)

    if m_coverable == 0:
        return {
            "run_id": run_id,
            "pr_number": pr_number,
            "changed_lines_mapped": len(changed_ids),
            "coverable_changed_lines": 0,
            "max_change_cov": 0.0,
            "pareto_points": 0,
            "status": "skip_zero_coverable",
        }

    problem = EnergyCoverageProblem(energy_j=energy_j, test_bits=test_bits2, m_coverable=m_coverable)

    algo = NSGA2(
        pop_size=pop,
        sampling=BinaryRandomSampling(),
        crossover=HalfUniformCrossover(prob=0.9),
        mutation=BitflipMutation(prob=0.02),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algo, ("n_gen", gens), seed=seed, verbose=False)

    F = res.F         # [energy, -cov]
    X = res.X.astype(bool)

    cov = -F[:, 1]
    energy = F[:, 0]
    size = X.sum(axis=1)

    os.makedirs(os.path.join(out_dir, "pareto"), exist_ok=True)
    pareto_path = os.path.join(out_dir, "pareto", f"{run_id}.csv")

    dfp = pd.DataFrame(
        {
            "run_id": run_id,
            "pr_number": pr_number,
            "energy_j": energy,
            "change_cov": cov,
            "suite_size": size,
        }
    ).sort_values(["energy_j", "change_cov"], ascending=[True, False])

    dfp.to_csv(pareto_path, index=False)

    max_cov = float(dfp["change_cov"].max())
    targets = [0.80, 0.90, 0.95]

    summary = {
        "run_id": run_id,
        "pr_number": pr_number,
        "changed_lines_mapped": len(changed_ids),
        "coverable_changed_lines": int(m_coverable),
        "max_change_cov": max_cov,
        "pareto_points": int(len(dfp)),
        "status": "ok",
    }

    for t in targets:
        need = t * max_cov
        sub = dfp[dfp["change_cov"] >= need]
        if len(sub) == 0:
            summary[f"energy_j_at_{int(t*100)}pct"] = np.nan
            summary[f"suite_size_at_{int(t*100)}pct"] = np.nan
        else:
            best = sub.sort_values(["energy_j", "suite_size"]).iloc[0]
            summary[f"energy_j_at_{int(t*100)}pct"] = float(best["energy_j"])
            summary[f"suite_size_at_{int(t*100)}pct"] = int(best["suite_size"])

    return summary


# ------------------------------------------------------------
# Main: iterate all PR runs
# ------------------------------------------------------------

def main():
    """
    Orchestrate:
    - load line index (coverage -> tests)
    - load energy per test and align to index tests
    - load PR run table and iterate eligible rows
    - run NSGA-II per run, write pareto csv, collect summaries
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-csv", required=True)
    ap.add_argument("--line-index-prefix", required=True)
    ap.add_argument("--energy-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-prefix", default="src/")
    ap.add_argument("--only-success", action="store_true")
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit; otherwise process first N eligible runs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    idx = load_line_index(args.line_index_prefix)
    energy_df = load_energy(args.energy_csv)
    energy_j, _wall_s = align_tests(idx.test_names, energy_df)

    df = pd.read_csv(args.runs_csv)

    if args.only_success:
        df = df[(df["status"] == "completed") & (df["conclusion"] == "success")].copy()

    # only rows that actually have changed statements
    df = df[df["modified_files_and_statements"].notna()].copy()

    summaries: List[Dict] = []
    count = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="NSGA-II runs"):
        run_id = int(r["run_id"])
        pr_number = None if pd.isna(r.get("pr_number", np.nan)) else int(r["pr_number"])
        changed_str = str(r["modified_files_and_statements"])

        s = run_nsga2_for_run(
            run_id=run_id,
            pr_number=pr_number,
            changed_str=changed_str,
            idx=idx,
            energy_j=energy_j,
            out_dir=args.out_dir,
            keep_prefix=args.keep_prefix,
            pop=args.pop,
            gens=args.gens,
            seed=args.seed,
        )
        summaries.append(s)

        count += 1
        if args.limit and count >= args.limit:
            break

    out_sum = os.path.join(args.out_dir, "summary.csv")
    pd.DataFrame(summaries).to_csv(out_sum, index=False)

    print("Wrote:", out_sum)
    print("Pareto CSVs in:", os.path.join(args.out_dir, "pareto"))


if __name__ == "__main__":
    main()