#!/usr/bin/env python3
"""
Run NSGA-II for every PR run in unittest_linux_runs_with_pr_changes.csv,
but in a way that matches the paper description (selection + ordering).

Key idea
- Represent a solution as an execution order (a permutation) plus a cut point k.
- The executed subset T' is the prefix of length k of that ordering.
- Evaluate objectives on that prefix.

We implement this using a "random-keys" encoding:
- Decision variables: keys[0:n_tests] in [0,1] induce a permutation pi = argsort(keys)
- One extra variable cut_u in [0,1] sets k = 1 + floor(cut_u * (n_tests - 1))

Objectives (all minimized):
- f1: total_energy(prefix)         [J]
- f2: -change_coverage(prefix)     [maximize coverage]
- f3: total_time(prefix)           [s]
- f4: prefix_length k              [#tests]

Change-aware coverage is computed efficiently using a prebuilt line->tests index
and per-test bitsets over ONLY the changed lines of each run.

Option A (added in this version)
- After NSGA-II selects a prefix, we reorder ONLY those selected tests to front-load
  marginal changed-line coverage per Joule (proxy for earlier change exposure).
- We then compute energy-to-reach absolute change coverage thresholds:
  80%, 90%, 95%, 100% (columns: energy_to_80cov_J, ..., energy_to_100cov_J)

Inputs
- --runs-csv: PR runs with modified files/lines
- --line-index-prefix: prefix to line_index files (.npz + _lines.txt + _tests.txt)
- --energy-csv: per-test energy/time measurements

Outputs
- out/pareto/{run_id}.csv : pareto solutions for that run (now includes energy_to_*cov_J)
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
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
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

    Missing values => NaN (will be penalized in objective).
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
    Reduce representation so we count coverage EXACTLY over coverable changed lines only.

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

        hits = (test_bits[:, old_word] & old_mask) != 0
        out[hits, new_word] |= new_mask

    return out, m


# ------------------------------------------------------------
# Option A: reorder selected prefix to front-load change coverage per Joule
# ------------------------------------------------------------

def _popcount_u64(arr_u64: np.ndarray) -> int:
    """Count 1-bits in a uint64 vector."""
    return int(np.unpackbits(arr_u64.view(np.uint8)).sum())


def greedy_reorder_by_marginal_cov_per_joule(
    sel: np.ndarray,
    test_bits: np.ndarray,     # (n_tests, n_words) uint64
    energy_j: np.ndarray,      # (n_tests,) float
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Greedily reorder 'sel' to maximize marginal new changed-line coverage per Joule.

    This does NOT change total energy/time if all tests in sel run.
    It improves coverage-vs-energy progression (proxy for early termination benefits).
    """
    if len(sel) <= 1 or test_bits.shape[1] == 0:
        return sel

    e = np.where(np.isnan(energy_j), 1e12, energy_j)

    remaining = sel.astype(int).tolist()
    ordered: List[int] = []
    covered = np.zeros((test_bits.shape[1],), dtype=np.uint64)

    while remaining:
        best_t: Optional[int] = None
        best_score = -1.0
        best_gain = -1
        best_energy = None

        for t in remaining:
            bits_t = test_bits[t]
            new_bits = np.bitwise_and(bits_t, np.bitwise_not(covered))
            gain = _popcount_u64(new_bits)
            score = gain / (float(e[t]) + eps)

            t_energy = float(e[t])
            if (score > best_score) or (score == best_score and gain > best_gain) or (
                score == best_score and gain == best_gain and (best_energy is None or t_energy < best_energy)
            ):
                best_score = score
                best_gain = gain
                best_t = t
                best_energy = t_energy

        assert best_t is not None
        ordered.append(best_t)
        covered |= test_bits[best_t]
        remaining.remove(best_t)

    return np.array(ordered, dtype=int)


def energy_to_reach_cov_thresholds(
    sel_ordered: np.ndarray,
    test_bits: np.ndarray,
    m_coverable: int,
    energy_j: np.ndarray,
    thresholds=(0.80, 0.90, 0.95, 1.00),
) -> Dict[str, float]:
    """
    Compute energy required to reach absolute change coverage thresholds (80/90/95/100%).

    Returns NaN for unreachable thresholds.
    """
    out: Dict[str, float] = {}
    for th in thresholds:
        out[f"energy_to_{int(th*100)}cov_J"] = np.nan

    if m_coverable <= 0 or test_bits.shape[1] == 0 or len(sel_ordered) == 0:
        return out

    e = np.where(np.isnan(energy_j), 1e12, energy_j)

    covered = np.zeros((test_bits.shape[1],), dtype=np.uint64)
    cum_energy = 0.0

    pending = {float(th): None for th in thresholds}

    for t in sel_ordered.astype(int):
        cum_energy += float(e[t])
        covered |= test_bits[t]
        cov = _popcount_u64(covered) / float(m_coverable)

        for th in list(pending.keys()):
            if pending[th] is None and cov + 1e-15 >= th:
                pending[th] = cum_energy

        if all(v is not None for v in pending.values()):
            break

    for th, val in pending.items():
        out[f"energy_to_{int(th*100)}cov_J"] = float(val) if val is not None else np.nan

    return out


# ------------------------------------------------------------
# NSGA-II problem: random-keys ordering + cut
# ------------------------------------------------------------

class EnergyCoverageOrderingProblem(Problem):
    """
    Random-keys ordering + cut (prefix) optimization.

    Decision variables per individual:
      - keys[0:n_tests] in [0,1]  -> induces permutation pi = argsort(keys)
      - cut_u in [0,1]            -> prefix length k = 1 + floor(cut_u * (n_tests-1))

    Executed subset is prefix pi[:k].

    Objectives (all minimized):
      f1 = total_energy(prefix)
      f2 = -change_coverage(prefix)  (maximize coverage)
      f3 = total_time(prefix)
      f4 = prefix_length k
    """

    def __init__(
        self,
        energy_j: np.ndarray,
        wall_s: np.ndarray,
        test_bits: np.ndarray,
        m_coverable: int,
        penalize_missing: float = 1e12,
    ):
        self.energy_j = energy_j
        self.wall_s = wall_s
        self.test_bits = test_bits
        self.m_coverable = int(m_coverable)
        self.penalty = float(penalize_missing)

        n_tests = len(energy_j)
        super().__init__(n_var=n_tests + 1, n_obj=4, xl=0.0, xu=1.0, type_var=float)

    def _evaluate(self, X, out, *args, **kwargs):
        n_tests = len(self.energy_j)

        e = np.where(np.isnan(self.energy_j), self.penalty, self.energy_j)
        t = np.where(np.isnan(self.wall_s), self.penalty, self.wall_s)

        pop = X.shape[0]
        energy = np.zeros(pop, dtype=np.float64)
        time_s = np.zeros(pop, dtype=np.float64)
        cov = np.zeros(pop, dtype=np.float64)
        k_arr = np.zeros(pop, dtype=np.float64)

        for i in range(pop):
            keys = X[i, :n_tests]
            cut_u = float(X[i, n_tests])

            pi = np.argsort(keys)

            k = 1 + int(np.floor(cut_u * (n_tests - 1)))
            k = max(1, min(n_tests, k))
            k_arr[i] = float(k)

            sel = pi[:k]

            energy[i] = float(e[sel].sum())
            time_s[i] = float(t[sel].sum())

            if self.m_coverable > 0 and self.test_bits.shape[1] > 0:
                union = np.bitwise_or.reduce(self.test_bits[sel], axis=0)
                covered = int(np.unpackbits(union.view(np.uint8)).sum())
                cov[i] = covered / float(self.m_coverable)
            else:
                cov[i] = 0.0

        out["F"] = np.column_stack([energy, -cov, time_s, k_arr])


# ------------------------------------------------------------
# One run -> run NSGA-II and write pareto
# ------------------------------------------------------------

def run_nsga2_for_run(
    run_id: int,
    pr_number: Optional[int],
    changed_str: str,
    idx: LineIndex,
    energy_j: np.ndarray,
    wall_s: np.ndarray,
    out_dir: str,
    keep_prefix: str = "src/",
    pop: int = 80,
    gens: int = 120,
    seed: int = 1,
    penalize_missing: float = 1e12,
) -> Dict:
    """
    Execute pipeline for one PR run:
    - parse changed lines
    - map to indexed line IDs
    - build per-test bitsets for these changed lines
    - restrict to coverable changed lines
    - run NSGA-II with ordering+cut encoding
    - save pareto CSV (now includes energy_to_{80,90,95,100}cov_J computed after Option A reordering)
    - compute summary stats (energy/suite/time at 80/90/95% of max attainable coverage)
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

    problem = EnergyCoverageOrderingProblem(
        energy_j=energy_j,
        wall_s=wall_s,
        test_bits=test_bits2,
        m_coverable=m_coverable,
        penalize_missing=penalize_missing,
    )

    algo = NSGA2(
        pop_size=pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(prob=1.0 / (len(energy_j) + 1), eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algo, ("n_gen", gens), seed=seed, verbose=False)

    X = res.X  # decision variables
    F = res.F  # [energy, -cov, time, k]

    if F is None or len(F) == 0:
        return {
            "run_id": run_id,
            "pr_number": pr_number,
            "changed_lines_mapped": len(changed_ids),
            "coverable_changed_lines": int(m_coverable),
            "max_change_cov": 0.0,
            "pareto_points": 0,
            "status": "skip_no_solutions",
        }

    energy = F[:, 0].astype(float)
    cov = (-F[:, 1]).astype(float)
    time_s = F[:, 2].astype(float)
    k = F[:, 3].astype(int)

    # Option A metrics: energy-to-reach absolute coverage thresholds under reordering
    n_tests = len(energy_j)
    e_to_80, e_to_90, e_to_95, e_to_100 = [], [], [], []

    for i in range(X.shape[0]):
        keys = X[i, :n_tests]
        cut_u = float(X[i, n_tests])

        pi = np.argsort(keys)

        k_i = 1 + int(np.floor(cut_u * (n_tests - 1)))
        k_i = max(1, min(n_tests, k_i))

        sel = pi[:k_i]

        sel2 = greedy_reorder_by_marginal_cov_per_joule(sel, test_bits2, energy_j)

        metrics = energy_to_reach_cov_thresholds(
            sel_ordered=sel2,
            test_bits=test_bits2,
            m_coverable=m_coverable,
            energy_j=energy_j,
            thresholds=(0.80, 0.90, 0.95, 1.00),
        )

        e_to_80.append(metrics["energy_to_80cov_J"])
        e_to_90.append(metrics["energy_to_90cov_J"])
        e_to_95.append(metrics["energy_to_95cov_J"])
        e_to_100.append(metrics["energy_to_100cov_J"])

    os.makedirs(os.path.join(out_dir, "pareto"), exist_ok=True)
    pareto_path = os.path.join(out_dir, "pareto", f"{run_id}.csv")

    dfp = pd.DataFrame(
        {
            "run_id": run_id,
            "pr_number": pr_number,
            "energy_j": energy,
            "change_cov": cov,
            "time_s": time_s,
            "suite_size": k,
            "energy_to_80cov_J": e_to_80,
            "energy_to_90cov_J": e_to_90,
            "energy_to_95cov_J": e_to_95,
            "energy_to_100cov_J": e_to_100,
        }
    ).sort_values(["energy_j", "change_cov", "suite_size"], ascending=[True, False, True])

    dfp.to_csv(pareto_path, index=False)

    max_cov = float(dfp["change_cov"].max()) if len(dfp) else 0.0
    targets = [0.80, 0.90, 0.95]

    summary: Dict = {
        "run_id": run_id,
        "pr_number": pr_number,
        "changed_lines_mapped": len(changed_ids),
        "coverable_changed_lines": int(m_coverable),
        "max_change_cov": max_cov,
        "pareto_points": int(len(dfp)),
        "status": "ok" if len(dfp) else "skip_no_solutions",
    }

    # For each target fraction of attainable coverage, choose min-energy solution
    for tfrac in targets:
        if max_cov <= 0:
            summary[f"energy_j_at_{int(tfrac*100)}pct"] = np.nan
            summary[f"time_s_at_{int(tfrac*100)}pct"] = np.nan
            summary[f"suite_size_at_{int(tfrac*100)}pct"] = np.nan
            continue

        need = tfrac * max_cov
        sub = dfp[dfp["change_cov"] >= need]
        if len(sub) == 0:
            summary[f"energy_j_at_{int(tfrac*100)}pct"] = np.nan
            summary[f"time_s_at_{int(tfrac*100)}pct"] = np.nan
            summary[f"suite_size_at_{int(tfrac*100)}pct"] = np.nan
        else:
            best = sub.sort_values(["energy_j", "suite_size", "time_s"]).iloc[0]
            summary[f"energy_j_at_{int(tfrac*100)}pct"] = float(best["energy_j"])
            summary[f"time_s_at_{int(tfrac*100)}pct"] = float(best["time_s"])
            summary[f"suite_size_at_{int(tfrac*100)}pct"] = int(best["suite_size"])

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
    ap.add_argument("--penalize-missing", type=float, default=1e12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    idx = load_line_index(args.line_index_prefix)
    energy_df = load_energy(args.energy_csv)
    energy_j, wall_s = align_tests(idx.test_names, energy_df)

    df = pd.read_csv(args.runs_csv)

    if args.only_success:
        df = df[(df["status"] == "completed") & (df["conclusion"] == "success")].copy()

    # only rows that actually have changed statements
    df = df[df["modified_files_and_statements"].notna()].copy()

    summaries: List[Dict] = []
    count = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="NSGA-II runs (ordering+cut)"):
        run_id = int(r["run_id"])
        pr_number = None if pd.isna(r.get("pr_number", np.nan)) else int(r["pr_number"])
        changed_str = str(r["modified_files_and_statements"])

        s = run_nsga2_for_run(
            run_id=run_id,
            pr_number=pr_number,
            changed_str=changed_str,
            idx=idx,
            energy_j=energy_j,
            wall_s=wall_s,
            out_dir=args.out_dir,
            keep_prefix=args.keep_prefix,
            pop=args.pop,
            gens=args.gens,
            seed=args.seed,
            penalize_missing=args.penalize_missing,
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