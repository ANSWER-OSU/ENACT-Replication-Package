#!/usr/bin/env python3
"""
Compute three baseline test orderings for comparison against ENACT (NSGA-II).

Baselines (all operate on the full test suite of n tests):
  1. Random selection      -- random ordering, repeated n_trials times, averaged
  2. Greedy cov/second     -- greedy ordering by marginal coverage per second
  3. Greedy cov/joule      -- greedy ordering by marginal coverage per joule (no NSGA-II)

For each baseline and each run in the PR runs CSV, we compute:
  energy_to_{80,90,95,100}cov_J

These can be compared directly against ENACT's pareto/summary.csv output.

Inputs (same as run_nsga2.py):
  --runs-csv             PR runs CSV (same file used for NSGA-II)
  --line-index-prefix    prefix to line_index files (.npz + _lines.txt + _tests.txt)
  --energy-csv           per-test energy/time measurements
  --out-dir              output directory (baseline summary written here)

Output:
  {out_dir}/baseline_summary.csv   one row per run with all three baselines
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ------------------------------------------------------------
# Path + change parsing (copied from run_nsga2.py)
# ------------------------------------------------------------

def normalize_to_lammps_relpath(path: str) -> str:
    if path is None:
        return ""
    p = str(path).strip().replace("\\", "/")
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    marker = "/lammps/"
    if marker in p:
        p = p.split(marker, 1)[1].lstrip("/")
    if p.startswith("lammps/"):
        p = p[len("lammps/"):]
    return p


def parse_modified_files_and_statements(s: str) -> Dict[str, Set[int]]:
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
                a, b = int(m2.group(1)), int(m2.group(2))
                if b < a:
                    a, b = b, a
                out.setdefault(rel, set()).update(range(a, b + 1))
    return out


# ------------------------------------------------------------
# Line index
# ------------------------------------------------------------

@dataclass
class LineIndex:
    offsets: np.ndarray
    test_ids: np.ndarray
    test_names: List[str]
    line_map: Dict[Tuple[str, int], int]


def load_line_index(prefix: str) -> LineIndex:
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
            rel = normalize_to_lammps_relpath(rel)
            line_map[(rel, int(lno))] = i
    return LineIndex(offsets=offsets, test_ids=test_ids, test_names=test_names, line_map=line_map)


# ------------------------------------------------------------
# Energy alignment
# ------------------------------------------------------------

def load_energy(energy_csv: str) -> pd.DataFrame:
    df = pd.read_csv(energy_csv)
    if "rc" in df.columns:
        df = df[df["rc"] == 0].copy()
    keep = ["test", "energy_j"]
    if "wall_s" in df.columns:
        keep.append("wall_s")
    return df[keep].copy()


def align_tests(index_tests: List[str], energy_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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
# Change lines -> bitsets (copied from run_nsga2.py)
# ------------------------------------------------------------

def changed_lines_to_ids(
    changed: Dict[str, Set[int]],
    line_map: Dict[Tuple[str, int], int],
    keep_prefix: str = "src/",
) -> List[int]:
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


def build_test_bitsets_for_changed_lines(
    changed_line_ids: List[int], idx: LineIndex
) -> Tuple[np.ndarray, np.ndarray]:
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


def restrict_to_coverable_lines(
    test_bits: np.ndarray, coverable_mask: np.ndarray
) -> Tuple[np.ndarray, int]:
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
# Shared utilities
# ------------------------------------------------------------

def _popcount_u64(arr_u64: np.ndarray) -> int:
    return int(np.unpackbits(arr_u64.view(np.uint8)).sum())


def energy_to_reach_cov_thresholds(
    sel_ordered: np.ndarray,
    test_bits: np.ndarray,
    m_coverable: int,
    energy_j: np.ndarray,
    thresholds=(0.80, 0.90, 0.95, 1.00),
) -> Dict[str, float]:
    """
    For each threshold, compute:
      - energy_to_{th}cov_J:        cumulative energy up to the point threshold is hit
      - suite_size_at_{th}cov:      number of tests selected up to threshold
      - total_subset_energy_{th}cov_J: total energy of all tests in subset up to threshold
    """
    out: Dict[str, float] = {}
    for th in thresholds:
        out[f"energy_to_{int(th*100)}cov_J"] = np.nan
        out[f"suite_size_at_{int(th*100)}cov"] = np.nan
    if m_coverable <= 0 or test_bits.shape[1] == 0 or len(sel_ordered) == 0:
        return out
    e = np.where(np.isnan(energy_j), 1e12, energy_j)
    covered = np.zeros((test_bits.shape[1],), dtype=np.uint64)
    cum_energy = 0.0
    n_tests_so_far = 0
    pending = {float(th): None for th in thresholds}
    pending_size = {float(th): None for th in thresholds}
    for t in sel_ordered.astype(int):
        cum_energy += float(e[t])
        n_tests_so_far += 1
        covered |= test_bits[t]
        cov = _popcount_u64(covered) / float(m_coverable)
        for th in list(pending.keys()):
            if pending[th] is None and cov + 1e-15 >= th:
                pending[th] = cum_energy
                pending_size[th] = n_tests_so_far
        if all(v is not None for v in pending.values()):
            break
    for th, val in pending.items():
        out[f"energy_to_{int(th*100)}cov_J"] = float(val) if val is not None else np.nan
        out[f"suite_size_at_{int(th*100)}cov"] = float(pending_size[th]) if pending_size[th] is not None else np.nan
    return out


# ------------------------------------------------------------
# Baseline 1: Random
# ------------------------------------------------------------

def baseline_random(
    energy_j: np.ndarray,
    test_bits: np.ndarray,
    m_coverable: int,
    thresholds=(0.80, 0.90, 0.95, 1.00),
    n_trials: int = 30,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Random ordering of all tests with valid energy measurements,
    repeated n_trials times, results averaged.
    """
    rng = np.random.default_rng(seed)
    # Only include tests with valid energy measurements
    valid = np.where(~np.isnan(energy_j))[0]
    results = {f"energy_to_{int(th*100)}cov_J": [] for th in thresholds}
    for _ in range(n_trials):
        ordering = valid[rng.permutation(len(valid))]
        metrics = energy_to_reach_cov_thresholds(
            sel_ordered=ordering,
            test_bits=test_bits,
            m_coverable=m_coverable,
            energy_j=energy_j,
            thresholds=thresholds,
        )
        for th in thresholds:
            key = f"energy_to_{int(th*100)}cov_J"
            results[key].append(metrics[key])
    return {k: float(np.nanmean(v)) for k, v in results.items()}


# ------------------------------------------------------------
# Baseline 2: Greedy coverage-per-second
# ------------------------------------------------------------

def greedy_order_by_cov_per_second(
    test_bits: np.ndarray,
    wall_s: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Greedy ordering of ALL tests by marginal coverage per second.
    Traditional time-based proxy baseline.
    """
    n_tests = test_bits.shape[0]
    if test_bits.shape[1] == 0:
        return np.arange(n_tests)

    t = np.where(np.isnan(wall_s), 1e12, wall_s)
    # Only include tests with valid energy AND time measurements
    valid = set(np.where(~np.isnan(wall_s))[0].tolist())
    remaining = [i for i in range(n_tests) if i in valid]
    ordered: List[int] = []
    covered = np.zeros((test_bits.shape[1],), dtype=np.uint64)

    while remaining:
        best_t: Optional[int] = None
        best_score = -1.0
        best_gain = -1
        best_time = None

        for t_idx in remaining:
            bits_t = test_bits[t_idx]
            new_bits = np.bitwise_and(bits_t, np.bitwise_not(covered))
            gain = _popcount_u64(new_bits)
            score = gain / (float(t[t_idx]) + eps)
            t_time = float(t[t_idx])
            if (
                score > best_score
                or (score == best_score and gain > best_gain)
                or (score == best_score and gain == best_gain
                    and (best_time is None or t_time < best_time))
            ):
                best_score = score
                best_gain = gain
                best_t = t_idx
                best_time = t_time

        assert best_t is not None
        ordered.append(best_t)
        covered |= test_bits[best_t]
        remaining.remove(best_t)

    return np.array(ordered, dtype=int)


# ------------------------------------------------------------
# Baseline 3: Greedy coverage-per-joule (no NSGA-II)
# ------------------------------------------------------------

def greedy_order_by_cov_per_joule(
    test_bits: np.ndarray,
    energy_j: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Greedy ordering of ALL tests by marginal coverage per joule.
    Energy-aware baseline without NSGA-II subset selection.
    """
    n_tests = test_bits.shape[0]
    if test_bits.shape[1] == 0:
        return np.arange(n_tests)

    e = np.where(np.isnan(energy_j), 1e12, energy_j)
    # Only include tests with valid energy measurements
    valid = set(np.where(~np.isnan(energy_j))[0].tolist())
    remaining = [i for i in range(n_tests) if i in valid]
    ordered: List[int] = []
    covered = np.zeros((test_bits.shape[1],), dtype=np.uint64)

    while remaining:
        best_t: Optional[int] = None
        best_score = -1.0
        best_gain = -1
        best_energy = None

        for t_idx in remaining:
            bits_t = test_bits[t_idx]
            new_bits = np.bitwise_and(bits_t, np.bitwise_not(covered))
            gain = _popcount_u64(new_bits)
            score = gain / (float(e[t_idx]) + eps)
            t_energy = float(e[t_idx])
            if (
                score > best_score
                or (score == best_score and gain > best_gain)
                or (score == best_score and gain == best_gain
                    and (best_energy is None or t_energy < best_energy))
            ):
                best_score = score
                best_gain = gain
                best_t = t_idx
                best_energy = t_energy

        assert best_t is not None
        ordered.append(best_t)
        covered |= test_bits[best_t]
        remaining.remove(best_t)

    return np.array(ordered, dtype=int)


# ------------------------------------------------------------
# Per-run baseline computation
# ------------------------------------------------------------

def run_baselines_for_run(
    run_id: int,
    pr_number: Optional[int],
    changed_str: str,
    idx: LineIndex,
    energy_j: np.ndarray,
    wall_s: np.ndarray,
    keep_prefix: str = "src/",
    thresholds=(0.80, 0.90, 0.95, 1.00),
    n_random_trials: int = 30,
    seed: int = 42,
) -> Dict:
    """
    For one PR run, compute all three baselines and return summary row.
    """
    changed = parse_modified_files_and_statements(changed_str)
    changed_ids = changed_lines_to_ids(changed, idx.line_map, keep_prefix=keep_prefix)

    base = {
        "run_id": run_id,
        "pr_number": pr_number,
        "changed_lines_mapped": len(changed_ids),
    }

    if len(changed_ids) == 0:
        base["status"] = "skip_no_changed_lines"
        base["coverable_changed_lines"] = 0
        return base

    test_bits, coverable_mask = build_test_bitsets_for_changed_lines(changed_ids, idx)
    test_bits2, m_coverable = restrict_to_coverable_lines(test_bits, coverable_mask)

    base["coverable_changed_lines"] = int(m_coverable)

    if m_coverable == 0:
        base["status"] = "skip_zero_coverable"
        return base

    base["status"] = "ok"

    # Baseline 1: Random
    rand_metrics = baseline_random(
        energy_j=energy_j,
        test_bits=test_bits2,
        m_coverable=m_coverable,
        thresholds=thresholds,
        n_trials=n_random_trials,
        seed=seed,
    )
    for k, v in rand_metrics.items():
        base[f"random_{k}"] = v

    # Baseline 2: Greedy cov/second
    ordered_by_time = greedy_order_by_cov_per_second(
        test_bits=test_bits2,
        wall_s=wall_s,
    )
    time_metrics = energy_to_reach_cov_thresholds(
        sel_ordered=ordered_by_time,
        test_bits=test_bits2,
        m_coverable=m_coverable,
        energy_j=energy_j,
        thresholds=thresholds,
    )
    for k, v in time_metrics.items():
        base[f"greedy_cov_per_second_{k}"] = v

    # Baseline 3: Greedy cov/joule (no NSGA-II)
    ordered_by_joule = greedy_order_by_cov_per_joule(
        test_bits=test_bits2,
        energy_j=energy_j,
    )
    joule_metrics = energy_to_reach_cov_thresholds(
        sel_ordered=ordered_by_joule,
        test_bits=test_bits2,
        m_coverable=m_coverable,
        energy_j=energy_j,
        thresholds=thresholds,
    )
    for k, v in joule_metrics.items():
        base[f"greedy_cov_per_joule_{k}"] = v

    return base


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-csv", required=True)
    ap.add_argument("--line-index-prefix", required=True)
    ap.add_argument("--energy-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-prefix", default="src/")
    ap.add_argument("--only-success", action="store_true")
    ap.add_argument("--n-random-trials", type=int, default=30,
                    help="Number of random trials to average for random baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = no limit; otherwise process first N eligible runs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading line index...")
    idx = load_line_index(args.line_index_prefix)

    print("Loading energy data...")
    energy_df = load_energy(args.energy_csv)
    energy_j, wall_s = align_tests(idx.test_names, energy_df)

    print("Loading PR runs...")
    df = pd.read_csv(args.runs_csv)

    if args.only_success:
        df = df[(df["status"] == "completed") & (df["conclusion"] == "success")].copy()

    df = df[df["modified_files_and_statements"].notna()].copy()
    print(f"Eligible runs: {len(df)}")

    summaries: List[Dict] = []
    count = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Baselines"):
        run_id = int(r["run_id"])
        pr_number = None if pd.isna(r.get("pr_number", np.nan)) else int(r["pr_number"])
        changed_str = str(r["modified_files_and_statements"])

        s = run_baselines_for_run(
            run_id=run_id,
            pr_number=pr_number,
            changed_str=changed_str,
            idx=idx,
            energy_j=energy_j,
            wall_s=wall_s,
            keep_prefix=args.keep_prefix,
            n_random_trials=args.n_random_trials,
            seed=args.seed,
        )
        summaries.append(s)

        count += 1
        if args.limit and count >= args.limit:
            break

    out_path = os.path.join(args.out_dir, "baseline_summary.csv")
    pd.DataFrame(summaries).to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()