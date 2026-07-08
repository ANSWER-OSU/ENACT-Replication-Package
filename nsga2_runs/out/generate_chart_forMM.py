#!/usr/bin/env python3
"""
Energy across runs/commits: Full suite vs Selected subset.

Produces FOUR plots:
  1) 95% target, linear y
  2) 95% target, log y
  3) 100% target, linear y
  4) 100% target, log y

Behavior:
- 95% plots: SORT by selected energy to reach 95% (ascending)
- 100% plots: SORT by selected energy to reach 100% (ascending)
- DROP runs with missing/unreachable/penalized selected energy (and optionally > baseline)

Presentation requirements:
1) All fonts 2x larger
2) X-axis label says "Commits (sorted)"
3) Each commit is a small dot on the chart line (markers)
4) Full-suite baseline (1-run energy) appears as a Y-AXIS TICK value (explicitly labeled)
   - MUST appear on BOTH linear AND log plots
5) NO totals / NO plotted run counts / NO extra annotation boxes
6) NO x-axis numeric tick labels at all
7) Linear plots y-axis starts at 0 (no negative energy)
8) Log plots MUST include baseline line/value (force ylim)
9) Log plots y tick labels must NOT overlap; show decades + baseline label

Inputs:
  --pareto-dir : directory with per-run pareto CSVs (*.csv) including columns:
                 energy_to_95cov_J, energy_to_100cov_J (and ideally run_id)
  --runs-csv   : CSV mapping run_id -> commit SHA (tries: head_sha, commit, commit_sha, sha, git_sha)
  --energy-csv : per-test energy file used to compute full-suite baseline energy (ONE run)
  --out-prefix : output file prefix
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

COMMIT_COL_CANDIDATES = ["head_sha", "commit", "commit_sha", "sha", "git_sha"]

# ---- Global plotting style: 2x font sizes ----
_BASE_FONT = plt.rcParams.get("font.size", 10)
_FONT2X = _BASE_FONT * 2
plt.rcParams.update({
    "font.size": _FONT2X,
    "axes.labelsize": _FONT2X,
    "axes.titlesize": _FONT2X,
    "xtick.labelsize": _FONT2X * 0.9,
    "ytick.labelsize": _FONT2X * 0.9,
    "legend.fontsize": _FONT2X * 0.9,
})


def full_suite_energy_j(energy_csv: str) -> float:
    df = pd.read_csv(energy_csv)
    if "rc" in df.columns:
        df = df[df["rc"] == 0].copy()
    if "energy_j" not in df.columns:
        raise ValueError(f"energy-csv missing 'energy_j' column: {energy_csv}")
    e = pd.to_numeric(df["energy_j"], errors="coerce").dropna()
    return float(e.sum())


def load_run_to_commit(runs_csv: str):
    df = pd.read_csv(runs_csv)
    if "run_id" not in df.columns:
        raise ValueError(f"runs-csv missing 'run_id' column: {runs_csv}")

    commit_col = None
    for c in COMMIT_COL_CANDIDATES:
        if c in df.columns:
            commit_col = c
            break

    if commit_col is None:
        return {int(r): str(int(r)) for r in df["run_id"].dropna().astype(int).tolist()}, None

    m = {}
    for _, r in df.iterrows():
        if pd.isna(r["run_id"]):
            continue
        rid = int(r["run_id"])
        val = r.get(commit_col)
        if pd.isna(val):
            continue
        m[rid] = str(val)
    return m, commit_col


def best_energy_to_target(pareto_csv: str, col: str) -> float:
    df = pd.read_csv(pareto_csv)
    if col not in df.columns:
        raise ValueError(f"{pareto_csv} missing column '{col}'")
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.min())


def extract_run_id_from_path_or_file(fp: str):
    base = os.path.basename(fp).split(".")[0]
    if base.isdigit():
        return int(base)
    try:
        dfp = pd.read_csv(fp, nrows=1)
        if "run_id" in dfp.columns and len(dfp) == 1 and pd.notna(dfp["run_id"].iloc[0]):
            return int(dfp["run_id"].iloc[0])
    except Exception:
        pass
    return None


def drop_penalties_and_missing(sel: np.ndarray, penalty_thresh: float) -> np.ndarray:
    sel = sel.astype(float).copy()
    sel[~np.isfinite(sel)] = np.nan
    sel[sel >= float(penalty_thresh)] = np.nan
    return sel


def drop_above_baseline(sel: np.ndarray, baseline: float) -> np.ndarray:
    sel = sel.astype(float).copy()
    sel[sel > float(baseline) * 1.0000001] = np.nan
    return sel


def _ensure_linear_baseline_tick(ax, baseline: float):
    if not np.isfinite(baseline):
        return
    ticks = list(ax.get_yticks())
    ticks.append(baseline)
    ax.set_yticks(sorted(set(ticks)))


def _set_log_ticks_with_baseline(ax, baseline: float, numticks: int = 7):
    """
    Log ticks: show only decades (10^k) to avoid overlap, plus baseline as a plain number.
    This avoids using LogFormatter inside FuncFormatter (which can crash in tight_layout).
    """
    if not np.isfinite(baseline) or baseline <= 0:
        return

    # Only decades: 1, 10, 100, ...
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=numticks))

    # Add baseline as an extra tick
    ticks = list(ax.get_yticks())
    ticks.append(float(baseline))

    lo, hi = ax.get_ylim()
    ticks = [t for t in ticks if np.isfinite(t) and t > 0 and lo <= t <= hi]
    ticks = sorted(set(ticks))
    ax.set_yticks(ticks)

    baseline_f = float(baseline)

    def fmt(y, _pos):
        if y <= 0 or not np.isfinite(y):
            return ""

        # baseline printed plainly
        if abs(y - baseline_f) / baseline_f < 1e-9:
            return f"{baseline_f:,.0f}"

        # decades printed as 10^k (mathtext)
        k = np.log10(y)
        k_round = int(np.round(k))
        if abs(k - k_round) < 1e-10:
            if k_round == 0:
                return "1"
            return rf"$10^{{{k_round}}}$"

        # no labels for non-decade ticks (shouldn't appear with subs=(1.0,), but safe)
        return ""

    ax.yaxis.set_major_formatter(FuncFormatter(fmt))


def make_plot(
    baseline_one_run: float,
    n_points: int,
    full_energy: np.ndarray,
    selected_energy: np.ndarray,
    target_label: str,
    out_png: str,
    out_pdf: str,
    yscale: str,
):
    x = np.arange(1, n_points + 1)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    ax.plot(x, full_energy, label="Full suite", marker="o", markersize=3, linewidth=1.4)
    ax.plot(x, selected_energy, label=f"Selected (to {target_label})", marker="o", markersize=3, linewidth=1.4)

    ax.set_xlabel("Commits (sorted)")
    ax.set_ylabel("Energy (J)")

    # No x-axis numbers at all
    ax.set_xticks([])

    if yscale == "log":
        ax.set_yscale("log")

        # Force y-limits to include baseline
        sel_pos = selected_energy[np.isfinite(selected_energy) & (selected_energy > 0)]
        if sel_pos.size == 0:
            ymin = baseline_one_run / 10.0 if baseline_one_run > 0 else 1.0
            ymax = baseline_one_run * 1.2 if baseline_one_run > 0 else 10.0
        else:
            sel_min = float(np.min(sel_pos))
            sel_max = float(np.max(sel_pos))
            ymax = max(float(baseline_one_run), sel_max) * 1.15

            ymin = sel_min * 0.90
            floor = ymax / 1e4  # show up to 4 orders of magnitude
            ymin = max(ymin, floor, 1e-12)

        ax.set_ylim(ymin, ymax)

        # Log ticks that don't overlap + baseline label
        _set_log_ticks_with_baseline(ax, baseline_one_run, numticks=7)

    else:
        ax.set_ylim(bottom=0)
        _ensure_linear_baseline_tick(ax, baseline_one_run)

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print("Wrote:", out_png)
    print("Wrote:", out_pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto-dir", required=True)
    ap.add_argument("--runs-csv", required=True)
    ap.add_argument("--energy-csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--penalty-thresh", type=float, default=1e11,
                    help="Drop selected energies >= this as penalty/unreachable (default 1e11).")
    ap.add_argument("--no-drop-above-baseline", action="store_true",
                    help="If set, do NOT drop selected energies that exceed full-suite baseline.")
    args = ap.parse_args()

    baseline = full_suite_energy_j(args.energy_csv)
    run_to_commit, _commit_col = load_run_to_commit(args.runs_csv)

    pareto_files = sorted(glob.glob(os.path.join(args.pareto_dir, "*.csv")))
    if not pareto_files:
        raise SystemExit(f"No pareto CSVs found in {args.pareto_dir}")

    rows = []
    for fp in pareto_files:
        rid = extract_run_id_from_path_or_file(fp)
        if rid is None:
            continue
        _commit = run_to_commit.get(rid, str(rid))  # traceability only; not plotted
        e95 = best_energy_to_target(fp, "energy_to_95cov_J")
        e100 = best_energy_to_target(fp, "energy_to_100cov_J")
        rows.append((rid, e95, e100))

    df_all = pd.DataFrame(rows, columns=["run_id", "selected_95_J", "selected_100_J"])

    df_all["selected_95_J"] = drop_penalties_and_missing(df_all["selected_95_J"].to_numpy(), args.penalty_thresh)
    df_all["selected_100_J"] = drop_penalties_and_missing(df_all["selected_100_J"].to_numpy(), args.penalty_thresh)

    if not args.no_drop_above_baseline:
        df_all["selected_95_J"] = drop_above_baseline(df_all["selected_95_J"].to_numpy(), baseline)
        df_all["selected_100_J"] = drop_above_baseline(df_all["selected_100_J"].to_numpy(), baseline)

    # 95%: drop missing + sort by selected energy
    df95 = df_all[np.isfinite(df_all["selected_95_J"].to_numpy())].copy()
    df95 = df95.sort_values(by=["selected_95_J", "run_id"], ascending=[True, True])

    n95 = len(df95)
    full95 = np.full(n95, baseline, dtype=float)
    sel95 = df95["selected_95_J"].to_numpy(dtype=float)

    make_plot(baseline, n95, full95, sel95, "95% change coverage",
              args.out_prefix + "_95_linear.png", args.out_prefix + "_95_linear.pdf", "linear")
    make_plot(baseline, n95, full95, sel95, "95% change coverage",
              args.out_prefix + "_95_log.png", args.out_prefix + "_95_log.pdf", "log")

    # 100%: drop missing + sort by selected energy
    df100 = df_all[np.isfinite(df_all["selected_100_J"].to_numpy())].copy()
    df100 = df100.sort_values(by=["selected_100_J", "run_id"], ascending=[True, True])

    n100 = len(df100)
    full100 = np.full(n100, baseline, dtype=float)
    sel100 = df100["selected_100_J"].to_numpy(dtype=float)

    make_plot(baseline, n100, full100, sel100, "100% change coverage",
              args.out_prefix + "_100_linear.png", args.out_prefix + "_100_linear.pdf", "linear")
    make_plot(baseline, n100, full100, sel100, "100% change coverage",
              args.out_prefix + "_100_log.png", args.out_prefix + "_100_log.pdf", "log")


if __name__ == "__main__":
    main()