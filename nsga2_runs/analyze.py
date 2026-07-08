#!/usr/bin/env python3
"""
Analyze NSGA-II outputs:
- summary.csv  (one row per run)
- pareto/*.csv (pareto solutions per run)

Prints key insights and generates charts.

Example:
  python3 analyze_nsga2_outputs.py \
    --out-dir /path/to/ENACT-Replication-Package/nsga2_runs/out/nsga2_all_pop80_gen120_seed1 \
    --charts-dir /path/to/ENACT-Replication-Package/nsga2_runs/out/nsga2_all_pop80_gen120_seed1/charts
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ptiles(x, ps=(10, 50, 90)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {p: np.nan for p in ps}
    return {p: float(np.percentile(x, p)) for p in ps}


def safe_series(df, col):
    if col not in df.columns:
        return pd.Series([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def load_all_pareto(pareto_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(pareto_dir, "*.csv")))
    frames = []
    for p in paths:
        try:
            d = pd.read_csv(p)
            frames.append(d)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Numeric coercion for safety
    for c in ["energy_j", "change_cov", "time_s", "suite_size",
              "energy_to_80cov_J", "energy_to_90cov_J", "energy_to_95cov_J", "energy_to_100cov_J"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize_runs(summary_df: pd.DataFrame):
    print("\n=== SUMMARY.CSV INSIGHTS ===")
    print("Rows:", len(summary_df))
    if "status" in summary_df.columns:
        print("\nStatus counts:")
        print(summary_df["status"].value_counts(dropna=False).to_string())

    ok = summary_df[summary_df.get("status", "") == "ok"].copy() if "status" in summary_df.columns else summary_df.copy()
    print("\nOK runs:", len(ok))

    for col in ["changed_lines_mapped", "coverable_changed_lines", "max_change_cov", "pareto_points"]:
        if col in ok.columns:
            s = safe_series(ok, col)
            pts = ptiles(s.values, (10, 50, 90))
            print(f"\n{col}:")
            print(f"  p10={pts[10]:.3f}  median={pts[50]:.3f}  p90={pts[90]:.3f}  min={np.nanmin(s.values):.3f}  max={np.nanmax(s.values):.3f}")

    # Energy/time/suite at 80/90/95% of attainable max (relative thresholds in summary.csv)
    for pct in [80, 90, 95]:
        e_col = f"energy_j_at_{pct}pct"
        t_col = f"time_s_at_{pct}pct"
        k_col = f"suite_size_at_{pct}pct"

        if e_col in ok.columns:
            e = safe_series(ok, e_col).values
            t = safe_series(ok, t_col).values if t_col in ok.columns else np.array([])
            k = safe_series(ok, k_col).values if k_col in ok.columns else np.array([])

            reach = np.isfinite(e)
            print(f"\nReachability: {pct}% of attainable max coverage")
            print(f"  reachable runs: {int(reach.sum())}/{len(ok)} ({(reach.mean()*100 if len(ok)>0 else 0):.1f}%)")

            pe = ptiles(e[reach], (10, 50, 90))
            print(f"  energy_j: p10={pe[10]:.3f}  median={pe[50]:.3f}  p90={pe[90]:.3f}")

            if t.size:
                pt = ptiles(t[reach], (10, 50, 90))
                print(f"  time_s  : p10={pt[10]:.3f}  median={pt[50]:.3f}  p90={pt[90]:.3f}")
            if k.size:
                pk = ptiles(k[reach], (10, 50, 90))
                print(f"  suite_k : p10={pk[10]:.1f}  median={pk[50]:.1f}  p90={pk[90]:.1f}")

    # “Knee” behavior: 95% vs 80% extra energy (only where both exist)
    if "energy_j_at_80pct" in ok.columns and "energy_j_at_95pct" in ok.columns:
        e80 = safe_series(ok, "energy_j_at_80pct")
        e95 = safe_series(ok, "energy_j_at_95pct")
        mask = np.isfinite(e80) & np.isfinite(e95) & (e80 > 0)
        ratio = (e95[mask] / e80[mask]).values
        if ratio.size:
            pr = ptiles(ratio, (10, 50, 90))
            print("\nEnergy inflation (95% attainable / 80% attainable):")
            print(f"  p10={pr[10]:.3f}  median={pr[50]:.3f}  p90={pr[90]:.3f}")
            print(f"  (Interpretation: how much more energy you pay to go from “good enough” to “near max”.)")
        else:
            print("\nEnergy inflation (95/80): not enough runs with both values.")


def summarize_pareto(pareto_df: pd.DataFrame):
    print("\n=== PARETO/*.CSV INSIGHTS ===")
    if pareto_df.empty:
        print("No pareto files loaded.")
        return

    print("Pareto rows:", len(pareto_df))
    if "run_id" in pareto_df.columns:
        print("Unique runs:", pareto_df["run_id"].nunique())

    # Dedup by objective columns (common repeats from random-keys)
    dedup_cols = [c for c in ["run_id", "energy_j", "change_cov", "time_s", "suite_size",
                              "energy_to_80cov_J", "energy_to_90cov_J", "energy_to_95cov_J", "energy_to_100cov_J"]
                  if c in pareto_df.columns]
    pareto_dedup = pareto_df.drop_duplicates(subset=dedup_cols).copy()
    print("Deduped pareto rows:", len(pareto_dedup), f"({(len(pareto_dedup)/len(pareto_df)*100):.1f}% kept)")

    # How often absolute threshold metrics exist (non-NaN) across all solutions
    for col in ["energy_to_80cov_J", "energy_to_90cov_J", "energy_to_95cov_J", "energy_to_100cov_J"]:
        if col in pareto_dedup.columns:
            s = pareto_dedup[col]
            frac = float(np.isfinite(s).mean()) * 100.0
            print(f"{col}: finite in {frac:.1f}% of solutions")

    # Best (minimum) energy to reach 100% per run (absolute threshold)
    if "energy_to_100cov_J" in pareto_dedup.columns and "run_id" in pareto_dedup.columns:
        g = pareto_dedup.groupby("run_id")["energy_to_100cov_J"].min()
        g = g[np.isfinite(g)]
        if len(g):
            pg = ptiles(g.values, (10, 50, 90))
            print("\nBest energy_to_100cov_J per run (absolute 100%):")
            print(f"  runs with reachable 100%: {len(g)}")
            print(f"  p10={pg[10]:.3f}  median={pg[50]:.3f}  p90={pg[90]:.3f}")
        else:
            print("\nNo runs have reachable 100% absolute coverage (energy_to_100cov_J all NaN).")


def make_charts(summary_df: pd.DataFrame, pareto_df: pd.DataFrame, charts_dir: str, max_scatter_points: int = 200_000):
    os.makedirs(charts_dir, exist_ok=True)

    # 1) Histogram of max_change_cov
    if "max_change_cov" in summary_df.columns:
        x = safe_series(summary_df, "max_change_cov").values
        x = x[np.isfinite(x)]
        if x.size:
            plt.figure()
            plt.hist(x, bins=30)
            plt.xlabel("max_change_cov per run")
            plt.ylabel("count")
            plt.title("Distribution of max change coverage attainable (per run)")
            out = os.path.join(charts_dir, "hist_max_change_cov.png")
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            print("Wrote chart:", out)

    # 2) Boxplot: energy at 80/90/95% of attainable
    cols = []
    labels = []
    for pct in [80, 90, 95]:
        c = f"energy_j_at_{pct}pct"
        if c in summary_df.columns:
            v = safe_series(summary_df, c).values
            v = v[np.isfinite(v)]
            if v.size:
                cols.append(v)
                labels.append(f"{pct}%")
    if cols:
        plt.figure()
        plt.boxplot(cols, labels=labels, showfliers=False)
        plt.ylabel("energy (J)")
        plt.title("Energy to reach % of attainable max change coverage (per run)")
        out = os.path.join(charts_dir, "box_energy_at_80_90_95_attainable.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print("Wrote chart:", out)

    # 3) Scatter: pareto energy vs change_cov (downsample if huge)
    if not pareto_df.empty and "energy_j" in pareto_df.columns and "change_cov" in pareto_df.columns:
        d = pareto_df[["energy_j", "change_cov"]].dropna()
        # basic sanity
        d = d[np.isfinite(d["energy_j"]) & np.isfinite(d["change_cov"])]
        if len(d) > max_scatter_points:
            d = d.sample(max_scatter_points, random_state=1)
        if len(d):
            plt.figure()
            plt.scatter(d["energy_j"].values, d["change_cov"].values, s=4, alpha=0.2)
            plt.xlabel("energy (J)")
            plt.ylabel("change coverage")
            plt.title("Pareto solutions: energy vs change coverage")
            out = os.path.join(charts_dir, "scatter_energy_vs_change_cov.png")
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            print("Wrote chart:", out)

    # 4) Histogram of best energy_to_100cov_J per run (absolute)
    if not pareto_df.empty and "run_id" in pareto_df.columns and "energy_to_100cov_J" in pareto_df.columns:
        d = pareto_df.drop_duplicates(subset=["run_id", "energy_to_100cov_J", "energy_j", "change_cov", "suite_size"], keep="first")
        g = d.groupby("run_id")["energy_to_100cov_J"].min()
        g = g[np.isfinite(g)]
        if len(g):
            plt.figure()
            plt.hist(g.values, bins=30)
            plt.xlabel("best energy_to_100cov_J per run (J)")
            plt.ylabel("count of runs")
            plt.title("Distribution: energy to reach 100% absolute change coverage (best per run)")
            out = os.path.join(charts_dir, "hist_best_energy_to_100cov_per_run.png")
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            print("Wrote chart:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="NSGA output dir containing summary.csv and pareto/")
    ap.add_argument("--charts-dir", default="", help="Where to write PNG charts (default: <out-dir>/charts)")
    ap.add_argument("--max-scatter-points", type=int, default=200_000)
    args = ap.parse_args()

    out_dir = args.out_dir
    charts_dir = args.charts_dir or os.path.join(out_dir, "charts")

    summary_path = os.path.join(out_dir, "summary.csv")
    pareto_dir = os.path.join(out_dir, "pareto")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing {summary_path}")
    if not os.path.isdir(pareto_dir):
        raise FileNotFoundError(f"Missing {pareto_dir}")

    summary_df = pd.read_csv(summary_path)
    pareto_df = load_all_pareto(pareto_dir)

    summarize_runs(summary_df)
    summarize_pareto(pareto_df)
    make_charts(summary_df, pareto_df, charts_dir, max_scatter_points=args.max_scatter_points)

    print("\nDone.")
    print("Charts in:", charts_dir)


if __name__ == "__main__":
    main()