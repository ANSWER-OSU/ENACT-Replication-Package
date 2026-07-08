#!/usr/bin/env python3
"""
Analyze NSGA-II summary.csv and generate paper-friendly stats + plots.

Input columns (expected):
run_id, pr_number, changed_lines_mapped, coverable_changed_lines, max_change_cov,
pareto_points, status,
energy_j_at_80pct, suite_size_at_80pct,
energy_j_at_90pct, suite_size_at_90pct,
energy_j_at_95pct, suite_size_at_95pct

Outputs (in --out-dir):
- overview.txt                    : human-readable summary
- overview_table.tex              : small LaTeX table of key aggregates
- ok_runs.csv, skipped_runs.csv   : split datasets
- candidates_best_*.csv           : “showcase” runs for figures/tables
- plots/*.png                     : distributions + trade-off scatter plots
"""

import argparse
import os
import textwrap
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


THRESHOLDS = [80, 90, 95]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _to_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    out["n_total"] = len(df)
    out["n_ok"] = int((df["status"] == "ok").sum())
    out["n_skipped"] = out["n_total"] - out["n_ok"]

    # status breakdown
    out["status_counts"] = df["status"].value_counts(dropna=False).to_dict()

    ok = df[df["status"] == "ok"].copy()

    def stats(series: pd.Series):
        series = series.dropna()
        if len(series) == 0:
            return {"n": 0}
        return {
            "n": int(series.shape[0]),
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }

    out["ok_changed_lines_mapped"] = stats(ok.get("changed_lines_mapped", pd.Series(dtype=float)))
    out["ok_coverable_changed_lines"] = stats(ok.get("coverable_changed_lines", pd.Series(dtype=float)))
    out["ok_pareto_points"] = stats(ok.get("pareto_points", pd.Series(dtype=float)))
    out["ok_max_change_cov"] = stats(ok.get("max_change_cov", pd.Series(dtype=float)))

    # threshold summaries
    for t in THRESHOLDS:
        e = ok.get(f"energy_j_at_{t}pct", pd.Series(dtype=float))
        s = ok.get(f"suite_size_at_{t}pct", pd.Series(dtype=float))
        out[f"ok_energy_{t}"] = stats(e)
        out[f"ok_suite_{t}"] = stats(s)

        # energy per test (rough efficiency proxy)
        ratio = None
        if e is not None and s is not None:
            ratio = (e / s).replace([np.inf, -np.inf], np.nan)
        out[f"ok_energy_per_test_{t}"] = stats(ratio if ratio is not None else pd.Series(dtype=float))

    return out


def write_overview_txt(summary: dict, out_path: str) -> None:
    lines = []
    lines.append("NSGA-II summary.csv analysis\n")
    lines.append(f"Total runs: {summary['n_total']}")
    lines.append(f"OK runs:    {summary['n_ok']}")
    lines.append(f"Skipped:    {summary['n_skipped']}\n")

    lines.append("Status breakdown:")
    for k, v in summary["status_counts"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")

    def fmt_block(title: str, block: dict):
        if block.get("n", 0) == 0:
            return [f"{title}: (no data)"]
        return [
            f"{title}: n={block['n']}, "
            f"min={block['min']:.3g}, p25={block['p25']:.3g}, median={block['median']:.3g}, "
            f"p75={block['p75']:.3g}, max={block['max']:.3g}, mean={block['mean']:.3g}"
        ]

    lines += fmt_block("OK changed_lines_mapped", summary["ok_changed_lines_mapped"])
    lines += fmt_block("OK coverable_changed_lines", summary["ok_coverable_changed_lines"])
    lines += fmt_block("OK pareto_points", summary["ok_pareto_points"])
    lines += fmt_block("OK max_change_cov", summary["ok_max_change_cov"])
    lines.append("")

    for t in THRESHOLDS:
        lines += fmt_block(f"OK energy_j_at_{t}pct", summary[f"ok_energy_{t}"])
        lines += fmt_block(f"OK suite_size_at_{t}pct", summary[f"ok_suite_{t}"])
        lines += fmt_block(f"OK energy_per_test_at_{t}pct", summary[f"ok_energy_per_test_{t}"])
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_overview_table_tex(summary: dict, out_path: str) -> None:
    # Small table suitable for IVR: counts + medians for 80/90/95
    def med(block): return "—" if block.get("n", 0) == 0 else f"{block['median']:.3g}"
    def mean(block): return "—" if block.get("n", 0) == 0 else f"{block['mean']:.3g}"

    rows = []
    rows.append(r"\begin{table}[t]")
    rows.append(r"\centering")
    rows.append(r"\caption{Summary of NSGA-II results across runs (OK runs only).}")
    rows.append(r"\label{tab:nsga_summary}")
    rows.append(r"\begin{tabular}{lccc}")
    rows.append(r"\hline")
    rows.append(r"\textbf{Metric} & \textbf{80\%} & \textbf{90\%} & \textbf{95\%} \\")
    rows.append(r"\hline")

    rows.append(rf"Median energy (J) & {med(summary['ok_energy_80'])} & {med(summary['ok_energy_90'])} & {med(summary['ok_energy_95'])} \\")
    rows.append(rf"Mean energy (J) & {mean(summary['ok_energy_80'])} & {mean(summary['ok_energy_90'])} & {mean(summary['ok_energy_95'])} \\")
    rows.append(rf"Median suite size & {med(summary['ok_suite_80'])} & {med(summary['ok_suite_90'])} & {med(summary['ok_suite_95'])} \\")
    rows.append(rf"Mean suite size & {mean(summary['ok_suite_80'])} & {mean(summary['ok_suite_90'])} & {mean(summary['ok_suite_95'])} \\")
    rows.append(r"\hline")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(rows))


def plot_hist(series: pd.Series, title: str, out_path: str, xlabel: str):
    series = series.dropna()
    if len(series) == 0:
        return
    plt.figure()
    plt.hist(series.values, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(x: pd.Series, y: pd.Series, title: str, out_path: str, xlabel: str, ylabel: str):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) == 0:
        return
    plt.figure()
    plt.scatter(df["x"], df["y"], s=12)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def pick_showcase_runs(ok: pd.DataFrame, out_dir: str):
    """
    Create small CSVs for paper figures:
    - best energy efficiency at 90% (min energy)
    - smallest suite at 90% (min suite size)
    - typical run (closest to median energy at 90%)
    """
    t = 90
    e_col = f"energy_j_at_{t}pct"
    s_col = f"suite_size_at_{t}pct"

    cand = ok.dropna(subset=[e_col, s_col]).copy()
    if len(cand) == 0:
        return

    best_energy = cand.sort_values([e_col, s_col]).head(20)
    best_suite = cand.sort_values([s_col, e_col]).head(20)

    med_e = cand[e_col].median()
    cand["abs_diff_med_energy"] = (cand[e_col] - med_e).abs()
    typical = cand.sort_values(["abs_diff_med_energy", s_col]).head(20)

    best_energy.to_csv(os.path.join(out_dir, f"candidates_best_energy_{t}.csv"), index=False)
    best_suite.to_csv(os.path.join(out_dir, f"candidates_smallest_suite_{t}.csv"), index=False)
    typical.to_csv(os.path.join(out_dir, f"candidates_typical_{t}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True, help="Path to nsga2 summary.csv")
    ap.add_argument("--out-dir", required=True, help="Directory for outputs")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)
    plots_dir = os.path.join(args.out_dir, "plots")
    _ensure_dir(plots_dir)

    df = pd.read_csv(args.summary_csv)

    # Normalize numeric columns
    num_cols = [
        "changed_lines_mapped", "coverable_changed_lines", "max_change_cov", "pareto_points",
        "energy_j_at_80pct", "suite_size_at_80pct",
        "energy_j_at_90pct", "suite_size_at_90pct",
        "energy_j_at_95pct", "suite_size_at_95pct",
    ]
    df = _to_num(df, num_cols)

    # Split
    ok = df[df["status"] == "ok"].copy()
    skipped = df[df["status"] != "ok"].copy()

    ok.to_csv(os.path.join(args.out_dir, "ok_runs.csv"), index=False)
    skipped.to_csv(os.path.join(args.out_dir, "skipped_runs.csv"), index=False)

    # Summary stats
    summ = summarize(df)
    write_overview_txt(summ, os.path.join(args.out_dir, "overview.txt"))
    write_overview_table_tex(summ, os.path.join(args.out_dir, "overview_table.tex"))

    # Plots (OK only)
    plot_hist(ok.get("changed_lines_mapped", pd.Series(dtype=float)),
              "Changed lines mapped per run (OK runs)", os.path.join(plots_dir, "hist_changed_lines.png"),
              "changed_lines_mapped")

    plot_hist(ok.get("coverable_changed_lines", pd.Series(dtype=float)),
              "Coverable changed lines per run (OK runs)", os.path.join(plots_dir, "hist_coverable_lines.png"),
              "coverable_changed_lines")

    plot_hist(ok.get("pareto_points", pd.Series(dtype=float)),
              "Pareto points per run (OK runs)", os.path.join(plots_dir, "hist_pareto_points.png"),
              "pareto_points")

    plot_hist(ok.get("max_change_cov", pd.Series(dtype=float)),
              "Max change-aware coverage per run (OK runs)", os.path.join(plots_dir, "hist_max_change_cov.png"),
              "max_change_cov")

    # Trade-off proxy plots (energy vs suite size at thresholds)
    for t in THRESHOLDS:
        e = ok.get(f"energy_j_at_{t}pct", pd.Series(dtype=float))
        s = ok.get(f"suite_size_at_{t}pct", pd.Series(dtype=float))
        plot_scatter(e, s,
                     f"Energy vs suite size at {t}% of max change-aware coverage (OK runs)",
                     os.path.join(plots_dir, f"scatter_energy_vs_suite_{t}.png"),
                     f"energy_j_at_{t}pct", f"suite_size_at_{t}pct")

        ratio = (e / s).replace([np.inf, -np.inf], np.nan)
        plot_hist(ratio,
                  f"Energy per selected test at {t}% (OK runs)",
                  os.path.join(plots_dir, f"hist_energy_per_test_{t}.png"),
                  f"energy_j_at_{t}pct / suite_size_at_{t}pct")

    # Save some “showcase” candidates for paper figures/tables
    pick_showcase_runs(ok, args.out_dir)

    print("Wrote outputs to:", args.out_dir)


if __name__ == "__main__":
    main()