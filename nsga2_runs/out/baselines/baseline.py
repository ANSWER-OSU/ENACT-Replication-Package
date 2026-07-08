#!/usr/bin/env python3
"""
compare_baselines.py

Merges ENACT summary.csv with baseline_summary.csv and computes
aggregate statistics for each method at each coverage threshold.

Output:
  comparison_stats.csv   -- mean, median, p10, p90 for each method x threshold
  comparison_table.txt   -- formatted table ready for the paper
"""

import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enact-summary", required=True,
                    help="ENACT summary.csv from nsga2_final.py")
    ap.add_argument("--baseline-summary", required=True,
                    help="baseline_summary.csv from run_baselines.py")
    ap.add_argument("--full-suite-energy-j", type=float, default=188600.0,
                    help="Full test suite energy in joules (default 188.6 kJ)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    enact = pd.read_csv(args.enact_summary)
    base = pd.read_csv(args.baseline_summary)

    # Keep only successful runs in both
    enact_ok = enact[enact["status"] == "ok"].copy()
    base_ok = base[base["status"] == "ok"].copy()

    # Merge on run_id
    merged = enact_ok.merge(base_ok, on="run_id", suffixes=("_enact", "_base"))
    print(f"Matched runs: {len(merged)}")

    full_suite_j = args.full_suite_energy_j
    thresholds = [80, 90, 95, 100]

    # For ENACT, energy columns are energy_j_at_{80,90,95}pct
    # For 100% we use energy_to_100cov_J from the pareto — but summary.csv
    # doesn't have it directly. We'll use the baseline 100% columns only.

    methods = {
        "ENACT": {
            80: "energy_j_at_80pct",
            90: "energy_j_at_90pct",
            95: "energy_j_at_95pct",
        },
        "Random": {
            80: "random_energy_to_80cov_J",
            90: "random_energy_to_90cov_J",
            95: "random_energy_to_95cov_J",
            100: "random_energy_to_100cov_J",
        },
        "Greedy (cov/s)": {
            80: "greedy_cov_per_second_energy_to_80cov_J",
            90: "greedy_cov_per_second_energy_to_90cov_J",
            95: "greedy_cov_per_second_energy_to_95cov_J",
            100: "greedy_cov_per_second_energy_to_100cov_J",
        },
        "Greedy (cov/J)": {
            80: "greedy_cov_per_joule_energy_to_80cov_J",
            90: "greedy_cov_per_joule_energy_to_90cov_J",
            95: "greedy_cov_per_joule_energy_to_95cov_J",
            100: "greedy_cov_per_joule_energy_to_100cov_J",
        },
    }

    rows = []
    for method, cols in methods.items():
        for th, col in cols.items():
            # Find column in merged df
            if col in merged.columns:
                values = merged[col].dropna()
            elif col + "_base" in merged.columns:
                values = merged[col + "_base"].dropna()
            elif col + "_enact" in merged.columns:
                values = merged[col + "_enact"].dropna()
            else:
                print(f"[WARN] Column not found: {col}")
                continue

            # Compute energy savings % relative to full suite
            savings = ((full_suite_j - values) / full_suite_j * 100).clip(0, 100)

            rows.append({
                "Method": method,
                "Coverage Target (%)": th,
                "Energy Savings Mean (%)": savings.mean(),
                "Energy Savings Median (%)": savings.median(),
                "Energy Savings p10 (%)": savings.quantile(0.10),
                "Energy Savings p90 (%)": savings.quantile(0.90),
                "n": len(values),
            })

    df_stats = pd.DataFrame(rows)
    stats_path = os.path.join(args.out_dir, "comparison_stats.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"\nWrote: {stats_path}")

    # Print formatted table
    print("\n=== Comparison Table ===\n")
    print(f"{'Method':<20} {'Cov':>5} {'Mean':>8} {'Median':>8} {'p10':>8} {'p90':>8} {'n':>6}")
    print("-" * 65)
    for _, r in df_stats.iterrows():
        print(
            f"{r['Method']:<20} "
            f"{r['Coverage Target (%)']:>5.0f} "
            f"{r['Energy Savings Mean (%)']:>8.2f} "
            f"{r['Energy Savings Median (%)']:>8.2f} "
            f"{r['Energy Savings p10 (%)']:>8.2f} "
            f"{r['Energy Savings p90 (%)']:>8.2f} "
            f"{r['n']:>6.0f}"
        )

    # Also write LaTeX table
    latex_path = os.path.join(args.out_dir, "comparison_table.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\caption{Energy Savings Comparison. All values are percentages relative\n")
        f.write("  to full suite execution across matched runs.}\n")
        f.write("  \\label{tab:baseline-comparison}\n")
        f.write("  \\centering\n")
        f.write("  \\begin{tabular}{llcccc}\n")
        f.write("    \\toprule\n")
        f.write("    Method & Coverage & \\multicolumn{4}{c}{Energy Savings (\\%)} \\\\\n")
        f.write("    \\cmidrule(lr){3-6}\n")
        f.write("    & Target & Mean & Median & p10 & p90 \\\\\n")
        f.write("    \\midrule\n")

        current_method = None
        for _, r in df_stats.iterrows():
            method = r["Method"]
            if method != current_method:
                if current_method is not None:
                    f.write("    \\midrule\n")
                current_method = method
            f.write(
                f"    {method} & {r['Coverage Target (%)']:.0f}\\% & "
                f"{r['Energy Savings Mean (%)']:.2f} & "
                f"{r['Energy Savings Median (%)']:.2f} & "
                f"{r['Energy Savings p10 (%)']:.2f} & "
                f"{r['Energy Savings p90 (%)']:.2f} \\\\\n"
            )
            method = ""  # blank for subsequent rows of same method

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nWrote LaTeX table: {latex_path}")


if __name__ == "__main__":
    main()