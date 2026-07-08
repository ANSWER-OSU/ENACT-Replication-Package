#!/usr/bin/env python3
"""
Energy savings (%) relative to running the FULL test suite.

Baseline:
  E_full_suite = sum of energy_j over ALL tests

Savings% = (E_full_suite - E_target) / E_full_suite * 100
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pctiles(x, ps=(10, 50, 90)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {p: np.nan for p in ps}
    return {p: float(np.percentile(x, p)) for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--energy-csv", required=True,
                    help="Per-test energy CSV (codecarbon_energy_pertest_*.csv)")
    ap.add_argument("--charts-dir", default="")
    args = ap.parse_args()

    out_dir = args.out_dir
    charts_dir = args.charts_dir or os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # -------------------------------
    # Load per-test energy and compute full-suite energy
    # -------------------------------
    energy_df = pd.read_csv(args.energy_csv)
    if "rc" in energy_df.columns:
        energy_df = energy_df[energy_df["rc"] == 0]

    E_full_suite = energy_df["energy_j"].sum()

    print(f"\nFULL TEST SUITE ENERGY: {E_full_suite:.2f} J")

    # -------------------------------
    # Load summary.csv
    # -------------------------------
    summary = pd.read_csv(os.path.join(out_dir, "summary.csv"))
    summary = summary[summary["status"] == "ok"].copy()

    print("\n=== ENERGY SAVINGS RELATIVE TO FULL TEST SUITE ===")

    rows = []
    box_data = []
    labels = []

    for pct in [80, 90, 95]:
        col = f"energy_j_at_{pct}pct"
        if col not in summary.columns:
            continue

        e_target = pd.to_numeric(summary[col], errors="coerce")
        mask = np.isfinite(e_target) & (e_target <= E_full_suite)
        valid = mask.sum()

        if valid == 0:
            continue

        savings = (E_full_suite - e_target[mask]) / E_full_suite * 100.0
        ps = pctiles(savings.values, (10, 50, 90))
        mean = float(np.mean(savings.values))

        print(f"\nSavings at {pct}% change coverage:")
        print(f"  runs: {len(savings)}")
        print(f"  mean={mean:.1f}%")
        print(f"  p10={ps[10]:.1f}%  median={ps[50]:.1f}%  p90={ps[90]:.1f}%")

        rows.append({
            "coverage_pct": pct,
            "mean_savings_pct": mean,
            "median_savings_pct": ps[50],
            "p10_savings_pct": ps[10],
            "p90_savings_pct": ps[90],
        })

        box_data.append(savings.values)
        labels.append(f"{pct}%")

    # -------------------------------
    # Write table CSV
    # -------------------------------
    out_csv = os.path.join(out_dir, "energy_savings_vs_full_suite.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("\nWrote:", out_csv)

    # -------------------------------
    # Boxplot
    # -------------------------------
    if box_data:
        plt.figure()
        plt.boxplot(box_data, tick_labels=labels, showfliers=False)
        plt.ylabel("Energy savings (%) vs full test suite")
        plt.title("Energy savings from test selection (baseline: full suite)")
        out_png = os.path.join(charts_dir, "box_energy_savings_vs_full_suite.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("Wrote chart:", out_png)


if __name__ == "__main__":
    main()