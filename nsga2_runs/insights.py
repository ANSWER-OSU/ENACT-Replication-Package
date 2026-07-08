#!/usr/bin/env python3
"""
FSE-IVR analysis script for NSGA-II outputs.

What this script does (end-to-end)
1) Loads:
   - summary.csv (per-run NSGA-II summary)
   - codecarbon_energy_pertest_canonB.csv (per-test energy/time)
   - optional: pareto/ directory (per-run Pareto points) for example plots

2) Computes a clean "full suite" baseline:
   - total_energy_all_tests_j = sum(energy_j) across the per-test file
   - total_time_all_tests_s   = sum(wall_s) across the per-test file (if available)

3) Adds per-run derived metrics:
   - energy savings (%) at 80/90/95 targets vs full suite baseline
   - energy fraction at targets vs baseline
   - suite size fraction at targets vs total number of tests
   - diminishing returns deltas (80->90, 90->95) in energy and suite size

4) Produces paper-friendly artifacts in --out-dir:
   - per_run_summary_enriched.csv
   - tables/*.csv + some .tex snippets
   - figures/*.png
   - analysis_report.md

Assumptions (explicit)
- summary.csv contains columns:
    run_id,status,pareto_points,coverable_changed_lines,max_change_cov
    energy_j_at_80pct, suite_size_at_80pct
    energy_j_at_90pct, suite_size_at_90pct
    energy_j_at_95pct, suite_size_at_95pct
  (time_s_at_* optional)
- energy_pertest contains: test, energy_j, wall_s (wall_s optional)

Run example:
python3 analyze_nsga2_final.py \
  --summary /path/to/summary.csv \
  --energy-csv /path/to/codecarbon_energy_pertest_canonB.csv \
  --pareto-dir /path/to/pareto \
  --out-dir /path/to/out_analysis
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGETS = [80, 90, 95]


# -----------------------------
# Utilities
# -----------------------------

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(df.to_latex(index=False, escape=True))
    lines.append("\\end{table}")
    return "\n".join(lines)


def describe_series(s: pd.Series) -> Dict[str, float]:
    s = s.dropna().astype(float)
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": float(len(s)),
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "median": float(s.median()),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
    }


def save_boxplot(df: pd.DataFrame, cols: List[str], title: str, ylabel: str, outpath: Path) -> None:
    data = [df[c].dropna().values for c in cols]
    labels = cols
    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_hist(series: pd.Series, title: str, xlabel: str, outpath: Path, bins: int = 30) -> None:
    s = series.dropna().astype(float)
    plt.figure()
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_scatter(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    d = df[[x, y]].dropna()
    plt.figure()
    plt.scatter(d[x].values, d[y].values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def read_pareto(pareto_dir: Path, run_id: int) -> Optional[pd.DataFrame]:
    f = pareto_dir / f"{run_id}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    for c in ["energy_j", "change_cov", "suite_size", "time_s"]:
        if c not in df.columns:
            df[c] = np.nan
    to_numeric(df, ["energy_j", "change_cov", "suite_size", "time_s"])
    return df


def plot_pareto_front(pareto_df: pd.DataFrame, run_id: int, outpath: Path) -> None:
    plt.figure()
    plt.scatter(pareto_df["energy_j"], pareto_df["change_cov"])
    plt.title(f"Pareto front (run_id={run_id})")
    plt.xlabel("Energy (J)")
    plt.ylabel("Change-aware coverage")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def pick_example_runs(df_ok: pd.DataFrame) -> Dict[str, int]:
    """
    Pick 2-3 example runs for Pareto front visuals.
    """
    out: Dict[str, int] = {}
    d = df_ok[df_ok["pareto_points"].fillna(0) > 1].copy()
    if len(d) == 0:
        return out

    # richest Pareto front
    out["rich_front"] = int(d.sort_values("pareto_points", ascending=False).iloc[0]["run_id"])

    # best savings at 90
    if "energy_savings_pct_at_90pct" in d.columns:
        d90 = d.dropna(subset=["energy_savings_pct_at_90pct"])
        if len(d90) > 0:
            out["high_savings_90"] = int(d90.sort_values("energy_savings_pct_at_90pct", ascending=False).iloc[0]["run_id"])

    # low savings at 90 (hard case)
    if "energy_savings_pct_at_90pct" in d.columns:
        d90 = d.dropna(subset=["energy_savings_pct_at_90pct"])
        if len(d90) > 0:
            out["low_savings_90"] = int(d90.sort_values("energy_savings_pct_at_90pct", ascending=True).iloc[0]["run_id"])

    # ensure uniqueness
    seen = set()
    uniq = {}
    for k, v in out.items():
        if v not in seen:
            uniq[k] = v
            seen.add(v)
    return uniq


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to summary.csv")
    ap.add_argument("--energy-csv", required=True, help="Path to per-test energy csv (codecarbon_energy_pertest_*.csv)")
    ap.add_argument("--pareto-dir", default="", help="Optional path to pareto/ directory (run_id.csv files)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--only-ok", action="store_true", help="Analyze only status==ok runs")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    energy_path = Path(args.energy_csv)
    pareto_dir = Path(args.pareto_dir) if args.pareto_dir else None
    out_dir = Path(args.out_dir)

    safe_mkdir(out_dir)
    safe_mkdir(out_dir / "tables")
    safe_mkdir(out_dir / "figures")
    safe_mkdir(out_dir / "per_run")

    # Load inputs
    df = pd.read_csv(summary_path)
    energy_df = pd.read_csv(energy_path)

    if "status" not in df.columns:
        df["status"] = "ok"

    # Coerce numeric fields if present
    base_num_cols = [
        "run_id", "pr_number", "changed_lines_mapped", "coverable_changed_lines",
        "max_change_cov", "pareto_points"
    ]
    target_cols = []
    for t in TARGETS:
        target_cols += [f"energy_j_at_{t}pct", f"suite_size_at_{t}pct", f"time_s_at_{t}pct"]
        for c in target_cols:
            if c not in df.columns:
                df[c] = np.nan

    to_numeric(df, base_num_cols + target_cols)

    # Full-suite baseline from per-test file
    if "energy_j" not in energy_df.columns:
        raise SystemExit("energy-csv must contain an 'energy_j' column")

    energy_df["energy_j"] = pd.to_numeric(energy_df["energy_j"], errors="coerce")
    total_energy_all_tests_j = float(energy_df["energy_j"].sum(skipna=True))

    n_tests_all = None
    if "test" in energy_df.columns:
        n_tests_all = int(energy_df["test"].astype(str).nunique())

    total_time_all_tests_s = np.nan
    if "wall_s" in energy_df.columns:
        energy_df["wall_s"] = pd.to_numeric(energy_df["wall_s"], errors="coerce")
        total_time_all_tests_s = float(energy_df["wall_s"].sum(skipna=True))

    # Filter runs
    if args.only_ok:
        df = df[df["status"] == "ok"].copy()

    # Derived: energy fraction and savings vs full suite baseline
    for t in TARGETS:
        e_col = f"energy_j_at_{t}pct"

        df[f"energy_frac_of_full_at_{t}pct"] = np.nan
        df[f"energy_savings_pct_at_{t}pct"] = np.nan

        mask = df[e_col].notna() & (total_energy_all_tests_j > 0)
        df.loc[mask, f"energy_frac_of_full_at_{t}pct"] = df.loc[mask, e_col] / total_energy_all_tests_j
        df.loc[mask, f"energy_savings_pct_at_{t}pct"] = (1.0 - df.loc[mask, e_col] / total_energy_all_tests_j) * 100.0

    # Derived: suite size fraction vs total number of tests (if available)
    if n_tests_all is not None and n_tests_all > 0:
        for t in TARGETS:
            s_col = f"suite_size_at_{t}pct"
            df[f"suite_frac_of_full_at_{t}pct"] = np.nan
            mask = df[s_col].notna()
            df.loc[mask, f"suite_frac_of_full_at_{t}pct"] = df.loc[mask, s_col] / float(n_tests_all)

    # Diminishing returns deltas
    df["delta_energy_80_to_90_j"] = df["energy_j_at_90pct"] - df["energy_j_at_80pct"]
    df["delta_energy_90_to_95_j"] = df["energy_j_at_95pct"] - df["energy_j_at_90pct"]
    df["delta_suite_80_to_90"] = df["suite_size_at_90pct"] - df["suite_size_at_80pct"]
    df["delta_suite_90_to_95"] = df["suite_size_at_95pct"] - df["suite_size_at_90pct"]

    # Save enriched per-run summary
    enriched_path = out_dir / "per_run" / "per_run_summary_enriched.csv"
    df.to_csv(enriched_path, index=False)

    # -----------------------------
    # Tables (CSV + LaTeX)
    # -----------------------------
    # Basic stats
    basic = {
        "runs_total": int(len(df)),
        "runs_ok": int((df["status"] == "ok").sum()) if "status" in df.columns else int(len(df)),
        "total_energy_all_tests_j": total_energy_all_tests_j,
        "total_time_all_tests_s": total_time_all_tests_s,
        "n_tests_all": n_tests_all if n_tests_all is not None else np.nan,
        "median_coverable_changed_lines": float(df["coverable_changed_lines"].median(skipna=True)) if "coverable_changed_lines" in df.columns else np.nan,
        "median_pareto_points": float(df["pareto_points"].median(skipna=True)) if "pareto_points" in df.columns else np.nan,
    }
    basic_df = pd.DataFrame([basic])
    basic_df.to_csv(out_dir / "tables" / "table_basic_stats.csv", index=False)

    # Aggregate savings table
    savings_rows = []
    for t in TARGETS:
        s = df[f"energy_savings_pct_at_{t}pct"]
        stats = describe_series(s)
        savings_rows.append({
            "Coverage target": f"{t}%",
            "Savings median (%)": stats.get("median", np.nan),
            "Savings p10 (%)": stats.get("p10", np.nan),
            "Savings p90 (%)": stats.get("p90", np.nan),
            "Savings max (%)": stats.get("max", np.nan),
            "Savings mean (%)": stats.get("mean", np.nan),
        })
    savings_df = pd.DataFrame(savings_rows)
    savings_df.to_csv(out_dir / "tables" / "table_energy_savings.csv", index=False)
    (out_dir / "tables" / "table_energy_savings.tex").write_text(
        latex_table(savings_df, "Energy savings relative to executing the full test suite.", "tab:energy_savings_full"),
        encoding="utf-8"
    )

    # Aggregate energy fractions table (how much of full energy you used)
    frac_rows = []
    for t in TARGETS:
        s = df[f"energy_frac_of_full_at_{t}pct"]
        stats = describe_series(s)
        frac_rows.append({
            "Coverage target": f"{t}%",
            "Energy fraction median": stats.get("median", np.nan),
            "Energy fraction p10": stats.get("p10", np.nan),
            "Energy fraction p90": stats.get("p90", np.nan),
        })
    frac_df = pd.DataFrame(frac_rows)
    frac_df.to_csv(out_dir / "tables" / "table_energy_fraction.csv", index=False)
    (out_dir / "tables" / "table_energy_fraction.tex").write_text(
        latex_table(frac_df, "Energy required at coverage targets (fraction of full suite energy).", "tab:energy_frac_full"),
        encoding="utf-8"
    )

    # Diminishing returns table
    dim_df = pd.DataFrame([
        {
            "Metric": "Δ Energy (80→90) J",
            **describe_series(df["delta_energy_80_to_90_j"])
        },
        {
            "Metric": "Δ Energy (90→95) J",
            **describe_series(df["delta_energy_90_to_95_j"])
        },
        {
            "Metric": "Δ Suite size (80→90)",
            **describe_series(df["delta_suite_80_to_90"])
        },
        {
            "Metric": "Δ Suite size (90→95)",
            **describe_series(df["delta_suite_90_to_95"])
        },
    ])
    # Keep a compact subset of columns
    keep_cols = ["Metric", "n", "min", "p10", "median", "p90", "max", "mean"]
    dim_df = dim_df[[c for c in keep_cols if c in dim_df.columns]]
    dim_df.to_csv(out_dir / "tables" / "table_diminishing_returns.csv", index=False)

    # -----------------------------
    # Figures
    # -----------------------------
    save_boxplot(
        df,
        [f"energy_savings_pct_at_{t}pct" for t in TARGETS],
        "Energy savings at coverage targets (vs full suite)",
        "Energy savings (%)",
        out_dir / "figures" / "energy_savings_box.png",
    )

    save_boxplot(
        df,
        [f"energy_frac_of_full_at_{t}pct" for t in TARGETS],
        "Energy used at coverage targets (fraction of full suite)",
        "Energy fraction",
        out_dir / "figures" / "energy_fraction_box.png",
    )

    save_hist(
        df["pareto_points"].fillna(0),
        "Pareto front size across code changes",
        "Pareto points",
        out_dir / "figures" / "pareto_points_hist.png",
        bins=30,
    )

    # Change size vs savings at 90% (if coverable_changed_lines exists)
    if "coverable_changed_lines" in df.columns:
        save_scatter(
            df,
            "coverable_changed_lines",
            "energy_savings_pct_at_90pct",
            "Change size vs energy savings (90% target)",
            "Coverable changed lines",
            "Energy savings at 90% (%)",
            out_dir / "figures" / "changed_lines_vs_savings90.png",
        )

    # If suite size fraction exists, plot it too
    if n_tests_all is not None and f"suite_frac_of_full_at_90pct" in df.columns:
        save_boxplot(
            df,
            [f"suite_frac_of_full_at_{t}pct" for t in TARGETS],
            "Test suite fraction at coverage targets",
            "Fraction of all tests",
            out_dir / "figures" / "suite_fraction_box.png",
        )

    # Example Pareto fronts (optional)
    example_runs: Dict[str, int] = {}
    if pareto_dir is not None and pareto_dir.exists():
        df_ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
        example_runs = pick_example_runs(df_ok)
        for tag, rid in example_runs.items():
            pdf = read_pareto(pareto_dir, rid)
            if pdf is None or len(pdf) == 0:
                continue
            plot_pareto_front(pdf, rid, out_dir / "figures" / f"example_pareto_{tag}_run_{rid}.png")

    # -----------------------------
    # Report (Markdown)
    # -----------------------------
    def fmt(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{x:.3f}"

    s90 = describe_series(df["energy_savings_pct_at_90pct"])
    s80 = describe_series(df["energy_savings_pct_at_80pct"])
    s95 = describe_series(df["energy_savings_pct_at_95pct"])

    report = []
    report.append("# NSGA-II Energy–Coverage Analysis (Full-Suite Baseline)")
    report.append("")
    report.append(f"- Full-suite baseline energy: **{total_energy_all_tests_j:.2f} J**")
    if not np.isnan(total_time_all_tests_s):
        report.append(f"- Full-suite baseline time: **{total_time_all_tests_s:.2f} s** (sum of per-test wall time)")
    if n_tests_all is not None:
        report.append(f"- Test universe size (from energy csv): **{n_tests_all}** tests")
    report.append("")
    report.append(f"- Runs analyzed: **{len(df)}**")
    if "status" in df.columns:
        report.append(f"  - ok: **{int((df['status']=='ok').sum())}**")
        report.append(f"  - non-ok/skip: **{int((df['status']!='ok').sum())}**")
    report.append("")
    report.append("## Energy savings (% vs full suite)")
    report.append(f"- 80% target: median **{fmt(s80.get('median', np.nan))}**, p10 **{fmt(s80.get('p10', np.nan))}**, p90 **{fmt(s80.get('p90', np.nan))}**, max **{fmt(s80.get('max', np.nan))}**")
    report.append(f"- 90% target: median **{fmt(s90.get('median', np.nan))}**, p10 **{fmt(s90.get('p10', np.nan))}**, p90 **{fmt(s90.get('p90', np.nan))}**, max **{fmt(s90.get('max', np.nan))}**")
    report.append(f"- 95% target: median **{fmt(s95.get('median', np.nan))}**, p10 **{fmt(s95.get('p10', np.nan))}**, p90 **{fmt(s95.get('p90', np.nan))}**, max **{fmt(s95.get('max', np.nan))}**")
    report.append("")
    report.append("## Diminishing returns indicators")
    d1 = describe_series(df["delta_energy_80_to_90_j"])
    d2 = describe_series(df["delta_energy_90_to_95_j"])
    report.append(f"- ΔEnergy 80→90 (J): median **{fmt(d1.get('median', np.nan))}**, p90 **{fmt(d1.get('p90', np.nan))}**")
    report.append(f"- ΔEnergy 90→95 (J): median **{fmt(d2.get('median', np.nan))}**, p90 **{fmt(d2.get('p90', np.nan))}**")
    report.append("")
    report.append("## Artifacts")
    report.append(f"- Enriched per-run summary: `{enriched_path}`")
    report.append("- Tables: `tables/` (CSV + LaTeX snippets)")
    report.append("- Figures: `figures/` (boxplots/histograms/scatters)")
    if example_runs:
        report.append(f"- Example Pareto fronts: {', '.join([f'{k}={v}' for k,v in example_runs.items()])}")
    report.append("")
    report.append("## Notes for writing (FSE-IVR tone)")
    report.append("- Interpret results as **trade-offs** and **decision support**, not as a single best solution.")
    report.append("- Emphasize **variance across changes** and **diminishing returns** as the main takeaways.")
    report.append("- Be explicit that savings are relative to the measured full-suite baseline (sum of per-test energy).")

    report_path = out_dir / "analysis_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("✅ Done.")
    print("Output dir:", out_dir)
    print("Report:", report_path)
    print("Enriched summary:", enriched_path)


if __name__ == "__main__":
    main()