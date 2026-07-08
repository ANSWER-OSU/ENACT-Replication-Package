# NSGA-II Energy–Coverage Analysis (Full-Suite Baseline)

- Full-suite baseline energy: **188600.30 J**
- Full-suite baseline time: **5812.09 s** (sum of per-test wall time)
- Test universe size (from energy csv): **613** tests

- Runs analyzed: **1257**
  - ok: **1257**
  - non-ok/skip: **0**

## Energy savings (% vs full suite)
- 80% target: median **96.174**, p10 **90.217**, p90 **97.089**, max **97.739**
- 90% target: median **96.157**, p10 **90.077**, p90 **97.089**, max **97.739**
- 95% target: median **96.147**, p10 **89.474**, p90 **97.089**, max **97.739**

## Diminishing returns indicators
- ΔEnergy 80→90 (J): median **0.000**, p90 **156.733**
- ΔEnergy 90→95 (J): median **0.000**, p90 **1012.215**

## Artifacts
- Enriched per-run summary: `/scratch/projects/anklesan/energy_aware_testing/Analysis/nsga2_runs/out/nsga2_80-120-order/analysis/per_run/per_run_summary_enriched.csv`
- Tables: `tables/` (CSV + LaTeX snippets)
- Figures: `figures/` (boxplots/histograms/scatters)
- Example Pareto fronts: rich_front=13383639497, high_savings_90=16206726524, low_savings_90=14501378990

## Notes for writing (FSE-IVR tone)
- Interpret results as **trade-offs** and **decision support**, not as a single best solution.
- Emphasize **variance across changes** and **diminishing returns** as the main takeaways.
- Be explicit that savings are relative to the measured full-suite baseline (sum of per-test energy).