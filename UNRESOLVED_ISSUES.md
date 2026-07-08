# Unresolved Issues

This document tracks open questions found while preparing the repository
for publication. Each item states whether it **blocks exact reproduction**
of the paper's headline numbers, or only affects **historical provenance**
(i.e., traceability of how a number was produced, without necessarily
changing the number itself).

## 1. Claimed 2,746 executions vs. 2,718 rows found

- The claimed figure is **2,746 commit-associated executions**.
- `Workflow_runs/unittest_linux_runs.csv` (and `unittest_linux_all_history.csv`)
  currently contain **2,718 data rows** (2,705 unique `head_sha` values).
- The file that would document the raw, pre-filter execution count
  (`unittest_linux_runs_with_pr_changes.csv`, see item 2) is missing from
  the repository, so the 2,746 figure could not be traced to a concrete
  artifact during this audit.
- **Impact: provenance.** This does not block re-running the existing
  NSGA-II pipeline against the existing `nsga2_runs/out/nsga2_canonB/`
  results, but it means the "2,746" figure as stated cannot currently be
  regenerated from what's in the repo. Needs reconciliation against the
  original data source (was 2,746 measured at a later/earlier point in
  time than this CSV snapshot? does it include a different workflow or
  event type?).

## 2. Missing `unittest_linux_runs_with_pr_changes.csv`

- Every NSGA-II script (`nsga2_final.py`, `nsga2_with_order.py`,
  `nsga2_all_prs.py`, `nsga_calculate_en.py`, `baselines.py`) requires a
  `--runs-csv` input with a `modified_files_and_statements` column. This
  file is referenced by name in multiple docstrings but does not exist
  anywhere in the repository.
- `Workflow_runs/unittest_linux_runs.csv` has run metadata (id, status,
  conclusion, timestamps, head_sha) but **not** per-run modified
  files/lines.
- **Impact: blocks exact reproduction.** Without this file, none of the
  NSGA-II run scripts can currently be re-executed end-to-end from the
  files in this repository. The existing `nsga2_runs/out/*/summary.csv`
  and `pareto/*.csv` outputs are present and usable as-is, but the
  "run" pipeline stage is not independently reproducible until this input
  (or the script that derives it) is added.

## 3. Incomplete-looking seed directories

- `nsga2_runs/out/nsga2_all_pop80_gen120_seed1/` — 1,257 pareto CSVs,
  consistent with the canonical `nsga2_canonB` run (1,257 "ok" runs out of
  1,822 total).
- `nsga2_runs/out/nsga2_all_pop80_gen120_seed2/` — **1,144** pareto CSVs
  (~91% of seed1's count).
- `nsga2_runs/out/nsga2_all_pop80_gen120_seed2_rapl/` — **11** pareto CSVs
  only; this looks like a small RAPL-instrumentation smoke test rather than
  a full production seed run.
- No command/invocation log was found for any of these directories to
  confirm whether seed2/seed2_rapl were intentionally partial (e.g.,
  still running, or a scoped RAPL sanity check) or genuinely incomplete
  runs that were meant to finish.
- **Impact: blocks exact reproduction of "three random seeds" claim as
  currently evidenced.** If the paper's cross-seed statistics depend on
  three full runs, only one (`seed1`) is currently verifiable as complete
  in this snapshot. Does not affect the single-seed `nsga2_canonB` results
  already reported elsewhere in `nsga2_runs/out/`.

## 4. Missing `analyze_nsga2_final.py`

- `nsga2_runs/insights.py` references (in an example command, line 37)
  a script called `analyze_nsga2_final.py`. No file by that name exists
  anywhere in the repository.
- It is unclear whether this was renamed to one of the existing analysis
  scripts (`analyze.py`, `analyze2.py`, `nsga2_runs/out/nsga_summary.py`)
  or was never committed.
- **Impact: provenance.** `insights.py` itself is present and runnable
  independently; this only affects the ability to follow the exact example
  invocation documented inside it.

## 5. `baselines/` vs. `baselines_v2/`

- `nsga2_runs/out/baselines/baseline_summary.csv` and
  `nsga2_runs/out/baselines_v2/baseline_summary.csv` have the **same row
  count** (1,823 lines including header).
- `baselines_v2/` has additional `*_suite_size_at_*cov` columns not present
  in `baselines/`; otherwise the two appear to cover the same runs.
- No documentation states which one was used for any published comparison
  numbers.
- **Impact: provenance.** Both directories are present and usable;
  `baselines/` looks like a stale/superseded duplicate of `baselines_v2/`,
  but this has not been confirmed, so neither has been removed or renamed
  in this pass.

## Summary table

| # | Issue | Blocks exact reproduction? |
|---|---|---|
| 1 | 2,746 vs. 2,718 executions | No — provenance only |
| 2 | Missing `unittest_linux_runs_with_pr_changes.csv` | **Yes** |
| 3 | Incomplete seed2 / seed2_rapl directories | **Yes**, for the 3-seed claim specifically |
| 4 | Missing `analyze_nsga2_final.py` | No — provenance only |
| 5 | `baselines/` vs `baselines_v2/` | No — provenance only |
