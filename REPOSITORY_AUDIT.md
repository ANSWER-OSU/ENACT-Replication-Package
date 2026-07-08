# Repository Audit

Snapshot taken during repository preparation (pre-first-push). This
document reflects the state of the working tree at audit time; re-run the
inventory commands below if the tree changes materially.

## Current repository size

- Total working tree: **~45 GB**
- Two prior commits on `main` (`Initial commit`, `Add project description
  to README`); the six data directories listed below were **untracked** at
  audit time.

## Directory inventory

| Directory | Size | Contents |
|---|---|---|
| `unittest_linux_bigbig_coverage/` | ~36 GB | Raw per-test gcov `.info` files (`pertest/`, 615 files, ~5.4 GB), processed coverage matrices (`matrix/`, dominated by a ~31 GB `pertest_line_coverage.csv`), ctest lists, per-test energy CSV |
| `coverage_results/` | ~8.7 GB | `per_test/` (small, ~13 MB summaries+logs) and `quickmax/` (dominated by ~8.7 GB of raw `gcovr_verbose_*.log`, 289 files; summary CSVs themselves are small) |
| `nsga2_runs/` | ~304 MB | Core algorithm scripts (`scripts/`), NSGA-II run outputs (`out/`: pareto CSVs, `summary.csv`, tables, charts), and a captured Python venv (`out/venv_nsga2/`, ~254 MB, now gitignored) |
| `energy_results/` | ~14 MB | CodeCarbon energy CSVs, per-run Slurm/driver logs, `.tar` regression artifacts (now gitignored) |
| `Workflow_runs/` | ~4 MB | GitHub Actions run metadata (CSV/JSON) pulled via `Workflow_runs/runs_linux_bigbig.py` |
| `Workflow_Energy_Consumption/` | ~172 KB | Per-test energy summaries plus a small CSV-summation script |

## Canonical implementation files

- **`nsga2_runs/scripts/nsga2_final.py`** is currently treated as the
  canonical ENACT NSGA-II implementation: random-keys permutation + cut-point
  encoding, 4 objectives (energy, coverage, time, suite size), pymoo `NSGA2`.
- Related/earlier variants kept for provenance, not canonical:
  `nsga2_all_prs.py` (binary-selection encoding, earliest), `nsga2_with_order.py`
  (ordering+cut, no Option A reordering/threshold metrics), `nsga_calculate_en.py`
  (same as `nsga2_final.py` plus optional RAPL self-instrumentation of the
  optimizer).
- `nsga2_runs/scripts/baselines.py` — the three comparison baselines (random,
  greedy-cov/sec, greedy-cov/joule).
- `nsga2_runs/scripts/build_line_index.py` — builds the line→tests CSR index
  consumed by all of the above.

## Canonical result directory

- **`nsga2_runs/out/nsga2_canonB/`** is currently treated as the canonical
  result set (1,257 "ok" pareto runs out of 1,822 total, matching the
  reported `1257,1257,188600.29600000003,5812.0916050000005,613,...`
  basic-stats row).
- Other `nsga2_runs/out/nsga2_*` directories (`nsga2_80-120`, `nsga2_80-120-order`,
  `nsga2_all_prs`, `nsga2_all_pop80_gen120_seed1/2/2_rapl`, `nsga2_test`) appear to
  be earlier/partial/exploratory runs of the same pipeline and are not treated
  as canonical. See `UNRESOLVED_ISSUES.md` for the seed-completeness question.
- `nsga2_runs/out/baselines_v2/` supersedes `nsga2_runs/out/baselines/`
  (same row count, `_v2` adds `suite_size_at_*cov` columns); `baselines/`
  looks like a stale duplicate, not removed at this stage.

## Large files excluded from version control

Explicitly gitignored (see `.gitignore` for full rationale):

- `nsga2_runs/out/venv_nsga2/` — ~254 MB captured Python venv
- `nsga2_runs/scripts/__pycache__/` — compiled bytecode
- `coverage_results/quickmax/logs/` — ~8.7 GB raw gcovr verbose logs
- `unittest_linux_bigbig_coverage/pertest/` — ~5.4 GB raw per-test `.info` files
- `unittest_linux_bigbig_coverage/matrix/pertest_line_coverage.csv` — ~31 GB, largest single file in the repo
- `energy_results/**/*.out`, `energy_results/**/*.err` — Slurm scheduler stdout/stderr
- `energy_results/**/*.tar` — raw regression-test artifacts
- `unittest_linux_bigbig_coverage/matrix/line_to_tests.csv` — 249 MB
  intermediate processed artifact. **Confirmed 2026-07-08**: this file is
  consumed only by `nsga2_runs/scripts/build_line_index.py`
  (`--line-to-tests`) to build the compact runtime line-index. No
  optimization or baseline script (`nsga2_final.py`, `nsga_calculate_en.py`,
  `nsga2_with_order.py`, `nsga2_all_prs.py`, `baselines.py`) reads this CSV
  directly — all of them load only `{prefix}.npz` + `_tests.txt` +
  `_lines.txt` via `--line-index-prefix`. The generated index is already
  committed as `nsga2_runs/out/line_index.npz` (+ `_tests.txt`/`_lines.txt`
  siblings; 180,474 line keys, 8,754,453 edges) and is sufficient on its
  own to rerun the canonical optimization without this CSV. It also
  exceeds GitHub's 100 MB hard per-file limit. It will be made available
  alongside the external data archive for users who want to rebuild the
  index from scratch.
  **Path correction:** the task that prompted this verification described
  the runtime index as living at
  `unittest_linux_bigbig_coverage/matrix/line_index.npz` — no such file
  exists at that path. The actual generated index lives at
  `nsga2_runs/out/line_index.npz`, alongside the `nsga2_final.py` /
  `nsga2_canonB` outputs it was built to feed.

## Missing inputs

- **`unittest_linux_runs_with_pr_changes.csv`** — the `--runs-csv` input
  referenced by every NSGA-II script's docstring (must contain a
  `modified_files_and_statements` column). Not present anywhere in the
  repository; `Workflow_runs/unittest_linux_runs.csv` has run metadata only,
  without per-run modified-line data. This is the missing link between the
  "collect" and "run" pipeline stages.
- No collection script for the CodeCarbon energy CSVs was found in-repo
  (likely produced by an external CI/Slurm pipeline not included here).
- `nsga2_runs/insights.py` references `analyze_nsga2_final.py`, which does
  not exist anywhere in the repository.
- No record of the exact CLI invocations (pop/gens/seed/paths) used to
  produce each `nsga2_runs/out/nsga2_*` directory; provenance must be
  inferred from directory names.

## Unresolved discrepancies

See `UNRESOLVED_ISSUES.md` for full detail. Summary:

- Claimed **2,746 commit-associated executions** vs. **2,718 rows** currently
  found in `Workflow_runs/unittest_linux_runs.csv` (2,705 unique commit SHAs).
- Apparent incompleteness of the "three random seeds": `seed1` looks complete
  (1,257 pareto files, matches canonical `nsga2_canonB`); `seed2` has 1,144;
  `seed2_rapl` has only 11.

## Data archival plan — still to be decided

The following is **not yet decided** and requires your input before any
large-data `git add`:

1. Whether `unittest_linux_bigbig_coverage/` (~36 GB) and
   `coverage_results/quickmax/logs/` (~8.7 GB) go to Git LFS or to an
   external archive (e.g., Zenodo/OSF) with a DOI referenced from the README.
2. Whether `unittest_linux_bigbig_coverage/matrix/line_to_tests.csv`
   (~249 MB, now gitignored) and
   `unittest_linux_bigbig_coverage/matrix/pertest_line_coverage.csv`
   (~31 GB) follow the same external-archive path as item 1, or are
   packaged separately since they are "processed" rather than "raw."
3. Whether the missing `unittest_linux_runs_with_pr_changes.csv` will be
   added to the repo, generated by a script to be added, or documented as
   an external/private input with access instructions.

No deletion, move, rename, or compression of any experiment data or result
directory has been performed as part of this audit.
