# ENACT-Replication-Package

ENACT FSE Replication Package

> **Status: under preparation.** This repository is being readied for
> public replication and is **not yet claiming full end-to-end
> reproducibility**. See `REPOSITORY_AUDIT.md` and `UNRESOLVED_ISSUES.md`
> for the current state, known gaps, and open questions before relying on
> any number here.

## What this is

ENACT applies NSGA-II (via [pymoo](https://pymoo.org/)) to select and order
test-suite subsets for the [LAMMPS](https://www.lammps.org/) molecular
dynamics simulator, optimizing four objectives simultaneously:
modified-statement (change-aware) coverage (maximize), energy consumption,
execution time, and selected test count (all minimize).

## Canonical artifacts (current working assumption)

While the repository is being prepared, the following are treated as
canonical:

- **Implementation:** [`nsga2_runs/scripts/nsga2_final.py`](nsga2_runs/scripts/nsga2_final.py)
  is the canonical ENACT NSGA-II implementation (random-keys permutation +
  cut-point encoding, 4-objective `pymoo.NSGA2`). Related scripts in the
  same directory (`nsga2_all_prs.py`, `nsga2_with_order.py`,
  `nsga_calculate_en.py`) are earlier iterations or instrumented variants,
  kept for provenance rather than as the primary implementation.
- **Results:** [`nsga2_runs/out/nsga2_canonB/`](nsga2_runs/out/nsga2_canonB/)
  is the canonical result set (1,257 "ok" pareto runs; full-suite baseline
  of 188,600.30 J / 5,812.09 s over 613 tests).

These assignments may change as the two unresolved issues below are
addressed — treat them as the current best understanding, not a final
statement.

## Repository layout

| Path | Role |
|---|---|
| `nsga2_runs/scripts/` | Core NSGA-II algorithm + baselines + line-index builder |
| `nsga2_runs/out/` | NSGA-II run outputs (pareto CSVs, summaries, tables, charts) |
| `nsga2_runs/analyze.py`, `analyze2.py`, `insights.py` | Analysis scripts over `summary.csv` outputs |
| `Workflow_runs/` | GitHub Actions run metadata collection (`runs_linux_bigbig.py`) and pulled CSV/JSON |
| `Workflow_Energy_Consumption/` | Per-test energy summaries + summation utility |
| `energy_results/` | Per-test/per-run energy measurement artifacts (CodeCarbon, Slurm logs) |
| `coverage_results/`, `unittest_linux_bigbig_coverage/` | Raw and processed test-coverage data for LAMMPS |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions recovered from a captured environment
found in this repository; `pymoo`, `tqdm`, and `requests` could not be
version-pinned from that source (see `requirements.txt` header comment and
`UNRESOLVED_ISSUES.md`).

## Large data artifacts

This repository contains raw and processed measurement data up to tens of
gigabytes per file (e.g., a ~31 GB per-test-per-line coverage matrix).
**Large raw measurement artifacts are not intended for ordinary Git
storage** — see `.gitignore` for what is currently excluded, and
`REPOSITORY_AUDIT.md` for the archival plan that is still being decided
(Git LFS vs. external archive with a DOI).

One artifact is worth calling out specifically:
`unittest_linux_bigbig_coverage/matrix/line_to_tests.csv` is a **249 MB
intermediate processed artifact** — the raw line→tests table consumed only
by `nsga2_runs/scripts/build_line_index.py` to build the compact runtime
index. It is excluded from ordinary Git storage. The **compact runtime
input actually used for optimization**, `nsga2_runs/out/line_index.npz`
(with its `_tests.txt`/`_lines.txt` siblings), is committed and is
sufficient on its own to rerun the canonical optimization — no script that
runs NSGA-II or the baselines reads `line_to_tests.csv` directly. The CSV
will be made available alongside the external data archive for users who
want to rebuild the index from scratch.

## Known open issues

Two issues currently limit exact reproducibility and are not yet resolved:

1. **Execution-count provenance** — the input file needed to reproduce the
   NSGA-II "run" stage end-to-end (`unittest_linux_runs_with_pr_changes.csv`)
   is not present in the repository, and a reported count of 2,746
   commit-associated executions does not match the 2,718 rows currently
   found in `Workflow_runs/unittest_linux_runs.csv`.
2. **Seed completeness** — of the three random seeds referenced by the
   `nsga2_all_pop80_gen120_seed*` output directories, only `seed1` appears
   fully computed (1,257 pareto files); `seed2` (1,144) and `seed2_rapl`
   (11) appear incomplete in this snapshot.

Full detail on both, plus three lower-severity provenance-only issues, is
in `UNRESOLVED_ISSUES.md`.
