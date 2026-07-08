# ENACT Replication Package

This repository contains the code, processed inputs, experiment outputs, and analysis artifacts for:

**Energy-Aware Test Prioritization for High-Performance Computing: A Multi-Objective Approach**

ENACT uses NSGA-II to select and order LAMMPS tests while optimizing four objectives:

- maximize modified-statement coverage
- minimize energy consumption
- minimize execution time
- minimize selected test count

## Main implementation

The primary ENACT implementation is:

```text
nsga2_runs/scripts/nsga2_final.py
```

Related baseline and preprocessing scripts are under:

```text
nsga2_runs/scripts/
```

Important scripts include:

- `baselines.py`
- `build_line_index.py`
- `nsga2_all_prs.py`
- `nsga2_with_order.py`
- `nsga_calculate_en.py`

## Experiment configuration

The paper configuration uses:

```text
Population size: 80
Generations: 120
```

The included result set contains:

- 1,257 retained changes
- 613 tests
- full-suite energy of 188,600.30 J
- full-suite execution time of 5,812.09 s

## Repository structure

| Path | Contents |
|---|---|
| `nsga2_runs/scripts/` | ENACT, baseline, and preprocessing scripts |
| `nsga2_runs/out/` | Runtime inputs, Pareto fronts, summaries, tables, and figures |
| `Workflow_runs/` | GitHub Actions workflow metadata |
| `Workflow_Energy_Consumption/` | Per-test energy summaries and utilities |
| `energy_results/` | Energy measurement results |
| `coverage_results/` | Coverage collection results |
| `unittest_linux_bigbig_coverage/` | Test lists and processed coverage artifacts |

## Setup

Create a Python environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

View the available command-line options:

```bash
python nsga2_runs/scripts/nsga2_final.py --help
```

## Runtime inputs

The optimizer uses the compact line-to-test index:

```text
nsga2_runs/out/line_index.npz
nsga2_runs/out/line_index_tests.txt
nsga2_runs/out/line_index_lines.txt
```

These files are sufficient to run ENACT and the included baselines.

## Results

The canonical result directory is:

```text
nsga2_runs/out/nsga2_canonB/
```

Per-change Pareto fronts are available under:

```text
nsga2_runs/out/nsga2_canonB/pareto/
```

The aggregate summary is:

```text
nsga2_runs/out/nsga2_canonB/summary.csv
```

Additional tables, figures, baseline outputs, and analysis files are under:

```text
nsga2_runs/out/
```

## Raw data archive

The large raw coverage and energy artifacts are available on Zenodo:

[https://doi.org/10.5281/zenodo.21266181](https://doi.org/10.5281/zenodo.21266181)

The archive contains:

- raw per-test coverage files
- coverage matrices
- the line-to-test mapping
- coverage logs
- energy measurement artifacts
- SHA-256 checksums

## License

The source code is released under the MIT License.

## Citation

Citation metadata is available in `CITATION.cff`.