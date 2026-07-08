ENACT Replication Package

This repository contains the code, processed inputs, experiment outputs, and analysis artifacts for the paper:

Energy-Aware Test Prioritization for High-Performance Computing: A Multi-Objective Approach

ENACT uses NSGA-II to select and order tests for LAMMPS while optimizing four objectives:

* maximize modified-statement coverage
* minimize energy consumption
* minimize execution time
* minimize selected test count

Main implementation

The primary ENACT implementation is:

nsga2_runs/scripts/nsga2_final.py

The main baseline and preprocessing scripts are also under:

nsga2_runs/scripts/

This directory includes:

* baselines.py
* build_line_index.py
* nsga2_all_prs.py
* nsga2_with_order.py
* nsga_calculate_en.py

Experiment configuration

The paper configuration uses:

Population size: 80
Generations: 120

The full test suite contains 613 tests.

The canonical result set contains successful outputs for 1,257 retained changes.

The full-suite reference values used in the analysis are:

Energy: 188,600.30 J
Execution time: 5,812.09 s
Test count: 613

Repository structure

Path	Contents
nsga2_runs/scripts/	ENACT, baseline, and preprocessing scripts
nsga2_runs/out/	Runtime inputs, Pareto fronts, summaries, tables, and figures
Workflow_runs/	GitHub Actions workflow metadata
Workflow_Energy_Consumption/	Per-test energy summaries and utilities
energy_results/	Energy measurement results and logs
coverage_results/	Coverage collection results
unittest_linux_bigbig_coverage/	Test lists and processed coverage artifacts

Setup

Create a Python virtual environment and install the required packages:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

To view the available command-line options:

python nsga2_runs/scripts/nsga2_final.py --help

Runtime inputs

The optimizer uses the compact line-to-test index stored in:

nsga2_runs/out/line_index.npz
nsga2_runs/out/line_index_tests.txt
nsga2_runs/out/line_index_lines.txt

These files contain the processed mapping between modified lines and tests and are sufficient to run ENACT and the included baselines.

The larger intermediate file used to build this index, line_to_tests.csv, is stored with the raw data archive rather than in GitHub.

Results

The canonical result directory is:

nsga2_runs/out/nsga2_canonB/

Per-change Pareto fronts are available under:

nsga2_runs/out/nsga2_canonB/pareto/

The aggregate result summary is:

nsga2_runs/out/nsga2_canonB/summary.csv

Additional outputs, including baseline results, tables, figures, and analysis files, are available under:

nsga2_runs/out/

Raw data archive

The large raw coverage and energy artifacts are hosted separately on Zenodo:

https://doi.org/10.5281/zenodo.21266181

The Zenodo archive contains:

* raw per-test coverage files
* coverage matrices
* the line-to-test mapping
* verbose coverage logs
* energy measurement artifacts
* checksums for the uploaded archives

License

The source code in this repository is released under the MIT License.

Citation

Citation metadata is available in:

CITATION.cff