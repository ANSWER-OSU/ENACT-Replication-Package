ENACT Replication Package

Replication package for:

Energy-Aware Test Prioritization for High-Performance Computing: A Multi-Objective Approach

ENACT uses NSGA-II to select and order tests for LAMMPS while optimizing four objectives:

* maximize modified-statement coverage
* minimize energy consumption
* minimize execution time
* minimize the number of selected tests

Main files

Primary implementation:

nsga2_runs/scripts/nsga2_final.py

Canonical results:

nsga2_runs/out/nsga2_canonB/

The included result set contains outputs for 1,257 retained changes and 613 tests.

Full-suite reference values:

* Energy: 188,600.30 J
* Time: 5,812.09 s
* Tests: 613

Repository structure

Path	Contents
nsga2_runs/scripts/	ENACT, baselines, and preprocessing scripts
nsga2_runs/out/	Pareto fronts, summaries, tables, and figures
Workflow_runs/	GitHub Actions workflow metadata
Workflow_Energy_Consumption/	Per-test energy summaries
energy_results/	Energy measurement results
coverage_results/	Coverage measurement results
unittest_linux_bigbig_coverage/	Test lists and processed coverage data

Setup

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

To view the available ENACT options:

python nsga2_runs/scripts/nsga2_final.py --help

The paper configuration uses a population size of 80 and 120 generations.

Runtime inputs

The optimizer uses the compact line-to-test index:

nsga2_runs/out/line_index.npz
nsga2_runs/out/line_index_tests.txt
nsga2_runs/out/line_index_lines.txt

The original line_to_tests.csv file is about 249 MB and is not included in Git. It is only needed to rebuild the compact index above.

Results

Per-change Pareto fronts:

nsga2_runs/out/nsga2_canonB/pareto/

Aggregate summary:

nsga2_runs/out/nsga2_canonB/summary.csv

Additional tables and figures are available under nsga2_runs/out/.

Data notes

Large raw coverage files, verbose logs, virtual environments, and archive files are excluded from Git because the full experiment data is about 45 GB.

The repository includes the main code, compact processed inputs, canonical outputs, and analysis files. Large raw artifacts will be archived separately.

Two provenance issues are still being checked:

* project notes mention 2,746 executions, while the included workflow CSV currently has 2,718 rows
* some separately named seed directories appear incomplete, although the canonical result set is complete

More details are available in:

REPOSITORY_AUDIT.md
UNRESOLVED_ISSUES.md

License

The source code is released under the MIT License.

Citation

Please cite the associated paper when using ENACT or this repository. Machine-readable citation information is available in CITATION.cff.