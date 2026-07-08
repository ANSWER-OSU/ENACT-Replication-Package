#!/usr/bin/env bash
set -euo pipefail

CSV_DIR="/nfs/stak/users/anklesan/hpc-share/lammps/energy_results/full_regression"
PATTERN="fullreg_energy_idx_*.csv"

awk -F',' '
BEGIN {
    printf "\n=== TOTALS PER PHASE ===\n"
}

NR > 1 {
    phase = $1
    wall[phase] += $4
    uj[phase]   += $5
    j[phase]    += $6
    kwh[phase]  += $7

    g_wall += $4
    g_uj   += $5
    g_j    += $6
    g_kwh  += $7
}

END {
    for (p in wall) {
        printf "%-8s wall_s=%8d  energy_j=%12.2f  energy_kwh=%10.6f\n",
               p, wall[p], j[p], kwh[p]
    }

    printf "\n=== GRAND TOTAL (ALL PHASES) ===\n"
    printf "TOTAL    wall_s=%8d  energy_j=%12.2f  energy_kwh=%10.6f\n",
           g_wall, g_j, g_kwh
}
' "$CSV_DIR"/$PATTERN