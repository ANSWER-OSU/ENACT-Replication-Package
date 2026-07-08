#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

NUM_FIELDS = ["wall_s", "energy_kwh", "energy_j", "emissions_kg"]

def parse_float(x: str) -> float:
    x = (x or "").strip()
    if x == "":
        return 0.0
    return float(x)

def sum_csv(path: str | None) -> dict[str, float]:
    if path is None or path == "-":
        f = sys.stdin
    else:
        f = open(path, "r", newline="", encoding="utf-8")

    totals = {k: 0.0 for k in NUM_FIELDS}
    rows = 0

    try:
        reader = csv.DictReader(f)
        missing = [k for k in NUM_FIELDS if k not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing required columns: {missing}\nFound: {reader.fieldnames}")

        for row in reader:
            rows += 1
            for k in NUM_FIELDS:
                totals[k] += parse_float(row.get(k, "0"))
    finally:
        if f is not sys.stdin:
            f.close()

    totals["rows"] = float(rows)
    return totals

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "-"
    totals = sum_csv(path)

    print(f"rows={int(totals['rows'])}")
    print(f"total_wall_s={totals['wall_s']:.6f}")
    print(f"total_energy_kwh={totals['energy_kwh']:.12f}")
    print(f"total_energy_j={totals['energy_j']:.6f}")
    print(f"total_emissions_kg={totals['emissions_kg']:.12f}")

if __name__ == "__main__":
    main()