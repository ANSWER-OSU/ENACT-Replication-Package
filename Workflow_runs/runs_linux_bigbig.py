#!/usr/bin/env python3
"""
Fetch GitHub Actions workflow runs and compute run counts + avg runs/day for
last 1d, 7d, 30d, 365d. Optionally save ALL run data to JSON/CSV.

Examples:
  export GITHUB_TOKEN=ghp_xxx

  # All history + save
  python3 workflow_run_rates.py --repo lammps/lammps --workflow unittest-linux.yml \
    --all-history --save-json all_runs.json --save-csv all_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

API_BASE = "https://api.github.com"


@dataclass
class Window:
    name: str
    days: int


WINDOWS = [
    Window("1_day", 1),
    Window("7_days", 7),
    Window("30_days", 30),
    Window("365_days", 365),
]


def iso_to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def gh_headers(token: Optional[str]) -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "workflow-run-rates-script",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def request_with_retry(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> requests.Response:
    for attempt in range(1, 6):
        r = requests.get(url, headers=headers, params=params, timeout=30)

        if r.status_code == 403:
            reset = r.headers.get("X-RateLimit-Reset")
            remaining = r.headers.get("X-RateLimit-Remaining")
            if remaining == "0" and reset:
                wait_s = max(1, int(reset) - int(time.time()) + 1)
                print(f"[rate-limit] waiting {wait_s}s until reset...", file=sys.stderr)
                time.sleep(wait_s)
                continue

        if r.status_code in (500, 502, 503, 504):
            backoff = min(10, 2 ** attempt)
            print(f"[retry] server error {r.status_code}; sleeping {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
            continue

        return r

    return r


def fetch_runs_page(
    owner: str,
    repo: str,
    workflow: str,
    token: Optional[str],
    page: int,
    per_page: int,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    url = f"{API_BASE}/repos/{owner}/{repo}/actions/workflows/{workflow}/runs"
    headers = gh_headers(token)
    params = {"per_page": per_page, "page": page}

    r = request_with_retry(url, headers, params)
    if r.status_code != 200:
        raise SystemExit(f"GitHub API error {r.status_code}: {r.text}")

    data = r.json()
    runs = data.get("workflow_runs", [])
    next_page = (page + 1) if len(runs) == per_page else None
    return runs, next_page


def normalize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": run.get("id"),
        "name": run.get("name"),
        "event": run.get("event"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "run_attempt": run.get("run_attempt"),
        "run_number": run.get("run_number"),
        "head_branch": run.get("head_branch"),
        "head_sha": run.get("head_sha"),
        "html_url": run.get("html_url"),
    }


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] JSON -> {path}", file=sys.stderr)


def save_csv(path: str, runs: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "id", "name", "event", "status", "conclusion",
        "created_at", "updated_at",
        "run_attempt", "run_number",
        "head_branch", "head_sha", "html_url",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"[saved] CSV -> {path}", file=sys.stderr)


def compute_stats_and_collect(
    owner: str,
    repo: str,
    workflow: str,
    token: Optional[str],
    per_page: int,
    max_pages: int,
    all_history: bool,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoffs = {w.name: now - timedelta(days=w.days) for w in WINDOWS}

    counts = {w.name: 0 for w in WINDOWS}
    newest_run: Optional[datetime] = None
    earliest_run: Optional[datetime] = None

    collected: List[Dict[str, Any]] = []

    page = 1
    pages_fetched = 0

    while True:
        if page is None:
            break
        if pages_fetched >= max_pages:
            print(f"[warn] reached max_pages={max_pages}. Results may be incomplete.", file=sys.stderr)
            break

        runs, next_page = fetch_runs_page(owner, repo, workflow, token, page, per_page)
        pages_fetched += 1
        if not runs:
            break

        for run in runs:
            created = iso_to_dt(run["created_at"])

            if newest_run is None or created > newest_run:
                newest_run = created
            if earliest_run is None or created < earliest_run:
                earliest_run = created

            for name, cutoff in cutoffs.items():
                if created >= cutoff:
                    counts[name] += 1

            collected.append(normalize_run(run))

        # If not all_history, we could stop early once older than 365d,
        # but user requested ALL history, so we always paginate.
        page = next_page if all_history else next_page  # keep explicit for clarity

    averages_per_day = {w.name: counts[w.name] / float(w.days) for w in WINDOWS}

    return {
        "repo": f"{owner}/{repo}",
        "workflow": workflow,
        "now_utc": now.isoformat(),
        "counts": counts,
        "averages_per_day": averages_per_day,
        "newest_run_utc": newest_run.isoformat() if newest_run else None,
        "earliest_run_utc": earliest_run.isoformat() if earliest_run else None,
        "pages_fetched": pages_fetched,
        "runs_collected": len(collected),
        "runs": collected,
        "all_history": all_history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/repo (e.g., lammps/lammps)")
    ap.add_argument("--workflow", required=True, help="workflow file name or workflow id (e.g., unittest-linux.yml)")
    ap.add_argument("--token", default=os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"),
                    help="GitHub token (or set env GITHUB_TOKEN/GH_TOKEN)")
    ap.add_argument("--per-page", type=int, default=100, help="API page size (max 100)")
    ap.add_argument("--max-pages", type=int, default=20000, help="safety cap for pagination (all-history can be large)")
    ap.add_argument("--all-history", action="store_true", help="fetch ALL workflow runs (entire history)")
    ap.add_argument("--save-json", default=None, help="path to write JSON (summary + runs)")
    ap.add_argument("--save-csv", default=None, help="path to write CSV (runs)")
    args = ap.parse_args()

    if "/" not in args.repo:
        raise SystemExit("--repo must be in owner/repo format")
    owner, repo = args.repo.split("/", 1)

    stats = compute_stats_and_collect(
        owner=owner,
        repo=repo,
        workflow=args.workflow,
        token=args.token,
        per_page=max(1, min(100, args.per_page)),
        max_pages=args.max_pages,
        all_history=True,  # user requested all history
    )

    # Save artifacts
    if args.save_json:
        save_json(args.save_json, {
            "repo": stats["repo"],
            "workflow": stats["workflow"],
            "now_utc": stats["now_utc"],
            "newest_run_utc": stats["newest_run_utc"],
            "earliest_run_utc": stats["earliest_run_utc"],
            "pages_fetched": stats["pages_fetched"],
            "all_history": stats["all_history"],
            "counts": stats["counts"],
            "averages_per_day": stats["averages_per_day"],
            "runs": stats["runs"],
        })

    if args.save_csv:
        save_csv(args.save_csv, stats["runs"])

    # Print summary
    print(f"repo: {stats['repo']}")
    print(f"workflow: {stats['workflow']}")
    print(f"now_utc: {stats['now_utc']}")
    print(f"newest_run_utc: {stats['newest_run_utc']}")
    print(f"earliest_run_utc: {stats['earliest_run_utc']}")
    print(f"pages_fetched: {stats['pages_fetched']}")
    print(f"runs_saved: {stats['runs_collected']}")
    print()

    for w in WINDOWS:
        c = stats["counts"][w.name]
        avg = stats["averages_per_day"][w.name]
        print(f"{w.name}: runs={c}, avg_runs_per_day={avg:.6f}")


if __name__ == "__main__":
    main()