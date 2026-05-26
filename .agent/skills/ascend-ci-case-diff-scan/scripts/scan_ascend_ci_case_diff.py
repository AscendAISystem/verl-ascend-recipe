#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI entrypoint for scanning workflow-level Ascend CI case differences."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from modules.compare import compare_cases_by_pair, summarize_scanned_workflows
from modules.config import ST_KIND, UT_KIND, load_config, validate_repo_root
from modules.excel import write_excel_report, write_past_commit_excel_report
from modules.past_commits import build_past_commit_report
from modules.render import render_past_commit_report, render_report
from modules.workflows import build_workflow_groups, collect_scan_data


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Scan Ascend CI case differences for a target verl repository.")
    parser.add_argument(
        "--repo-root",
        dest="repo_root",
        required=True,
        help="Path to the target verl repository root to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the generated report.md and report.xlsx will be written.",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=None,
        help=(
            "Optional positive integer for past-N-days analysis. "
            "When omitted, only report.md/xlsx are written; when set, also write report-past-N.md/xlsx."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the workflow parity scan."""
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    validate_repo_root(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    workflow_infos, cases, ignored_paths = collect_scan_data(repo_root, config)
    grouped_workflows = build_workflow_groups(workflow_infos)
    report = {
        "repo_root": str(repo_root),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ignored_workflows": ignored_paths,
        "scanned_workflows": summarize_scanned_workflows(grouped_workflows, cases),
        "ut_details": compare_cases_by_pair(cases, UT_KIND),
        "st_details": compare_cases_by_pair(cases, ST_KIND),
    }
    report_path = output_dir / "report.md"
    report_path.write_text(render_report(report), encoding="utf-8")
    excel_path = output_dir / "report.xlsx"
    write_excel_report(excel_path, report)
    print(report_path)
    print(excel_path)
    if args.since_days is not None:
        if args.since_days <= 0:
            raise ValueError("--since-days must be a positive integer")
        past_report = build_past_commit_report(repo_root, config, args.since_days, cases)
        past_report_path = output_dir / f"report-past-{args.since_days}.md"
        past_report_path.write_text(render_past_commit_report(past_report), encoding="utf-8")
        past_excel_path = output_dir / f"report-past-{args.since_days}.xlsx"
        write_past_commit_excel_report(past_excel_path, past_report)
        print(past_report_path)
        print(past_excel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
