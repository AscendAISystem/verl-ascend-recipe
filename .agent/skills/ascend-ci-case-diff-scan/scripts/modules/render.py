#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Markdown report rendering for the Ascend CI case diff scanner."""

from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest


def markdown_multiline(value: str) -> str:
    """Render internal multiline markers without relying on embedded HTML."""
    return value.replace("<br>", "; ")


def render_reference_details(indent: str, label: str, ref: dict | None) -> list[str]:
    """Render one workflow location with enough context for manual navigation."""
    lines = [f"{indent}- {label}:"]
    if not ref:
        lines.append(f"{indent}  - None")
        return lines
    lines.append(f"{indent}  - Workflow file: `{ref['workflow_path']}`")
    lines.append(f"{indent}  - Line number: `{ref['line_number']}`")
    lines.append(f"{indent}  - Workflow context: `{ref['workflow_name']} / {ref['job_name']} / {ref['step_name']}`")
    lines.append(f"{indent}  - Command: `{ref['raw_command']}`")
    return lines


def render_adjacent_pairs(cpu_gpu_refs: list[dict], npu_refs: list[dict]) -> list[str]:
    """Render CPU/GPU and NPU references next to each other."""
    lines: list[str] = []
    if not cpu_gpu_refs and not npu_refs:
        lines.append("  - None")
        return lines
    sorted_cpu_gpu = sorted(cpu_gpu_refs, key=lambda item: (item["name"], item["workflow_path"], item["line_number"]))
    sorted_npu = sorted(npu_refs, key=lambda item: (item["name"], item["workflow_path"], item["line_number"]))
    for index, (cpu_gpu_ref, npu_ref) in enumerate(zip_longest(sorted_cpu_gpu, sorted_npu), start=1):
        lines.append(f"  - Pair {index}")
        lines.extend(render_reference_details("    ", "CPU/GPU", cpu_gpu_ref))
        lines.extend(render_reference_details("    ", "NPU", npu_ref))
    return lines


def render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a Markdown table."""
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render_case_section(title: str, items: list[dict]) -> list[str]:
    """Render one details subsection."""
    lines = [f"### {title}", ""]
    if not items:
        lines.append("- None")
        lines.append("")
        return lines
    for item in items:
        lines.append(f"- Case path: `{item['name']}`")
        lines.extend(render_adjacent_pairs(item["cpu_gpu_refs"], item["npu_refs"]))
    lines.append("")
    return lines


def render_report(report: dict) -> str:
    """Render the final Markdown report."""
    lines = [
        "# Ascend CI Workflow and Case Alignment Report",
        "",
        "Read this report in order: Ignored Workflows, Scanned Workflows, UT Case Details, and ST Case Details.",
        "",
        (
            "UT cases are counted and displayed at the individual test function or test method level. "
            "ST cases are counted and displayed at the executed test command or script level."
        ),
        "",
        (
            "Within both UT/ST detail sections, cases are grouped into Matched Cases, CPU/GPU-only Cases, "
            "NPU-only Cases, and Manual Review."
        ),
        "",
        f"## Ignored Workflows ({len(report['ignored_workflows'])})",
        "",
    ]
    ignored_rows = [[path] for path in report["ignored_workflows"]]
    if ignored_rows:
        lines.extend(render_table(["Workflow Name"], ignored_rows))
    else:
        lines.append("No workflows were ignored.")
    lines.extend(
        [
            "",
            "## Scanned Workflows",
            "",
        ]
    )
    scanned_rows = [
        [
            markdown_multiline(row["workflow_name"]),
            str(row["cpu_gpu_case_count"]),
            str(row["npu_supported_case_count"]),
        ]
        for row in report["scanned_workflows"]
    ]
    if scanned_rows:
        lines.extend(
            render_table(
                ["Workflow Name", "CPU/GPU Case Count", "NPU Supported Case Count"],
                scanned_rows,
            )
        )
    else:
        lines.append("No workflows were scanned.")
    for title, key in (("UT Case Details", "ut_details"), ("ST Case Details", "st_details")):
        lines.extend([f"## {title}", ""])
        for section_title, section_key in (
            ("Matched Cases", "matched"),
            ("CPU/GPU-only Cases", "cpu_gpu_only"),
            ("NPU-only Cases", "npu_only"),
            ("Manual Review", "manual_review"),
        ):
            lines.extend(render_case_section(section_title, report[key][section_key]))
    return "\n".join(lines).rstrip() + "\n"


def render_past_commit_report(report: dict) -> str:
    """Render the past-N-days commit analysis as Markdown."""
    lines = [
        "# Ascend CI Past N Days Commit Analysis",
        "",
        f"Repository: `{report['repo_root']}`",
        f"Window: past `{report['since_days']}` days",
        f"Commits scanned: `{report['commit_count']}`",
        "",
        "## Summary",
        "",
    ]
    summary_rows = [
        [
            row["affected_path"],
            str(row["ut_gap_count"]),
            str(row["st_gap_count"]),
        ]
        for row in report["summary"]
    ]
    if summary_rows:
        lines.extend(
            render_table(
                ["CPU/GPU and NPU Not Fully Aligned Cases", "UT Gap Count", "ST Gap Count"],
                summary_rows,
            )
        )
    else:
        lines.append("No CI-related commits were found in the selected window.")
    lines.extend(["", "## Commit Details", ""])
    details = report["details"]
    if not details:
        lines.append("- None")
    else:
        for commit_hash, commit_rows in _group_rows_by_commit(details).items():
            first = commit_rows[0]
            lines.append(f"- Commit `{first['commit_hash']}` - {first['commit_title']}")
            lines.append(f"  - Time: `{first['commit_time']}`")
            lines.append("  - Effective changed files:")
            for changed_file in first["effective_changed_files"]:
                lines.append(f"    - `{changed_file}`")
            lines.append("  - Cases:")
            for row in commit_rows:
                lines.append(f"    - Case: `{row['case_name']}`")
                lines.append(f"      - Kind: `{row['case_kind']}`")
                lines.append(f"      - Affected path: `{row['affected_path']}`")
                lines.append(f"      - Line number: `{row['line_number']}`")
                lines.append(f"      - Workflow context: `{row['workflow_context']}`")
                lines.append(f"      - Source type: `{row['source_type']}`")
                lines.append(f"      - Status: `{row['npu_status']}`")
                if row["npu_refs"]:
                    lines.append("      - NPU refs:")
                    for ref in row["npu_refs"]:
                        lines.append(
                            f"        - `{ref['workflow_name']} / {ref['job_name']} / {ref['step_name']}` "
                            f"`{ref['workflow_path']}` line `{ref['line_number']}`"
                        )
                else:
                    lines.append("      - NPU refs: None")
    return "\n".join(lines).rstrip() + "\n"


def _group_rows_by_commit(details: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        grouped[row["commit_hash"]].append(row)
    return dict(
        sorted(
            grouped.items(),
            key=lambda item: (item[1][0]["commit_time"], item[0]),
        )
    )
