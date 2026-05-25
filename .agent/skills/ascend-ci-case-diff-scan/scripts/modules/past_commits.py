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

"""Past-N-days commit analysis for workflow and test case changes."""

from __future__ import annotations

import datetime as dt
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .compare import compare_cases_by_pair
from .config import ST_KIND, UT_KIND, WorkflowConfig
from .extractors import normalize_path_text
from .workflows import parse_workflow_content

WORKFLOW_PREFIX = ".github/workflows/"
REPORT_STATUSES = {
    "matched": "aligned",
    "cpu_gpu_only": "missing_in_npu_workflows",
    "manual_review": "manual_review_needed",
    "npu_only": "npu_only",
}


@dataclass(frozen=True)
class CommitInfo:
    commit_hash: str
    commit_time: str
    commit_title: str
    changed_files: tuple[str, ...]


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _is_ci_related(path_text: str) -> bool:
    normalized = normalize_path_text(path_text)
    return normalized.startswith(WORKFLOW_PREFIX)


def _is_workflow_path(path_text: str) -> bool:
    normalized = normalize_path_text(path_text)
    return normalized.startswith(".github/workflows/") and normalized.endswith((".yml", ".yaml"))


def list_recent_commits(repo_root: Path, since_days: int) -> list[CommitInfo]:
    """Return commits in HEAD history during the last N days that touch CI-related paths."""
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=since_days)).isoformat()
    output = _run_git(
        repo_root,
        "log",
        "--since",
        cutoff,
        "--pretty=format:%H%x1f%cI%x1f%s",
        "--name-only",
        "--no-renames",
        "HEAD",
    )
    commits: list[CommitInfo] = []
    current_hash = ""
    current_time = ""
    current_title = ""
    current_files: list[str] = []

    def flush() -> None:
        if not current_hash:
            return
        related_files = tuple(path for path in current_files if _is_ci_related(path))
        if related_files:
            commits.append(
                CommitInfo(
                    commit_hash=current_hash,
                    commit_time=current_time,
                    commit_title=current_title,
                    changed_files=related_files,
                )
            )

    for line in output.splitlines():
        if not line.strip():
            continue
        if "\x1f" in line:
            flush()
            current_hash, current_time, current_title = line.split("\x1f", 2)
            current_files = []
            continue
        current_files.append(normalize_path_text(line))
    flush()
    return commits


def build_past_commit_report(repo_root: Path, config: WorkflowConfig, since_days: int, head_cases: list[dict]) -> dict:
    """Build a past-N-days report using the current HEAD scan as the NPU baseline."""
    commits = list_recent_commits(repo_root, since_days)
    status_index = _build_head_status_index(head_cases)
    details: list[dict] = []
    seen_details: set[tuple[str, str]] = set()
    for commit in commits:
        cases = _collect_commit_cases(repo_root, config, commit)
        for case in cases:
            if not _is_effective_case(repo_root, config, case):
                continue
            status, npu_refs = _lookup_npu_support(case, status_index)
            if status == "aligned":
                continue
            detail_key = (commit.commit_hash, case["case_id"])
            if detail_key in seen_details:
                continue
            seen_details.add(detail_key)
            details.append(
                {
                    "commit_hash": commit.commit_hash,
                    "commit_time": commit.commit_time,
                    "commit_title": commit.commit_title,
                    "changed_files": tuple(commit.changed_files),
                    "affected_path": case["source_path"],
                    "source_type": case["source_type"],
                    "case_id": case["case_id"],
                    "case_kind": case["case_kind"],
                    "case_name": case["display_name"],
                    "workflow_context": f"{case['workflow_name']} / {case['job_name']} / {case['step_name']}",
                    "line_number": case["line_number"],
                    "npu_status": status,
                    "npu_refs": npu_refs,
                }
            )

    summary = _summarize_details(details)
    _attach_effective_changed_files(details)
    return {
        "repo_root": str(repo_root),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "since_days": since_days,
        "commit_count": len(commits),
        "summary": summary,
        "details": sorted(
            details,
            key=lambda row: (
                row["commit_time"],
                row["commit_hash"],
                row["case_kind"],
                row["affected_path"],
                row["case_name"],
            ),
        ),
    }


def _collect_commit_cases(repo_root: Path, config: WorkflowConfig, commit: CommitInfo) -> list[dict]:
    cases: list[dict] = []
    for path_text in commit.changed_files:
        if _is_workflow_path(path_text):
            cases.extend(_collect_changed_workflow_cases(repo_root, config, commit, path_text))

    deduped: list[dict] = []
    seen: set[str] = set()
    for case in cases:
        if case["workflow_kind"] == "npu":
            continue
        key = case["case_id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def _collect_changed_workflow_cases(
    repo_root: Path,
    config: WorkflowConfig,
    commit: CommitInfo,
    path_text: str,
) -> list[dict]:
    after_content = _load_git_file(repo_root, commit.commit_hash, path_text)
    if after_content is None:
        return []
    after_cases = _parse_workflow_cases(repo_root, config, path_text, after_content)
    before_content = _load_git_file(repo_root, f"{commit.commit_hash}^", path_text)
    before_cases = _parse_workflow_cases(repo_root, config, path_text, before_content) if before_content else []
    before_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for case in before_cases:
        before_counts[_case_change_key(case)] += 1

    changed_cases: list[dict] = []
    after_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for case in after_cases:
        key = _case_change_key(case)
        after_counts[key] += 1
        if after_counts[key] <= before_counts.get(key, 0):
            continue
        case["source_path"] = path_text
        case["source_type"] = "workflow"
        changed_cases.append(case)
    return changed_cases


def _parse_workflow_cases(
    repo_root: Path,
    config: WorkflowConfig,
    path_text: str,
    content: str,
) -> list[dict]:
    _workflow_info, workflow_cases = parse_workflow_content(
        Path(path_text).name,
        path_text,
        content,
        repo_root,
        config,
    )
    return workflow_cases


def _case_change_key(case: dict) -> tuple[str, ...]:
    return (
        case["case_kind"],
        case["command_type"],
        case["target"],
        case["signature"],
        case["workflow_name"],
        case["job_name"],
        case["step_name"],
        case["raw_command"],
    )


def _load_git_file(repo_root: Path, commit_hash: str, path_text: str) -> str | None:
    try:
        return _run_git(repo_root, "show", f"{commit_hash}:{path_text}")
    except subprocess.CalledProcessError:
        return None


def _build_head_status_index(head_cases: list[dict]) -> dict[str, dict[str, dict[str, tuple[str, list[dict]]]]]:
    index: dict[str, dict[str, dict[str, tuple[str, list[dict]]]]] = {
        UT_KIND: {},
        ST_KIND: {},
    }
    cases_by_pair: dict[str, list[dict]] = defaultdict(list)
    for case in head_cases:
        cases_by_pair[case["pair_key"]].append(case)

    for pair_key, pair_cases in cases_by_pair.items():
        for case_kind in (UT_KIND, ST_KIND):
            details = compare_cases_by_pair(pair_cases, case_kind)
            pair_index = index[case_kind].setdefault(pair_key, {})
            for section_key, status in REPORT_STATUSES.items():
                for item in details[section_key]:
                    current = pair_index.get(item["name"])
                    if current and _status_rank(current[0]) <= _status_rank(status):
                        continue
                    pair_index[item["name"]] = (status, item["npu_refs"])
    return index


def _lookup_npu_support(
    case: dict, status_index: dict[str, dict[str, dict[str, tuple[str, list[dict]]]]]
) -> tuple[str, list[dict]]:
    pair_key = case.get("pair_key", "")
    case_bucket = status_index.get(case["case_kind"], {}).get(pair_key, {})
    return case_bucket.get(case["target"], ("missing_in_npu_workflows", []))


def _status_rank(status: str) -> int:
    order = {
        "aligned": 0,
        "manual_review_needed": 1,
        "missing_in_npu_workflows": 2,
        "npu_only": 3,
    }
    return order.get(status, 99)


def _summarize_details(details: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], dict] = defaultdict(lambda: {"ut_case_ids": set(), "st_case_ids": set()})
    for row in details:
        if row["npu_status"] == "aligned":
            continue
        key = (row["affected_path"], row["npu_status"])
        if row["case_kind"] == UT_KIND:
            buckets[key]["ut_case_ids"].add(row["case_id"])
        else:
            buckets[key]["st_case_ids"].add(row["case_id"])
    return [
        {
            "affected_path": affected_path,
            "npu_status": status,
            "ut_gap_count": len(payload["ut_case_ids"]),
            "st_gap_count": len(payload["st_case_ids"]),
        }
        for (affected_path, status), payload in sorted(buckets.items())
    ]


def _attach_effective_changed_files(details: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        grouped[row["commit_hash"]].append(row)
    for commit_rows in grouped.values():
        seen_paths: set[str] = set()
        effective_paths: list[str] = []
        for row in commit_rows:
            path = row["affected_path"]
            if path in seen_paths:
                continue
            seen_paths.add(path)
            effective_paths.append(path)
        for row in commit_rows:
            row["effective_changed_files"] = tuple(effective_paths)


def _is_effective_case(repo_root: Path, config: WorkflowConfig, case: dict) -> bool:
    source_path = case.get("source_path", "")
    source_type = case.get("source_type", "")
    if source_type == "workflow":
        content = _load_repo_file(repo_root, source_path)
        if content is None:
            return False
        workflow_info, workflow_cases = parse_workflow_content(
            Path(source_path).name, source_path, content, repo_root, config
        )
        if workflow_info is None:
            return False
        current_case_keys = {_case_change_key(current_case) for current_case in workflow_cases}
        return _case_change_key(case) in current_case_keys
    return False


def _load_repo_file(repo_root: Path, path_text: str) -> str | None:
    path = repo_root / path_text
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
