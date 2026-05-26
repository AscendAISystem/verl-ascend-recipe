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

"""Past-N-days commit analysis for workflow and test case changes."""

from __future__ import annotations

import datetime as dt
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .compare import compare_cases_by_pair
from .config import ST_KIND, UT_KIND, WorkflowConfig
from .extractors import normalize_path_text
from .workflows import build_workflow_groups, collect_scan_data

WORKFLOW_PREFIX = ".github/workflows/"
RELEVANT_PREFIXES = (".github/workflows/", "tests/")
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


def _normalize_commit_commit_hash(commit_hash: str) -> str:
    return commit_hash.strip()


def _is_relevant_path(path_text: str) -> bool:
    normalized = normalize_path_text(path_text)
    return normalized.startswith(RELEVANT_PREFIXES) or (
        normalized.startswith("examples/") and normalized.endswith(".sh")
    )


def _is_workflow_path(path_text: str) -> bool:
    normalized = normalize_path_text(path_text)
    return normalized.startswith(WORKFLOW_PREFIX) and normalized.endswith((".yml", ".yaml"))


def _get_first_parent_commits(repo_root: Path, since_days: int) -> list[str]:
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=since_days)).isoformat()
    output = _run_git(
        repo_root,
        "rev-list",
        "--first-parent",
        "--reverse",
        "--since",
        cutoff,
        "HEAD",
    )
    return [_normalize_commit_commit_hash(line) for line in output.splitlines() if line.strip()]


def _get_commit_info(repo_root: Path, commit_hash: str) -> CommitInfo:
    payload = _run_git(repo_root, "show", "-s", "--format=%H%x1f%cI%x1f%s", commit_hash).strip()
    commit_id, commit_time, commit_title = payload.split("\x1f", 2)
    changed_files = _list_changed_files(repo_root, commit_hash)
    return CommitInfo(
        commit_hash=commit_id,
        commit_time=commit_time,
        commit_title=commit_title,
        changed_files=changed_files,
    )


def _list_changed_files(repo_root: Path, commit_hash: str) -> tuple[str, ...]:
    output = _run_git(repo_root, "diff-tree", "--no-commit-id", "--name-only", "--no-renames", "-r", commit_hash)
    return tuple(normalize_path_text(line) for line in output.splitlines() if line.strip() and _is_relevant_path(line))


def _get_commit_sequence(repo_root: Path, since_days: int) -> list[CommitInfo]:
    commits = [
        _get_commit_info(repo_root, commit_hash) for commit_hash in _get_first_parent_commits(repo_root, since_days)
    ]
    return commits


def _get_base_commit(repo_root: Path, oldest_commit: str) -> str | None:
    try:
        return _run_git(repo_root, "rev-parse", f"{oldest_commit}^").strip()
    except subprocess.CalledProcessError:
        return None


def _collect_relevant_tree_paths(repo_root: Path, commit_hash: str) -> list[str]:
    output = _run_git(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        commit_hash,
        "--",
        ".github/workflows",
        "tests",
        "examples",
    )
    return [normalize_path_text(line) for line in output.splitlines() if line.strip() and _is_relevant_path(line)]


def _materialize_snapshot(repo_root: Path, commit_hash: str, snapshot_root: Path) -> None:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for rel_path in _collect_relevant_tree_paths(repo_root, commit_hash):
        try:
            content = _run_git(repo_root, "show", f"{commit_hash}:{rel_path}")
        except subprocess.CalledProcessError:
            continue
        target = snapshot_root / Path(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", errors="ignore")


def _load_snapshot_scan(
    snapshot_root: Path, config: WorkflowConfig
) -> tuple[dict[str, dict], list[dict], dict[str, list[dict]], dict[str, object]]:
    workflow_infos, cases, _ignored_paths = collect_scan_data(snapshot_root, config)
    grouped = build_workflow_groups(workflow_infos)
    workflow_infos_by_path = {info.workflow_path: info for info in workflow_infos}
    cases_by_path: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        cases_by_path[case["workflow_path"]].append(case)
    return grouped, cases, cases_by_path, workflow_infos_by_path


def build_past_commit_report(repo_root: Path, config: WorkflowConfig, since_days: int, head_cases: list[dict]) -> dict:
    """Build a past-N-days report by comparing the base snapshot with the current HEAD snapshot."""
    commits = _get_commit_sequence(repo_root, since_days)
    if not commits:
        return {
            "repo_root": str(repo_root),
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "since_days": since_days,
            "commit_count": 0,
            "summary": [],
            "workflow_changes": [],
            "case_details": [],
            "commit_details": [],
        }

    base_commit = _get_base_commit(repo_root, commits[0].commit_hash)
    head_commit = _run_git(repo_root, "rev-parse", "HEAD").strip()

    with tempfile.TemporaryDirectory(prefix="ascend-ci-past-") as temp_dir:
        temp_root = Path(temp_dir)
        base_root = temp_root / "base"
        head_root = temp_root / "head"
        if base_commit:
            _materialize_snapshot(repo_root, base_commit, base_root)
        _materialize_snapshot(repo_root, head_commit, head_root)

        _base_workflow_groups, _base_cases, base_cases_by_path, base_workflow_infos = _load_snapshot_scan(
            base_root, config
        )
        _head_workflow_groups, head_cases_snapshot, head_cases_by_path, head_workflow_infos = _load_snapshot_scan(
            head_root, config
        )

    status_index = _build_head_status_index(head_cases if head_cases else head_cases_snapshot)
    changed_files = _collect_window_changed_files(commits)
    workflow_changes: list[dict] = []
    case_details: list[dict] = []
    workflow_paths = sorted(set(head_cases_by_path) | set(base_cases_by_path))
    for workflow_path in workflow_paths:
        head_info = head_workflow_infos.get(workflow_path)
        base_info = base_workflow_infos.get(workflow_path)
        workflow_kind = head_info.workflow_kind if head_info else base_info.workflow_kind
        if workflow_kind == "npu":
            continue
        head_workflow_cases = head_cases_by_path.get(workflow_path, [])
        base_case_keys = {_case_change_key(case) for case in base_cases_by_path.get(workflow_path, [])}
        changed_cases = _collect_changed_head_cases(head_workflow_cases, base_case_keys, changed_files)
        if not _workflow_changed(base_info, head_info, base_case_keys, head_workflow_cases) and not changed_cases:
            continue

        touched_commits = _commits_touching_workflow(workflow_path, changed_cases, commits)
        workflow_row = {
            "workflow_path": workflow_path,
            "head_workflow_name": head_info.workflow_name if head_info else "",
            "base_workflow_name": base_info.workflow_name if base_info else "",
            "workflow_status": _workflow_status(base_info, head_info),
            "commit_hashes": tuple(touched_commits),
            "case_count_head": len(head_workflow_cases),
            "case_count_base": len(base_cases_by_path.get(workflow_path, [])),
            "ut_gap_count": 0,
            "st_gap_count": 0,
        }

        for case in changed_cases:
            status, npu_refs = _lookup_npu_support(case, status_index)
            case_detail = {
                "workflow_path": workflow_path,
                "workflow_name": head_info.workflow_name if head_info else case["workflow_name"],
                "case_id": case["case_id"],
                "case_name": case["display_name"],
                "case_kind": case["case_kind"],
                "command_type": case["command_type"],
                "workflow_context": f"{case['workflow_name']} / {case['job_name']} / {case['step_name']}",
                "line_number": case["line_number"],
                "target": case["target"],
                "signature": case["signature"],
                "raw_command": case["raw_command"],
                "npu_status": status,
                "npu_refs": npu_refs,
                "commit_hashes": tuple(_commits_touching_case(case, commits)),
            }
            case_details.append(case_detail)
            if status != "aligned":
                if case["case_kind"] == UT_KIND:
                    workflow_row["ut_gap_count"] += 1
                else:
                    workflow_row["st_gap_count"] += 1

        workflow_row["cases"] = sorted(
            case_details_for_workflow(case_details, workflow_path), key=lambda row: (row["case_kind"], row["case_name"])
        )
        workflow_changes.append(workflow_row)

    workflow_changes = sorted(workflow_changes, key=_workflow_change_sort_key)
    summary = _summarize_details(case_details)
    commit_details = _build_commit_details(commits, workflow_changes, case_details)
    return {
        "repo_root": str(repo_root),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "since_days": since_days,
        "commit_count": len(commits),
        "summary": summary,
        "workflow_changes": workflow_changes,
        "case_details": sorted(
            case_details, key=lambda row: (row["workflow_path"], row["case_kind"], row["case_name"])
        ),
        "commit_details": commit_details,
    }


def case_details_for_workflow(case_details: list[dict], workflow_path: str) -> list[dict]:
    return [row for row in case_details if row["workflow_path"] == workflow_path]


def _workflow_changed(
    base_info: object | None,
    head_info: object | None,
    base_case_keys: set[tuple[str, ...]],
    head_cases: list[dict],
) -> bool:
    if base_info is None or head_info is None:
        return True
    if (
        base_info.workflow_name != head_info.workflow_name
        or base_info.workflow_kind != head_info.workflow_kind
        or base_info.pair_key != head_info.pair_key
    ):
        return True
    head_case_keys = {_case_change_key(case) for case in head_cases}
    return head_case_keys != base_case_keys


def _workflow_status(base_info: object | None, head_info: object | None) -> str:
    if base_info is None and head_info is not None:
        return "added"
    if base_info is not None and head_info is None:
        return "removed"
    return "modified"


def _workflow_change_sort_key(row: dict) -> tuple[int, str]:
    status_order = {"added": 0, "modified": 1, "removed": 2}
    return status_order.get(row["workflow_status"], 99), row["workflow_path"]


def _collect_window_changed_files(commits: list[CommitInfo]) -> set[str]:
    return {path for commit in commits for path in commit.changed_files}


def _collect_changed_head_cases(
    head_cases: list[dict],
    base_case_keys: set[tuple[str, ...]],
    changed_files: set[str],
) -> list[dict]:
    changed_cases: list[dict] = []
    seen_case_ids: set[str] = set()
    for case in head_cases:
        if case["workflow_kind"] == "npu":
            continue
        if _case_change_key(case) not in base_case_keys or _case_target_changed(case, changed_files):
            if case["case_id"] in seen_case_ids:
                continue
            seen_case_ids.add(case["case_id"])
            changed_cases.append(case)
    return changed_cases


def _case_target_changed(case: dict, changed_files: set[str]) -> bool:
    case_path = normalize_path_text(case["target"].split("::", 1)[0])
    if case["workflow_path"] in changed_files:
        return True
    if case["command_type"] == "pytest":
        if case_path in {"tests", "tests/"}:
            return any(path.startswith("tests/") for path in changed_files)
        return case_path in changed_files
    if case["command_type"] == "bash":
        return case_path in changed_files
    if case["command_type"] == "torchrun":
        return case_path in changed_files or any(path.startswith(case_path.rstrip("/") + "/") for path in changed_files)
    return False


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


def _commits_touching_workflow(workflow_path: str, added_cases: list[dict], commits: list[CommitInfo]) -> list[str]:
    touched: list[str] = []
    seen: set[str] = set()
    for commit in commits:
        if workflow_path in commit.changed_files:
            if commit.commit_hash not in seen:
                seen.add(commit.commit_hash)
                touched.append(commit.commit_hash)
            continue
        if any(_commit_touches_case_path(commit, case) for case in added_cases):
            if commit.commit_hash not in seen:
                seen.add(commit.commit_hash)
                touched.append(commit.commit_hash)
    return touched


def _commits_touching_case(case: dict, commits: list[CommitInfo]) -> list[str]:
    touched: list[str] = []
    seen: set[str] = set()
    for commit in commits:
        if _commit_touches_case_path(commit, case):
            if commit.commit_hash not in seen:
                seen.add(commit.commit_hash)
                touched.append(commit.commit_hash)
    return touched


def _commit_touches_case_path(commit: CommitInfo, case: dict) -> bool:
    case_path = normalize_path_text(case["target"].split("::", 1)[0])
    if case["workflow_path"] in commit.changed_files:
        return True
    if case["command_type"] == "bash":
        return case_path in commit.changed_files
    if case["command_type"] == "torchrun":
        return any(path.startswith(case_path.rstrip("/")) for path in commit.changed_files)
    if case["command_type"] == "pytest":
        if case_path in {"tests", "tests/"}:
            return any(path.startswith("tests/") for path in commit.changed_files)
        return case_path in commit.changed_files
    return False


def _build_head_status_index(
    head_cases: list[dict],
) -> dict[str, dict[str, dict[tuple[str, str], tuple[str, list[dict]]]]]:
    index: dict[str, dict[str, dict[tuple[str, str], tuple[str, list[dict]]]]] = {
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
                    key = (item["command_type"], item["name"], item["signature"])
                    current = pair_index.get(key)
                    if current and _status_rank(current[0]) <= _status_rank(status):
                        continue
                    pair_index[key] = (status, item["npu_refs"])
    return index


def _lookup_npu_support(
    case: dict, status_index: dict[str, dict[str, dict[tuple[str, str], tuple[str, list[dict]]]]]
) -> tuple[str, list[dict]]:
    pair_key = case.get("pair_key", "")
    case_bucket = status_index.get(case["case_kind"], {}).get(pair_key, {})
    return case_bucket.get((case["command_type"], case["target"], case["signature"]), ("missing_in_npu_workflows", []))


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
        key = (row["workflow_path"], row["npu_status"])
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


def _build_commit_details(
    commits: list[CommitInfo], workflow_changes: list[dict], case_details: list[dict]
) -> list[dict]:
    workflow_map: dict[str, list[str]] = defaultdict(list)
    for change in workflow_changes:
        for commit_hash in change["commit_hashes"]:
            workflow_map[commit_hash].append(change["workflow_path"])
    case_map: dict[str, list[str]] = defaultdict(list)
    for detail in case_details:
        for commit_hash in detail["commit_hashes"]:
            case_map[commit_hash].append(detail["workflow_path"])

    rows: list[dict] = []
    for commit in commits:
        affected_workflows = sorted(
            set(workflow_map.get(commit.commit_hash, []) + case_map.get(commit.commit_hash, []))
        )
        if not affected_workflows:
            continue
        rows.append(
            {
                "commit_hash": commit.commit_hash,
                "commit_time": commit.commit_time,
                "commit_title": commit.commit_title,
                "changed_files": commit.changed_files,
                "affected_workflows": tuple(affected_workflows),
            }
        )
    return rows
