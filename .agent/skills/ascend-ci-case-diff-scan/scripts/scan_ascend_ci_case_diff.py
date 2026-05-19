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

"""Scan workflow-level Ascend CI case differences for a target verl repo."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import shlex
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatch
from itertools import zip_longest
from pathlib import Path

ENV_PREFIX_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S+)\s+)+")
TORCHRUN_RE = re.compile(r"\btorchrun\b")
WORKFLOW_NAME_RE = re.compile(r"^\s*name:\s*(.+?)\s*$")
JOB_RE = re.compile(r"^(\s{2})([A-Za-z0-9_-]+):\s*$")
STEP_NAME_RE = re.compile(r"^(\s*)-\s+name:\s*(.+?)\s*$")
RUN_RE = re.compile(r"^(\s*)run:\s*(.*)$")
RUNS_ON_ASCEND_RE = re.compile(r"runs-on:\s+.*(?:aarch64|a2|a3)", re.IGNORECASE)
IMAGE_ASCEND_RE = re.compile(r"image:\s+.*ascend", re.IGNORECASE)
PATH_VALUE_OPTIONS = {"--ignore", "--ignore-glob"}
PYTEST_OPTIONS_WITH_VALUE = {
    "-k",
    "-m",
    "--maxfail",
    "--rootdir",
    "--log-level",
    "--capture",
    "--asyncio-mode",
    "--durations",
}
IGNORED_CASE_TARGETS = {"tests/"}
PYTEST_DIRECTORY_TARGETS = {"tests", "tests/"}
UT_KIND = "ut"
ST_KIND = "st"
CASE_COMPARISON_SECTIONS = ("matched", "cpu_gpu_only", "npu_only", "manual_review")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "workflow_scope.json"


@dataclass(frozen=True)
class WorkflowConfig:
    ignored_workflows: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowInfo:
    workflow_name: str
    workflow_path: str
    file_name: str
    workflow_kind: str
    pair_key: str


def normalize_path_text(value: str) -> str:
    """Normalize filesystem-like text so matching stays stable across platforms."""
    return value.replace("\\", "/").strip()


def load_text(path: Path) -> str:
    """Read workflow text with a small encoding fallback chain for local repos."""
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def load_config(config_path: Path = CONFIG_PATH) -> WorkflowConfig:
    """Load the workflow-scope configuration from JSON."""
    payload = json.loads(load_text(config_path))
    ignored_workflows = tuple(payload.get("ignored_workflows", []))
    return WorkflowConfig(ignored_workflows=ignored_workflows)


def is_ignored_workflow(file_name: str, config: WorkflowConfig) -> bool:
    """Return whether the workflow file should be excluded from the scan."""
    lowered = file_name.lower()
    return any(fnmatch(lowered, pattern.lower()) for pattern in config.ignored_workflows)


def classify_workflow(file_stem: str, content: str) -> str:
    """Classify workflows into cpu, gpu, or npu buckets."""
    stem = file_stem.lower()
    if stem == "cpu_unit_tests":
        return "cpu"
    if stem == "npu_unit_tests" or stem.endswith("_ascend") or "nightly_ascend" in stem:
        return "npu"
    if RUNS_ON_ASCEND_RE.search(content) or IMAGE_ASCEND_RE.search(content):
        return "npu"
    return "gpu"


def workflow_pair_key(file_stem: str) -> str:
    """Map workflow file names into CPU/GPU vs NPU pairing groups."""
    stem = file_stem.lower()
    if stem == "npu_unit_tests":
        return "gpu_unit_tests"
    if stem.endswith("_ascend"):
        return stem[: -len("_ascend")]
    return stem


def normalize_signature(command: str, target: str) -> str:
    """Keep only the command prefixes that materially affect parity matching."""
    prefix = command
    idx = command.find(target) if target else -1
    if idx != -1:
        prefix = command[:idx]
    env_match = ENV_PREFIX_RE.match(prefix or "")
    env_prefix = env_match.group(0).strip() if env_match else ""
    parts: list[str] = []
    for key in ("ROLLOUT_NAME", "ENGINE", "STRATEGY", "MODE", "RESUME_MODE", "BACKEND"):
        match = re.search(rf"{key}=([^\s]+)", env_prefix)
        if match:
            parts.append(match.group(0))
    if " -m pytest " in command:
        parts.append("-m pytest")
    if "--nproc_per_node" in command or "--nproc-per-node" in command:
        parts.append("torchrun-distributed")
    return " ".join(parts).strip()


def split_shell_commands(command: str) -> list[str]:
    """Split a shell line on common shell separators while respecting simple quote scopes."""
    parts: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    idx = 0
    while idx < len(command):
        ch = command[idx]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if not in_single and not in_double:
            separator = None
            if ch == ";":
                separator = ";"
            elif command.startswith("&&", idx):
                separator = "&&"
            elif command.startswith("||", idx):
                separator = "||"
            if separator is not None:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                idx += len(separator)
                continue

        current.append(ch)
        idx += 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def tokenize_command(command: str) -> list[str]:
    """Tokenize shell text conservatively and fall back when quoting is malformed."""
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def extract_step_python_file_patterns(run_entries: list[tuple[str, int]]) -> list[str]:
    """Extract pytest file-name filters configured earlier in the same run block."""
    patterns: list[str] = []
    for raw, _line_number in run_entries:
        match = re.search(r"python_files\s*=\s*([^\s'\"`]+)", raw)
        if match:
            patterns.append(match.group(1).strip())
    return patterns


def extract_pytest_ignore_specs(tokens: list[str], pytest_idx: int) -> tuple[list[str], list[str]]:
    """Extract ignore path and ignore glob options from a pytest command."""
    ignore_paths: list[str] = []
    ignore_globs: list[str] = []
    idx = pytest_idx + 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in PATH_VALUE_OPTIONS and idx + 1 < len(tokens):
            value = normalize_path_text(tokens[idx + 1].strip("\"'"))
            if token == "--ignore":
                ignore_paths.append(value)
            else:
                ignore_globs.append(value)
            idx += 2
            continue
        if token.startswith("--ignore="):
            ignore_paths.append(normalize_path_text(token.split("=", 1)[1].strip("\"'")))
            idx += 1
            continue
        if token.startswith("--ignore-glob="):
            ignore_globs.append(normalize_path_text(token.split("=", 1)[1].strip("\"'")))
            idx += 1
            continue
        idx += 1
    return ignore_paths, ignore_globs


def extract_pytest_targets(tokens: list[str], pytest_idx: int) -> list[str]:
    """Extract test targets from a pytest command without mistaking option values for paths."""
    targets: list[str] = []
    skip_next = False
    idx = pytest_idx + 1
    while idx < len(tokens):
        token = tokens[idx]
        if skip_next:
            skip_next = False
            idx += 1
            continue
        if token in PATH_VALUE_OPTIONS or token in PYTEST_OPTIONS_WITH_VALUE:
            skip_next = True
            idx += 1
            continue
        if token.startswith("--ignore=") or token.startswith("--ignore-glob="):
            idx += 1
            continue
        if token.startswith("-"):
            idx += 1
            continue
        normalized = normalize_path_text(token.strip("\"'"))
        if normalized in PYTEST_DIRECTORY_TARGETS or normalized.startswith("tests/"):
            targets.append(normalized)
        idx += 1
    return targets


def extract_torchrun_targets(tokens: list[str], torchrun_idx: int) -> list[str]:
    """Extract test targets from torchrun commands, including `torchrun -m pytest`."""
    for idx in range(torchrun_idx + 1, len(tokens) - 1):
        if tokens[idx] == "-m" and tokens[idx + 1] == "pytest":
            return extract_pytest_targets(tokens, idx + 1)

    targets: list[str] = []
    idx = torchrun_idx + 1
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("-"):
            idx += 1
            continue
        normalized = normalize_path_text(token.strip("\"'"))
        if normalized.startswith("tests/"):
            targets.append(normalized)
        idx += 1
    return targets


def extract_bash_target(tokens: list[str]) -> str | None:
    """Extract bash-invoked test scripts when the target is an explicit tests/*.sh path."""
    for idx, token in enumerate(tokens[:-1]):
        if token == "bash":
            target = normalize_path_text(tokens[idx + 1].strip("\"'"))
            if target.startswith("tests/") and target.endswith(".sh"):
                return target
    return None


def should_keep_target(target: str, command_type: str) -> bool:
    """Drop placeholder paths that do not identify a real test case."""
    normalized_target = normalize_path_text(target)
    return (command_type == "pytest" and normalized_target in PYTEST_DIRECTORY_TARGETS) or (
        normalized_target not in IGNORED_CASE_TARGETS
    )


def matches_python_file_patterns(path_text: str, python_file_patterns: list[str]) -> bool:
    """Check whether a discovered file name matches the configured pytest file patterns."""
    file_name = Path(path_text).name
    if not python_file_patterns:
        return fnmatch(file_name, "test_*.py")
    return any(fnmatch(file_name, pattern) for pattern in python_file_patterns)


def is_ignored_pytest_target(path_text: str, ignore_paths: list[str], ignore_globs: list[str]) -> bool:
    """Check whether a discovered test file should be excluded by pytest ignore options."""
    normalized = normalize_path_text(path_text)
    for ignore_path in ignore_paths:
        candidate = normalize_path_text(ignore_path)
        if normalized == candidate or normalized.startswith(candidate.rstrip("/") + "/"):
            return True
    return any(fnmatch(normalized, pattern) for pattern in ignore_globs)


def extract_test_functions_from_file(path: Path) -> list[str]:
    """Extract unittest-style and pytest-style test functions from one Python file."""
    try:
        module = ast.parse(load_text(path))
    except SyntaxError:
        return []

    names: list[str] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                    names.append(f"{node.name}::{child.name}")
    return sorted(dict.fromkeys(names))


def expand_python_test_file(path_text: str, repo_root: Path) -> list[str]:
    """Expand one Python test file into function-level UT identifiers."""
    path = repo_root / path_text
    if not path.is_file():
        return [path_text]
    functions = extract_test_functions_from_file(path)
    if not functions:
        return [path_text]
    return [f"{path_text}::{function_name}" for function_name in functions]


def expand_pytest_directory(
    target_path: Path,
    repo_root: Path,
    python_file_patterns: list[str],
    ignore_paths: list[str],
    ignore_globs: list[str],
) -> list[str]:
    """Expand one pytest directory target using pattern-based Python file discovery."""
    expanded: list[str] = []
    for path in sorted(target_path.rglob("*.py")):
        if not path.is_file():
            continue
        path_text = normalize_path_text(path.relative_to(repo_root).as_posix())
        if not matches_python_file_patterns(path_text, python_file_patterns):
            continue
        if is_ignored_pytest_target(path_text, ignore_paths, ignore_globs):
            continue
        expanded.extend(expand_python_test_file(path_text, repo_root))
    return expanded


def expand_pytest_targets(
    target: str,
    repo_root: Path,
    python_file_patterns: list[str],
    ignore_paths: list[str],
    ignore_globs: list[str],
) -> list[str]:
    """Expand pytest targets into function-level UT cases."""
    normalized = normalize_path_text(target)
    target_path = repo_root / normalized
    if target_path.is_dir():
        return expand_pytest_directory(
            target_path,
            repo_root,
            python_file_patterns,
            ignore_paths,
            ignore_globs,
        )
    if normalized.endswith(".py"):
        return expand_python_test_file(normalized, repo_root)
    return [normalized]


def dedupe_case_candidates(cases: list[dict]) -> list[dict]:
    """Remove duplicate command interpretations produced from the same shell line."""
    seen = set()
    deduped = []
    for case in cases:
        key = (case["command_type"], tuple(case["targets"]), case["signature"], case["raw_command"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def extract_cases_from_command(
    command: str,
    repo_root: Path,
    python_file_patterns: list[str] | None = None,
) -> list[dict]:
    """Turn one shell command into one or more normalized workflow case records."""
    command = command.strip()
    if not command or command.startswith("#"):
        return []

    tokens = tokenize_command(command)
    if not tokens:
        return []

    cases: list[dict] = []
    bash_target = extract_bash_target(tokens)
    if bash_target and should_keep_target(bash_target, "bash"):
        cases.append(
            {
                "command_type": "bash",
                "case_kind": ST_KIND,
                "target": bash_target,
                "targets": [bash_target],
                "raw_command": command,
                "signature": normalize_signature(command, bash_target),
            }
        )

    if "torchrun" in tokens or TORCHRUN_RE.search(command):
        torchrun_idx = tokens.index("torchrun") if "torchrun" in tokens else 0
        for target in extract_torchrun_targets(tokens, torchrun_idx):
            if not should_keep_target(target, "torchrun"):
                continue
            cases.append(
                {
                    "command_type": "torchrun",
                    "case_kind": ST_KIND,
                    "target": target,
                    "targets": [target],
                    "raw_command": command,
                    "signature": normalize_signature(command, target),
                }
            )
    elif "pytest" in tokens:
        pytest_idx = tokens.index("pytest")
        ignore_paths, ignore_globs = extract_pytest_ignore_specs(tokens, pytest_idx)
        for target in extract_pytest_targets(tokens, pytest_idx):
            if not should_keep_target(target, "pytest"):
                continue
            normalized_target = normalize_path_text(target)
            if normalized_target in PYTEST_DIRECTORY_TARGETS or Path(normalized_target).suffix == "":
                target_shape = "pytest-dir"
            elif normalized_target.endswith(".py"):
                target_shape = "pytest-file"
            else:
                target_shape = "pytest-target"

            signature = normalize_signature(command, target)
            if signature:
                signature = f"{signature} {target_shape}"
            else:
                signature = target_shape

            cases.append(
                {
                    "command_type": "pytest",
                    "case_kind": UT_KIND,
                    "target": target,
                    "targets": expand_pytest_targets(
                        target,
                        repo_root,
                        python_file_patterns or [],
                        ignore_paths,
                        ignore_globs,
                    ),
                    "raw_command": command,
                    "signature": signature,
                }
            )

    return dedupe_case_candidates(cases)


def build_case_id(case: dict) -> str:
    """Build a per-occurrence case key that preserves workflow/job/step distinctions."""
    return "|".join(
        [
            case["case_kind"],
            case["command_type"],
            case["target"],
            case["signature"],
            case["workflow_name"],
            case["job_name"],
            case["step_name"],
            str(case["line_number"]),
        ]
    )


def build_display_name(case: dict) -> str:
    """Produce the report-facing case name for UT and ST entries."""
    if case["case_kind"] == UT_KIND:
        return case["target"]
    return f"{case['target']} [{case['workflow_name']} / {case['job_name']} / {case['step_name']}]"


def parse_workflow(path: Path, repo_root: Path, config: WorkflowConfig) -> tuple[WorkflowInfo | None, list[dict]]:
    """Parse one workflow file and extract test cases from each run block."""
    file_name = path.name
    if is_ignored_workflow(file_name, config):
        return None, []

    content = load_text(path)
    workflow_name = path.stem
    name_match = WORKFLOW_NAME_RE.search(content)
    if name_match:
        workflow_name = name_match.group(1).strip().strip("\"'")
    workflow_kind = classify_workflow(path.stem, content)
    workflow_info = WorkflowInfo(
        workflow_name=workflow_name,
        workflow_path=normalize_path_text(path.relative_to(repo_root).as_posix()),
        file_name=file_name,
        workflow_kind=workflow_kind,
        pair_key=workflow_pair_key(path.stem),
    )

    lines = content.splitlines()
    job_name = "unknown"
    step_name = "(unnamed step)"
    cases: list[dict] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        job_match = JOB_RE.match(line)
        if job_match:
            job_name = job_match.group(2)
        step_match = STEP_NAME_RE.match(line)
        if step_match:
            step_name = step_match.group(2).strip()
        run_match = RUN_RE.match(line)
        if run_match:
            indent = len(run_match.group(1))
            inline = run_match.group(2).strip()
            run_entries: list[tuple[str, int]] = []
            if inline and inline not in {"|", ">"}:
                run_entries.append((inline, idx + 1))
            idx += 1
            while idx < len(lines):
                next_line = lines[idx]
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_line.strip() == "":
                    idx += 1
                    continue
                if next_indent <= indent:
                    break
                stripped = next_line[indent + 2 :] if len(next_line) >= indent + 2 else next_line.lstrip()
                run_entries.append((stripped, idx + 1))
                idx += 1
            python_file_patterns = extract_step_python_file_patterns(run_entries)
            for raw, line_number in run_entries:
                for command in split_shell_commands(raw):
                    for case in extract_cases_from_command(command, repo_root, python_file_patterns):
                        for target in case["targets"]:
                            expanded_case = {
                                "workflow_name": workflow_name,
                                "workflow_path": workflow_info.workflow_path,
                                "file_name": file_name,
                                "workflow_kind": workflow_kind,
                                "pair_key": workflow_info.pair_key,
                                "job_name": job_name,
                                "step_name": step_name,
                                "line_number": line_number,
                                "command_type": case["command_type"],
                                "case_kind": case["case_kind"],
                                "target": target,
                                "raw_command": case["raw_command"],
                                "signature": case["signature"],
                            }
                            expanded_case["case_id"] = build_case_id(expanded_case)
                            expanded_case["display_name"] = build_display_name(expanded_case)
                            cases.append(expanded_case)
            continue
        idx += 1
    return workflow_info, cases


def collect_scan_data(repo_root: Path, config: WorkflowConfig) -> tuple[list[WorkflowInfo], list[dict], list[str]]:
    """Scan workflows and return parsed workflow metadata, cases, and ignored paths."""
    workflow_dir = repo_root / ".github" / "workflows"
    workflow_infos: list[WorkflowInfo] = []
    cases: list[dict] = []
    ignored_paths: list[str] = []
    for path in sorted(workflow_dir.glob("*.yml")):
        rel_path = normalize_path_text(path.relative_to(repo_root).as_posix())
        if is_ignored_workflow(path.name, config):
            ignored_paths.append(rel_path)
            continue
        workflow_info, workflow_cases = parse_workflow(path, repo_root, config)
        if workflow_info is None:
            ignored_paths.append(rel_path)
            continue
        workflow_infos.append(workflow_info)
        cases.extend(workflow_cases)
    return workflow_infos, cases, ignored_paths


def build_workflow_groups(workflow_infos: list[WorkflowInfo]) -> dict[str, dict[str, list[WorkflowInfo]]]:
    """Group workflows by their CPU/GPU vs NPU pairing key."""
    grouped: dict[str, dict[str, list[WorkflowInfo]]] = defaultdict(lambda: {"cpu_gpu": [], "npu": []})
    for info in workflow_infos:
        side = "npu" if info.workflow_kind == "npu" else "cpu_gpu"
        grouped[info.pair_key][side].append(info)
    return grouped


def summarize_scanned_workflows(
    grouped_workflows: dict[str, dict[str, list[WorkflowInfo]]], cases: list[dict]
) -> list[dict]:
    """Build table rows for scanned workflow coverage counts."""
    rows: list[dict] = []
    for pair_key, workflows in sorted(grouped_workflows.items()):
        cpu_gpu_paths = {info.workflow_path for info in workflows["cpu_gpu"]}
        npu_paths = {info.workflow_path for info in workflows["npu"]}
        cpu_gpu_case_ids = {case["case_id"] for case in cases if case["workflow_path"] in cpu_gpu_paths}
        npu_case_ids = {case["case_id"] for case in cases if case["workflow_path"] in npu_paths}
        workflow_names = [
            f"{info.workflow_path} ({info.workflow_name})" for info in workflows["cpu_gpu"] + workflows["npu"]
        ]
        rows.append(
            {
                "workflow_name": "<br>".join(sorted(workflow_names)) if workflow_names else pair_key,
                "cpu_gpu_case_count": len(cpu_gpu_case_ids),
                "npu_supported_case_count": len(npu_case_ids),
            }
        )
    return rows


def make_ref(case: dict) -> dict:
    """Keep only the fields needed for report rendering."""
    return {
        "name": case["display_name"],
        "workflow_name": case["workflow_name"],
        "workflow_path": case["workflow_path"],
        "job_name": case["job_name"],
        "step_name": case["step_name"],
        "line_number": case["line_number"],
        "raw_command": case["raw_command"],
    }


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


def compare_group_cases(cpu_gpu_cases: list[dict], npu_cases: list[dict]) -> dict[str, list[dict]]:
    """Compare CPU/GPU and NPU cases inside one UT/ST bucket."""
    grouped_cpu_gpu: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    grouped_npu: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    results = {section_name: [] for section_name in CASE_COMPARISON_SECTIONS}

    for case in cpu_gpu_cases:
        grouped_cpu_gpu[(case["command_type"], case["target"], case["signature"])].append(case)
    for case in npu_cases:
        grouped_npu[(case["command_type"], case["target"], case["signature"])].append(case)

    cpu_gpu_base_index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    npu_base_index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in cpu_gpu_cases:
        cpu_gpu_base_index[(case["command_type"], case["target"])].append(case)
    for case in npu_cases:
        npu_base_index[(case["command_type"], case["target"])].append(case)

    matched_keys = sorted(set(grouped_cpu_gpu) & set(grouped_npu))
    for key in matched_keys:
        command_type, target, signature = key
        results["matched"].append(
            {
                "name": target,
                "command_type": command_type,
                "signature": signature,
                "cpu_gpu_refs": [make_ref(case) for case in grouped_cpu_gpu[key]],
                "npu_refs": [make_ref(case) for case in grouped_npu[key]],
            }
        )

    processed_base_keys = set()
    for base_key in sorted(set(cpu_gpu_base_index) | set(npu_base_index)):
        command_type, target = base_key
        cpu_gpu_signatures = {case["signature"] for case in cpu_gpu_base_index.get(base_key, [])}
        npu_signatures = {case["signature"] for case in npu_base_index.get(base_key, [])}
        shared_signatures = cpu_gpu_signatures & npu_signatures
        only_cpu_gpu_signatures = cpu_gpu_signatures - shared_signatures
        only_npu_signatures = npu_signatures - shared_signatures
        if only_cpu_gpu_signatures and only_npu_signatures:
            results["manual_review"].append(
                {
                    "name": target,
                    "command_type": command_type,
                    "signature": "",
                    "cpu_gpu_refs": [
                        make_ref(case)
                        for case in cpu_gpu_base_index[base_key]
                        if case["signature"] in only_cpu_gpu_signatures
                    ],
                    "npu_refs": [
                        make_ref(case) for case in npu_base_index[base_key] if case["signature"] in only_npu_signatures
                    ],
                }
            )
            processed_base_keys.add(base_key)

    for base_key in sorted(set(cpu_gpu_base_index) - processed_base_keys):
        command_type, target = base_key
        cpu_gpu_signatures = {case["signature"] for case in cpu_gpu_base_index[base_key]}
        npu_signatures = {case["signature"] for case in npu_base_index.get(base_key, [])}
        unmatched_signatures = cpu_gpu_signatures - npu_signatures
        if not unmatched_signatures:
            continue
        results["cpu_gpu_only"].append(
            {
                "name": target,
                "command_type": command_type,
                "signature": "",
                "cpu_gpu_refs": [
                    make_ref(case) for case in cpu_gpu_base_index[base_key] if case["signature"] in unmatched_signatures
                ],
                "npu_refs": [],
            }
        )

    for base_key in sorted(set(npu_base_index) - processed_base_keys):
        command_type, target = base_key
        npu_signatures = {case["signature"] for case in npu_base_index[base_key]}
        cpu_gpu_signatures = {case["signature"] for case in cpu_gpu_base_index.get(base_key, [])}
        unmatched_signatures = npu_signatures - cpu_gpu_signatures
        if not unmatched_signatures:
            continue
        results["npu_only"].append(
            {
                "name": target,
                "command_type": command_type,
                "signature": "",
                "cpu_gpu_refs": [],
                "npu_refs": [
                    make_ref(case) for case in npu_base_index[base_key] if case["signature"] in unmatched_signatures
                ],
            }
        )

    for section_name in CASE_COMPARISON_SECTIONS:
        results[section_name] = sorted(results[section_name], key=lambda item: (item["name"], item["command_type"]))
    return results


def compare_cases_by_pair(cases: list[dict], case_kind: str) -> dict[str, list[dict]]:
    """Compare cases inside each workflow pairing group and merge the results."""
    aggregated = {section_name: [] for section_name in CASE_COMPARISON_SECTIONS}
    pair_keys = sorted({case["pair_key"] for case in cases})
    for pair_key in pair_keys:
        pair_cases = [case for case in cases if case["pair_key"] == pair_key and case["case_kind"] == case_kind]
        cpu_gpu_cases = [case for case in pair_cases if case["workflow_kind"] in {"cpu", "gpu"}]
        npu_cases = [case for case in pair_cases if case["workflow_kind"] == "npu"]
        comparison = compare_group_cases(cpu_gpu_cases, npu_cases)
        for section_name in CASE_COMPARISON_SECTIONS:
            aggregated[section_name].extend(comparison[section_name])
    for section_name in CASE_COMPARISON_SECTIONS:
        aggregated[section_name] = sorted(
            aggregated[section_name],
            key=lambda item: (item["name"], item["command_type"]),
        )
    return aggregated


def render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


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


def render_details_block(title: str, comparison: dict[str, list[dict]]) -> list[str]:
    """Render UT or ST comparison details in the requested order."""
    lines = [f"## {title}", ""]
    lines.extend(render_case_section("Matched Cases", comparison["matched"]))
    lines.extend(render_case_section("CPU/GPU-only Cases", comparison["cpu_gpu_only"]))
    lines.extend(render_case_section("NPU-only Cases", comparison["npu_only"]))
    lines.extend(render_case_section("Manual Review", comparison["manual_review"]))
    return lines


def render_report(report: dict) -> str:
    """Render the final Markdown report."""
    lines = [
        "# Ascend CI Workflow and Case Alignment Report",
        "",
        "Read this report in order: Ignored Workflows, Scanned Workflows, UT Case Details, and ST Case Details.",
        "",
        (
            "Within both UT Case Details and ST Case Details, cases are grouped into Matched Cases, "
            "CPU/GPU-only Cases, NPU-only Cases, and Manual Review."
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
            row["workflow_name"],
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
    lines.extend(["", *render_details_block("UT Case Details", report["ut_details"])])
    lines.extend(["", *render_details_block("ST Case Details", report["st_details"])])
    return "\n".join(lines).rstrip() + "\n"


def validate_repo_root(repo_root: Path) -> None:
    """Fail fast when the target repository does not look like a workflow source repo."""
    if not repo_root.exists():
        raise FileNotFoundError(f"Target repository root does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise NotADirectoryError(f"Target repository root is not a directory: {repo_root}")
    workflow_dir = repo_root / ".github" / "workflows"
    if not workflow_dir.is_dir():
        raise FileNotFoundError(
            f"Target repository does not contain '.github/workflows'. Expected workflow directory: {workflow_dir}"
        )


def build_report(repo_root: Path, config: WorkflowConfig) -> dict:
    """Build the in-memory report data."""
    workflow_infos, cases, ignored_paths = collect_scan_data(repo_root, config)
    grouped_workflows = build_workflow_groups(workflow_infos)
    return {
        "repo_root": str(repo_root),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ignored_workflows": ignored_paths,
        "scanned_workflows": summarize_scanned_workflows(grouped_workflows, cases),
        "ut_details": compare_cases_by_pair(cases, UT_KIND),
        "st_details": compare_cases_by_pair(cases, ST_KIND),
    }


def main() -> int:
    """CLI entrypoint for running the workflow parity scan."""
    parser = argparse.ArgumentParser(description="Scan Ascend CI case differences for a target verl repository.")
    parser.add_argument(
        "--repo-root",
        "--target-repo-root",
        dest="repo_root",
        required=True,
        help="Path to the target verl repository root to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the generated report.md will be written.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    validate_repo_root(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    report = build_report(repo_root, config)
    report_path = output_dir / "report.md"
    report_path.write_text(render_report(report), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
