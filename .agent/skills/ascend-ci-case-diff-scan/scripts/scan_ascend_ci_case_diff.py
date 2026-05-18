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
import datetime as dt
import hashlib
import re
import shlex
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

IGNORED_WORKFLOW_NAMES = {
    "check-pr-title",
    "doc",
    "pre-commit",
    "precommit-autofix",
    "sanity",
    "scorecard",
    "secrets_scan",
    "type-coverage-check",
}

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


def normalize_path_text(value: str) -> str:
    """Normalize filesystem-like text so matching stays stable across platforms."""
    return value.replace("\\", "/").strip()


def stable_id(*parts: str) -> str:
    """Build a deterministic short identifier for report items."""
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def load_text(path: Path) -> str:
    """Read workflow text with a small encoding fallback chain for local repos."""
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def should_ignore_workflow(workflow_name: str, workflow_path: Path) -> bool:
    """Filter out repo workflows that are clearly not test execution workflows."""
    stem = workflow_path.stem.lower()
    name = workflow_name.strip().strip("\"'").lower()
    return stem in IGNORED_WORKFLOW_NAMES or name in IGNORED_WORKFLOW_NAMES


def classify_workflow(workflow_name: str, workflow_path: Path, content: str) -> str | None:
    """Classify a workflow into cpu/gpu/npu buckets based on stable naming signals."""
    if should_ignore_workflow(workflow_name, workflow_path):
        return None
    stem = workflow_path.stem.lower()
    if stem == "cpu_unit_tests":
        return "cpu"
    if stem.endswith("_ascend") or stem == "npu_unit_tests" or "nightly_ascend" in stem:
        return "npu"
    if RUNS_ON_ASCEND_RE.search(content) or IMAGE_ASCEND_RE.search(content):
        return "npu"
    return "gpu"


def normalize_signature(command: str, target: str) -> str:
    """Keep only the command prefixes that materially affect parity matching."""
    prefix = command
    idx = command.find(target) if target else -1
    if idx != -1:
        prefix = command[:idx]
    env_match = ENV_PREFIX_RE.match(prefix or "")
    env_prefix = env_match.group(0).strip() if env_match else ""
    parts: list[str] = []
    # Preserve only high-signal environment knobs to avoid treating every
    # incidental argument difference as a distinct workflow case.
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
    """Split a shell line on semicolons while respecting simple quote scopes."""
    parts: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    for ch in command:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == ";" and not in_single and not in_double:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)
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
        if normalized.startswith("tests/"):
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


def should_keep_target(target: str) -> bool:
    """Drop placeholder paths that do not identify a real test case."""
    return target not in IGNORED_CASE_TARGETS


def dedupe_case_candidates(cases: list[dict]) -> list[dict]:
    """Remove duplicate command interpretations produced from the same shell line."""
    seen = set()
    deduped = []
    for case in cases:
        key = (case["command_type"], case["target"], case["signature"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def extract_cases_from_command(command: str) -> list[dict]:
    """Turn one shell command into one or more normalized workflow case records."""
    command = command.strip()
    if not command or command.startswith("#"):
        return []

    tokens = tokenize_command(command)
    if not tokens:
        return []

    cases: list[dict] = []
    bash_target = extract_bash_target(tokens)
    if bash_target and should_keep_target(bash_target):
        cases.append(
            {
                "command_type": "bash_script",
                "target": bash_target,
                "raw_command": command,
                "signature": normalize_signature(command, bash_target),
            }
        )

    if "torchrun" in tokens or TORCHRUN_RE.search(command):
        torchrun_idx = tokens.index("torchrun") if "torchrun" in tokens else 0
        for target in extract_torchrun_targets(tokens, torchrun_idx):
            if not should_keep_target(target):
                continue
            cases.append(
                {
                    "command_type": "torchrun",
                    "target": target,
                    "raw_command": command,
                    "signature": normalize_signature(command, target),
                }
            )
    elif "pytest" in tokens:
        pytest_idx = tokens.index("pytest")
        for target in extract_pytest_targets(tokens, pytest_idx):
            if not should_keep_target(target):
                continue
            cases.append(
                {
                    "command_type": "pytest",
                    "target": target,
                    "raw_command": command,
                    "signature": normalize_signature(command, target),
                }
            )

    return dedupe_case_candidates(cases)


def parse_workflow(path: Path, repo_root: Path) -> tuple[list[dict], str | None]:
    """Parse one workflow file and extract test cases from each run block."""
    content = load_text(path)
    workflow_name = path.stem
    name_match = WORKFLOW_NAME_RE.search(content)
    if name_match:
        workflow_name = name_match.group(1).strip().strip("\"'")
    workflow_kind = classify_workflow(workflow_name, path, content)
    if workflow_kind is None:
        return [], None

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
            # Consume the indented body of the run block so each physical line
            # can later be mapped back to its original workflow line number.
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
            for raw, line_number in run_entries:
                for command in split_shell_commands(raw):
                    for case in extract_cases_from_command(command):
                        case.update(
                            {
                                "workflow_name": workflow_name,
                                "workflow_path": normalize_path_text(path.relative_to(repo_root).as_posix()),
                                "workflow_kind": workflow_kind,
                                "job_name": job_name,
                                "step_name": step_name,
                                "line_number": line_number,
                            }
                        )
                        cases.append(case)
            continue
        idx += 1
    return cases, workflow_kind


def collect_workflow_cases(repo_root: Path) -> tuple[list[dict], list[str]]:
    """Scan every workflow file under .github/workflows and aggregate extracted cases."""
    workflow_dir = repo_root / ".github" / "workflows"
    cases: list[dict] = []
    sources: list[str] = []
    for path in sorted(workflow_dir.glob("*.yml")):
        sources.append(normalize_path_text(path.relative_to(repo_root).as_posix()))
        workflow_cases, _ = parse_workflow(path, repo_root)
        cases.extend(workflow_cases)
    return cases, sources


def ref_from_case(case: dict) -> dict:
    """Keep only the fields needed for a human-readable report reference."""
    return {
        "workflow_name": case["workflow_name"],
        "workflow_path": case["workflow_path"],
        "job_name": case["job_name"],
        "step_name": case["step_name"],
        "line_number": case["line_number"],
        "raw_command": case["raw_command"],
    }


def build_case_index(cases: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Index cases by command kind and target for coarse parity matching."""
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in cases:
        index[(case["command_type"], case["target"])].append(case)
    return index


def signature_set(cases: list[dict]) -> set[str]:
    """Collect distinct normalized signatures from a case list."""
    return {case["signature"] for case in cases}


def group_cases_by_signature(cases: list[dict]) -> dict[str, list[dict]]:
    """Group cases by normalized signature so exact and partial matches can be separated."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        grouped[case["signature"]].append(case)
    return grouped


def display_signature(signature: str) -> str:
    """Render empty signatures explicitly to avoid ambiguous report output."""
    return signature if signature else "<none>"


def format_signature_summary(prefix: str, signatures: list[str]) -> str:
    """Format a labeled signature summary for manual review report entries."""
    rendered = ", ".join(display_signature(signature) for signature in signatures)
    return f"{prefix}={rendered}"


def compare_case_sets(cpu_gpu_cases: list[dict], npu_cases: list[dict]) -> list[dict]:
    """Compare baseline CPU/GPU cases against NPU cases and build report items."""
    cpu_gpu_index = build_case_index(cpu_gpu_cases)
    npu_index = build_case_index(npu_cases)
    items: list[dict] = []

    for key, baseline_cases in sorted(cpu_gpu_index.items()):
        command_type, target = key
        npu_matches = npu_index.get(key, [])
        if not npu_matches:
            items.append(
                {
                    "id": stable_id("missing", command_type, target),
                    "category": "missing_in_npu_workflows",
                    "command_type": command_type,
                    "target": target,
                    "signature": "",
                    "cpu_gpu_refs": [ref_from_case(case) for case in baseline_cases],
                    "npu_refs": [],
                    "reason": "CPU/GPU workflow case has no NPU workflow match.",
                    "confidence": "high",
                }
            )
            continue

        baseline_by_signature = group_cases_by_signature(baseline_cases)
        npu_by_signature = group_cases_by_signature(npu_matches)

        # First emit exact signature matches. Any remaining unmatched signatures
        # become manual-review candidates instead of being forced into aligned.
        shared_signatures = sorted(set(baseline_by_signature) & set(npu_by_signature))
        for signature in shared_signatures:
            items.append(
                {
                    "id": stable_id("aligned", command_type, target, signature),
                    "category": "aligned",
                    "command_type": command_type,
                    "target": target,
                    "signature": signature,
                    "cpu_gpu_refs": [ref_from_case(case) for case in baseline_by_signature[signature]],
                    "npu_refs": [ref_from_case(case) for case in npu_by_signature[signature]],
                    "reason": "Matching target and compatible signatures found in NPU workflows.",
                    "confidence": "high",
                }
            )

        baseline_only_signatures = sorted(set(baseline_by_signature) - set(npu_by_signature))
        npu_only_signatures = sorted(set(npu_by_signature) - set(baseline_by_signature))
        if baseline_only_signatures or npu_only_signatures:
            unmatched_baseline_cases = [
                case for signature in baseline_only_signatures for case in baseline_by_signature[signature]
            ]
            unmatched_npu_cases = [case for signature in npu_only_signatures for case in npu_by_signature[signature]]
            items.append(
                {
                    "id": stable_id(
                        "manual",
                        command_type,
                        target,
                        "|".join(baseline_only_signatures),
                        "|".join(npu_only_signatures),
                    ),
                    "category": "manual_review_needed",
                    "command_type": command_type,
                    "target": target,
                    "signature": " | ".join(
                        part
                        for part in (
                            format_signature_summary("baseline-only", baseline_only_signatures)
                            if baseline_only_signatures
                            else "",
                            format_signature_summary("npu-only", npu_only_signatures) if npu_only_signatures else "",
                        )
                        if part
                    ),
                    "cpu_gpu_refs": [ref_from_case(case) for case in unmatched_baseline_cases],
                    "npu_refs": [ref_from_case(case) for case in unmatched_npu_cases],
                    "reason": (
  			  "Target exists in both baseline and NPU workflows, "
			    "but some command signatures remain unmatched."
           	    ),
                    "confidence": "medium",
                }
            )

    for key, npu_only_cases in sorted(npu_index.items()):
        if key in cpu_gpu_index:
            continue
        command_type, target = key
        items.append(
            {
                "id": stable_id("npu-only", command_type, target),
                "category": "npu_only",
                "command_type": command_type,
                "target": target,
                "signature": ", ".join(sorted(sig for sig in signature_set(npu_only_cases) if sig)),
                "cpu_gpu_refs": [],
                "npu_refs": [ref_from_case(case) for case in npu_only_cases],
                "reason": "Case appears only in NPU workflows.",
                "confidence": "high",
            }
        )

    return items


def build_summary(
    case_items: list[dict],
    cpu_gpu_cases: list[dict],
    npu_cases: list[dict],
) -> dict:
    """Compute top-level counts using deduplicated case identities."""
    counts = Counter(item["category"] for item in case_items)
    return {
        "cpu_gpu_cases": len({(case["command_type"], case["target"], case["signature"]) for case in cpu_gpu_cases}),
        "npu_cases": len({(case["command_type"], case["target"], case["signature"]) for case in npu_cases}),
        "aligned": counts["aligned"],
        "missing_in_npu_workflows": counts["missing_in_npu_workflows"],
        "manual_review_needed": counts["manual_review_needed"],
        "npu_only": counts["npu_only"],
    }


def render_ref_block(title: str, refs: list[dict]) -> list[str]:
    """Render a sorted block of workflow references for one side of the comparison."""
    lines = [f"  - {title}:"]
    if not refs:
        lines.append("    - None")
        return lines
    for ref in sorted(
        refs,
        key=lambda item: (item["raw_command"], item["workflow_path"], item["line_number"]),
    ):
        lines.extend(render_reference_details("    ", "Reference", ref))
    return lines


def render_reference_details(indent: str, label: str, ref: dict | None) -> list[str]:
    """Render one workflow location with enough context for manual navigation."""
    lines = [f"{indent}- {label}:"]
    if not ref:
        lines.append(f"{indent}  - None")
        return lines
    lines.append(f"{indent}  - Workflow file: `{ref['workflow_path']}`")
    lines.append(f"{indent}  - Line: `{ref['line_number']}`")
    lines.append(f"{indent}  - Workflow context: `{ref['workflow_name']} / {ref['job_name']} / {ref['step_name']}`")
    lines.append(f"{indent}  - Command: `{ref['raw_command']}`")
    return lines


def render_manual_review_block(item: dict) -> list[str]:
    """Render CPU/GPU and NPU references in adjacent pairs for easier review."""
    cpu_gpu_refs = sorted(
        item["cpu_gpu_refs"],
        key=lambda ref: (ref["raw_command"], ref["workflow_path"], ref["line_number"]),
    )
    npu_refs = sorted(
        item["npu_refs"],
        key=lambda ref: (ref["raw_command"], ref["workflow_path"], ref["line_number"]),
    )

    lines = ["  - Comparison context:"]
    if not cpu_gpu_refs and not npu_refs:
        lines.append("    - None")
        return lines

    for idx, (cpu_gpu_ref, npu_ref) in enumerate(zip_longest(cpu_gpu_refs, npu_refs), start=1):
        lines.append(f"    - Pair {idx}:")
        lines.extend(render_reference_details("      ", "CPU/GPU reference", cpu_gpu_ref))
        lines.extend(render_reference_details("      ", "NPU reference", npu_ref))
    return lines


def render_report(report: dict) -> str:
    """Render the final Markdown report consumed by developers."""
    summary = report["summary"]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in report["items"]:
        by_category[item["category"]].append(item)

    lines = [
        "# Ascend CI Case Diff Scan Report",
        "",
        "## Summary",
        "",
        f"- CPU/GPU workflow cases: {summary['cpu_gpu_cases']}",
        f"- NPU workflow cases: {summary['npu_cases']}",
        f"- aligned: {summary['aligned']}",
        f"- missing_in_npu_workflows: {summary['missing_in_npu_workflows']}",
        f"- manual_review_needed: {summary['manual_review_needed']}",
        f"- npu_only: {summary['npu_only']}",
        "",
    ]

    sections = [
        ("CPU/GPU Cases Covered by NPU", "aligned"),
        ("CPU/GPU Cases Missing in NPU Workflows", "missing_in_npu_workflows"),
        ("Cases Requiring Manual Review", "manual_review_needed"),
        ("NPU-only Cases", "npu_only"),
    ]

    for title, category in sections:
        lines.append(f"## {title}")
        lines.append("")
        items = by_category.get(category, [])
        if not items:
            lines.append("- None")
            lines.append("")
            continue
        for item in items:
            lines.append(f"- `{item['target']}`")
            lines.append(f"  - Original file location: `{item['target']}`")
            lines.append(f"  - Difference: {item['reason']}")
            if item["signature"]:
                lines.append(f"  - Signature summary: `{item['signature']}`")
            if category == "manual_review_needed":
                lines.extend(render_manual_review_block(item))
            else:
                lines.extend(render_ref_block("CPU/GPU case locations", item["cpu_gpu_refs"]))
                lines.extend(render_ref_block("NPU case locations", item["npu_refs"]))
        lines.append("")
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

    workflow_cases, workflow_sources = collect_workflow_cases(repo_root)
    cpu_gpu_cases = [case for case in workflow_cases if case["workflow_kind"] in {"cpu", "gpu"}]
    npu_cases = [case for case in workflow_cases if case["workflow_kind"] == "npu"]

    case_items = compare_case_sets(cpu_gpu_cases, npu_cases)
    items = sorted(case_items, key=lambda item: (item["category"], item["target"], item["id"]))

    report = {
        "repo_root": str(repo_root),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "workflow-case-diff",
        "baseline": "cpu_gpu_workflow_union",
        "sources": {
            "workflows": workflow_sources,
            "tests": ["tests/README.md"],
            "docs": ["docs/ascend_tutorial/contribution_guide/ascend_ci_guide_zh.rst"],
        },
        "summary": build_summary(case_items, cpu_gpu_cases, npu_cases),
        "items": items,
    }

    report_path = output_dir / "report.md"
    report_path.write_text(render_report(report), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
