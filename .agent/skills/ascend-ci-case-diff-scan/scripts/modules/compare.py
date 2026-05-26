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

"""Case comparison and workflow summary helpers."""

from __future__ import annotations

from collections import defaultdict

from .config import CASE_COMPARISON_SECTIONS, WorkflowInfo


def make_ref(case: dict) -> dict:
    """Keep only the fields needed for report rendering."""
    return {
        key: case[source]
        for key, source in (
            ("name", "display_name"),
            ("workflow_name", "workflow_name"),
            ("workflow_path", "workflow_path"),
            ("job_name", "job_name"),
            ("step_name", "step_name"),
            ("line_number", "line_number"),
            ("signature", "signature"),
            ("raw_command", "raw_command"),
        )
    }


def summarize_scanned_workflows(
    grouped_workflows: dict[str, dict[str, list[WorkflowInfo]]], cases: list[dict]
) -> list[dict]:
    """Build table rows for scanned workflow coverage counts."""
    case_ids_by_path: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        case_ids_by_path[case["workflow_path"]].add(case["case_id"])
    rows: list[dict] = []
    for pair_key, workflows in sorted(grouped_workflows.items()):
        cpu_gpu_paths = {info.workflow_path for info in workflows["cpu_gpu"]}
        npu_paths = {info.workflow_path for info in workflows["npu"]}
        cpu_gpu_case_ids = {
            case_id for workflow_path in cpu_gpu_paths for case_id in case_ids_by_path.get(workflow_path, set())
        }
        npu_case_ids = {
            case_id for workflow_path in npu_paths for case_id in case_ids_by_path.get(workflow_path, set())
        }
        workflow_names = [
            f"{info.workflow_path} ({info.workflow_name})" for info in workflows["cpu_gpu"] + workflows["npu"]
        ]
        rows.append(
            {
                "workflow_names": sorted(workflow_names) if workflow_names else [pair_key],
                "workflow_name": "<br>".join(sorted(workflow_names)) if workflow_names else pair_key,
                "cpu_gpu_case_count": len(cpu_gpu_case_ids),
                "npu_supported_case_count": len(npu_case_ids),
            }
        )
    return rows


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
    cases_by_pair: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        if case["case_kind"] == case_kind:
            cases_by_pair[case["pair_key"]].append(case)
    for pair_key in sorted(cases_by_pair):
        pair_cases = cases_by_pair[pair_key]
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
