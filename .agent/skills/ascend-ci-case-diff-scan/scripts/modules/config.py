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

"""Config loading and repository validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

UT_KIND = "ut"
ST_KIND = "st"
CASE_COMPARISON_SECTIONS = ("matched", "cpu_gpu_only", "npu_only", "manual_review")
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "workflow_scope.json"


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


def load_text(path: Path) -> str:
    """Read text with a small encoding fallback chain for local repos."""
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
