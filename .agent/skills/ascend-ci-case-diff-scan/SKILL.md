---
name: ascend-ci-case-diff-scan
description: Scan an external verl repository for Ascend/NPU CI coverage gaps by comparing CPU/GPU workflow cases against NPU workflows. Use when Codex needs to audit workflow and case-level parity or generate a Markdown report about workflow execution differences.
---

# Ascend CI Case Diff Scan

Use this skill from the current repository to audit Ascend CI coverage in a target `verl` repository.

## Overview

The scanner performs a static analysis of workflow files and reports:

1. Which workflows are out of scope and excluded by configuration.
2. Which CPU/GPU workflows are paired with NPU workflows.
3. How many function-level UT cases and command-level ST cases each workflow contributes.
4. Which cases are matched, missing, NPU-only, or need manual review.

## Scope

The scanner reads workflow `run:` commands as the source of truth.

- Ignore `examples/**`.
- Ignore commented workflow lines.
- Ignore `cpu_unit_tests.yml` because it is treated as shared CPU/GPU/NPU baseline coverage, not an NPU-adaptation target.
- Do not execute tests.
- Do not expand GitHub Actions matrices.
- Keep the final report in English only.

## Workflow

1. Read [references/repo-signals.md](./references/repo-signals.md) to refresh repo-specific boundaries.
2. Identify the target `verl` repository root, for example `{PATH}/verl`.
3. Run the scanner from the current repository:

```shell
python .agent/skills/ascend-ci-case-diff-scan/scripts/scan_ascend_ci_case_diff.py \
  --repo-root {PATH}/verl \
  --output-dir ./report/ascend-ci-case-diff-scan
```

You can also use the alias:

```shell
python .agent/skills/ascend-ci-case-diff-scan/scripts/scan_ascend_ci_case_diff.py \
  --target-repo-root {PATH}/verl \
  --output-dir ./report/ascend-ci-case-diff-scan
```

## Extraction Rules

Recognize only workflow commands that visibly execute tests:

- `pytest ... tests/...`
- `bash tests/.../*.sh`
- `torchrun ... tests/...`
- `torchrun ... -m pytest tests/...`

For each extracted case, preserve:

- `workflow_name`
- `job_name`
- `step_name`
- `command_type`
- `target`
- `raw_command`
- `signature`

When the same command is repeated, keep cases distinct by `workflow_name`, `job_name`, and `step_name`.
For UT expansion, treat Python test functions and `Test*::test_*` methods as the reporting unit whenever the target file can be parsed.

## Classification Rules

Use these output categories:

- `aligned`
- `missing_in_npu_workflows`
- `manual_review_needed`
- `npu_only`

Treat matching conservatively:

- Exact target matches are the strongest signal.
- Compatible signatures can be aligned.
- Different signatures for the same target should fall back to manual review.

## Reporting

The scanner writes `report.md` to the requested output directory.

The report contains:

- ignored workflows
- scanned workflows with CPU/GPU and NPU case counts
- UT details
- ST details

Within the UT and ST sections, matched, CPU/GPU-only, NPU-only, and manual-review cases are shown in that order, with CPU/GPU and NPU references adjacent for easier comparison.
