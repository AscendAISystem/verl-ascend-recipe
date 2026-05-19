# Ascend CI Case Diff Scan

This skill statically audits a target `verl` repository and reports workflow/case alignment between CPU/GPU and NPU workflows.

## What it does

- Excludes non-test workflows through `config/workflow_scope.json`
- Excludes `cpu_unit_tests.yml` from the scan because it is treated as shared baseline coverage rather than an NPU-adapted workflow
- Pairs CPU/GPU workflows with their NPU counterparts
- Counts UT cases at the test-function or test-method level
- Counts ST cases at the command level
- Writes an English `report.md`

## Workflow pairing

- `foo.yml` pairs with `foo_ascend.yml`
- `gpu_unit_tests.yml` pairs with `npu_unit_tests.yml`
- Standalone NPU workflows such as `e2e_ascend.yml` and `nightly_ascend.yml` remain NPU-only

## Case rules

- `pytest` means UT
- UT entries are expanded to concrete test functions or test methods whenever the target Python file can be parsed
- `torchrun` and `bash tests/*.sh` mean ST
- Repeated commands are kept distinct by `workflow name`, `job name`, and `step name`

## Output

The report includes:

- ignored workflows
- scanned workflows with CPU/GPU and NPU case counts
- UT details
- ST details

Within UT and ST sections, the report shows matched, CPU/GPU-only, NPU-only, and manual-review cases in that order, with adjacent CPU/GPU and NPU references for easy comparison.
