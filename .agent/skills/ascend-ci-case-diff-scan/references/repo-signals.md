# Repo Signals

Use these target-repo facts when running `ascend-ci-case-diff-scan` against an external `verl` checkout.

## Primary inputs

- `.github/workflows/*.yml`

## Workflow scope

Exclude workflows that are not part of CPU/GPU/NPU test coverage. The default ignored set is maintained in `config/workflow_scope.json`.

Examples:

- `check-pr-title.yml`
- `cpu_unit_tests.yml`
- `doc.yml`
- `docker-*.yml`
- `pre-commit.yml`
- `precommit-autofix.yml`
- `sanity.yml`
- `scorecard.yml`
- `secrets_scan.yml`
- `type-coverage-check.yml`

## Workflow families

- `foo.yml` pairs with `foo_ascend.yml`
- `gpu_unit_tests.yml` pairs with `npu_unit_tests.yml`
- Standalone NPU workflows such as `e2e_ascend.yml` and `nightly_ascend.yml` are treated as NPU-only unless a matching CPU/GPU workflow exists

## Extractable command forms

Recognize only workflow commands that visibly execute tests:

- `pytest ... tests/...`
- `bash tests/.../*.sh`
- `torchrun ... tests/...`
- `torchrun ... -m pytest tests/...`

## Matching expectations

- Exact target matches are the strongest signal.
- Same target with materially different env or argument prefixes should usually become `manual_review_needed`.
- Repeated commands should remain distinct when they appear in different workflow, job, or step contexts.
- For UT, prefer function-level or test-method-level comparison over file-level comparison because broad `pytest tests/...` commands and `--ignore-glob` options can hide partial support.
