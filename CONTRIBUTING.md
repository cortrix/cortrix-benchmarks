# Contributing

Contributions should keep benchmark code reproducible, public documentation clear, and result artifacts independently verifiable.

## Expectations

- Keep runner changes scoped to benchmark behavior.
- Add or update tests when changing metrics, sampling, manifests, source-document mapping, or result parsing.
- Do not commit downloaded datasets, derived dataset fixtures, raw runs, logs, model assets, credentials, or local infrastructure details.
- Keep capped validation separate from published full-corpus measurements.
- Do not change curated metrics or source locks without a new measurement review.

## Before opening a pull request

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q benchmarks/beir-retrieval-quality/tests
python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py selftest
python3 benchmarks/beir-retrieval-quality/tools/validate_release_candidate.py \
  results/published/beir-three-full-corpus-2026-07-v1
git diff --check
```
