# Local Reproduction

This document describes sanitized local reproduction steps for the BEIR retrieval quality benchmark.

## Prerequisites

- A running Cortrix API server.
- Python 3.
- Enough disk space for downloaded BEIR archives and run artifacts.
- A writable runtime directory outside Git. The examples use `CORTRIX_BENCH_WORKDIR`, defaulting to `/tmp/cortrix-benchmarks/beir-retrieval-quality`.

## Commands

```bash
python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py selftest
```

```bash
export CORTRIX_BENCH_WORKDIR="${CORTRIX_BENCH_WORKDIR:-/tmp/cortrix-benchmarks/beir-retrieval-quality}"

python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py run \
  --base-url http://127.0.0.1:8080 \
  --work-dir "$CORTRIX_BENCH_WORKDIR" \
  --datasets scifact,nfcorpus,fiqa,webis-touche2020,hotpotqa \
  --profiles vector_only,full_stack \
  --min-queries 20 \
  --poll-timeout-seconds 1800
```

For a smaller gate, use explicit caps on an upstream dataset:

```bash
python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py run \
  --base-url http://127.0.0.1:8080 \
  --work-dir "$CORTRIX_BENCH_WORKDIR" \
  --datasets scifact \
  --profiles vector_only \
  --seed 44044 \
  --max-queries 30 \
  --max-corpus-docs 1000 \
  --poll-timeout-seconds 1800
```

The runner stores the upstream archive and generated sample under `--work-dir`. The fixed seed and `sample_manifest.json` make the query and corpus boundary auditable. Treat capped output as quick validation only: its scores are not comparable to the full-corpus measurements and cannot support a published performance claim.

## Artifacts

The runner writes JSON manifests, per-query logs, and resource snapshots under the selected `--work-dir`. Do not commit those runtime artifacts.
