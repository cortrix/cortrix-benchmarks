# BEIR Retrieval Quality Benchmark

This benchmark evaluates Cortrix retrieval quality against supported BEIR datasets. It provides two deliberately separate paths:

1. Full-corpus measurement for a curated, citable result bundle.
2. Deterministic capped validation for checking a local installation before committing full-corpus resources.

## Supported upstream datasets

- SciFact
- NFCorpus
- FiQA
- Webis-Touche2020
- HotpotQA

No dataset corpus, queries, qrels, or derived fixture is distributed here. The runner obtains supported archives from the configured upstream source and stores them under `--work-dir`.

## Profiles

- `vector_only`: sampled vector retrieval baseline.
- `full_stack`: sampled full-stack validation.
- `bge_m3_rerank_baseline_v2`: BGE-M3 embedding with `bge-reranker-v2-m3`, without LLM-dependent ingestion or query stages.
- `bge_m3_rf_pool_listwise`: Full Stack profile with the pinned LLM-dependent ingestion and query contract.

A requested profile does not prove that every runtime feature was active. Published results therefore report scientific validity and strict feature completeness separately.

## Self-test

```bash
python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py selftest
```

## Capped validation

Use a fixed seed and explicit caps on an upstream dataset:

```bash
export CORTRIX_BENCH_WORKDIR="${CORTRIX_BENCH_WORKDIR:-/tmp/cortrix-benchmarks/beir-retrieval-quality}"

python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py run \
  --base-url http://127.0.0.1:8080 \
  --work-dir "$CORTRIX_BENCH_WORKDIR" \
  --datasets scifact \
  --profiles vector_only \
  --seed 44044 \
  --max-queries 30 \
  --max-corpus-docs 1000
```

The generated `sample_manifest.json` records the selected boundary. Capped validation can verify installation, pipeline behavior, configuration, source-document mapping, and metric generation. It is not comparable to a published full-corpus result and cannot support a performance claim.

## Full-corpus result

The curated six-cell bundle is at [results/published/beir-three-full-corpus-2026-07-v1](../../results/published/beir-three-full-corpus-2026-07-v1/README.md). Follow its own reproduction procedure, source locks, hardware boundaries, checksums, and strict-status limitations.

## Local artifacts

Runs write manifests, query logs, task snapshots, latency, metrics, and resource evidence under `--work-dir`. Keep all runtime output outside Git.

See [Methodology](docs/methodology.md), [Local reproduction](docs/local-reproduction.md), and [Result interpretation](docs/result-interpretation.md).
