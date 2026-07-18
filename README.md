# Cortrix Benchmarks

Cortrix Benchmarks provides reproducible retrieval-quality methodology, runner code, validation contracts, and curated result bundles for Cortrix.

## Published result bundle

The [three-dataset full-corpus retrieval bundle](results/published/beir-three-full-corpus-2026-07-v1/README.md) records six cells across SciFact, FiQA, and NFCorpus:

- Full Stack
- Embedding + Reranking

The bundle includes exact scorecards, profile identity, source locks, validity status, limitations, reproduction steps, and a SHA-256 inventory. It measures retrieval quality only; it does not establish end-to-end answer quality, universal domain performance, or business outcomes.

## Verify the bundle

```bash
python3 benchmarks/beir-retrieval-quality/tools/validate_release_candidate.py \
  results/published/beir-three-full-corpus-2026-07-v1

cd results/published/beir-three-full-corpus-2026-07-v1
shasum -a 256 -c checksums.sha256
```

For citation, pin the bundle directory to a 40-character Git commit rather than a mutable branch.

## Try the runner on a bounded sample

Start with a supported BEIR dataset obtained from its upstream distributor, then cap the run:

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

This capped path is quick validation for installation, pipeline behavior, configuration, source-document mapping, and metric generation. Its scores are not comparable to the published full-corpus measurements and cannot support a performance claim.

Downloaded datasets, raw runs, logs, model files, and runtime databases are not distributed in this repository and must remain outside Git.

## Documentation

- [Benchmark overview](benchmarks/beir-retrieval-quality/README.md)
- [Methodology](benchmarks/beir-retrieval-quality/docs/methodology.md)
- [Local reproduction](benchmarks/beir-retrieval-quality/docs/local-reproduction.md)
- [Result interpretation](benchmarks/beir-retrieval-quality/docs/result-interpretation.md)
- [Source equivalence](provenance/README.md)

## License

- Code under `benchmarks/beir-retrieval-quality/runner/`, `tests/`, and `tools/` is licensed under `AGPL-3.0-only`.
- Documentation, schemas, curated results, and provenance are licensed under `CC-BY-4.0`.
- Third-party datasets are not included. Their original terms continue to apply.

See [LICENSES/README.md](LICENSES/README.md) for the exact path mapping and license texts.
