# Cortrix Benchmarks

Cortrix Benchmarks provides retrieval-quality methodology, runner code, validation contracts, and curated maintainer-reported result bundles for Cortrix.

## Published result bundle

The [three-dataset full-corpus retrieval bundle](results/published/beir-three-full-corpus-2026-07-v1/README.md) records six cells across SciFact, FiQA, and NFCorpus:

- Full Stack
- Embedding + Reranking

The bundle includes exact scorecards, profile identity, source locks, validity status, limitations, a maintainer rerun procedure, and a SHA-256 inventory. It measures retrieval quality only; it does not establish end-to-end answer quality, universal domain performance, or business outcomes.

The published scores are maintainer-reported full-corpus measurements. The repository lets readers inspect and checksum the scorecards, requested profiles, method, and source identities. It does not yet provide a completed independent public rerun or independent per-query recomputation. A requested Full Stack profile included the configured embedding, reranking, and query-processing stages; the profile comparison does not isolate the causal contribution of any one feature.

## Source access and historical provenance

The Core snapshot used for these measurements, [`4a6299ca86c7bec21ed7b8989a729a198fe5a42a`](https://github.com/cortrix/cortrix/tree/4a6299ca86c7bec21ed7b8989a729a198fe5a42a), is now publicly inspectable. The immutable v1 bundle's reproduction guide and [`provenance/source-equivalence.json`](provenance/source-equivalence.json) retain the source-availability wording recorded before the Core repository became public. Read that wording as historical snapshot metadata, not as the current availability state.

This availability update does not change the measurement source locks, runner or profile files, scores, or result checksums. It also does not claim that an independent public rerun has been completed.

## Inspect and verify the bundle files

```bash
python3 benchmarks/beir-retrieval-quality/tools/validate_release_candidate.py \
  results/published/beir-three-full-corpus-2026-07-v1

cd results/published/beir-three-full-corpus-2026-07-v1
shasum -a 256 -c checksums.sha256
```

These checks verify bundle structure and file identity, not an independent rerun of the measurements. For citation, pin the bundle directory to a 40-character Git commit rather than a mutable branch.

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

## Project governance

- [Contributing](CONTRIBUTING.md): contribution workflow, DCO 1.1, and benchmark challenges.
- [Maintainers](MAINTAINERS.md): ownership, acknowledgment targets, and recusal rules.
- [Security policy](SECURITY.md): private vulnerability reporting and supported versions.
- [Code of Conduct](CODE_OF_CONDUCT.md): community standards and private reporting.

## License

- Code under `benchmarks/beir-retrieval-quality/runner/`, `tests/`, and `tools/` is licensed under `Apache-2.0`.
- Documentation, schemas, curated results, and provenance are licensed under `CC-BY-4.0`.
- Third-party datasets are not included. Their original terms continue to apply.

Historical code revisions before this license mapping changed remain available under `AGPL-3.0-only`; history is not rewritten or dual-licensed. See [LICENSES/README.md](LICENSES/README.md) for the current path mapping, current and historical license texts, and [COPYRIGHT.md](COPYRIGHT.md) for copyright information.
