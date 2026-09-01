# Four-corpus CPU retrieval measurements

Sixteen measured cells across four BEIR corpora and five pipeline arms, on CPU only, measured 2026-08-28 to 2026-08-31 against Cortrix core `79a4eb17c62521338d1ac47a9749e6230e87e69b`.

This bundle is immutable. The July bundle in `../beir-three-full-corpus-2026-07-v1/` is untouched; its values are superseded for the newer build, not withdrawn.

## What this bundle is for

A measurement round, not a release candidate. It records what was measured, including arms and a corpus the earlier bundle did not have, and including the depths it did **not** measure. It uses its own schema (`measurement-bundle` / `measurement-scorecard`) rather than the release-candidate schema, because forcing a round into a contract written for a different shape is how numbers quietly become wrong.

## Results

nDCG@10 / recall@10. Every cell evaluated every selected query, with no missing query text and no request failures.

### Without LLM stages

| Corpus | Dense + BM25 | + cross-encoder rerank | + rerank (full profile) |
|---|---|---|---|
| SciFact | 0.5942 / 0.7728 | 0.6184 / 0.8111 | 0.6206 / 0.8044 |
| NFCorpus | 0.2991 / 0.1495 | 0.3121 / 0.1569 | 0.3091 / 0.1539 |
| FiQA | 0.2540 / 0.3703 | 0.3125 / 0.4523 | 0.3156 / 0.4554 |

### Quora, 522,931 documents across 8 namespaces, 2,000 of 10,000 judged queries

| Arm | nDCG@10 | recall@10 | mean latency |
|---|---|---|---|
| Dense + BM25 | 0.5003 | 0.7142 | 1.7 s |
| + cross-encoder rerank | 0.2031 | 0.3078 | 19.0 s |
| + LLM listwise rerank | 0.3039 | 0.3092 | 21.6 s |

These three arms query the same eight namespaces, so they are the only strictly controlled comparison in the bundle: the arm is the sole variable.

**Reranking costs 0.297 nDCG here.** A cross-encoder scores how relevant a passage is to a query. Quora's ground truth is a *duplicate* — a different phrasing of the same question. When the criterion and the task diverge, the near-identical document is scored below topically broader ones and can leave the result set entirely. The LLM listwise stage is a partial repair on top of that loss, recovering 0.101, not the cause of it.

The same reranker is a small gain on the three question-answering corpora above. The direction is a property of the task, not of the model.

### With ingest-time LLM enrichment

| Corpus | Ingest LLM + rerank | + LLM listwise |
|---|---|---|
| SciFact | 0.6449 / 0.8214 | 0.7784 / 0.8752 |
| NFCorpus | 0.3124 / 0.1589 | 0.3772 / 0.1768 |

## What you must know before quoting any of this

**Measured at top_k=10 only.** Retrieval depth is top_k × 5, so a deeper cutoff would have used a different candidate pool and would not have reproduced these values. Cutoffs beyond 10 are absent rather than estimated.

**Two independent ingests of the same corpus do not agree.** Measured here on NFCorpus with one binary: 0.2991 / 0.1495 against 0.3001 / 0.1524. About 0.001–0.003 nDCG is therefore the floor for any comparison across separate ingests. On SciFact, NFCorpus and FiQA the arms were ingested separately, so those arm-to-arm differences carry that floor; the NFCorpus rerank gain of 0.013 is only a few times it. The Quora arms share namespaces and carry none of it.

**An unexplained residual of 0.0011 nDCG.** Holding code, configuration, tool and vector graph fixed against an earlier measurement of the same revision leaves 0.0011 on NFCorpus and 0.0027 on FiQA. About one document in 323 queries. Disclosed rather than rounded away.

**The published p50 and p95 are one observation above nearest rank on some cells.** The measuring runner indexed `floor(p*n)`, which equals nearest rank unless `p*n` is an integer. At n=300 and n=2000 both percentiles are affected, at n=648 only p50, at n=323 neither. Per-query samples were not retained, so these are documented exactly rather than recomputed; each scorecard names which of its own percentiles are affected. The published runner computes nearest rank, so a reproduction differs by at most that one observation. Means, minima and maxima are unaffected, as are all retrieval metrics.

**Absolute values.** Quora dense-only at 0.50 is below published dense-retrieval baselines for that corpus, and FiQA carries a known loss from the score fusion constant in this build. Arm-to-arm comparison within this bundle is valid; the absolute numbers need those boundaries stated first.

**Comparability with July.** Only the cross-encoder arm on SciFact, NFCorpus and FiQA has a like-for-like cell in the July bundle — same corpus, same runner profile, same cutoff. Every other cell here has no predecessor, and each scorecard's `comparability` block says which case it is. The July round ran on GPU hardware with a query-complexity model that this round did not use, so even the comparable cells differ by build **and** hardware **and** configuration.

## What these numbers are not evidence of

Answer quality of any system built on retrieval. Production throughput, latency or capacity under concurrent load. Security, privacy or compliance properties. Competitive ranking against other retrieval systems. Business outcomes of any kind.

## Provenance

Models are pinned by content hash, not by name: `models.content_sha256` on every scorecard carries the sha256 of each ONNX graph, its external weight file and its tokenizer. Corpora are pinned by archive hash in `dataset_archive_sha256`. The SciFact archive hash matches the one recorded in the July bundle, so both rounds demonstrably read the same corpus.

LLM identity is recorded per cell and split by stage, because ingest-time enrichment and query-time listwise reranking are different claims. The Quora listwise cells ran on vector-only namespaces: query-time LLM, no ingest-time one. The SciFact and NFCorpus listwise cells ran on enriched namespaces and therefore had both. `arm_config` carries the retrieval knobs each profile sent, so an arm is reproducible from the record rather than from a profile name.

The runner that produced these cells was not a committed revision. Both measuring nodes ran an uncommitted working tree on top of a named commit, so the honest identity is a content hash, recorded per cell in `source_locks.measuring_runner_content_sha256`. The published runner is a rewrite for publication and is **not** presented as the revision that produced these numbers.

The published runner is pinned by commit **and** by content hash, so `source_equivalence` can be checked rather than taken on trust:

```bash
git show <public_runner_sha>:benchmarks/beir-retrieval-quality/runner/query_only_eval.py | sha256sum
```

Two differences between the published and measuring runners are known, and both are stated rather than smoothed over. The published runner counts failed requests, which the measuring runners did not — every cell here recorded zero failures and zero missing query text, so the two behave identically over this data. And the published runner computes nearest-rank percentiles while the measuring runners indexed `floor(p*n)`, which is the one-observation difference described above.

The published profiles file has a different content hash from either measuring file. The five profiles used here were compared field by field and every retrieval parameter is identical; the differences are comment wording and one diagnostic profile, unused here, that is absent from the published file.

## Files

| File | Contents |
|---|---|
| `manifest.json` | Build, datasets, arms, declared cell count, protocol, limitations |
| `summaries.json` | Per-corpus, per-arm headline metrics |
| `scorecards/` | One file per measured cell, 16 total |
| `reproduction.md` | How to obtain the corpora and re-run each cell |
| `checksums.sha256` | sha256 of every file above |

Verify with:

```
python3 benchmarks/beir-retrieval-quality/tools/validate_measurement_bundle.py \
    results/published/beir-four-corpus-cpu-2026-08-v1
```
