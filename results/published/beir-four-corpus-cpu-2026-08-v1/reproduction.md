# Reproducing these measurements

Every command here uses only public files and the runner in this repository. Nothing depends on the environment the original round ran in.

Expect the numbers to land close but not identical. The reasons are in the bundle's limitations and are worth reading before concluding that a difference means something: an independently ingested namespace has its own vector graph, and the floor for that difference measured here is roughly 0.001–0.003 nDCG.

## 1. Obtain the corpora

Each corpus is the public BEIR archive. The manifest records the URL, the corpus size, the judged query count and the qrels row count for each.

| Corpus | Documents | Judged queries | Used here |
|---|---|---|---|
| SciFact | 5,183 | 300 | all 300 |
| NFCorpus | 3,633 | 323 | all 323 |
| FiQA | 57,638 | 648 | all 648 |
| Quora | 522,931 | 10,000 | first 2,000 |

Unpack so each corpus sits at `<work-dir>/datasets/<name>/<name>/` with `corpus.jsonl`, `queries.jsonl` and `qrels/test.tsv`.

## 2. Check you have the same queries

The query subset is the judged qrels ids for the split, sorted lexicographically, then truncated. The runner prints the sha256 of that list as `query_subset`. It must match the value in `manifest.json` for that corpus:

| Corpus | query_subset sha256 |
|---|---|
| SciFact | `26cc483d5ed08c9559e5f99948281accbb9aac17b8c2ea6e1e857286d9ebd684` |
| NFCorpus | `f3950f7914eaf889ccceeaa0c7d53c787e674db1e442f8fa07ff8d2aec991d44` |
| FiQA | `afde57664f4b3702f65a94218a2dbd831c5461b3aa3106e7ffd40820b22d8f6c` |
| Quora | `826e3aa1ba6b464e43e12c6e11afcc2669e8327d20f832f8b33cbbf1a1740e0f` |

If yours differs, stop: you are scoring a different set of queries and no comparison below is meaningful.

## 3. Ingest

Ingest each corpus into its own namespace with the Cortrix build named in `manifest.json`. Wait for every task to reach a terminal state before querying; these measurements were taken with no ingestion in flight.

Quora was ingested across eight namespaces and queried as one merged set. A single namespace holding the whole corpus is also valid, but it is not what was measured, and cross-namespace merge is not identical to a single-namespace top-k.

## 4. Score each cell

One command per cell. `--profile` selects the arm; `manifest.json` maps every arm id to its runner profile.

```bash
RUNNER=benchmarks/beir-retrieval-quality/runner/query_only_eval.py

# Dense + BM25, no reranking
python3 $RUNNER <namespace> --dataset scifact --max-queries 300 --top-k 10 \
    --profile vector_full --timeout-seconds 300 \
    --base-url http://127.0.0.1:8420 --work-dir <work-dir>

# Cross-encoder rerank
python3 $RUNNER <namespace> --dataset scifact --max-queries 300 --top-k 10 \
    --profile bge_m3_rerank_baseline_v2 --timeout-seconds 300 \
    --base-url http://127.0.0.1:8420 --work-dir <work-dir>

# Quora: eight namespaces merged, 2,000 queries
python3 $RUNNER ns1,ns2,ns3,ns4,ns5,ns6,ns7,ns8 --dataset quora \
    --max-queries 2000 --top-k 10 --profile bge_m3_rerank_full \
    --timeout-seconds 300 --base-url http://127.0.0.1:8420 --work-dir <work-dir>
```

`--timeout-seconds 300` is not optional for the reranking and LLM arms. The client default of 30s cuts those requests off mid-flight; that is what the flag exists for.

The arms with `ingest_llm_rerank` and `llm_listwise` require an LLM role configured on the server. Without one, the stage does not run and the arm is not the arm.

## 5. Compare

Each scorecard carries `comparability`, saying whether a cell in the July bundle measured the same thing. Where `directly_comparable` is false, there is no predecessor — that is a statement about coverage, not a missing value to be filled in with the nearest-looking number.

## Protocol details that change the result if you change them

| Setting | Value | Effect if changed |
|---|---|---|
| `top_k` | 10 | Retrieval depth is `top_k × 5`; a different top_k changes the candidate pool, so the @10 figures will not reproduce |
| query mode | serial | Concurrent querying changes latency and can change results under load |
| models | bge-m3, bge-reranker-v2-m3, ONNX fp32 | A different precision or execution provider changes scores in the last digits |
| dedup | by document id, best rank kept | Scoring chunks instead of documents inflates recall |
