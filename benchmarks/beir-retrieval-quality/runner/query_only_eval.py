#!/usr/bin/env python3
"""Query-only recall/nDCG eval against an ALREADY-INGESTED Cortrix namespace.

Why this exists: full-corpus BEIR ingest on CPU bge-m3 is ~8h, far longer than the
runner's poll-timeout. The runner has no skip-ingest/reuse-namespace mode, so this
script reuses the runner's own loaders + metrics + doc-id extraction to score an
existing namespace without re-ingesting. Same doc-collapse logic as run_benchmark
(extract_result_doc_ids + dedup) so the numbers are comparable.

Usage:
  python3 query_only_eval.py <namespace> [--dataset scifact] [--max-queries 50] \
      [--base-url http://127.0.0.1:8420] [--top-k 10]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import beir_loader
import metrics
from cortrix_client import CortrixClient, extract_result_doc_ids
from profiles import PROFILES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("namespace")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--max-queries", type=int, default=50)
    ap.add_argument("--base-url", default="http://127.0.0.1:8420")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--profile", default="vector_full")
    ap.add_argument(
        "--work-dir", default="/tmp/cortrix-benchmarks/beir"
    )
    args = ap.parse_args()

    root = Path(args.work_dir) / "datasets" / args.dataset / args.dataset
    qrels = beir_loader.load_qrels(root, "test")
    # Deterministic query subset: sorted qrels ids, capped.
    qids = [q for q in sorted(qrels.keys()) if qrels[q]]
    if args.max_queries > 0:
        qids = qids[: args.max_queries]
    queries = beir_loader.load_queries(root, set(qids))

    profile = PROFILES[args.profile]
    top_k = args.top_k
    retrieval_k = max(top_k, top_k * 5)
    client = CortrixClient(args.base_url)

    runs: dict[str, list[str]] = {}
    missing = 0
    for qid in qids:
        qtext = queries.get(qid)
        if not qtext:
            missing += 1
            continue
        body = profile.query_body(query=qtext, namespace=args.namespace, top_k=retrieval_k)
        response, _lat = client.query(body)
        seen: set[str] = set()
        doc_ids: list[str] = []
        for d in extract_result_doc_ids(response):
            if d and d not in seen:
                seen.add(d)
                doc_ids.append(d)
        runs[qid] = doc_ids[:top_k]

    scored_qrels = {q: qrels[q] for q in qids if q in queries}
    rec = metrics.recall_at_k(scored_qrels, runs, top_k)
    ndcg = metrics.ndcg_at_k(scored_qrels, runs, top_k)

    def mean(d: dict) -> float:
        return statistics.mean(d.values()) if d else 0.0

    print(f"namespace      : {args.namespace}")
    print(f"dataset        : {args.dataset} (profile={args.profile})")
    print(f"queries scored : {len(rec)} (requested {len(qids)}, missing_text={missing})")
    print(f"recall@{top_k}      : {mean(rec):.4f}")
    print(f"ndcg@{top_k}        : {mean(ndcg):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
