from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence


Qrels = Mapping[str, Mapping[str, float]]
Runs = Mapping[str, Sequence[str]]


def recall_at_k(qrels: Qrels, runs: Runs, k: int = 10) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for query_id, relevant in qrels.items():
        rel_ids = {doc_id for doc_id, score in relevant.items() if score > 0}
        if not rel_ids:
            continue
        retrieved = set(runs.get(query_id, [])[:k])
        scores[query_id] = len(rel_ids & retrieved) / len(rel_ids)
    return scores


def ndcg_at_k(qrels: Qrels, runs: Runs, k: int = 10) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for query_id, relevant in qrels.items():
        ranking = list(runs.get(query_id, [])[:k])
        dcg = 0.0
        for rank, doc_id in enumerate(ranking, start=1):
            rel = float(relevant.get(doc_id, 0.0))
            if rel > 0:
                dcg += rel / math.log2(rank + 1)
        ideal_rels = sorted((float(v) for v in relevant.values() if v > 0), reverse=True)[:k]
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
        scores[query_id] = dcg / idcg if idcg > 0 else 0.0
    return scores


def summarize(values: Iterable[float]) -> Dict[str, float]:
    values_list = sorted(float(v) for v in values)
    if not values_list:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values_list),
        "mean": sum(values_list) / len(values_list),
        "p50": percentile(values_list, 50),
        "p95": percentile(values_list, 95),
        "min": values_list[0],
        "max": values_list[-1],
    }


def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def aggregate(qrels: Qrels, runs: Runs, latencies_ms: Sequence[float], k: int = 10) -> Dict[str, object]:
    recall = recall_at_k(qrels, runs, k=k)
    ndcg = ndcg_at_k(qrels, runs, k=k)
    return {
        f"recall@{k}": summarize(recall.values()),
        f"ndcg@{k}": summarize(ndcg.values()),
        "latency_ms": summarize(latencies_ms),
        "evaluated_queries": len(recall),
    }
