from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Set

from beir_loader import count_corpus_rows, iter_corpus, load_qrels, load_queries


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class SamplePlan:
    dataset: str
    fraction: float
    seed: int
    split: str
    full_corpus_rows: int
    full_qrels_queries: int
    selected_queries: int
    selected_relevant_docs: int
    uncapped_target_corpus_rows: int
    max_queries: Optional[int]
    max_corpus_rows: Optional[int]
    target_corpus_rows: int
    actual_corpus_rows: int
    sampled_corpus_path: Path
    sampled_queries_path: Path
    sampled_qrels_path: Path
    manifest_path: Path

    def to_json(self) -> Dict[str, object]:
        data = dict(self.__dict__)
        for key in [
            "sampled_corpus_path",
            "sampled_queries_path",
            "sampled_qrels_path",
            "manifest_path",
        ]:
            data[key] = str(data[key])
        return data


def make_sample(
    dataset: str,
    root: Path,
    output_dir: Path,
    fraction: float,
    seed: int,
    split: str = "test",
    min_queries: int = 20,
    max_queries: Optional[int] = None,
    max_corpus_rows: Optional[int] = None,
    log: LogFn = print,
) -> SamplePlan:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    output_dir.mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(root, split=split)
    query_ids = sorted(qrels)
    selected_count = max(1, math.ceil(len(query_ids) * fraction))
    selected_count = max(min_queries, selected_count)
    if max_queries is not None:
        selected_count = min(selected_count, max(1, max_queries))
    selected_count = min(selected_count, len(query_ids))
    rng = random.Random(seed)
    selected_queries = sorted(rng.sample(query_ids, selected_count))

    selected_qrels: Dict[str, Dict[str, float]] = {
        query_id: qrels[query_id] for query_id in selected_queries
    }
    relevant_doc_ids: Set[str] = set()
    for docs in selected_qrels.values():
        relevant_doc_ids.update(docs)

    log(f"[sample] count corpus rows for {dataset}")
    full_rows = count_corpus_rows(root)
    uncapped_target_rows = max(len(relevant_doc_ids), math.ceil(full_rows * fraction))
    target_rows = uncapped_target_rows
    if max_corpus_rows is not None:
        target_rows = min(uncapped_target_rows, max(max_corpus_rows, len(relevant_doc_ids)))
    distractor_count = max(0, target_rows - len(relevant_doc_ids))

    log(
        f"[sample] {dataset}: queries={selected_count}/{len(query_ids)} "
        f"target_docs={target_rows}/{full_rows} relevant_docs={len(relevant_doc_ids)}"
    )
    distractor_ids = _select_distractors(root, relevant_doc_ids, distractor_count, seed)
    selected_doc_ids = relevant_doc_ids | distractor_ids

    sampled_corpus = output_dir / "sampled_corpus.jsonl"
    sampled_queries = output_dir / "sampled_queries.jsonl"
    sampled_qrels = output_dir / "sampled_qrels.tsv"
    manifest = output_dir / "sample_manifest.json"

    actual_docs = 0
    with sampled_corpus.open("w", encoding="utf-8") as out:
        for doc_id, title, text in iter_corpus(root):
            if doc_id not in selected_doc_ids:
                continue
            record = {"_id": doc_id, "title": title, "text": text}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            actual_docs += 1

    query_texts = load_queries(root, set(selected_queries))
    with sampled_queries.open("w", encoding="utf-8") as out:
        for query_id in selected_queries:
            out.write(
                json.dumps(
                    {"_id": query_id, "text": query_texts.get(query_id, "")},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with sampled_qrels.open("w", encoding="utf-8") as out:
        out.write("query-id\tcorpus-id\tscore\n")
        for query_id in selected_queries:
            for doc_id, score in sorted(selected_qrels[query_id].items()):
                out.write(f"{query_id}\t{doc_id}\t{score:g}\n")

    plan = SamplePlan(
        dataset=dataset,
        fraction=fraction,
        seed=seed,
        split=split,
        full_corpus_rows=full_rows,
        full_qrels_queries=len(query_ids),
        selected_queries=len(selected_queries),
        selected_relevant_docs=len(relevant_doc_ids),
        uncapped_target_corpus_rows=uncapped_target_rows,
        max_queries=max_queries,
        max_corpus_rows=max_corpus_rows,
        target_corpus_rows=target_rows,
        actual_corpus_rows=actual_docs,
        sampled_corpus_path=sampled_corpus,
        sampled_queries_path=sampled_queries,
        sampled_qrels_path=sampled_qrels,
        manifest_path=manifest,
    )
    manifest.write_text(json.dumps(plan.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def load_sampled_corpus(path: Path) -> List[Mapping[str, str]]:
    rows: List[Mapping[str, str]] = []
    for row in iter_sampled_corpus(path):
        rows.append(row)
    return rows


def iter_sampled_corpus(path: Path) -> Iterator[Mapping[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            yield {
                "_id": str(item.get("_id", "")),
                "title": str(item.get("title", "") or ""),
                "text": str(item.get("text", "") or ""),
            }


def load_sampled_queries(path: Path) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            queries[str(item.get("_id", ""))] = str(item.get("text", "") or "")
    return queries


def _select_distractors(root: Path, relevant_doc_ids: Set[str], count: int, seed: int) -> Set[str]:
    if count <= 0:
        return set()
    heap: List[tuple[int, str]] = []
    for doc_id, _, _ in iter_corpus(root):
        if doc_id in relevant_doc_ids:
            continue
        rank = _stable_rank(doc_id, seed)
        item = (-rank, doc_id)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return {doc_id for _, doc_id in heap}


def _stable_rank(doc_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
