from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER_DIR))

import score_signal_overlay_eval as overlay  # noqa: E402


def test_overlay_uses_sqlite_score_signals_to_reorder_candidates(tmp_path: Path) -> None:
    store_db = tmp_path / "store.db"
    conn = sqlite3.connect(store_db)
    conn.execute(
        "create table blocks (child_id text primary key, enriched_score real, semantic_score real)"
    )
    conn.execute(
        "insert into blocks(child_id, enriched_score, semantic_score) values (?, ?, ?)",
        ("c1", 0.0, 0.2),
    )
    conn.execute(
        "insert into blocks(child_id, enriched_score, semantic_score) values (?, ?, ?)",
        ("c2", 1.0, 1.0),
    )
    conn.commit()
    conn.close()

    rows = [
        {
            "query_id": "q1",
            "retrieved_doc_ids": ["d1", "d2"],
            "response": {
                "results": [
                    {
                        "child_id": "c1",
                        "score": 0.1,
                        "rerank_score": 1.0,
                        "metadata": {"beir_corpus_id": "d1"},
                    },
                    {
                        "child_id": "c2",
                        "score": 0.1,
                        "rerank_score": 0.9,
                        "metadata": {"beir_corpus_id": "d2"},
                    },
                ]
            },
        }
    ]
    queries = tmp_path / "queries.jsonl"
    queries.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    qrels = tmp_path / "sampled_qrels.tsv"
    qrels.write_text("q1\td2\t1\n", encoding="utf-8")

    loaded_rows = list(overlay.iter_query_rows(queries))
    signals = overlay.load_signals(store_db, ["c1", "c2"])
    runs, coverage = overlay.build_runs(loaded_rows, signals, top_k=2)
    qrels_data = overlay.load_sampled_qrels(qrels)

    assert coverage["results_matched_in_db"] == 2
    assert coverage["results_with_any_signal"] == 2
    assert coverage["queries_top10_changed_vs_response"] == 1
    assert runs["response_order"]["q1"] == ["d1", "d2"]
    assert runs["fixed_candidate_score_desc"]["q1"] == ["d2", "d1"]

    fixed_metrics = overlay.metric_summary(qrels_data, runs["fixed_candidate_score_desc"], 1)
    baseline_metrics = overlay.metric_summary(qrels_data, runs["response_order"], 1)
    assert fixed_metrics["recall@1"] == 1.0
    assert baseline_metrics["recall@1"] == 0.0
