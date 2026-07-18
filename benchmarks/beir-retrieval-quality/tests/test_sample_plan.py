from __future__ import annotations

import json
import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER_DIR))

from sample_plan import make_sample  # noqa: E402


def test_make_sample_honors_query_and_corpus_caps(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    qrels_dir = dataset_root / "qrels"
    qrels_dir.mkdir(parents=True)
    with (dataset_root / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(1, 31):
            handle.write(json.dumps({"_id": f"d{index}", "title": "", "text": f"doc {index}"}) + "\n")
    with (dataset_root / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(1, 11):
            handle.write(json.dumps({"_id": f"q{index}", "text": f"query {index}"}) + "\n")
    with (qrels_dir / "test.tsv").open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for index in range(1, 11):
            handle.write(f"q{index}\td{index}\t1\n")

    sample = make_sample(
        dataset="tiny",
        root=dataset_root,
        output_dir=tmp_path / "sample",
        fraction=1.0,
        seed=44,
        min_queries=1,
        max_queries=3,
        max_corpus_rows=5,
        log=lambda _: None,
    )

    assert sample.selected_queries == 3
    assert sample.target_corpus_rows == 5
    assert sample.actual_corpus_rows == 5
    assert sample.uncapped_target_corpus_rows == 30
