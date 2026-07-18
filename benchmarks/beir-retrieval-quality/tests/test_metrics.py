from __future__ import annotations

import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER_DIR))

from metrics import ndcg_at_k, recall_at_k  # noqa: E402


def test_recall_and_ndcg_smoke() -> None:
    qrels = {"q1": {"d1": 1.0, "d2": 1.0}, "q2": {"d3": 2.0}}
    runs = {"q1": ["d2", "d9"], "q2": ["d4", "d3"]}

    recall = recall_at_k(qrels, runs, k=2)
    ndcg = ndcg_at_k(qrels, runs, k=2)

    assert recall["q1"] == 0.5
    assert recall["q2"] == 1.0
    assert 0.0 < ndcg["q1"] <= 1.0
    assert 0.0 < ndcg["q2"] <= 1.0
