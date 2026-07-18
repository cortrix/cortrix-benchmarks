from __future__ import annotations

import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER_DIR))

from profiles import PROFILES  # noqa: E402

ISOLATION_ARMS = [
    "bge_m3_rerank_baseline_v2",
    "bge_m3_rerank_ingest_llm",
    "bge_m3_rerank_all_llm_v2",
    "bge_m3_llm_listwise",
]


def test_legacy_profiles_keep_five_field_body() -> None:
    for name in ["vector_full", "vector_only", "bge_m3_rerank_full", "full_stack"]:
        body = PROFILES[name].query_body(query="q", namespace="ns", top_k=50)
        assert set(body) == {"query", "namespaces", "top_k", "rerank", "rag_fusion"}


def test_isolation_arms_pin_route_and_explain() -> None:
    for name in ISOLATION_ARMS:
        body = PROFILES[name].query_body(query="q", namespace="ns", top_k=50)
        assert body["route"] == "complex"
        assert body["explain"] is True
        assert body["rerank"] is True
        assert PROFILES[name].query_fraction == 1.0


def test_arm_a_and_b_request_no_query_time_llm() -> None:
    for name in ["bge_m3_rerank_baseline_v2", "bge_m3_rerank_ingest_llm"]:
        body = PROFILES[name].query_body(query="q", namespace="ns", top_k=50)
        assert body["rag_fusion"] is False
        assert "llm_rerank" not in body


def test_arm_c_keeps_rag_fusion_without_llm_rerank() -> None:
    body = PROFILES["bge_m3_rerank_all_llm_v2"].query_body(query="q", namespace="ns", top_k=50)
    assert body["rag_fusion"] is True
    assert "llm_rerank" not in body


def test_arm_d_sends_llm_rerank_config() -> None:
    body = PROFILES["bge_m3_llm_listwise"].query_body(query="q", namespace="ns", top_k=50)
    assert body["llm_rerank"] is True
    assert body["llm_rerank_config"] == {"top_n": 30}
    assert body["rag_fusion"] is False


def test_profile_to_json_carries_new_fields() -> None:
    payload = PROFILES["bge_m3_llm_listwise"].to_json()
    assert payload["route"] == "complex"
    assert payload["explain"] is True
    assert payload["llm_rerank"] is True
    assert payload["llm_rerank_top_n"] == 30
