from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER_DIR))

from cortrix_client import (  # noqa: E402
    diff_query_latency_metric_values,
    extract_result_doc_ids,
    extract_result_traces,
    parse_query_latency_metrics,
)


def test_extract_result_traces_preserves_beir_mapping_and_scores() -> None:
    response = {
        "results": [
            {
                "child_id": "chunk-1",
                "score": 0.7,
                "rerank_score": "0.9",
                "metadata": {"beir_corpus_id": "d1"},
                "score_signals": {"semantic_score": 1.0, "enriched_score": 0.8},
            }
        ]
    }

    traces = extract_result_traces(response, query_id="q1", qrels_relevant=["d1"])

    assert extract_result_doc_ids(response) == ["d1"]
    assert traces[0]["query_id"] == "q1"
    assert traces[0]["rank"] == 1
    assert traces[0]["dedup_rank"] == 1
    assert traces[0]["returned_doc_id"] == "d1"
    assert traces[0]["beir_corpus_id"] == "d1"
    assert traces[0]["child_id"] == "chunk-1"
    assert traces[0]["rerank_score"] == 0.9
    assert traces[0]["semantic_score"] == 1.0
    assert traces[0]["enriched_score"] == 0.8
    assert traces[0]["source_role"] is None
    assert traces[0]["source_stage"] is None
    assert traces[0]["id_mapping_decision"] == "mapped_to_beir"
    assert traces[0]["is_qrels_relevant"] is True


def test_extract_result_traces_uses_metadata_json_fallback() -> None:
    response = {
        "results": [
            {
                "child_id": "chunk-1",
                "metadata": {
                    "metadata_json": json.dumps({"beir_corpus_id": "d2"}),
                },
            }
        ]
    }

    traces = extract_result_traces(response)

    assert extract_result_doc_ids(response) == ["d2"]
    assert traces[0]["returned_doc_id"] == "d2"
    assert traces[0]["beir_corpus_id"] == "d2"
    assert traces[0]["id_mapping_decision"] == "mapped_to_beir"


def test_extract_result_traces_uses_beir_alias_fields() -> None:
    response = {
        "results": [
            {"child_id": "chunk-1", "metadata": {"original_doc_id": "d3"}},
            {"child_id": "chunk-2", "metadata": {"beir_doc_id": "d4"}},
        ]
    }

    traces = extract_result_traces(response)

    assert [trace["beir_corpus_id"] for trace in traces] == ["d3", "d4"]
    assert extract_result_doc_ids(response) == ["d3", "d4"]


def test_extract_result_traces_ignores_internal_alias_id() -> None:
    response = {
        "results": [
            {
                "child_id": "01KWJTWK8EZHRWNSZJ0130KE4M",
                "metadata": {"id": "01KWJTWK8EZHRWNSZJ0130KE4M"},
            }
        ]
    }

    traces = extract_result_traces(response)

    assert traces[0]["beir_corpus_id"] is None
    assert traces[0]["returned_doc_id"] == "01KWJTWK8EZHRWNSZJ0130KE4M"
    assert traces[0]["id_mapping_decision"] == "internal_only"


def test_extract_result_traces_marks_internal_only_when_mapping_missing() -> None:
    response = {
        "results": [
            {
                "child_id": "01KWJTWK8EZHRWNSZJ0130KE4M",
                "score": 0.2,
                "metadata": {},
            }
        ]
    }

    traces = extract_result_traces(response, query_id="q1", qrels_relevant=["d1"])

    assert extract_result_doc_ids(response) == ["01KWJTWK8EZHRWNSZJ0130KE4M"]
    assert traces[0]["returned_doc_id"] == "01KWJTWK8EZHRWNSZJ0130KE4M"
    assert traces[0]["beir_corpus_id"] is None
    assert traces[0]["id_mapping_decision"] == "internal_only"
    assert traces[0]["is_qrels_relevant"] is False
    assert "missing_beir_corpus_id" in traces[0]["trace_gap"]


def test_extract_result_traces_uses_child_id_alias_mapping() -> None:
    response = {
        "results": [
            {
                "child_id": "01KWJTWK8EZHRWNSZJ0130KE4M",
                "score": 0.2,
                "metadata": {},
            }
        ]
    }
    aliases = {"01KWJTWK8EZHRWNSZJ0130KE4M": "68239"}

    traces = extract_result_traces(response, query_id="q1", qrels_relevant=["68239"], child_id_aliases=aliases)

    assert extract_result_doc_ids(response, child_id_aliases=aliases) == ["68239"]
    assert traces[0]["returned_doc_id"] == "68239"
    assert traces[0]["beir_corpus_id"] == "68239"
    assert traces[0]["id_mapping_decision"] == "mapped_to_beir"
    assert traces[0]["is_qrels_relevant"] is True
    assert "missing_beir_corpus_id" not in traces[0]["trace_gap"]


def test_extract_result_traces_uses_identity_alias_for_doc_summary_id() -> None:
    response = {
        "results": [
            {
                "child_id": "434788",
                "metadata": {"via_path": "doc_summary"},
            }
        ]
    }
    aliases = {"434788": "434788"}

    traces = extract_result_traces(response, child_id_aliases=aliases)

    assert extract_result_doc_ids(response, child_id_aliases=aliases) == ["434788"]
    assert traces[0]["returned_doc_id"] == "434788"
    assert traces[0]["beir_corpus_id"] == "434788"
    assert traces[0]["id_mapping_decision"] == "mapped_to_beir"
    assert "missing_beir_corpus_id" not in traces[0]["trace_gap"]


def test_extract_result_traces_uses_metadata_source_path() -> None:
    response = {
        "results": [
            {
                "child_id": "chunk-1",
                "metadata": {"beir_corpus_id": "d1", "via_path": "doc_summary"},
            }
        ]
    }

    traces = extract_result_traces(response, source_role="doc_summary_llm", source_stage="doc_summary_read")

    assert traces[0]["source_type"] == "doc_summary"
    assert traces[0]["source_role"] == "doc_summary_llm"
    assert traces[0]["source_stage"] == "doc_summary_read"
    assert traces[0]["source_type_evidence"] == "metadata.via_path"
    assert "missing_explicit_source_type" not in traces[0]["trace_gap"]


def test_extract_result_traces_does_not_treat_source_path_as_source_type() -> None:
    response = {
        "results": [
            {
                "child_id": "chunk-1",
                "rerank_score": 0.6,
                "metadata": {
                    "beir_corpus_id": "d1",
                    "source_path": "fiqa-107045.txt",
                },
                "score_signals": {"semantic_score": 0.8},
            }
        ]
    }

    traces = extract_result_traces(
        response,
        source_role="semantic_llm",
        source_stage="query_time_rag_fusion",
        rag_fusion=True,
    )

    assert traces[0]["source_type"] == "semantic_fusion"
    assert traces[0]["source_type_evidence"] == "runner_context"
    assert "source_type_metadata.source_path" not in traces[0]["trace_gap"]


def test_extract_result_traces_infers_semantic_fusion_from_runner_context() -> None:
    response = {
        "results": [
            {
                "child_id": "chunk-1",
                "rerank_score": 0.6,
                "metadata": {"beir_corpus_id": "d1"},
                "score_signals": {"semantic_score": 0.8},
            }
        ]
    }

    traces = extract_result_traces(
        response,
        source_role="semantic_llm",
        source_stage="query_time_rag_fusion",
        rag_fusion=True,
    )

    assert traces[0]["source_type"] == "semantic_fusion"
    assert traces[0]["source_type_evidence"] == "runner_context"
    assert "source_type_runner_context" in traces[0]["trace_gap"]


def test_extract_result_traces_marks_all_llm_as_hybrid_when_scores_overlap() -> None:
    response = {
        "results": [
            {
                "child_id": "chunk-1",
                "rerank_score": 0.6,
                "metadata": {"beir_corpus_id": "d1"},
                "score_signals": {"semantic_score": 0.8, "enriched_score": 0.4},
            }
        ]
    }

    traces = extract_result_traces(
        response,
        source_role="doc_summary_llm,enricher_llm,semantic_llm",
        source_stage="query_time_rag_fusion+doc_summary_read+ingest_enrichment_read",
        rag_fusion=True,
    )

    assert traces[0]["source_type"] == "hybrid"
    assert traces[0]["source_type_evidence"] == "runner_context"


def test_extract_result_traces_assigns_dedup_rank_once_per_doc() -> None:
    response = {
        "results": [
            {"child_id": "c1", "metadata": {"beir_corpus_id": "d1"}, "score": 0.9},
            {"child_id": "c2", "metadata": {"beir_corpus_id": "d1"}, "score": 0.8},
            {"child_id": "c3", "metadata": {"beir_corpus_id": "d2"}, "score": 0.7},
        ]
    }

    traces = extract_result_traces(response)

    assert [trace["rank"] for trace in traces] == [1, 2, 3]
    assert [trace["dedup_rank"] for trace in traces] == [1, None, 2]
    assert [trace["score_order_consistent"] for trace in traces] == [True, True, True]


def test_extract_result_traces_preserves_zero_final_score() -> None:
    response = {
        "results": [
            {
                "child_id": "c1",
                "metadata": {"beir_corpus_id": "d1"},
                "final_score": 0.0,
                "score": 0.7,
            }
        ]
    }

    traces = extract_result_traces(response)

    assert traces[0]["final_score"] == 0.0


def test_parse_query_latency_metrics_sums_relevant_openmetrics_series() -> None:
    text = """
# TYPE cortrix_scatter_requests_total counter
cortrix_scatter_requests_total{reason="single",category="permanent"} 7
cortrix_scatter_requests_total{reason="multi",category="permanent"} 3
# TYPE cortrix_scatter_duration_seconds histogram
cortrix_scatter_duration_seconds_bucket{namespace_count_bucket="1",le="1"} 9
cortrix_scatter_duration_seconds_sum{namespace_count_bucket="1"} 12.5
cortrix_scatter_duration_seconds_count{namespace_count_bucket="1"} 10
# TYPE cortrix_reranker_score_duration_seconds histogram
cortrix_reranker_score_duration_seconds_sum 4.25
cortrix_reranker_score_duration_seconds_count 10
# TYPE cortrix_rag_fusion_llm_latency_seconds histogram
cortrix_rag_fusion_llm_latency_seconds_sum{model="deepseek"} 2.5
cortrix_rag_fusion_llm_latency_seconds_count{model="deepseek"} 1
unrelated_metric_total 999
"""

    values = parse_query_latency_metrics(text)

    assert values["cortrix_scatter_requests_total"] == 10
    assert values["cortrix_scatter_duration_seconds_sum"] == 12.5
    assert values["cortrix_scatter_duration_seconds_count"] == 10
    assert values["cortrix_reranker_score_duration_seconds_sum"] == 4.25
    assert values["cortrix_reranker_score_duration_seconds_count"] == 10
    assert values["cortrix_rag_fusion_llm_latency_seconds_sum"] == 2.5
    assert values["cortrix_rag_fusion_llm_latency_seconds_count"] == 1
    assert "unrelated_metric_total" not in values


def test_diff_query_latency_metric_values_derives_query_stage_fields() -> None:
    before = {
        "cortrix_scatter_requests_total": 10,
        "cortrix_scatter_duration_seconds_sum": 12.0,
        "cortrix_scatter_duration_seconds_count": 10,
        "cortrix_reranker_score_duration_seconds_sum": 4.0,
        "cortrix_reranker_score_duration_seconds_count": 10,
        "cortrix_rag_fusion_llm_latency_seconds_sum": 1.0,
        "cortrix_rag_fusion_llm_latency_seconds_count": 1,
        "cortrix_rag_fusion_variant_count_sum": 3,
        "cortrix_rag_fusion_rrf_fusion_duration_seconds_sum": 0.01,
    }
    after = {
        "cortrix_scatter_requests_total": 11,
        "cortrix_scatter_duration_seconds_sum": 13.5,
        "cortrix_scatter_duration_seconds_count": 11,
        "cortrix_reranker_score_duration_seconds_sum": 4.8,
        "cortrix_reranker_score_duration_seconds_count": 11,
        "cortrix_rag_fusion_llm_latency_seconds_sum": 2.25,
        "cortrix_rag_fusion_llm_latency_seconds_count": 2,
        "cortrix_rag_fusion_variant_count_sum": 6,
        "cortrix_rag_fusion_rrf_fusion_duration_seconds_sum": 0.015,
    }

    delta = diff_query_latency_metric_values(before, after)
    derived = delta["derived"]

    assert derived["retrieval_calls"] == 1
    assert derived["retrieval_wall_seconds"] == 1.5
    assert derived["retrieval_duration_observations"] == 1
    assert derived["rerank_calls"] == 1
    assert math.isclose(derived["rerank_wall_seconds"], 0.8)
    assert math.isclose(derived["rerank_wall_ms"], 800.0)
    assert derived["llm_call_wall_time_seconds"] == 1.25
    assert derived["llm_call_count"] == 1
    assert derived["query_variants"] == 3
    assert math.isclose(derived["rrf_fusion_seconds"], 0.005)
