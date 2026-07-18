from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class CortrixClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class JsonResponse:
    status: int
    body: Mapping[str, object]
    response_bytes: int
    latency_ms: float


QUERY_LATENCY_METRIC_FAMILIES = {
    "cortrix_scatter_requests_total",
    "cortrix_scatter_duration_seconds",
    "cortrix_reranker_score_duration_seconds",
    "cortrix_rag_fusion_invocation_total",
    "cortrix_rag_fusion_variant_count",
    "cortrix_rag_fusion_llm_latency_seconds",
    "cortrix_rag_fusion_degraded_total",
    "cortrix_rag_fusion_token_total",
    "cortrix_rag_fusion_rrf_fusion_duration_seconds",
}


def parse_query_latency_metrics(text: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        raw_name = parts[0].split("{", 1)[0]
        if raw_name.endswith("_bucket"):
            continue
        family = metric_family(raw_name)
        if family not in QUERY_LATENCY_METRIC_FAMILIES:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        values[raw_name] = values.get(raw_name, 0.0) + value
    return values


def metric_family(name: str) -> str:
    for suffix in ["_sum", "_count"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def diff_query_latency_metric_values(
    before_values: Mapping[str, object],
    after_values: Mapping[str, object],
) -> Dict[str, Any]:
    deltas: Dict[str, float] = {}
    for key in sorted(set(before_values) | set(after_values)):
        before_value = before_values.get(key, 0.0)
        after_value = after_values.get(key, 0.0)
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            deltas[key] = float(after_value) - float(before_value)
    derived = derive_query_latency_fields(deltas)
    return {"status": "ok", "values": deltas, "derived": derived}


def collect_query_latency_metrics(metrics_url: str, timeout_seconds: float = 5.0) -> Dict[str, Any]:
    if not metrics_url:
        return {"status": "not_observable", "reason": "metrics_url_not_configured"}
    try:
        request = urllib.request.Request(metrics_url, method="GET", headers={"Accept": "text/plain"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"status": "not_observable", "error": repr(exc), "url": metrics_url}
    return {"status": "ok", "url": metrics_url, "values": parse_query_latency_metrics(text)}


def diff_query_latency_metric_snapshots(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> Dict[str, Any]:
    if before.get("status") != "ok" or after.get("status") != "ok":
        return {"status": "not_observable", "before": dict(before), "after": dict(after)}
    before_values = before.get("values") if isinstance(before.get("values"), dict) else {}
    after_values = after.get("values") if isinstance(after.get("values"), dict) else {}
    return diff_query_latency_metric_values(before_values, after_values)


def derive_query_latency_fields(deltas: Mapping[str, float]) -> Dict[str, Optional[float]]:
    retrieval_calls = deltas.get("cortrix_scatter_requests_total")
    retrieval_wall_seconds = deltas.get("cortrix_scatter_duration_seconds_sum")
    retrieval_duration_observations = deltas.get("cortrix_scatter_duration_seconds_count")
    rerank_calls = deltas.get("cortrix_reranker_score_duration_seconds_count")
    rerank_wall_seconds = deltas.get("cortrix_reranker_score_duration_seconds_sum")
    query_variants = (
        deltas["cortrix_rag_fusion_variant_count_sum"]
        if "cortrix_rag_fusion_variant_count_sum" in deltas
        else deltas.get("cortrix_rag_fusion_variant_count")
    )
    return {
        "retrieval_calls": retrieval_calls,
        "retrieval_wall_seconds": retrieval_wall_seconds,
        "retrieval_duration_observations": retrieval_duration_observations,
        "rerank_calls": rerank_calls,
        "rerank_wall_seconds": rerank_wall_seconds,
        "rerank_wall_ms": rerank_wall_seconds * 1000.0 if isinstance(rerank_wall_seconds, (int, float)) else None,
        "llm_call_wall_time_seconds": deltas.get("cortrix_rag_fusion_llm_latency_seconds_sum"),
        "llm_call_count": deltas.get("cortrix_rag_fusion_llm_latency_seconds_count"),
        "query_variants": query_variants,
        "rrf_fusion_seconds": deltas.get("cortrix_rag_fusion_rrf_fusion_duration_seconds_sum"),
        "rrf_fusion_count": deltas.get("cortrix_rag_fusion_rrf_fusion_duration_seconds_count"),
        "degraded_total": deltas.get("cortrix_rag_fusion_degraded_total"),
        "token_total": deltas.get("cortrix_rag_fusion_token_total"),
    }


class CortrixClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> Mapping[str, object]:
        return self.request_json("GET", "/api/v1/health")

    def readiness(self) -> Mapping[str, object]:
        return self.request_json("GET", "/api/v1/system/health/ready")

    def create_namespace(self, name: str) -> Mapping[str, object]:
        return self.request_json("POST", "/api/v1/namespaces", {"name": name}, ok_statuses={200, 201, 409})

    def submit_documents_batch(
        self,
        namespace: str,
        documents: Iterable[Mapping[str, object]],
        on_duplicate: str = "skip",
    ) -> Mapping[str, object]:
        return self.submit_documents_batch_with_meta(namespace, documents, on_duplicate=on_duplicate).body

    def submit_documents_batch_with_meta(
        self,
        namespace: str,
        documents: Iterable[Mapping[str, object]],
        on_duplicate: str = "skip",
    ) -> JsonResponse:
        payload = {
            "namespace": namespace,
            "documents": list(documents),
            "options": {"async": True, "on_duplicate": on_duplicate},
        }
        return self.request_json_with_meta("POST", "/api/v1/documents/batch", payload, ok_statuses={200, 201, 202})

    def task_progress(self, task_id: str) -> Mapping[str, object]:
        return self.request_json("GET", f"/api/v1/documents/tasks/{task_id}/progress")

    def poll_tasks(self, task_ids: List[str], timeout_seconds: float, interval_seconds: float = 1.0) -> Dict[str, Mapping[str, object]]:
        deadline = time.time() + timeout_seconds
        pending = set(task_ids)
        final: Dict[str, Mapping[str, object]] = {}
        while pending:
            if time.time() > deadline:
                raise CortrixClientError(f"timed out waiting for {len(pending)} tasks")
            for task_id in list(pending):
                progress = self.task_progress(task_id)
                status = str(progress.get("status", "")).lower()
                if status in {"completed", "complete", "failed", "error", "cancelled"}:
                    final[task_id] = progress
                    pending.remove(task_id)
            if pending:
                time.sleep(interval_seconds)
        return final

    def query(self, body: Mapping[str, object]) -> Tuple[Mapping[str, object], float]:
        start = time.perf_counter()
        response = self.request_json("POST", "/api/v1/query", body)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return response, elapsed_ms

    def request_json(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, object]] = None,
        ok_statuses: Optional[set[int]] = None,
    ) -> Mapping[str, object]:
        return self.request_json_with_meta(method, path, body, ok_statuses).body

    def request_json_with_meta(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, object]] = None,
        ok_statuses: Optional[set[int]] = None,
    ) -> JsonResponse:
        ok_statuses = ok_statuses or {200}
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers=headers,
        )
        try:
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
                status = response.status
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            message = payload.decode("utf-8", errors="replace")
            raise CortrixClientError(f"{method} {path} failed with HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise CortrixClientError(f"{method} {path} failed: {exc}") from exc
        if status not in ok_statuses:
            raise CortrixClientError(f"{method} {path} returned HTTP {status}: {payload[:500]!r}")
        if not payload:
            return JsonResponse(status=status, body={}, response_bytes=0, latency_ms=elapsed_ms)
        try:
            item = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CortrixClientError(f"{method} {path} returned invalid JSON: {payload[:500]!r}") from exc
        body_item = item if isinstance(item, dict) else {"data": item}
        return JsonResponse(status=status, body=body_item, response_bytes=len(payload), latency_ms=elapsed_ms)


def extract_task_ids(batch_response: Mapping[str, object]) -> List[str]:
    task_ids: List[str] = []
    for key in ["task_ids", "tasks"]:
        value = batch_response.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    task_ids.append(item)
                elif isinstance(item, dict) and isinstance(item.get("task_id"), str):
                    task_ids.append(str(item["task_id"]))
    results = batch_response.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                task_id = item.get("task_id")
                if isinstance(task_id, str):
                    task_ids.append(task_id)
    single = batch_response.get("task_id")
    if isinstance(single, str):
        task_ids.append(single)
    return sorted(set(task_ids))


def extract_result_doc_ids(
    response: Mapping[str, object],
    child_id_aliases: Optional[Mapping[str, str]] = None,
) -> List[str]:
    doc_ids: List[str] = []
    for trace in extract_result_traces(response, child_id_aliases=child_id_aliases):
        doc_id = trace.get("returned_doc_id")
        if isinstance(doc_id, str) and doc_id:
            doc_ids.append(doc_id)
    return doc_ids


def extract_result_traces(
    response: Mapping[str, object],
    query_id: Optional[str] = None,
    qrels_relevant: Optional[Sequence[str]] = None,
    source_role: Optional[str] = None,
    source_stage: Optional[str] = None,
    rag_fusion: Optional[bool] = None,
    child_id_aliases: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, object]]:
    candidates = response.get("results")
    if candidates is None:
        candidates = response.get("data")
    if isinstance(candidates, dict):
        candidates = candidates.get("results")
    if not isinstance(candidates, list):
        return []
    relevant = set(qrels_relevant or [])
    seen_docs: set[str] = set()
    traces: List[Dict[str, object]] = []
    for rank, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue
        beir_corpus_id = _beir_corpus_id_from_item(item, child_id_aliases=child_id_aliases)
        returned_doc_id = beir_corpus_id or _internal_doc_id_from_item(item)
        dedup_rank: Optional[int] = None
        if returned_doc_id and returned_doc_id not in seen_docs:
            seen_docs.add(returned_doc_id)
            dedup_rank = len(seen_docs)
        trace_gap: List[str] = []
        if not beir_corpus_id:
            trace_gap.append("missing_beir_corpus_id")
        signals = item.get("score_signals")
        signals_map = signals if isinstance(signals, dict) else {}
        source_type, source_type_evidence = _source_type_from_item(
            item,
            source_role=source_role,
            source_stage=source_stage,
            rag_fusion=rag_fusion,
            score_signals=signals_map,
        )
        if source_type == "unknown":
            trace_gap.append("missing_explicit_source_type")
        elif source_type_evidence != "explicit":
            trace_gap.append(f"source_type_{source_type_evidence}")
        if _as_float(item.get("rerank_score")) is None:
            trace_gap.append("missing_rerank_score")
        trace = {
            "query_id": query_id,
            "rank": rank,
            "dedup_rank": dedup_rank,
            "returned_doc_id": returned_doc_id,
            "beir_corpus_id": beir_corpus_id,
            "internal_doc_id": _top_level_doc_id_from_item(item),
            "child_id": _string_or_none(item.get("child_id")),
            "parent_id": _string_or_none(item.get("parent_id")),
            "source_type": source_type,
            "source_role": source_role,
            "source_stage": source_stage,
            "source_type_evidence": source_type_evidence,
            "score": _as_float(item.get("score")),
            "vector_score": _as_float(item.get("vector_score")),
            "rerank_score": _as_float(item.get("rerank_score")),
            "semantic_score": _as_float(signals_map.get("semantic_score")),
            "enriched_score": _as_float(signals_map.get("enriched_score")),
            "final_score": _first_float(item.get("final_score"), item.get("score")),
            "score_order_consistent": None,
            "id_mapping_decision": "mapped_to_beir" if beir_corpus_id else ("internal_only" if returned_doc_id else "missing"),
            "is_qrels_relevant": bool(returned_doc_id and returned_doc_id in relevant),
            "trace_gap": trace_gap,
        }
        traces.append(trace)
    _annotate_score_order_consistency(traces)
    return traces


def _doc_id_from_item(item: Mapping[str, object]) -> str:
    doc_id = _beir_corpus_id_from_item(item) or _internal_doc_id_from_item(item)
    return doc_id or ""


def _beir_corpus_id_from_item(
    item: Mapping[str, object],
    child_id_aliases: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in [
            "beir_corpus_id",
            "beir_doc_id",
            "corpus_id",
            "corpus_doc_id",
            "original_doc_id",
            "original_id",
            "doc_id",
            "id",
        ]:
            value = metadata.get(key)
            stable_id = _stable_corpus_id(value)
            if stable_id:
                return stable_id
        metadata_json = metadata.get("metadata_json")
        if isinstance(metadata_json, str) and metadata_json:
            doc_id = _doc_id_from_metadata_json(metadata_json)
            if doc_id:
                return doc_id
    aliases = child_id_aliases or {}
    for key in ["child_id", "doc_id", "document_id", "parent_id"]:
        value = item.get(key)
        if isinstance(value, str) and value:
            stable_id = _stable_corpus_id(aliases.get(value))
            if stable_id:
                return stable_id
    return None


def _internal_doc_id_from_item(item: Mapping[str, object]) -> Optional[str]:
    for key in ["doc_id", "document_id", "parent_id", "child_id"]:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _top_level_doc_id_from_item(item: Mapping[str, object]) -> Optional[str]:
    for key in ["doc_id", "document_id"]:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _source_type_from_item(
    item: Mapping[str, object],
    source_role: Optional[str] = None,
    source_stage: Optional[str] = None,
    rag_fusion: Optional[bool] = None,
    score_signals: Optional[Mapping[str, object]] = None,
) -> Tuple[str, str]:
    for key in ["source_type", "source", "result_source"]:
        value = item.get(key)
        if isinstance(value, str) and value:
            return _normalize_source_type(value), "explicit"
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in ["source_type", "source", "result_source"]:
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return _normalize_source_type(value), "explicit"
        for key in ["via_path", "hybrid_source_path"]:
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return _normalize_source_type(value), f"metadata.{key}"
        if metadata.get("hybrid_paths") or metadata.get("hybrid_source_score"):
            return "hybrid", "metadata.hybrid"
        if metadata.get("doc_summary_match_score") is not None:
            return "doc_summary", "metadata.doc_summary"
    signals = score_signals or {}
    role_text = (source_role or "").lower()
    stage_text = (source_stage or "").lower()
    has_semantic_score = _as_float(signals.get("semantic_score")) is not None
    has_enriched_score = _as_float(signals.get("enriched_score")) is not None
    if "semantic_llm" in role_text and "enricher_llm" in role_text and has_semantic_score and has_enriched_score:
        return "hybrid", "runner_context"
    if rag_fusion or "semantic_llm" in role_text or "rag_fusion" in stage_text:
        if has_semantic_score:
            return "semantic_fusion", "runner_context"
    if "enricher_llm" in role_text and has_enriched_score:
        return "enricher", "runner_context"
    if _as_float(item.get("rerank_score")) is not None:
        return "rerank", "runner_context"
    return "unknown", "missing"


def _string_or_none(value: object) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _first_float(*values: object) -> Optional[float]:
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _normalize_source_type(value: str) -> str:
    lowered = value.strip().lower()
    if not lowered:
        return "unknown"
    if "doc_summary" in lowered or "summary" in lowered:
        return "doc_summary"
    if "hybrid" in lowered:
        return "hybrid"
    if "rag_fusion" in lowered or "semantic" in lowered:
        return "semantic_fusion"
    if "enrich" in lowered:
        return "enricher"
    if "rerank" in lowered:
        return "rerank"
    if lowered in {"chunk", "raw_chunk", "dense", "vector", "bm25", "fts", "keyword"}:
        return "raw_chunk"
    return lowered


_INTERNAL_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def _stable_corpus_id(value: object) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    if _looks_internal_id(value):
        return None
    return value


def _looks_internal_id(value: str) -> bool:
    return bool(_INTERNAL_ULID_RE.match(value))


def _annotate_score_order_consistency(traces: List[Dict[str, object]]) -> None:
    scored = [trace for trace in traces if isinstance(trace.get("final_score"), float)]
    if len(scored) < 2:
        return
    previous: Optional[float] = None
    for trace in traces:
        score = trace.get("final_score")
        if not isinstance(score, float):
            continue
        if previous is None:
            trace["score_order_consistent"] = True
        else:
            trace["score_order_consistent"] = score <= previous
        previous = score


def _doc_id_from_metadata_json(metadata_json: str) -> str:
    try:
        data = json.loads(metadata_json)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    for key in ["beir_corpus_id", "beir_doc_id", "corpus_id", "corpus_doc_id", "original_doc_id", "original_id", "doc_id", "id"]:
        value = data.get(key)
        stable_id = _stable_corpus_id(value)
        if stable_id:
            return stable_id
    filename = data.get("filename")
    if not isinstance(filename, str) or not filename:
        return ""
    name = filename.rsplit("/", 1)[-1]
    if name.endswith(".txt"):
        name = name[:-4]
    doc_id = _strip_known_dataset_prefix(name)
    if doc_id:
        return doc_id
    return name


def _strip_known_dataset_prefix(name: str) -> str:
    for dataset in sorted(
        {"scifact", "nfcorpus", "fiqa", "fiqa-mini-120", "fiqa-mini-600", "hotpotqa", "webis_touche2020", "webis-touche2020"},
        key=len,
        reverse=True,
    ):
        prefix = f"{dataset}-"
        if name.startswith(prefix):
            return name[len(prefix) :]
    return ""
