from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    query_fraction: float
    rerank: bool
    rag_fusion: bool
    intended_semantics: str
    public_boundary: str
    # Isolation-matrix knobs. All default to the legacy
    # off state so existing profiles keep sending the exact same query body.
    route: Optional[str] = None
    explain: bool = False
    llm_rerank: bool = False
    llm_rerank_top_n: Optional[int] = None
    # Tuned-probe knobs. None = key omitted from the body, so every
    # pre-existing profile keeps sending byte-identical queries.
    llm_rerank_max_doc_chars: Optional[int] = None
    llm_rerank_consensus_runs: Optional[int] = None
    # Extra rag_fusion_config body fields (for example, fusion-policy
    # knobs fusion_variant_weight / fusion_anchor_max, or candidate_multiplier).
    # Sent only when rag_fusion is on and supported by the target server.
    rag_fusion_config: Optional[Dict[str, object]] = None

    def query_body(self, query: str, namespace: str, top_k: int) -> Dict[str, object]:
        body: Dict[str, object] = {
            "query": query,
            "namespaces": [namespace],
            "top_k": top_k,
            "rerank": self.rerank,
            "rag_fusion": self.rag_fusion,
        }
        if self.route is not None:
            body["route"] = self.route
        if self.explain:
            body["explain"] = True
        if self.llm_rerank:
            body["llm_rerank"] = True
            lr_cfg: Dict[str, object] = {}
            if self.llm_rerank_top_n is not None:
                lr_cfg["top_n"] = self.llm_rerank_top_n
            if self.llm_rerank_max_doc_chars is not None:
                lr_cfg["max_doc_chars"] = self.llm_rerank_max_doc_chars
            if self.llm_rerank_consensus_runs is not None:
                lr_cfg["consensus_runs"] = self.llm_rerank_consensus_runs
            if lr_cfg:
                body["llm_rerank_config"] = lr_cfg
        if self.rag_fusion and self.rag_fusion_config:
            body["rag_fusion_config"] = dict(self.rag_fusion_config)
        return body

    def to_json(self) -> Dict[str, object]:
        return dict(self.__dict__)


PROFILES: Dict[str, BenchmarkProfile] = {
    "vector_full": BenchmarkProfile(
        name="vector_full",
        query_fraction=1.0,
        rerank=False,
        rag_fusion=False,
        intended_semantics=(
            "Full-corpus dense+BM25 recall baseline (no LLM enrichment). Ingests "
            "the entire dataset corpus (query_fraction=1.0) so recall@k reflects "
            "real retrieval quality rather than a sampled, structurally-inflated cell."
        ),
        public_boundary="local diagnostic",
    ),
    "vector_only": BenchmarkProfile(
        name="vector_only",
        query_fraction=0.05,
        rerank=False,
        rag_fusion=False,
        intended_semantics=(
            "Baseline sampled retrieval profile. It records the requested "
            "vector-only baseline, but does not patch Cortrix core "
            "routing if the API does not expose a hard vector-only switch."
        ),
        public_boundary=(
            "Use as a local diagnostic unless runtime evidence confirms "
            "the API-level routing exactly maps to vector-only execution."
        ),
    ),
    "bge_m3_rerank_full": BenchmarkProfile(
        name="bge_m3_rerank_full",
        query_fraction=1.0,
        rerank=True,
        rag_fusion=False,
        intended_semantics=(
            "Full-corpus, full-query FiQA profile for BGE-M3 embedding plus "
            "bge-reranker-v2-m3 reranking. It sends rerank=true through the "
            "public query API and keeps rag_fusion=false so no LLM path is "
            "intended to run."
        ),
        public_boundary=(
            "Local diagnostic; use only with a run manifest that "
            "confirms real BGE-M3, real bge-reranker-v2-m3, CUDA EP, and zero "
            "LLM calls"
        ),
    ),
    # ---- Isolation matrix. Every arm pins route=complex
    # (the complexity router silently drops the hype_question and
    # contextualized RRF paths for simple-classified queries) and requests
    # explain=true (per-result via_path plus rrf_path_counts land verbatim in
    # queries.jsonl for attribution). The recall regression gate compares arm
    # B against arm A only; arms C and D are diagnostic.
    "bge_m3_rerank_baseline_v2": BenchmarkProfile(
        name="bge_m3_rerank_baseline_v2",
        query_fraction=1.0,
        rerank=True,
        rag_fusion=False,
        route="complex",
        explain=True,
        intended_semantics=(
            "Isolation-matrix arm A (control). Same pinned query shape as the "
            "LLM arms (route=complex, explain=true) but no LLM feature is "
            "requested. Run it against a namespace ingested without LLM "
            "enrichment."
        ),
        public_boundary=(
            "Local diagnostic for arm-to-arm comparison; the gate "
            "compares arm B against this arm"
        ),
    ),
    "bge_m3_rerank_ingest_llm": BenchmarkProfile(
        name="bge_m3_rerank_ingest_llm",
        query_fraction=1.0,
        rerank=True,
        rag_fusion=False,
        route="complex",
        explain=True,
        intended_semantics=(
            "Isolation-matrix arm B (ingest-LLM only). Identical query shape "
            "to arm A; the only difference is the target namespace was "
            "ingested with the doc_summary/enricher/semantic LLM roles "
            "enabled, so the hype_question and contextualized RRF paths carry "
            "real artifacts. rag_fusion stays false so no query-time LLM "
            "runs. The recall regression gate is evaluated as this arm versus "
            "arm A."
        ),
        public_boundary="local diagnostic; comparison arm",
    ),
    "bge_m3_rerank_all_llm_v2": BenchmarkProfile(
        name="bge_m3_rerank_all_llm_v2",
        query_fraction=1.0,
        rerank=True,
        rag_fusion=True,
        route="complex",
        explain=True,
        intended_semantics=(
            "Isolation-matrix arm C (arm B plus query-time rag_fusion). "
            "Uses the same comparison shape but pins "
            "route=complex and with explain captured, so the rag_fusion "
            "increment over arm B is attributable."
        ),
        public_boundary="local diagnostic; no publication claim",
    ),
    "bge_m3_llm_listwise": BenchmarkProfile(
        name="bge_m3_llm_listwise",
        query_fraction=1.0,
        rerank=True,
        rag_fusion=False,
        route="complex",
        explain=True,
        llm_rerank=True,
        llm_rerank_top_n=30,
        intended_semantics=(
            "Isolation-matrix arm D (arm B plus LLM listwise rerank with "
            "llm_rerank_config.top_n=30). Measures the ordering-LLM "
            "contribution on top of ingest-LLM artifacts."
        ),
        public_boundary="local diagnostic; no publication claim",
    ),
    # ---- Tuned probes. Same pinned shape as the isolation matrix
    # (route=complex + explain, full corpus/queries); each varies exactly ONE
    # knob against arm D so the delta is attributable.
    "bge_m3_llm_listwise_ctx1500": BenchmarkProfile(
        name="bge_m3_llm_listwise_ctx1500",
        query_fraction=1.0,
        rerank=True,
        rag_fusion=False,
        route="complex",
        explain=True,
        llm_rerank=True,
        llm_rerank_top_n=30,
        llm_rerank_max_doc_chars=1500,
        intended_semantics=(
            "Arm D variant: listwise judge sees 1500 chars per passage instead "
            "of the built-in 600. FiQA answers are long forum posts; at 600 the "
            "LLM ranks many passages on their first paragraph only. Isolates "
            "the passage-truncation lever (top_n stays 30)."
        ),
        public_boundary="local diagnostic; no publication claim",
    ),
    "bge_m3_llm_listwise_deep100": BenchmarkProfile(
        name="bge_m3_llm_listwise_deep100",
        query_fraction=1.0,
        rerank=True,
        rag_fusion=False,
        route="complex",
        explain=True,
        llm_rerank=True,
        llm_rerank_top_n=100,
        llm_rerank_max_doc_chars=600,
        intended_semantics=(
            "Deep-window diagnostic that expands the listwise candidate "
            "window to 100 while keeping 600-character passages. It tests "
            "whether relevant documents below the first 50 candidates can "
            "be recovered when the target server supports the wider window."
        ),
        public_boundary="local diagnostic; no publication claim",
    ),
    "bge_m3_llm_listwise_t20": BenchmarkProfile(
        name="bge_m3_llm_listwise_t20",
        query_fraction=1.0,
        rerank=True,
        rag_fusion=False,
        route="complex",
        explain=True,
        llm_rerank=True,
        llm_rerank_top_n=20,
        intended_semantics=(
            "Arm D variant: llm_rerank_config.top_n=20 (single window, no "
            "sliding-window bubbling from CE ranks 20-30). Tests whether the "
            "smaller window avoids tail distractors admitted by a wider "
            "window. Truncation stays at the 600-character default."
        ),
        public_boundary="local diagnostic; no publication claim",
    ),
    "bge_m3_rf_pool_listwise": BenchmarkProfile(
        name="bge_m3_rf_pool_listwise",
        query_fraction=1.0,
        rerank=True,
        rag_fusion=True,
        route="complex",
        explain=True,
        llm_rerank=True,
        llm_rerank_top_n=30,
        llm_rerank_max_doc_chars=1500,
        rag_fusion_config={
            "fusion_variant_weight": 0.2,
            "fusion_anchor_max": 10,
            "candidate_multiplier": 2,
            "max_candidates": 200,
        },
        intended_semantics=(
            "Full Stack measurement profile: RAG Fusion widens the candidate "
            "pool with variant weight 0.2, the original head anchored to 10, "
            "and per-variant candidates multiplied by two. The listwise "
            "judge orders the widened pool. The target server must support "
            "the declared fusion-policy fields."
        ),
        public_boundary=(
            "Published measurement profile only when the exact server "
            "contract and run manifest match; otherwise local diagnostic."
        ),
    ),
    "full_stack": BenchmarkProfile(
        name="full_stack",
        query_fraction=0.02,
        rerank=True,
        rag_fusion=False,
        intended_semantics=(
            "Sampled full-stack profile that asks Cortrix to use the "
            "available retrieval stack through the public query API. In this "
            "sampled runner, rag_fusion is disabled unless a benchmark-specific "
            "runbook explicitly enables it."
        ),
        public_boundary=(
            "Use as a local diagnostic until artifact checks and runtime "
            "confirmation are complete. Do not present this local "
            "run as the final full-stack benchmark until the intended full-stack "
            "switch set is confirmed."
        ),
    ),
}
