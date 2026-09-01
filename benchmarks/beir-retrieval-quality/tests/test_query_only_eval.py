"""Targeted tests for query_only_eval.

The module had no coverage before, while carrying the parts a published result
depends on: which queries get scored, how the request addresses namespaces, what
the timeout is, and how latency and failures are counted. Each test below pins
one of those so a later edit cannot move a published number silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER_DIR))

import beir_loader  # noqa: E402
import query_only_eval as qoe  # noqa: E402
from cortrix_client import CortrixClientError  # noqa: E402
from profiles import PROFILES  # noqa: E402


class _RecordingProfile:
    """Minimal stand-in for a BenchmarkProfile that records what it was asked for."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def query_body(self, query: str, namespace: str, top_k: int) -> dict[str, object]:
        self.calls.append({"query": query, "namespace": namespace, "top_k": top_k})
        return {"query": query, "namespace": namespace, "top_k": top_k, "rerank": False}


# --------------------------------------------------------------------------
# namespace parsing and request construction
# --------------------------------------------------------------------------

def test_single_namespace_parses_to_one_element() -> None:
    assert qoe.parse_namespaces("only-ns") == ["only-ns"]


def test_comma_separated_namespaces_keep_their_order() -> None:
    assert qoe.parse_namespaces("c,a,b") == ["c", "a", "b"]


def test_namespace_parsing_drops_empty_segments_and_trims() -> None:
    # A trailing comma or a shell-wrapped list must not produce an empty or
    # space-padded namespace, which the server would treat as a distinct name.
    assert qoe.parse_namespaces("a, b ,,c,") == ["a", "b", "c"]


def test_body_always_sends_namespaces_as_a_list_even_for_one() -> None:
    # Every published measurement was produced with the list form, including the
    # single-namespace rows. Sending a scalar for one namespace would take a
    # different server path and stop the two being comparable.
    profile = _RecordingProfile()
    body = qoe.build_query_body(profile, "q", ["solo"], top_k=50)
    assert body["namespaces"] == ["solo"]
    assert isinstance(body["namespaces"], list)


def test_body_carries_every_namespace_in_order() -> None:
    profile = _RecordingProfile()
    body = qoe.build_query_body(profile, "q", ["s1", "s2", "s3"], top_k=50)
    assert body["namespaces"] == ["s1", "s2", "s3"]


def test_body_preserves_the_profile_retrieval_knobs() -> None:
    # The profile owns retrieval settings; namespace handling must not drop them.
    profile = _RecordingProfile()
    body = qoe.build_query_body(profile, "hello", ["a", "b"], top_k=50)
    assert body["query"] == "hello"
    assert body["top_k"] == 50
    assert body["rerank"] is False
    assert profile.calls[0]["top_k"] == 50


def test_body_does_not_mutate_the_profile_result() -> None:
    # build_query_body copies before adding namespaces, so a profile returning a
    # shared dict cannot accumulate state across queries.
    class _SharedDictProfile:
        def __init__(self) -> None:
            self.shared: dict[str, object] = {"query": "", "top_k": 0}

        def query_body(self, query: str, namespace: str, top_k: int) -> dict[str, object]:
            return self.shared

    profile = _SharedDictProfile()
    qoe.build_query_body(profile, "q", ["a", "b"], top_k=50)
    assert "namespaces" not in profile.shared


# --------------------------------------------------------------------------
# query subset determinism
# --------------------------------------------------------------------------

def test_query_subset_is_sorted_not_insertion_ordered() -> None:
    qrels = {"q3": {"d": 1.0}, "q1": {"d": 1.0}, "q2": {"d": 1.0}}
    assert qoe.select_query_ids(qrels, max_queries=0) == ["q1", "q2", "q3"]


def test_query_subset_cap_takes_the_first_ids_after_sorting() -> None:
    # Sorting before truncating is what makes --max-queries reproducible; taking
    # two ids from dict order would give a different pair on a different run.
    qrels = {"q3": {"d": 1.0}, "q1": {"d": 1.0}, "q2": {"d": 1.0}}
    assert qoe.select_query_ids(qrels, max_queries=2) == ["q1", "q2"]


def test_query_subset_excludes_ids_with_no_relevance_judgement() -> None:
    qrels = {"q1": {"d": 1.0}, "q2": {}, "q3": {"d": 1.0}}
    assert qoe.select_query_ids(qrels, max_queries=0) == ["q1", "q3"]


def test_query_subset_cap_of_zero_means_no_cap() -> None:
    qrels = {f"q{i}": {"d": 1.0} for i in range(5)}
    assert len(qoe.select_query_ids(qrels, max_queries=0)) == 5


def test_query_subset_digest_is_stable_and_order_sensitive() -> None:
    # The digest is published so a reader can prove which queries were scored.
    # It must not change run to run, and it must change if the set changes.
    assert qoe.query_subset_digest(["a", "b"]) == qoe.query_subset_digest(["a", "b"])
    assert qoe.query_subset_digest(["a", "b"]) != qoe.query_subset_digest(["b", "a"])
    assert qoe.query_subset_digest(["a", "b"]) != qoe.query_subset_digest(["a", "b", "c"])


def test_query_subset_digest_is_a_sha256_hex_string() -> None:
    digest = qoe.query_subset_digest(["q1", "q2"])
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# --------------------------------------------------------------------------
# document id collapse
# --------------------------------------------------------------------------

def _hits(*doc_ids: str) -> dict[str, object]:
    return {"results": [{"doc_id": d, "score": 1.0} for d in doc_ids]}


def test_duplicate_documents_collapse_to_their_best_rank() -> None:
    # A document retrieved as several chunks counts once, at its first position.
    assert qoe.dedupe_doc_ids(_hits("a", "b", "a", "c"), top_k=10) == ["a", "b", "c"]


def test_collapse_truncates_after_deduplicating_not_before() -> None:
    # Truncating first would let duplicates eat the top_k budget and lose real
    # documents, which would depress recall without any retrieval change.
    assert qoe.dedupe_doc_ids(_hits("a", "a", "a", "b"), top_k=2) == ["a", "b"]


def test_collapse_on_an_empty_response_returns_no_documents() -> None:
    assert qoe.dedupe_doc_ids({"results": []}, top_k=10) == []


# --------------------------------------------------------------------------
# latency aggregation
# --------------------------------------------------------------------------

def test_latency_summary_reports_every_published_statistic() -> None:
    stats = qoe.summarise_latency([10.0, 20.0, 30.0])
    assert set(stats) == {"mean", "p50", "p95", "min", "max", "n"}


def test_latency_summary_values_are_exact_on_a_known_sample() -> None:
    stats = qoe.summarise_latency([50.0, 10.0, 30.0, 20.0, 40.0])
    assert stats["mean"] == 30.0
    assert stats["p50"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["n"] == 5.0


def test_latency_summary_is_order_independent() -> None:
    ascending = qoe.summarise_latency([1.0, 2.0, 3.0, 4.0])
    shuffled = qoe.summarise_latency([3.0, 1.0, 4.0, 2.0])
    assert ascending == shuffled


def test_latency_p95_stays_inside_the_sample() -> None:
    # Nearest-rank on a small sample must not index past the end.
    stats = qoe.summarise_latency([1.0, 2.0])
    assert stats["p95"] == 2.0


# --- nearest rank, at the boundaries where the obvious spelling is wrong ------
#
# `ordered[int(p * n)]` and nearest rank agree unless p*n is an integer, where
# the former returns the next observation up. Every sample size below is one
# where they disagree, which is why they are the sizes worth pinning: at n=300,
# 648 and 2000 -- the real measured cell sizes -- p50 or p95 or both land on an
# integer rank.

def test_nearest_rank_is_the_ceil_rank_observation() -> None:
    import math

    ordered = [float(i) for i in range(1, 1001)]
    for p in (0.0001, 0.25, 0.5, 0.9, 0.95, 0.99, 1.0):
        expected = ordered[math.ceil(p * len(ordered)) - 1]
        assert qoe.nearest_rank(ordered, p) == expected


def test_p95_on_a_two_thousand_sample_is_observation_1900_not_1901() -> None:
    # The size of the Quora cells. 0.95 * 2000 = 1900 exactly, so the naive
    # index returns 1901 and every published p95 at this size would be one
    # observation too high.
    ordered = [float(i) for i in range(1, 2001)]
    assert qoe.nearest_rank(ordered, 0.95) == 1900.0


def test_p50_on_an_even_sample_is_the_lower_middle_observation() -> None:
    ordered = [float(i) for i in range(1, 2001)]
    assert qoe.nearest_rank(ordered, 0.50) == 1000.0


def test_p50_and_p95_on_a_three_hundred_sample() -> None:
    # The size of the SciFact cells: both percentiles land on integer ranks.
    ordered = [float(i) for i in range(1, 301)]
    assert qoe.nearest_rank(ordered, 0.50) == 150.0
    assert qoe.nearest_rank(ordered, 0.95) == 285.0


def test_p50_on_an_odd_sample_is_the_true_middle() -> None:
    ordered = [float(i) for i in range(1, 6)]
    assert qoe.nearest_rank(ordered, 0.50) == 3.0


def test_nearest_rank_clamps_to_the_ends() -> None:
    ordered = [10.0, 20.0, 30.0]
    assert qoe.nearest_rank(ordered, 0.0) == 10.0
    assert qoe.nearest_rank(ordered, 1.0) == 30.0


def test_nearest_rank_refuses_an_empty_sample() -> None:
    # Returning 0.0 would read as a measured latency of zero.
    import pytest

    with pytest.raises(ValueError):
        qoe.nearest_rank([], 0.5)


def test_summarised_percentiles_use_nearest_rank_at_a_real_cell_size() -> None:
    # The end-to-end check: the published statistic, not just the helper.
    stats = qoe.summarise_latency([float(i) for i in range(1, 649)])
    assert stats["p50"] == 324.0
    assert stats["p95"] == 616.0


def test_latency_summary_of_nothing_is_empty_rather_than_zero() -> None:
    # Reporting mean=0 for a run that measured nothing would read as a real
    # measurement of zero latency.
    assert qoe.summarise_latency([]) == {}


# --------------------------------------------------------------------------
# argument surface, including backward compatibility
# --------------------------------------------------------------------------

def _parse(argv: list[str]):
    """Parse with the parser the script actually uses.

    Deliberately not a rebuilt equivalent: a copied parser keeps passing while
    the production defaults or wiring drift away from it, which is exactly the
    drift these tests exist to catch.
    """
    import contextlib
    import io

    with contextlib.redirect_stderr(io.StringIO()):
        return qoe.build_parser().parse_args(argv)


def test_the_tests_exercise_the_production_parser() -> None:
    # Pins the seam itself. If build_parser stops being what main() calls, the
    # argument tests below stop meaning anything, and nothing else would say so.
    import inspect

    source = inspect.getsource(qoe.main)
    assert "build_parser()" in source


def test_the_parser_exposes_exactly_the_documented_options() -> None:
    # An option added to production without a test here would otherwise go
    # unnoticed; an option removed would break callers silently.
    options = {
        action.dest
        for action in qoe.build_parser()._actions
        if action.dest != "help"
    }
    assert options == {
        "namespace", "dataset", "split", "max_queries", "base_url",
        "top_k", "profile", "timeout_seconds", "work_dir",
    }


def test_timeout_default_is_the_long_arm_value() -> None:
    # The client default of 30s cuts off CPU rerank and LLM arms mid-request.
    assert qoe.DEFAULT_TIMEOUT_SECONDS == 300.0
    assert _parse(["ns"]).timeout_seconds == 300.0


def test_timeout_is_overridable_and_parsed_as_a_float() -> None:
    assert _parse(["ns", "--timeout-seconds", "12.5"]).timeout_seconds == 12.5


def test_existing_single_namespace_invocation_still_parses_unchanged() -> None:
    # The exact shape callers used before multi-namespace support existed.
    args = _parse(["my-ns", "--dataset", "scifact", "--max-queries", "300",
                   "--top-k", "10", "--profile", "vector_full"])
    assert args.namespace == "my-ns"
    assert args.dataset == "scifact"
    assert args.max_queries == 300
    assert args.top_k == 10
    assert args.profile == "vector_full"


def test_split_defaults_to_test() -> None:
    assert _parse(["ns"]).split == "test"


def test_quora_is_selectable_as_a_dataset() -> None:
    assert "quora" in beir_loader.DATASETS
    assert _parse(["ns", "--dataset", "quora"]).dataset == "quora"


def test_quora_dataset_spec_points_at_the_public_beir_archive() -> None:
    spec = beir_loader.DATASETS["quora"]
    assert spec.name == "quora"
    assert spec.url.endswith("/quora.zip")


def test_every_named_profile_is_selectable() -> None:
    # --profile is constrained to the registry, so a typo fails at parse time
    # rather than producing a run against the wrong retrieval settings.
    for name in PROFILES:
        assert _parse(["ns", "--profile", name]).profile == name


# --------------------------------------------------------------------------
# failure accounting
# --------------------------------------------------------------------------

class _FailingClient:
    """Client whose query() raises for the query texts it was told to fail."""

    def __init__(self, fail_for: set[str]) -> None:
        self.fail_for = fail_for
        self.calls = 0

    def query(self, body):  # noqa: ANN001 - duck-typed like CortrixClient
        self.calls += 1
        if body["query"] in self.fail_for:
            raise CortrixClientError("boom")
        return _hits("d1"), 5.0


def _score_with(client, qids: list[str], texts: dict[str, str]):
    """Run the scoring loop of main() over an injected client.

    Kept in the test rather than exported from the module so the production code
    does not grow a seam that exists only for tests.
    """
    profile = _RecordingProfile()
    runs: dict[str, list[str]] = {}
    latencies: list[float] = []
    missing = 0
    failed = 0
    for qid in qids:
        qtext = texts.get(qid)
        if not qtext:
            missing += 1
            continue
        body = qoe.build_query_body(profile, qtext, ["ns"], top_k=50)
        try:
            response, latency = client.query(body)
        except CortrixClientError:
            failed += 1
            continue
        latencies.append(latency)
        runs[qid] = qoe.dedupe_doc_ids(response, top_k=10)
    return runs, latencies, missing, failed


def test_a_failed_request_is_counted_and_not_scored_as_an_empty_result() -> None:
    # Scoring a failed request as an empty hit list would depress the metric and
    # make a broken run look like a bad one.
    client = _FailingClient(fail_for={"text-2"})
    runs, latencies, missing, failed = _score_with(
        client, ["q1", "q2", "q3"], {"q1": "text-1", "q2": "text-2", "q3": "text-3"}
    )
    assert failed == 1
    assert "q2" not in runs
    assert sorted(runs) == ["q1", "q3"]


def test_a_failed_request_contributes_no_latency_sample() -> None:
    client = _FailingClient(fail_for={"text-2"})
    _, latencies, _, _ = _score_with(
        client, ["q1", "q2", "q3"], {"q1": "text-1", "q2": "text-2", "q3": "text-3"}
    )
    assert len(latencies) == 2


def test_missing_query_text_is_counted_separately_from_a_failure() -> None:
    # The two mean different things: no text is a dataset problem, a failure is a
    # server problem, and a reader of a published result needs to tell them apart.
    client = _FailingClient(fail_for={"text-3"})
    runs, _, missing, failed = _score_with(
        client, ["q1", "q2", "q3"], {"q1": "text-1", "q3": "text-3"}
    )
    assert missing == 1
    assert failed == 1
    assert sorted(runs) == ["q1"]


def test_a_clean_run_reports_no_failures_and_scores_everything() -> None:
    # The counterpart to the failure tests: without it they would also pass on a
    # build where every request failed.
    client = _FailingClient(fail_for=set())
    runs, latencies, missing, failed = _score_with(
        client, ["q1", "q2"], {"q1": "text-1", "q2": "text-2"}
    )
    assert (missing, failed) == (0, 0)
    assert sorted(runs) == ["q1", "q2"]
    assert len(latencies) == 2
