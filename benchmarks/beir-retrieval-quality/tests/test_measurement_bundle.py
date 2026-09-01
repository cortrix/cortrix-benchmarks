"""Tests for the measurement bundle validator.

The validator is the thing standing between a miscounted or edited measurement
round and a published artifact that invites readers to verify it. A validator
that passes everything is worse than none, because it certifies. Each test below
breaks the shipped bundle in one specific way and asserts the validator says so.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import validate_measurement_bundle as vmb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE = REPO_ROOT / "results/published/beir-four-corpus-cpu-2026-08-v1"


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / BUNDLE.name
    shutil.copytree(BUNDLE, target)
    return target


def _rewrite(path: Path, mutate) -> None:
    """Edit a JSON file and refresh its checksum, so a test targets one failure.

    Without the refresh every mutation would also trip the checksum check and the
    test would pass for the wrong reason.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _refresh_checksums(path.parents[1] if path.parent.name == "scorecards" else path.parent)


def _refresh_checksums(bundle: Path) -> None:
    import hashlib

    lines = []
    for item in sorted(bundle.rglob("*")):
        if not item.is_file() or item.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        lines.append(f"{digest}  {item.relative_to(bundle)}")
    (bundle / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _errors_mentioning(errors: list[str], needle: str) -> list[str]:
    return [e for e in errors if needle in e]


# --------------------------------------------------------------------------
# the bundle as shipped
# --------------------------------------------------------------------------

def test_the_shipped_bundle_validates() -> None:
    # The counterpart to every failure test below: without this they would all
    # also pass on a validator that rejected everything.
    assert vmb.validate(BUNDLE) == []


def test_the_shipped_bundle_declares_the_number_of_cells_it_ships() -> None:
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    on_disk = sorted(p.name for p in (BUNDLE / "scorecards").glob("*.json"))
    assert manifest["cells"]["declared_count"] == len(on_disk)
    assert sorted(manifest["cells"]["scorecards"]) == on_disk


# --------------------------------------------------------------------------
# schema validation, in depth
#
# An earlier validator compared top-level key sets only. Every case below passed
# it. They are kept as a set because each exercises a different schema mechanism
# -- const, enum, pattern, numeric bound, type, and required at depth -- and a
# validator can lose any one of them independently.
# --------------------------------------------------------------------------

def test_an_invalid_schema_version_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "manifest.json", lambda d: d.__setitem__("schema_version", "9.9"))
    assert _errors_mentioning(vmb.validate(bundle), "schema_version")


def test_a_status_outside_its_enum_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "manifest.json", lambda d: d.__setitem__("status", "mostly-done"))
    assert _errors_mentioning(vmb.validate(bundle), "status")


def test_a_malformed_core_sha_is_rejected(tmp_path: Path) -> None:
    # A commit id that is not a commit id makes the whole provenance chain
    # unverifiable, which is the one thing a reader cannot work around.
    bundle = _copy(tmp_path)
    _rewrite(bundle / "manifest.json",
             lambda d: d["measured_build"].__setitem__("cortrix_core_sha", "not-a-sha"))
    assert _errors_mentioning(vmb.validate(bundle), "cortrix_core_sha")


def test_a_metric_outside_zero_to_one_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "scifact-dense-only.json",
             lambda d: d["metrics"].__setitem__("ndcg_at_10", 7.0))
    assert _errors_mentioning(vmb.validate(bundle), "ndcg_at_10")


def test_a_negative_metric_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "fiqa-dense-only.json",
             lambda d: d["metrics"].__setitem__("recall_at_10", -0.1))
    assert _errors_mentioning(vmb.validate(bundle), "recall_at_10")


def test_a_missing_nested_required_field_is_rejected(tmp_path: Path) -> None:
    # Depth is the point: hardware.cpu is required inside an object that is
    # itself present, so a top-level key check never sees it go missing.
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "scifact-dense-only.json",
             lambda d: d["hardware"].pop("cpu"))
    assert _errors_mentioning(vmb.validate(bundle), "cpu")


def test_a_wrong_type_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "nfcorpus-dense-only.json",
             lambda d: d["counts"].__setitem__("corpus_docs", "three thousand"))
    assert _errors_mentioning(vmb.validate(bundle), "corpus_docs")


def test_a_malformed_query_subset_digest_is_rejected_by_pattern(tmp_path: Path) -> None:
    # Distinct from the manifest-agreement check: this one catches a digest that
    # is not a sha256 at all, even if the manifest carries the same wrong value.
    bundle = _copy(tmp_path)

    def mutate(d):
        d["query_subset"]["sha256"] = "nope"

    _rewrite(bundle / "scorecards" / "fiqa-dense-only.json", mutate)
    assert _errors_mentioning(vmb.validate(bundle), "sha256")


def test_an_unexpected_top_level_field_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "quora-dense-only.json",
             lambda d: d.__setitem__("editorial_note", "looks good to me"))
    assert _errors_mentioning(vmb.validate(bundle), "editorial_note")


def test_an_empty_scorecard_is_rejected_with_every_missing_field(tmp_path: Path) -> None:
    # The blunt case: a scorecard reduced to nothing must not pass because the
    # checks happened to look only at fields it still had.
    bundle = _copy(tmp_path)
    card = bundle / "scorecards" / "scifact-dense-only.json"
    card.write_text("{}\n", encoding="utf-8")
    _refresh_checksums(bundle)
    errors = vmb.validate(bundle)
    assert len(_errors_mentioning(errors, "is a required property")) >= 10


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------

def test_a_declared_count_below_the_real_one_is_rejected(tmp_path: Path) -> None:
    # This is a real error that reached a draft of the source report: the count
    # predated a late-added measurement arm and was never updated.
    bundle = _copy(tmp_path)
    _rewrite(bundle / "manifest.json",
             lambda d: d["cells"].__setitem__("declared_count", d["cells"]["declared_count"] - 1))
    assert _errors_mentioning(vmb.validate(bundle), "declared_count")


def test_a_missing_scorecard_file_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    (bundle / "scorecards" / "quora-llm-listwise.json").unlink()
    _refresh_checksums(bundle)
    assert _errors_mentioning(vmb.validate(bundle), "do not match files on disk")


def test_a_duplicated_dataset_and_arm_cell_is_rejected(tmp_path: Path) -> None:
    # Two cells claiming the same dataset and arm means one measurement is being
    # counted twice, which inflates coverage without adding evidence.
    bundle = _copy(tmp_path)
    source = bundle / "scorecards" / "scifact-dense-only.json"
    clone = bundle / "scorecards" / "scifact-dense-only-copy.json"
    shutil.copy(source, clone)
    manifest = bundle / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["cells"]["scorecards"].append(clone.name)
    data["cells"]["declared_count"] += 1
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _refresh_checksums(bundle)
    assert _errors_mentioning(vmb.validate(bundle), "duplicate dataset/arm cell")


# --------------------------------------------------------------------------
# query accounting
# --------------------------------------------------------------------------

def test_an_unaccounted_query_is_rejected(tmp_path: Path) -> None:
    # evaluated + missing + failed must equal requested. A query that is none of
    # those was silently dropped.
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "fiqa-dense-only.json",
             lambda d: d["counts"].__setitem__("evaluated_queries", d["counts"]["evaluated_queries"] - 1))
    assert _errors_mentioning(vmb.validate(bundle), "queries accounted for")


def test_a_query_subset_digest_disagreeing_with_the_manifest_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "scifact-dense-only.json",
             lambda d: d["query_subset"].__setitem__("sha256", "0" * 64))
    assert _errors_mentioning(vmb.validate(bundle), "query_subset.sha256")


def test_counts_disagreeing_with_the_manifest_dataset_entry_are_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "nfcorpus-dense-only.json",
             lambda d: d["counts"].__setitem__("corpus_docs", 1))
    assert _errors_mentioning(vmb.validate(bundle), "counts.corpus_docs")


def test_a_latency_count_not_matching_evaluated_queries_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "scifact-cross-encoder-rerank.json",
             lambda d: d["latency_ms"].__setitem__("count", 1))
    assert _errors_mentioning(vmb.validate(bundle), "latency count")


# --------------------------------------------------------------------------
# declared shape
# --------------------------------------------------------------------------

def test_an_undeclared_arm_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "fiqa-dense-only.json",
             lambda d: d.__setitem__("arm", "an_arm_nobody_declared"))
    assert _errors_mentioning(vmb.validate(bundle), "is not declared in the manifest")


def test_a_scorecard_with_no_stated_limitation_is_rejected(tmp_path: Path) -> None:
    # A published result with no declared boundary reads as one with none.
    bundle = _copy(tmp_path)
    _rewrite(bundle / "scorecards" / "fiqa-dense-only.json",
             lambda d: d.__setitem__("limitations", []))
    assert _errors_mentioning(vmb.validate(bundle), "at least one limitation")


def test_a_bundle_that_declares_no_boundary_on_its_evidence_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "manifest.json", lambda d: d.__setitem__("not_evidence_of", []))
    assert _errors_mentioning(vmb.validate(bundle), "not_evidence_of")


# --------------------------------------------------------------------------
# integrity and public safety
# --------------------------------------------------------------------------

def test_an_edited_metric_without_a_refreshed_checksum_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    card = bundle / "scorecards" / "quora-cross-encoder-rerank-full.json"
    data = json.loads(card.read_text(encoding="utf-8"))
    data["metrics"]["ndcg_at_10"] = 0.99
    card.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert _errors_mentioning(vmb.validate(bundle), "hashes to")


def test_a_file_present_but_unlisted_in_the_checksums_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    (bundle / "stray.json").write_text("{}\n", encoding="utf-8")
    assert _errors_mentioning(vmb.validate(bundle), "present but unlisted")


def test_a_private_home_directory_path_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "manifest.json", lambda d: d.__setitem__("scope", "/home/someuser/work"))
    assert _errors_mentioning(vmb.validate(bundle), "home directory path")


def test_a_private_address_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "manifest.json", lambda d: d.__setitem__("scope", "measured on 10.1.104.44"))
    assert _errors_mentioning(vmb.validate(bundle), "private IPv4 address")


def test_a_leaked_api_key_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    _rewrite(bundle / "manifest.json",
             lambda d: d.__setitem__("scope", "key sk-0123456789abcdef0123"))
    assert _errors_mentioning(vmb.validate(bundle), "an API key")


def test_a_required_bundle_file_missing_is_rejected(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    (bundle / "reproduction.md").unlink()
    _refresh_checksums(bundle)
    assert _errors_mentioning(vmb.validate(bundle), "reproduction.md")
