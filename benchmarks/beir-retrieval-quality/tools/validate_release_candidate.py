#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping


BUNDLE_ID = "beir-three-full-corpus-2026-07"
PROFILE_MAPPING = {
    "full_llm": ("Full Stack", "bge_m3_rf_pool_listwise"),
    "no_llm": ("Embedding + Reranking", "bge_m3_rerank_baseline_v2"),
}
EXPECTED_CELLS = {
    ("scifact", "full_llm"): {
        "counts": (5183, 300, 339),
        "metrics": (
            0.8697777777777778,
            0.9303333333333335,
            0.7715628585309029,
            0.7859678770592412,
        ),
        "latency": (34232.35773449778, 39544.659186449644),
        "strict": "FAIL",
    },
    ("scifact", "no_llm"): {
        "counts": (5183, 300, 339),
        "metrics": (
            0.7794444444444444,
            0.9193333333333333,
            0.6009192031295701,
            0.6345248462308591,
        ),
        "latency": (7310.592896000344, 8766.053063300184),
        "strict": "PASS",
    },
    ("fiqa", "full_llm"): {
        "counts": (57638, 648, 1706),
        "metrics": (
            0.5204965730428691,
            0.6640487127755649,
            0.46629212707101036,
            0.5061861036723014,
        ),
        "latency": (106172.47638299887, 135121.1999856998),
        "strict": "NOT_EVALUATED",
    },
    ("fiqa", "no_llm"): {
        "counts": (57638, 648, 1706),
        "metrics": (
            0.3635011336400227,
            0.6342567454835977,
            0.2525741428783595,
            0.3323594258364845,
        ),
        "latency": (26520.88781350085, 33302.56768604977),
        "strict": "NOT_EVALUATED",
    },
    ("nfcorpus", "full_llm"): {
        "counts": (3633, 323, 12334),
        "metrics": (
            0.1717175200952775,
            0.2544556184064304,
            0.372061225020649,
            0.3175197600622023,
        ),
        "latency": (29503.740341002413, 32229.120364094706),
        "strict": "FAIL",
    },
    ("nfcorpus", "no_llm"): {
        "counts": (3633, 323, 12334),
        "metrics": (
            0.1481659822924788,
            0.2360874647660738,
            0.29823358400134947,
            0.26590052377340756,
        ),
        "latency": (5904.860682000162, 6963.527067099949),
        "strict": "PASS",
    },
}
EXPECTED_SOURCE_CHECKSUMS = {
    ("scifact", "full_llm"): {
        "dataset_archive_sha256": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        "source_scorecard_sha256": "117142997665b6fcbee68257f8c85aad6f5b2f679df3a168c2b3ac7bef64e072",
        "source_profile_record_sha256": "0e2765024eeabc5d1d022a7e34626274d5e53e4ee419a32e767ce11f60aa971f",
        "redacted_server_config_sha256": "6f813c98fe05113c72c7d58d2c23150fc82e2b8d88b78ba76f5f10234b23af28",
    },
    ("scifact", "no_llm"): {
        "dataset_archive_sha256": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        "source_scorecard_sha256": "06822729fc110919ac6a65d514215cf9294dafc807e47a03b41a2240dc43ebab",
        "source_profile_record_sha256": "97830a9d84b365b27f2e5d2f1783f66014f4054892e613e05c9a817c64ca9dec",
        "redacted_server_config_sha256": "67d827b8f4cd8955e6fc8c9a8a5af7d136b3c627bf276abb39cb180ccad77d62",
    },
    ("fiqa", "full_llm"): {
        "dataset_archive_sha256": "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
        "source_scorecard_sha256": "c9c92ce8a9685468c8c3cf94e811d06d5fd730567782c8b6375ed8d88867e476",
        "source_profile_record_sha256": "0e2765024eeabc5d1d022a7e34626274d5e53e4ee419a32e767ce11f60aa971f",
        "redacted_server_config_sha256": "2f5ee5755a6ca3b3c604e365eb7448d07b58ed8f3261aa9a21c67840dd98b310",
    },
    ("fiqa", "no_llm"): {
        "dataset_archive_sha256": "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
        "source_scorecard_sha256": "21312ab1b2cb4be15990897e8e036557a8e72f8c99f934ab07252f32647da83a",
        "source_profile_record_sha256": "97830a9d84b365b27f2e5d2f1783f66014f4054892e613e05c9a817c64ca9dec",
        "redacted_server_config_sha256": "4f69a8ac049ca104b57b47cc131db68bbdb462441a1d73b94156a2a9bdc17199",
    },
    ("nfcorpus", "full_llm"): {
        "dataset_archive_sha256": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        "source_scorecard_sha256": "19894ab397d737f5e481c4272322fc57c069c28c2528c536ab69d12763b40b7e",
        "source_profile_record_sha256": "0e2765024eeabc5d1d022a7e34626274d5e53e4ee419a32e767ce11f60aa971f",
        "redacted_server_config_sha256": "6f813c98fe05113c72c7d58d2c23150fc82e2b8d88b78ba76f5f10234b23af28",
    },
    ("nfcorpus", "no_llm"): {
        "dataset_archive_sha256": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        "source_scorecard_sha256": "6077445983397096440b4a8bd61f029abaa34f5c2ba45c57416472867ca85a7c",
        "source_profile_record_sha256": "97830a9d84b365b27f2e5d2f1783f66014f4054892e613e05c9a817c64ca9dec",
        "redacted_server_config_sha256": "67d827b8f4cd8955e6fc8c9a8a5af7d136b3c627bf276abb39cb180ccad77d62",
    },
}
METRIC_FIELDS = ("recall_at_10", "recall_at_50", "ndcg_at_10", "ndcg_at_50")
SUMMARY_FIELDS = {
    "schema_version",
    "bundle_id",
    "status",
    "public_label",
    "profile_display_mapping",
    "dataset_summaries",
    "macro_average",
    "hero_claims",
    "paired_latency",
    "scope_boundary",
}
FORBIDDEN_PUBLIC_PATTERNS = {
    "macOS user path": re.compile(r"/(?:Users|Volumes)/"),
    "Linux home path": re.compile(r"/home/[^/\s]+/"),
    "session path": re.compile(r"\.codex/sessions|session\.jsonl", re.I),
    "cloud object URL": re.compile(r"\bs3://", re.I),
    "AWS access-key shape": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "prohibited publication label": re.compile(r"\bpublished measured result\b", re.I),
    "private review material": re.compile(r"\b(?:private reviewer comment|human review checkbox|red[- ]blue verdict)\b", re.I),
    "internal release workflow": re.compile(
        r"\b(?:local[_ ]release[_ ]candidate|maintainer[_ ]review[_ ]evidence|remote publication pending|public-claim review)\b",
        re.I,
    ),
    "FiQA 5K result": re.compile(r"\bFiQA[ -]?5K\b|\b5,?000-document\b|supplemental_fiqa_5k_ablation", re.I),
    "numbered sampled configuration": re.compile(r"\bConfig(?:uration)?\s*[1-4]\b", re.I),
}
FORBIDDEN_PUBLIC_TOKEN_HASHES = {
    "private control-plane name": {
        "eb09772e51c5f1648c9984f3eec9123f17e726a1586207c6cf1be1f273c16714",
        "1e3150709ed290528eae1c2f0e4cb10fcd00cc629835e168123d30158ad60e1a",
        "263ea7077caca1680300cb5b5b244f2412b6df84d0358de1b8a8d702c63b2b47",
    },
    "historical sampled score": {
        "c77f027f777fb9b9f0c91ed8af25b9377f1ed634cef66ab89011843623c334df",
        "55eb145e504605d1b67fd81b0a3ac2e91f4c44c3d79a5b4fd697d1ca925f9d9b",
        "3da50754207e98ec5aa5b6f28e3c95c082713b7a03a3f93486ac50382f55ce62",
        "1956d058d22fd3d080e1a481dabf27e6163896999f3f1eef31d340a9c708dbdd",
        "3222c3b774655ee18614c801d675e1ea785dc466fa2218bc66a5bb3bf2e20417",
        "4fe348e6bd620e9a8d6bae8659bc2471c09bd295112f28366a6d7eaa6272df12",
        "bfdd19bfaa7abc9c35a9b738ac9007590d6a37e788ac480ce218aa6e7fce4f93",
        "e3938026a43a909781803423c423f22a009994a6a09a3c5c25d5f7ea612febbd",
    },
}


def _load_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def public_safety_errors(label: str, text: str) -> list[str]:
    errors = []
    for description, pattern in FORBIDDEN_PUBLIC_PATTERNS.items():
        if pattern.search(text):
            errors.append(f"{label}: contains {description}")
    token_digests = {
        hashlib.sha256(token.casefold().encode("utf-8")).hexdigest()
        for token in re.findall(r"[A-Za-z0-9_.-]+", text)
    }
    for description, forbidden_digests in FORBIDDEN_PUBLIC_TOKEN_HASHES.items():
        if token_digests & forbidden_digests:
            errors.append(f"{label}: contains {description}")
    return errors


def _validate_schema_contract(repo_root: Path, errors: list[str]) -> tuple[set[str], set[str]]:
    schema_root = repo_root / "benchmarks/beir-retrieval-quality/schemas"
    bundle_schema = _load_json(schema_root / "release-candidate-bundle.schema.json")
    score_schema = _load_json(schema_root / "release-candidate-scorecard.schema.json")
    for name, schema in (("bundle", bundle_schema), ("scorecard", score_schema)):
        _require(
            schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
            f"{name} schema: expected JSON Schema draft 2020-12",
            errors,
        )
        _require(schema.get("additionalProperties") is False, f"{name} schema: must reject extra top-level fields", errors)
    return set(bundle_schema.get("required", [])), set(score_schema.get("required", []))


def _validate_manifest(
    candidate: Path,
    required: set[str],
    errors: list[str],
) -> tuple[Mapping[str, object], list[Path]]:
    manifest = _load_json(candidate / "manifest.json")
    _require(set(manifest) == required, "manifest: top-level fields do not match the bundle schema", errors)
    _require(manifest.get("schema_version") == "1.1", "manifest: schema_version must be 1.1", errors)
    _require(manifest.get("bundle_id") == BUNDLE_ID, f"manifest: bundle_id must be {BUNDLE_ID}", errors)
    _require(manifest.get("status") == "complete", "manifest: measurement status must be complete", errors)
    _require(manifest.get("public_label") == "Full-corpus retrieval measurements", "manifest: invalid public label", errors)
    _require(manifest.get("scope") == "retrieval_quality_only", "manifest: invalid scope", errors)
    _require(
        manifest.get("profile_display_mapping")
        == {key: value[0] for key, value in PROFILE_MAPPING.items()},
        "manifest: profile display mapping mismatch",
        errors,
    )
    datasets = manifest.get("datasets")
    _require(isinstance(datasets, list) and len(datasets) == 3, "manifest: expected exactly three datasets", errors)
    scorecard_paths: list[Path] = []
    if isinstance(datasets, list):
        observed_datasets = set()
        for row in datasets:
            if not isinstance(row, dict):
                errors.append("manifest: dataset entry must be an object")
                continue
            dataset = row.get("dataset")
            observed_datasets.add(dataset)
            scorecards = row.get("scorecards")
            _require(isinstance(scorecards, list) and len(scorecards) == 2, f"manifest: {dataset} must have two scorecards", errors)
            if isinstance(scorecards, list):
                scorecard_paths.extend(candidate / str(path) for path in scorecards)
        _require(observed_datasets == {"scifact", "fiqa", "nfcorpus"}, "manifest: dataset identity mismatch", errors)
    return manifest, scorecard_paths


def _validate_scorecards(
    paths: Iterable[Path],
    required: set[str],
    errors: list[str],
) -> dict[tuple[str, str], Mapping[str, object]]:
    observed: dict[tuple[str, str], Mapping[str, object]] = {}
    for path in paths:
        if not path.is_file():
            errors.append(f"missing scorecard: {path}")
            continue
        card = _load_json(path)
        _require(set(card) == required, f"{path.name}: top-level fields do not match the scorecard schema", errors)
        key = (str(card.get("dataset")), str(card.get("source_profile_identity")))
        _require(key not in observed, f"{path.name}: duplicate cell {key}", errors)
        observed[key] = card
        expected = EXPECTED_CELLS.get(key)
        if expected is None:
            errors.append(f"{path.name}: unexpected cell {key}")
            continue
        _require(card.get("schema_version") == "1.1", f"{path.name}: schema version mismatch", errors)
        _require(card.get("bundle_id") == BUNDLE_ID, f"{path.name}: bundle ID mismatch", errors)
        _require(card.get("status") == "complete", f"{path.name}: measurement status must be complete", errors)
        _require(card.get("split") == "test", f"{path.name}: split must be test", errors)
        display, runner = PROFILE_MAPPING[key[1]]
        _require(card.get("public_display_label") == display, f"{path.name}: display label mismatch", errors)
        _require(card.get("runner_profile") == runner, f"{path.name}: runner profile mismatch", errors)

        counts = card.get("counts")
        if isinstance(counts, dict):
            actual_counts = (
                counts.get("corpus_docs"),
                counts.get("official_test_queries"),
                counts.get("qrels_rows"),
            )
            _require(actual_counts == expected["counts"], f"{path.name}: count tuple mismatch", errors)
            _require(counts.get("evaluated_queries") == expected["counts"][1], f"{path.name}: incomplete query evaluation", errors)
            _require(counts.get("query_failures") == 0, f"{path.name}: query failures must be zero", errors)
        else:
            errors.append(f"{path.name}: counts must be an object")

        metrics = card.get("metrics")
        if isinstance(metrics, dict):
            actual_metrics = tuple(metrics.get(field) for field in METRIC_FIELDS)
            _require(actual_metrics == expected["metrics"], f"{path.name}: exact metric tuple mismatch", errors)
        else:
            errors.append(f"{path.name}: metrics must be an object")

        latency = card.get("latency_ms")
        if isinstance(latency, dict):
            _require((latency.get("p50"), latency.get("p95")) == expected["latency"], f"{path.name}: latency tuple mismatch", errors)
            _require(latency.get("count") == expected["counts"][1], f"{path.name}: latency count mismatch", errors)
            _require(latency.get("comparison_scope") == "within_dataset_paired_hardware_only", f"{path.name}: latency scope mismatch", errors)
        else:
            errors.append(f"{path.name}: latency_ms must be an object")

        validity = card.get("validity")
        if isinstance(validity, dict):
            scientific = validity.get("scientific")
            strict = validity.get("strict_feature_completeness")
            _require(isinstance(scientific, dict) and scientific.get("status") == "VALID", f"{path.name}: scientific status must be VALID", errors)
            _require(isinstance(strict, dict) and strict.get("status") == expected["strict"], f"{path.name}: strict status mismatch", errors)
        else:
            errors.append(f"{path.name}: validity must be an object")

        checksums = card.get("checksums")
        if isinstance(checksums, dict):
            _require(len(checksums) >= 5, f"{path.name}: insufficient source checksums", errors)
            for field, value in checksums.items():
                _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{path.name}: invalid checksum {field}", errors)
            for field, value in EXPECTED_SOURCE_CHECKSUMS[key].items():
                _require(checksums.get(field) == value, f"{path.name}: source checksum mismatch for {field}", errors)
        else:
            errors.append(f"{path.name}: checksums must be an object")

        protocol = card.get("protocol")
        if isinstance(protocol, dict):
            _require(protocol.get("seed") == 44044, f"{path.name}: seed mismatch", errors)
            _require(protocol.get("retrieval_depth") == 250, f"{path.name}: retrieval depth mismatch", errors)
            _require(protocol.get("metric_cutoffs") == [10, 50], f"{path.name}: metric cutoffs mismatch", errors)
        else:
            errors.append(f"{path.name}: protocol must be an object")
    _require(set(observed) == set(EXPECTED_CELLS), "scorecards: six-cell identity mismatch", errors)
    return observed


def _validate_profiles(candidate: Path, errors: list[str]) -> None:
    paths = {
        "full_llm": candidate / "profiles/full-stack.json",
        "no_llm": candidate / "profiles/embedding-reranking.json",
    }
    for identity, path in paths.items():
        profile = _load_json(path)
        display, runner = PROFILE_MAPPING[identity]
        _require(profile.get("schema_version") == "1.1", f"{path.name}: schema version mismatch", errors)
        _require(profile.get("status") == "complete", f"{path.name}: measurement status must be complete", errors)
        _require(profile.get("measurement_scope") == "full_corpus", f"{path.name}: invalid measurement scope", errors)
        _require(profile.get("source_profile_identity") == identity, f"{path.name}: source identity mismatch", errors)
        _require(profile.get("public_display_label") == display, f"{path.name}: display label mismatch", errors)
        _require(profile.get("runner_profile") == runner, f"{path.name}: runner profile mismatch", errors)
        _require(profile.get("query_fraction") == 1.0, f"{path.name}: profile is not full corpus", errors)


def _validate_summary(candidate: Path, cards: Mapping[tuple[str, str], Mapping[str, object]], errors: list[str]) -> None:
    summary = _load_json(candidate / "summaries.json")
    _require(set(summary) == SUMMARY_FIELDS, "summaries.json: top-level fields must contain full-corpus summaries only", errors)
    _require("supplemental_fiqa_5k_ablation" not in summary, "summaries.json: sampled or 5K results are prohibited", errors)
    _require(summary.get("schema_version") == "1.1", "summaries.json: schema version mismatch", errors)
    _require(summary.get("bundle_id") == BUNDLE_ID, "summaries.json: bundle ID mismatch", errors)
    _require(summary.get("status") == "complete", "summaries.json: measurement status must be complete", errors)
    _require(summary.get("public_label") == "Full-corpus retrieval measurements", "summaries.json: invalid public label", errors)
    _require(summary.get("profile_display_mapping") == {key: value[0] for key, value in PROFILE_MAPPING.items()}, "summaries.json: display mapping mismatch", errors)
    rows = summary.get("dataset_summaries")
    _require(isinstance(rows, list) and len(rows) == 3, "summaries.json: expected three dataset summaries", errors)
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                errors.append("summaries.json: dataset summary must be an object")
                continue
            dataset = str(row.get("dataset"))
            profiles = row.get("profiles")
            if not isinstance(profiles, dict):
                errors.append(f"summaries.json: {dataset} profiles must be an object")
                continue
            for identity in PROFILE_MAPPING:
                card = cards.get((dataset, identity))
                metrics = profiles.get(identity)
                _require(
                    isinstance(card, dict)
                    and isinstance(metrics, dict)
                    and tuple(metrics.get(field) for field in METRIC_FIELDS)
                    == tuple(card["metrics"][field] for field in METRIC_FIELDS),
                    f"summaries.json: {dataset}/{identity} does not match its scorecard",
                    errors,
                )

    macro = summary.get("macro_average")
    if isinstance(macro, dict) and isinstance(macro.get("profiles"), dict):
        for identity in PROFILE_MAPPING:
            expected_means = {
                field: sum(float(cards[(dataset, identity)]["metrics"][field]) for dataset in ("scifact", "fiqa", "nfcorpus")) / 3
                for field in METRIC_FIELDS
            }
            actual = macro["profiles"].get(identity)
            for field, expected_value in expected_means.items():
                _require(isinstance(actual, dict) and math.isclose(actual.get(field, -1), expected_value, rel_tol=0, abs_tol=1e-15), f"summaries.json: macro {identity}/{field} mismatch", errors)
    else:
        errors.append("summaries.json: macro_average missing")

    hero = summary.get("hero_claims")
    _require(isinstance(hero, list) and len(hero) == 2, "summaries.json: expected exactly two hero claims", errors)
    expected_hero = {
        "ndcg_at_10": (84.61593960375356, "Up to 84.6% higher NDCG@10"),
        "recall_at_10": (43.189807368887045, "Up to 43.2% higher Recall@10"),
    }
    if isinstance(hero, list):
        for claim in hero:
            if not isinstance(claim, dict) or claim.get("metric") not in expected_hero:
                errors.append("summaries.json: unexpected hero claim")
                continue
            exact, display = expected_hero[str(claim["metric"])]
            _require(math.isclose(claim.get("exact_relative_improvement_percent", -1), exact, rel_tol=0, abs_tol=1e-12), f"summaries.json: {claim['metric']} hero math mismatch", errors)
            _require(claim.get("display") == display, f"summaries.json: {claim['metric']} hero display mismatch", errors)
            _require(claim.get("qualifier") == "Full Stack vs Embedding + Reranking on FiQA.", f"summaries.json: {claim['metric']} qualifier mismatch", errors)


def _validate_public_text(candidate: Path, errors: list[str]) -> None:
    for path in sorted(candidate.rglob("*")):
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"{path.relative_to(candidate)}: binary file is not allowed")
                continue
            errors.extend(public_safety_errors(str(path.relative_to(candidate)), text))

    readme = (candidate / "README.md").read_text(encoding="utf-8")
    visible_percentages = re.findall(r"\b\d+(?:\.\d+)?%", readme)
    _require(visible_percentages == ["84.6%", "43.2%"], "README.md: percentage claims must be exactly 84.6 and 43.2", errors)
    _require(re.search(r"\b(?:pp|percentage points?)\b", readme, re.I) is None, "README.md: percentage-point language is prohibited", errors)


def _validate_checksums(candidate: Path, errors: list[str]) -> None:
    checksum_path = candidate / "checksums.sha256"
    if not checksum_path.is_file():
        errors.append("checksums.sha256: missing")
        return
    entries: dict[str, str] = {}
    for line_no, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if not match:
            errors.append(f"checksums.sha256:{line_no}: invalid line")
            continue
        digest, relative = match.groups()
        if relative in entries:
            errors.append(f"checksums.sha256:{line_no}: duplicate path {relative}")
            continue
        entries[relative] = digest
    expected_paths = {
        str(path.relative_to(candidate))
        for path in candidate.rglob("*")
        if path.is_file() and path != checksum_path
    }
    _require(set(entries) == expected_paths, "checksums.sha256: file inventory mismatch", errors)
    for relative, digest in entries.items():
        path = candidate / relative
        try:
            path.resolve().relative_to(candidate.resolve())
        except ValueError:
            errors.append(f"checksums.sha256: path escapes candidate: {relative}")
            continue
        if path.is_file():
            _require(_sha256(path) == digest, f"checksums.sha256: digest mismatch for {relative}", errors)


def validate_candidate(candidate: Path) -> list[str]:
    candidate = candidate.resolve()
    errors: list[str] = []
    if not candidate.is_dir():
        return [f"candidate directory does not exist: {candidate}"]
    repo_root = Path(__file__).resolve().parents[3]
    bundle_required, score_required = _validate_schema_contract(repo_root, errors)
    _, scorecard_paths = _validate_manifest(candidate, bundle_required, errors)
    cards = _validate_scorecards(scorecard_paths, score_required, errors)
    _validate_profiles(candidate, errors)
    _validate_summary(candidate, cards, errors)
    _validate_public_text(candidate, errors)
    _validate_checksums(candidate, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the sanitized published three-dataset full-corpus result bundle.")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    errors = validate_candidate(args.bundle)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "bundle_id": BUNDLE_ID, "cells": len(EXPECTED_CELLS)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
