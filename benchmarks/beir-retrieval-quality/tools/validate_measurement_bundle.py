#!/usr/bin/env python3
"""Validate a measurement bundle against the measurement-* schemas.

Unlike validate_release_candidate.py, which is pinned to one release candidate's
bundle id, datasets, arms and headline percentages, this validator is generic: it
checks a bundle against its declared shape rather than against a remembered one.

What it enforces, and why each check exists:

- Every file listed in checksums.sha256 exists and hashes to the recorded value,
  and every file in the bundle is listed. A bundle that ships an unlisted file is
  a bundle whose contents nobody has to account for.
- The scorecard set matches the manifest's declared_count exactly, with no
  duplicates and nothing missing. A miscounted round must not be publishable as a
  complete one.
- Every scorecard's dataset appears in the manifest's dataset list, its arm in the
  arm list, and its per-dataset counts agree with the manifest. The same number
  must not be able to disagree with itself inside one bundle.
- requested_queries equals the dataset's selected_queries, and
  evaluated_queries + missing_query_text + query_failures accounts for all of
  them. An unaccounted query is a silently dropped measurement.
- Each scorecard declares at least one limitation, and the bundle declares both
  limitations and what it is not evidence of. A published result with no stated
  boundary reads as one with none.
- No private host names, addresses, user names or file system paths.

Usage:
  python3 validate_measurement_bundle.py results/published/<bundle-dir>
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"

# Shapes that must never reach a public artifact.
PRIVATE_PATTERNS = [
    (re.compile(r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "a private IPv4 address"),
    (re.compile(r"/home/[A-Za-z0-9._-]+"), "a home directory path"),
    (re.compile(r"/Users/[A-Za-z0-9._-]+"), "a home directory path"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "an API key"),
    (re.compile(r"\b[A-Za-z0-9._%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "an email address"),
]


def _fail(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _required(schema: Mapping[str, Any]) -> set[str]:
    return set(schema.get("required", []))


def _check_required_fields(obj: Mapping[str, Any], schema: Mapping[str, Any],
                           label: str, errors: list[str]) -> None:
    required = _required(schema)
    missing = required - set(obj)
    extra = set(obj) - set(schema.get("properties", {}))
    _fail(errors, not missing, f"{label}: missing required fields {sorted(missing)}")
    if schema.get("additionalProperties") is False:
        _fail(errors, not extra, f"{label}: unexpected fields {sorted(extra)}")


def _check_checksums(bundle: Path, errors: list[str]) -> None:
    listing = bundle / "checksums.sha256"
    if not listing.exists():
        errors.append("checksums.sha256: missing")
        return
    recorded: dict[str, str] = {}
    for line in listing.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        if not digest or not name:
            errors.append(f"checksums.sha256: cannot parse line {line!r}")
            continue
        recorded[name] = digest
    on_disk = {
        str(p.relative_to(bundle))
        for p in bundle.rglob("*")
        if p.is_file() and p.name != "checksums.sha256"
    }
    unlisted = on_disk - set(recorded)
    absent = set(recorded) - on_disk
    _fail(errors, not unlisted, f"checksums.sha256: files present but unlisted {sorted(unlisted)}")
    _fail(errors, not absent, f"checksums.sha256: listed but absent {sorted(absent)}")
    for name, digest in recorded.items():
        target = bundle / name
        if not target.exists():
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        _fail(errors, actual == digest, f"checksums.sha256: {name} hashes to {actual}, listed {digest}")


def _check_public_safety(bundle: Path, errors: list[str]) -> None:
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, description in PRIVATE_PATTERNS:
            if pattern.search(text):
                errors.append(f"{path.relative_to(bundle)}: contains {description}")


def validate(bundle: Path) -> list[str]:
    errors: list[str] = []
    bundle_schema = _load(SCHEMA_DIR / "measurement-bundle.schema.json")
    card_schema = _load(SCHEMA_DIR / "measurement-scorecard.schema.json")

    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        return ["manifest.json: missing"]
    manifest = _load(manifest_path)
    _check_required_fields(manifest, bundle_schema, "manifest.json", errors)

    datasets = {d["name"]: d for d in manifest.get("datasets", []) if isinstance(d, dict)}
    arms = {a["id"] for a in manifest.get("arms", []) if isinstance(a, dict)}
    cells = manifest.get("cells", {})
    declared = cells.get("declared_count")
    listed = cells.get("scorecards", [])

    card_dir = bundle / "scorecards"
    on_disk = sorted(p.name for p in card_dir.glob("*.json")) if card_dir.is_dir() else []

    _fail(errors, len(listed) == len(set(listed)), "manifest: duplicate scorecard names")
    _fail(errors, declared == len(listed),
          f"manifest: declared_count {declared} does not match {len(listed)} listed scorecards")
    _fail(errors, sorted(listed) == on_disk,
          f"manifest: listed scorecards {sorted(listed)} do not match files on disk {on_disk}")

    seen: set[tuple[str, str]] = set()
    for name in on_disk:
        card = _load(card_dir / name)
        label = f"scorecards/{name}"
        _check_required_fields(card, card_schema, label, errors)

        dataset = card.get("dataset")
        arm = card.get("arm")
        _fail(errors, dataset in datasets, f"{label}: dataset {dataset!r} is not declared in the manifest")
        _fail(errors, arm in arms, f"{label}: arm {arm!r} is not declared in the manifest")
        _fail(errors, card.get("bundle_id") == manifest.get("bundle_id"),
              f"{label}: bundle_id does not match the manifest")

        key = (str(dataset), str(arm))
        _fail(errors, key not in seen, f"{label}: duplicate dataset/arm cell {key}")
        seen.add(key)

        spec = datasets.get(dataset)
        counts = card.get("counts", {})
        if isinstance(spec, dict) and isinstance(counts, dict):
            for field, manifest_field in (("corpus_docs", "corpus_docs"),
                                          ("judged_queries", "judged_queries"),
                                          ("qrels_rows", "qrels_rows")):
                _fail(errors, counts.get(field) == spec.get(manifest_field),
                      f"{label}: counts.{field} disagrees with the manifest dataset entry")
            _fail(errors, counts.get("requested_queries") == spec.get("selected_queries"),
                  f"{label}: requested_queries disagrees with the manifest selected_queries")
            accounted = (counts.get("evaluated_queries", 0)
                         + counts.get("missing_query_text", 0)
                         + counts.get("query_failures", 0))
            _fail(errors, accounted == counts.get("requested_queries"),
                  f"{label}: {accounted} queries accounted for, {counts.get('requested_queries')} requested")
            subset = card.get("query_subset", {})
            _fail(errors, subset.get("sha256") == spec.get("query_subset_sha256"),
                  f"{label}: query_subset.sha256 disagrees with the manifest dataset entry")

        latency = card.get("latency_ms", {})
        if isinstance(latency, dict) and latency.get("count") is not None:
            _fail(errors, latency.get("count") == counts.get("evaluated_queries"),
                  f"{label}: latency count does not match evaluated_queries")
            lo, hi = latency.get("min"), latency.get("max")
            if lo is not None and hi is not None:
                _fail(errors, lo <= latency.get("p50", lo) <= latency.get("p95", hi) <= hi,
                      f"{label}: latency percentiles are not ordered within min..max")

        _fail(errors, len(card.get("limitations", [])) >= 1,
              f"{label}: at least one limitation must be stated")

    _fail(errors, len(manifest.get("limitations", [])) >= 1, "manifest: limitations must not be empty")
    _fail(errors, len(manifest.get("not_evidence_of", [])) >= 1, "manifest: not_evidence_of must not be empty")

    for required_file in ("README.md", "summaries.json", "reproduction.md", "checksums.sha256"):
        _fail(errors, (bundle / required_file).exists(), f"{required_file}: missing")

    _check_checksums(bundle, errors)
    _check_public_safety(bundle, errors)
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: validate_measurement_bundle.py <bundle-directory>", file=sys.stderr)
        return 2
    bundle = Path(argv[1])
    if not bundle.is_dir():
        print(f"not a directory: {bundle}", file=sys.stderr)
        return 2
    errors = validate(bundle)
    if errors:
        print(json.dumps({"status": "FAIL", "bundle": bundle.name, "errors": errors}, indent=2))
        return 1
    manifest = _load(bundle / "manifest.json")
    print(json.dumps({
        "status": "PASS",
        "bundle_id": manifest["bundle_id"],
        "cells": manifest["cells"]["declared_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
