from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "benchmarks/beir-retrieval-quality/tools"
sys.path.insert(0, str(TOOLS_DIR))

from validate_release_candidate import public_safety_errors, validate_candidate  # noqa: E402


class PublishedResultBundleTest(unittest.TestCase):
    def test_published_bundle_contract_and_public_safety(self) -> None:
        bundle = REPO_ROOT / "results/published/beir-three-full-corpus-2026-07-v1"
        self.assertEqual(validate_candidate(bundle), [])
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")

    def test_cloud_object_url_is_rejected(self) -> None:
        errors = public_safety_errors("fixture", "source=s3://example-bucket/result.json")
        self.assertTrue(any("cloud object URL" in error for error in errors))

    def test_prohibited_publication_label_is_rejected(self) -> None:
        errors = public_safety_errors("fixture", "status: published measured result")
        self.assertTrue(any("prohibited publication label" in error for error in errors))

    def test_unexpected_summary_field_fails_closed(self) -> None:
        source = REPO_ROOT / "results/published/beir-three-full-corpus-2026-07-v1"
        with tempfile.TemporaryDirectory() as temporary_dir:
            bundle = Path(temporary_dir) / "bundle"
            shutil.copytree(source, bundle)
            summary_path = bundle / "summaries.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["unexpected_sampled_result"] = {"status": "sampled"}
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            errors = validate_candidate(bundle)
        self.assertTrue(any("top-level fields" in error for error in errors))

    def test_checksum_tampering_fails_closed(self) -> None:
        source = REPO_ROOT / "results/published/beir-three-full-corpus-2026-07-v1"
        with tempfile.TemporaryDirectory() as temporary_dir:
            bundle = Path(temporary_dir) / "bundle"
            shutil.copytree(source, bundle)
            readme = bundle / "README.md"
            readme.write_text(readme.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            errors = validate_candidate(bundle)
        self.assertTrue(any("digest mismatch for README.md" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
