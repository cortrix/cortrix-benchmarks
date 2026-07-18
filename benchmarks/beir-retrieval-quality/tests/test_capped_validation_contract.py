from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "benchmarks/beir-retrieval-quality/runner/run_benchmark.py"


class CappedValidationContractTest(unittest.TestCase):
    def test_live_cli_exposes_deterministic_caps(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(RUNNER), "run", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--seed", completed.stdout)
        self.assertIn("--max-queries", completed.stdout)
        self.assertIn("--max-corpus-docs", completed.stdout)

    def test_public_docs_keep_claim_boundary(self) -> None:
        paths = [
            REPO_ROOT / "README.md",
            REPO_ROOT / "benchmarks/beir-retrieval-quality/README.md",
            REPO_ROOT / "benchmarks/beir-retrieval-quality/docs/local-reproduction.md",
        ]
        text = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
        self.assertIn("upstream dataset", text)
        self.assertIn("quick validation", text)
        self.assertIn("not comparable", text)
        self.assertIn("--max-queries", text)
        self.assertIn("--max-corpus-docs", text)

    def test_no_derived_dataset_fixture_is_distributed(self) -> None:
        datasets = REPO_ROOT / "benchmarks/beir-retrieval-quality/datasets"
        self.assertFalse(datasets.exists())


if __name__ == "__main__":
    unittest.main()
