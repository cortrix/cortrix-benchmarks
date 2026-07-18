from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_DIR = REPO_ROOT / "benchmarks/beir-retrieval-quality/runner"
sys.path.insert(0, str(RUNNER_DIR))

from profiles import PROFILES  # noqa: E402


class PublicProfileContractTest(unittest.TestCase):
    def test_published_profile_query_bodies_match_bundle_contracts(self) -> None:
        bundle = REPO_ROOT / "results/published/beir-three-full-corpus-2026-07-v1/profiles"
        cases = {
            "bge_m3_rerank_baseline_v2": bundle / "embedding-reranking.json",
            "bge_m3_rf_pool_listwise": bundle / "full-stack.json",
        }
        for profile_name, contract_path in cases.items():
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            profile = PROFILES[profile_name]
            actual = profile.query_body("query", "namespace", 50)
            actual.pop("query")
            actual.pop("namespaces")
            actual.pop("top_k")
            actual["llm_rerank"] = actual.get("llm_rerank", False)
            self.assertEqual(actual, contract["query_body"])
            self.assertEqual(profile.query_fraction, contract["query_fraction"])


if __name__ == "__main__":
    unittest.main()
