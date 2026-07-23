# Contributing

Contributions should keep benchmark code reproducible, public documentation clear, and result artifacts independently verifiable.

Cortrix Benchmarks does not require a Contributor License Agreement. Contributions use the same license that [the repository's path mapping](LICENSES/README.md) applies to the changed material and must be certified under the [Developer Certificate of Origin 1.1](DCO).

## Contribution certification

Every new commit must include a `Signed-off-by` trailer certifying DCO 1.1. Create it with:

```bash
git commit -s
```

For an existing local commit, add the trailer and update the branch before review:

```bash
git commit --amend --signoff --no-edit
```

For a multi-commit contribution, every commit must be signed off. Existing repository history before the policy effective date of 2026-07-20 is not retroactively re-signed. Maintainers verify sign-offs before merge; no GitHub App, required check, or branch-protection enforcement is claimed by this repository policy.

Contributors retain copyright in their contributions. No separate CLA grant is required. A different license for a contribution requires separate permission from the applicable copyright holder or holders.

## Expectations

- Keep runner changes scoped to benchmark behavior.
- Add or update tests when changing metrics, sampling, manifests, source-document mapping, or result parsing.
- Do not commit downloaded datasets, derived dataset fixtures, raw runs, logs, model assets, credentials, or local infrastructure details.
- Keep capped validation separate from published full-corpus measurements.
- Do not change curated metrics or source locks without a new measurement review.

## Challenge a published result

Open a [GitHub Issue](https://github.com/cortrix/cortrix-benchmarks/issues) for a disputed dataset, profile, metric cell, source identity, or reproduction outcome. Include exact repository, runner, bundle, manifest, and checksum identities rather than screenshots alone.

Cortrix Benchmarks Maintainers will acknowledge a sufficiently identified challenge within 10 business days and assign or explain an initial state such as `needs-reproduction` or `methodology-review`. Acknowledgment is not confirmation that the challenge is correct, a reproduction deadline, or permission to overwrite a published result. See [MAINTAINERS.md](MAINTAINERS.md) for ownership and correction boundaries.

## Security and conduct

- Follow [SECURITY.md](SECURITY.md) and email [security@cortrix.ai](mailto:security@cortrix.ai) for private vulnerability reports. The acknowledgment target is within 5 business days; it is not a remediation deadline.
- Participation is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Send conduct reports privately to [devrel@cortrix.ai](mailto:devrel@cortrix.ai), not through a public issue.

## License

The repository uses a path-based `AGPL-3.0-only` and `CC-BY-4.0` mapping. Contributions use the license applicable to the changed material, are certified under [DCO 1.1](DCO), and do not require a CLA. See [LICENSES/README.md](LICENSES/README.md) and [COPYRIGHT.md](COPYRIGHT.md).

## Before opening a pull request

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q benchmarks/beir-retrieval-quality/tests
python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py selftest
python3 benchmarks/beir-retrieval-quality/tools/validate_release_candidate.py \
  results/published/beir-three-full-corpus-2026-07-v1
git diff --check
```

Before opening a pull request, confirm that every new commit contains a valid `Signed-off-by` trailer.
