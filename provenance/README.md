# Source equivalence

The published result bundle preserves the original measurement identity while presenting public, content-addressed benchmark sources.

[source-equivalence.json](source-equivalence.json) records:

- the frozen Cortrix, runner, status-contract, orchestration, and dataset archive identities;
- the public runner, methodology, reproduction, and result paths;
- Git tree or blob identities and SHA-256 values for byte-equivalent source;
- the distinction between internal run-state anchors and public reproduction sources;
- the fact that no new full benchmark run was executed for publication packaging.

The orchestration anchors preserve run identity but are not presented as publicly accessible source history. External reproduction relies on the public protocol, runner, schemas, upstream datasets, and an exact Cortrix source snapshot.

Most runner files are byte-equivalent to the frozen source. `profiles.py` is a documented public overlay that removes non-measurement diagnostic narrative while preserving the executable fields for the two published profiles. The public test suite compares those fields directly with the bundle profile contracts.

The frozen Cortrix source lock is recorded, but an anonymous public Cortrix source URL is not yet available. Until that separate source publication exists, this repository supports result verification and method inspection but does not claim fully public source-level reproduction of the product under test.
