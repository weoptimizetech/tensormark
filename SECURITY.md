# Security reporting

Please do not publish exploit details, credentials, or private data in a public
issue. Contact WeOptimize privately through the company's **contact** mailbox
at **weoptimizetech.com**, with the subject `TensorMark security report`.

Include the affected revision/version, platform, a minimal reproduction, and
the security impact. Use synthetic inputs where possible; do not send live
secrets or sensitive model/data files. Coordinate privately before publishing
details. No response-time SLA or bug bounty is currently promised.

## Scope and precautions

This source edition contains a native numerical engine and Python bindings,
not standalone model-download or archive-decoding tools. Native operations are
not a sandbox for hostile inputs: validate shapes, sizes and values before use,
isolate untrusted workloads and apply resource limits. Any external model or
dataset loading code needs its own security review.

The current validation scope is Apple Silicon macOS with CPython 3.14. There
is no long-term-support or backport policy yet; reports should identify the
exact affected release or commit.
