# Contributing to TensorMark

TensorMark is a focused engine that complements PyTorch through a small,
documented compatibility surface. It does not aim to reproduce the complete
PyTorch or NumPy APIs. Discuss larger features before implementing them.

## Useful reports

Include the TensorMark version, Python version, operating system, CPU, install
method, a minimal example, and expected versus actual values. Do not attach
credentials, private datasets, proprietary model weights, or personal logs.
For security-sensitive issues, follow [SECURITY.md](SECURITY.md) instead.

## Changes

- Keep changes focused and add a regression test for changed behavior.
- Use small deterministic inputs and independent numerical references.
- Document unsupported cases explicitly; avoid silently falling back to a
  different library or an unintended development binary.
- Keep model downloads, large training runs, and timing thresholds out of
  correctness smoke tests.
- Preserve copyright and license notices. State the source and terms of any
  third-party code or assets you propose to include.

From a complete checkout on the documented platform, install the package and
run the model-free packaging and installed-package checks described in the
README. A test passing against an in-tree development extension is not enough
to establish that the distributed package works.

## Performance work

Treat PyTorch and NumPy as valued references, not adversaries. Report paired
same-machine measurements with versions, shapes, dtypes, thread/device settings,
reference execution mode, and numerical error. Include regressions and negative
results. Keep claims local to the workload actually measured.

## Contribution terms

Submit only material you have the right to contribute. Unless explicitly
stated otherwise, contributions intentionally submitted for inclusion are
under Apache-2.0 as described in [LICENSE](LICENSE). This does not change the
terms of separately identified third-party material.
