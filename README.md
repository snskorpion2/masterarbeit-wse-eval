# Masterarbeit WSE Evaluation Bundle

This public repository is a deliberately minimal, gold-free execution bundle
for one consumed-development experiment in a computer-science Master's thesis.
It is intended for anonymous checkout by the WSE vLLM job queue.

Included files are limited to:

- the evidence-local edge-mapping runtime;
- the frozen gpu01 development manifest;
- the manifest-bound AutoSchemaKG baseline checkpoint; and
- the four previously received pair-decision call records inside the bound
  technical-failure artifact.

Evaluator-only gold, thesis sources, private annotations, credentials, local
paths, environment files and unrelated experiment results are intentionally
absent. The runtime imports the existing digest-pinned WSE base-image modules;
this repository is not a standalone Python package.

The frozen raw bindings are:

- manifest SHA-256:
  `fa7c0697f04e9f88197a1681f94597f86e066a2ecfaa41a2862b1e464d13f8e4`
- baseline SHA-256:
  `a6618e2fa23dfe0aea88cea40fba838d7ea5ee9d42a315ab7008969606ddc435`
- replay evidence SHA-256:
  `422d1825abb2289ffdca084f740c4d3bf8a086c20134f2f9f937c8a999826fbd`

The queue job must pin both this repository commit and the compatible OCI base
image digest. Results remain consumed-development evidence and cannot be
reported as held-out confirmation or general AutoSchemaKG dominance.
