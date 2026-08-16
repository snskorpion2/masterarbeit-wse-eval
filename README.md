# Masterarbeit WSE Evaluation Bundle

This public repository is a deliberately minimal, gold-free execution bundle
for consumed-development experiments in a computer-science Master's thesis.
It is intended for anonymous checkout by the WSE vLLM job queue.

Included files are limited to:

- the evidence-local edge-mapping runtime and its four manifest-bound Scope
  runtime modules;
- the standard-library WSE client with a bounded per-call wall-clock deadline;
- the retained V1 and root-cause-repair V2 gpu01 development manifests;
- the manifest-bound AutoSchemaKG baseline checkpoint; and
- the four previously received pair-decision call records inside the bound
  technical-failure artifact.

Evaluator-only gold, thesis sources, private annotations, credentials, local
paths, environment files and unrelated experiment results are intentionally
absent. The runtime imports only lower-level dependencies from the
digest-pinned WSE base image;
this repository is not a standalone Python package.

The active V2 raw bindings are:

- manifest SHA-256:
  `5f984cb1b5fa764e9095a4f76c4de4cb22d72de9a8ee7ddfdcc06ea8118d88e1`
- baseline SHA-256:
  `a6618e2fa23dfe0aea88cea40fba838d7ea5ee9d42a315ab7008969606ddc435`
- replay evidence SHA-256:
  `422d1825abb2289ffdca084f740c4d3bf8a086c20134f2f9f937c8a999826fbd`

V2 changes only the failed wire representation from opaque integer indices to
exact labels from the same closed vocabulary. It preserves the V1 model,
inputs, evidence, budgets and packet memberships and remains a consumed
development root-cause repair.

The queue job must pin both this repository commit and the compatible OCI base
image digest. Results remain consumed-development evidence and cannot be
reported as held-out confirmation or general AutoSchemaKG dominance.
