# Masterarbeit WSE Evaluation Bundle

This public repository is a deliberately minimal, gold-free execution bundle
for consumed-development experiments in a computer-science Master's thesis.
It is intended for anonymous checkout by the WSE vLLM job queue.

Included files are limited to:

- the evidence-local and open-abstraction runtimes with their manifest-bound
  Scope support modules;
- the standard-library WSE client with a bounded per-call wall-clock deadline;
- the retained V1/V2, constrained-decoding V3 and open-abstraction V4 gpu01
  development manifests;
- the manifest-bound AutoSchemaKG baseline checkpoint; and
- the four previously received pair-decision call records inside the bound
  technical-failure artifact.

Evaluator-only gold, thesis sources, private annotations, credentials, local
paths, environment files and unrelated experiment results are intentionally
absent. The runtime imports only lower-level dependencies from the
digest-pinned WSE base image;
this repository is not a standalone Python package.

The active V3 raw bindings are:

- manifest SHA-256:
  `040fdba225002df965a91fd4b42639e9e6db76bbd9577fdf3842261e66198d42`
- baseline SHA-256:
  `a6618e2fa23dfe0aea88cea40fba838d7ea5ee9d42a315ab7008969606ddc435`
- replay evidence SHA-256:
  `422d1825abb2289ffdca084f740c4d3bf8a086c20134f2f9f937c8a999826fbd`

V3 changes only the failed V2 decoder boundary: every mapping request now
includes a packet-specific OpenAI JSON Schema that constrains source IDs,
closed type/relation labels, direction and row count during decoding. It
preserves the V2 model, prompts, inputs, evidence, budgets and all 34 packet
memberships and remains a consumed-development root-cause repair.

The active V4 raw bindings are:

- manifest SHA-256:
  `e57b63a0f653d21166df1b5b46c5068aaa4ccc8fcb48940ab52a7704773a33f1`
- runtime SHA-256:
  `084ad189770b620b4f49e7cbff482bf8bde6deb9e67c9c3551aee156e002ab5c`
- baseline SHA-256:
  `a6618e2fa23dfe0aea88cea40fba838d7ea5ee9d42a315ab7008969606ddc435`

V4 tests the diagnosed closed-vocabulary root cause with one open-label,
evidence-grounded mapping per source edge. It then performs only deterministic
exact-label consolidation with complete source-edge lineage. Evaluator gold is
not packaged and never enters model requests. V4 remains consumed development;
the unchanged post-hoc quality gate decides whether any later confirmation or
cross-model robustness run is warranted.

The queue job must pin both this repository commit and the compatible OCI base
image digest. Results remain consumed-development evidence and cannot be
reported as held-out confirmation or general AutoSchemaKG dominance.
