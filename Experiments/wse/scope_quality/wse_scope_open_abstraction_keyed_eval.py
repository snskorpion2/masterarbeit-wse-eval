"""Keyed wire repair for evidence-grounded open-label REDFM development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
for relative in (
    "Experiments/wse/triple_quality",
    "Experiments/wse/scope_quality",
    "Experiments/src",
):
    value = str(ROOT / relative)
    if value not in sys.path:
        sys.path.insert(0, value)

from wse_eval import (  # noqa: E402
    TransportIncomplete,
    WSEClient,
    canonical_json_bytes,
    write_checkpoint_atomic,
    write_json_atomic,
)
import wse_scope_edge_local_constrained_eval as constrained  # noqa: E402
import wse_scope_eval as legacy  # noqa: E402
import wse_scope_partition_eval as partition  # noqa: E402


CHECKPOINT_SCHEMA = "wse-scope-open-abstraction-keyed-checkpoint-v5"
MANIFEST_SCHEMA = "wse-scope-open-abstraction-keyed-manifest-v5"
PROTOCOL = "wse-scope-open-abstraction-keyed-development-v5"
SUMMARY_SCHEMA = "wse-scope-open-abstraction-keyed-summary-v5"
FAILURE_SCHEMA = "wse-scope-open-abstraction-keyed-technical-failure-v5"
SMOKE_SCHEMA = "wse-scope-open-abstraction-keyed-smoke-v5"
MAX_PROMPT_BYTES = 6000
MAX_EXCERPT_BYTES = 320
MAX_TYPE_LABEL_CHARS = 64
MAX_RELATION_LABEL_CHARS = 80
_MAPPING_SYSTEM_PROMPT = (
    "Induce a reusable corpus-level schema from evidence. For every raw edge, "
    "return a reusable English schema edge grounded only in its evidence. You "
    "may introduce labels absent from the raw vocabulary. Types must be reusable "
    "classes, never named entities. Relations must be reusable predicates, not "
    "document-specific wording. All labels must be lowercase. Use the same "
    "wording for equivalent concepts, preserve meaning, and do not generalize "
    "beyond the quote. Return JSON only: edge_mappings must be an object keyed "
    "by every supplied source_edge_id exactly once. Each value has head_type, "
    "relation_type, tail_type and reverse; use no other keys. Set reverse only "
    "when the reusable predicate requires swapped endpoints."
)
_V4_SYSTEM_PROMPT_BYTES = 723
if len(_MAPPING_SYSTEM_PROMPT.encode("utf-8")) > _V4_SYSTEM_PROMPT_BYTES:
    raise RuntimeError("keyed prompt exceeds the frozen V4 prompt boundary")
MAPPING_SYSTEM_PROMPT = _MAPPING_SYSTEM_PROMPT + " " * (
    _V4_SYSTEM_PROMPT_BYTES - len(_MAPPING_SYSTEM_PROMPT.encode("utf-8"))
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_manifest(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    observed = _sha256_bytes(raw)
    if observed != expected_sha256:
        raise ValueError("open-abstraction manifest bytes differ from binding")
    manifest = json.loads(raw.decode("utf-8"))
    semantic = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    selection = manifest.get("evidence_selection_contract", {})
    abstraction = manifest.get("open_abstraction_contract", {})
    predecessor = manifest.get("predecessor_binding", {})
    repair = manifest.get("serialization_repair_binding", {})
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("protocol_id") != PROTOCOL
        or manifest.get("artifact_sha256") != partition._semantic_sha(semantic)
        or manifest.get("response_contract")
        != "server-constrained-keyed-open-label-edge-mapping-v5"
        or manifest.get("projection_contract")
        != "exact-label-union-with-source-lineage-v4"
        or manifest.get("server_label") != "gpu01"
        or manifest.get("model") != "Qwen/Qwen3-14B"
        or selection.get("maximum_prompt_utf8_bytes") != MAX_PROMPT_BYTES
        or selection.get("maximum_excerpt_utf8_bytes") != MAX_EXCERPT_BYTES
        or selection.get("gold_access") is not False
        or abstraction.get("maximum_type_label_characters")
        != MAX_TYPE_LABEL_CHARS
        or abstraction.get("maximum_relation_label_characters")
        != MAX_RELATION_LABEL_CHARS
        or abstraction.get("consolidation")
        != "deterministic exact normalized edge union"
        or abstraction.get("gold_access") is not False
        or predecessor.get("status") != "quality_gate_fail"
        or not isinstance(predecessor.get("analysis_report_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", predecessor["analysis_report_sha256"])
        is None
        or repair.get("status") != "development_parse_failure"
        or repair.get("root_cause")
        != "duplicate_source_edge_id_and_missing_required_source_edge_id"
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(repair.get(key, ""))) is None
            for key in (
                "manifest_sha256",
                "checkpoint_sha256",
                "summary_sha256",
                "receipt_sha256",
            )
        )
        or manifest.get("request_parameters", {}).get("candidate", {}).get(
            "temperature"
        )
        != 0
        or manifest.get("request_parameters", {}).get("candidate", {}).get(
            "chat_template_kwargs"
        )
        != {"enable_thinking": False}
    ):
        raise ValueError("open-abstraction manifest contract mismatch")
    return manifest, observed


def _response_format(source_edge_ids: tuple[str, ...]) -> dict[str, Any]:
    if not source_edge_ids or len(set(source_edge_ids)) != len(source_edge_ids):
        raise ValueError("open-abstraction source-edge inventory is invalid")
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "head_type",
            "relation_type",
            "tail_type",
            "reverse",
        ],
        "properties": {
            "head_type": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TYPE_LABEL_CHARS,
            },
            "relation_type": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_RELATION_LABEL_CHARS,
            },
            "tail_type": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TYPE_LABEL_CHARS,
            },
            "reverse": {"type": "boolean"},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "evidence_grounded_open_schema_edges",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["edge_mappings"],
                "properties": {
                    "edge_mappings": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(source_edge_ids),
                        "properties": {
                            source_id: item for source_id in source_edge_ids
                        },
                    }
                },
            },
        },
    }


def _request(rows: list[dict[str, Any]]) -> dict[str, Any]:
    user_prompt = json.dumps(
        {"edges": rows},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    response_format = _response_format(
        tuple(row["source_edge_id"] for row in rows)
    )
    prompt_bytes = len(MAPPING_SYSTEM_PROMPT.encode("utf-8")) + len(
        user_prompt.encode("utf-8")
    )
    binding = {
        "system_prompt": MAPPING_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "response_format": response_format,
    }
    return {
        "system_prompt": MAPPING_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "prompt_utf8_bytes": prompt_bytes,
        "prompt_sha256": _sha256_bytes(canonical_json_bytes(binding)),
        "response_format": response_format,
        "response_format_sha256": _sha256_bytes(canonical_json_bytes(response_format)),
    }


def build_edge_packets(
    manifest: dict[str, Any],
    baseline: dict[str, Any],
    *,
    source_edge_limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    document_by_id = {row["doc_id"]: row for row in manifest["documents"]}
    source_edges = sorted(
        (legacy._source_edge(edge) for edge in baseline["edges"]),
        key=lambda row: row["source_edge_id"],
    )
    if len({row["source_edge_id"] for row in source_edges}) != len(source_edges):
        raise ValueError("development baseline source-edge IDs are not unique")
    if source_edge_limit is not None:
        if source_edge_limit < 1:
            raise ValueError("smoke source-edge limit is invalid")
        source_edges = source_edges[:source_edge_limit]

    prepared: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for source in source_edges:
        selections = constrained._evidence_selection(source, document_by_id)
        prepared.append(
            (
                source,
                {
                    "source_edge_id": source["source_edge_id"],
                    "raw": [
                        source["head_type"],
                        source["relation_type"],
                        source["tail_type"],
                    ],
                    "evidence": [
                        {
                            "doc_id": item["doc_id"],
                            "sentence_index": item["sentence_index"],
                            "quote": item["excerpt"],
                        }
                        for item in selections
                    ],
                },
                selections,
            )
        )

    def build(
        values: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]
    ) -> dict[str, Any]:
        return {
            "request": _request([row for _, row, _ in values]),
            "source_edges": tuple(source for source, _, _ in values),
            "evidence_selections": [
                {"source_edge_id": source["source_edge_id"], "documents": selections}
                for source, _, selections in values
            ],
        }

    packets: list[dict[str, Any]] = []
    current: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    maximum = manifest["evidence_selection_contract"][
        "maximum_prompt_utf8_bytes"
    ]
    for value in prepared:
        trial = build([*current, value])
        if trial["request"]["prompt_utf8_bytes"] <= maximum:
            current.append(value)
            continue
        if not current:
            raise TransportIncomplete("one open-abstraction edge exceeds 6 KB")
        packets.append(build(current))
        current = [value]
        if build(current)["request"]["prompt_utf8_bytes"] > maximum:
            raise TransportIncomplete("one open-abstraction edge exceeds 6 KB")
    if current:
        packets.append(build(current))
    return tuple(packets)


def _normal_label(value: Any, maximum: int) -> str:
    if type(value) is not str:
        raise ValueError("open schema label is not a string")
    normal = " ".join(value.split())
    if (
        value != normal
        or not normal
        or normal != normal.lower()
        or len(normal) > maximum
        or any(ord(character) < 32 for character in normal)
    ):
        raise ValueError("open schema label is invalid")
    return normal


def parse_mappings(
    response: dict[str, Any], source_edges: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("open-abstraction response contains duplicate keys")
            value[key] = item
        return value

    value = json.loads(
        legacy._finished_content(response), object_pairs_hook=strict_object
    )
    if not isinstance(value, dict) or set(value) != {"edge_mappings"}:
        raise ValueError("open-abstraction response schema is invalid")
    rows = value["edge_mappings"]
    if not isinstance(rows, dict):
        raise ValueError("open-abstraction mappings are not keyed")
    expected_ids = {row["source_edge_id"] for row in source_edges}
    if set(rows) != expected_ids:
        raise ValueError("open-abstraction mappings are incomplete")
    observed: dict[str, dict[str, Any]] = {}
    for source_id, row in rows.items():
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "head_type",
                "relation_type",
                "tail_type",
                "reverse",
            }
            or type(row.get("reverse")) is not bool
        ):
            raise ValueError("open-abstraction mapping binding is invalid")
        observed[source_id] = {
            "source_edge_id": source_id,
            "head_type": _normal_label(row.get("head_type"), MAX_TYPE_LABEL_CHARS),
            "relation_type": _normal_label(
                row.get("relation_type"), MAX_RELATION_LABEL_CHARS
            ),
            "tail_type": _normal_label(row.get("tail_type"), MAX_TYPE_LABEL_CHARS),
            "reverse": row["reverse"],
        }
    if set(observed) != expected_ids:
        raise ValueError("open-abstraction mappings are incomplete")
    return tuple(observed[source_id] for source_id in sorted(observed))


def _costs(
    baseline: dict[str, Any], records: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return partition._costs(baseline, records)


def _execute(
    *,
    manifest: dict[str, Any],
    manifest_sha: str,
    baseline: dict[str, Any],
    baseline_sha: str,
    results_root: Path,
    client: Any,
    lease: Any,
    smoke: bool,
) -> dict[str, Any]:
    packets = build_edge_packets(
        manifest, baseline, source_edge_limit=1 if smoke else None
    )
    records: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    parser_error: str | None = None
    try:
        for index, packet in enumerate(packets, start=1):
            request = packet["request"]
            lease.assert_healthy()
            result = client.chat(
                model=manifest["model"],
                messages=[
                    {"role": "system", "content": request["system_prompt"]},
                    {"role": "user", "content": request["user_prompt"]},
                ],
                parameters={
                    **manifest["request_parameters"]["candidate"],
                    "response_format": request["response_format"],
                },
                lease=lease,
            )
            lease.assert_healthy()
            record = {
                "stage": f"open-abstraction-{index:04d}",
                "prompt_sha256": request["prompt_sha256"],
                "prompt_utf8_bytes": request["prompt_utf8_bytes"],
                "response_format_sha256": request["response_format_sha256"],
                "response_sha256": _sha256_bytes(
                    canonical_json_bytes(result.response)
                ),
                "response": result.response,
                "usage": legacy._usage(result.response),
                "client_elapsed_seconds": result.client_elapsed_seconds,
                "retry_wait_seconds": result.retry_wait_seconds,
            }
            records.append(record)
            mappings.extend(parse_mappings(result.response, packet["source_edges"]))
            selections.extend(packet["evidence_selections"])
    except (json.JSONDecodeError, ValueError) as error:
        parser_error = str(error)

    if smoke:
        value = {
            "schema": SMOKE_SCHEMA,
            "status": "smoke_complete" if parser_error is None else "smoke_parse_failure",
            "manifest_sha256": manifest_sha,
            "source_baseline_sha256": baseline_sha,
            "server_label": manifest["server_label"],
            "model": manifest["model"],
            "fixed_denominator": 1,
            "received_calls": len(records),
            "mapping": mappings[0] if len(mappings) == 1 else None,
            "evidence_selection": selections[0] if len(selections) == 1 else None,
            "call_records": records,
            "parser_error": parser_error,
            "claim_guard": "Technical one-edge smoke only; no quality evidence.",
        }
        write_json_atomic(results_root / "smoke.json", value)
        return value

    status = "ok" if parser_error is None else "parse_error"
    edges = (
        legacy._project_global_mapping(baseline, tuple(mappings))
        if status == "ok"
        else []
    )
    marginal, total = _costs(baseline, records)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol_id": manifest["protocol_id"],
        "phase": manifest["phase"],
        "claim_guard": manifest["claim_guard"],
        "manifest_sha256": manifest_sha,
        "source_baseline_sha256": baseline_sha,
        "server_label": manifest["server_label"],
        "model": manifest["model"],
        "source_baseline": {
            "manifest_sha256": baseline["manifest_sha256"],
            "server_label": baseline["server_label"],
            "model": baseline["model"],
            "protocol_id": baseline["protocol_id"],
        },
        "method": "evidence-grounded open labels plus exact-label lineage union",
        "status": status,
        "fixed_denominator_source_edge_count": len(baseline["edges"]),
        "canonical_vocabulary": {
            "types": sorted(
                {
                    label
                    for mapping in mappings
                    for label in (mapping["head_type"], mapping["tail_type"])
                }
            ),
            "relations": sorted({mapping["relation_type"] for mapping in mappings}),
        },
        "evidence_selections": selections,
        "mappings": mappings,
        "edges": edges,
        "call_records": records,
        "actual_remote_calls_this_run": len(records),
        "parser_error": parser_error,
        "marginal_cost": marginal,
        "total_method_cost": total,
    }
    write_checkpoint_atomic(results_root / "open_abstraction.json", checkpoint)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "development_complete" if status == "ok" else "development_parse_failure",
        "runtime_complete": status == "ok",
        "quality_evaluated": False,
        "fixed_denominator_source_edge_count": len(baseline["edges"]),
        "mapped_source_edge_count": len(mappings) if status == "ok" else 0,
        "candidate_edge_count": len(edges),
        "actual_remote_calls_this_run": len(records),
        "marginal_cost": marginal,
        "total_method_cost": total,
        "claim_guard": manifest["claim_guard"],
    }
    write_json_atomic(results_root / "summary.json", summary)
    return summary


def run_open_abstraction(
    manifest_path: str | Path,
    baseline_path: str | Path,
    results_root: str | Path,
    client: Any,
    *,
    expected_manifest_sha256: str,
    smoke: bool = False,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(
        Path(manifest_path), expected_manifest_sha256
    )
    baseline, baseline_sha = partition.load_bound_baseline(
        Path(baseline_path), manifest
    )
    partition._validate_live_model(manifest, list(client.get_models()))
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        return _execute(
            manifest=manifest,
            manifest_sha=manifest_sha,
            baseline=baseline,
            baseline_sha=baseline_sha,
            results_root=Path(results_root),
            client=client,
            lease=lease,
            smoke=smoke,
        )


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--results-root", default="/results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        value = run_open_abstraction(
            args.manifest,
            args.baseline,
            args.results_root,
            WSEClient(),
            expected_manifest_sha256=args.manifest_sha256,
            smoke=args.smoke,
        )
    except TransportIncomplete as error:
        failure = {
            "schema": FAILURE_SCHEMA,
            "status": "technical_failure",
            "technical_error": str(error),
        }
        write_json_atomic(Path(args.results_root) / "technical_failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 4
    print(json.dumps(value, sort_keys=True))
    expected = "smoke_complete" if args.smoke else "development_complete"
    return 0 if value.get("status") == expected else 4


if __name__ == "__main__":
    raise SystemExit(_main())
