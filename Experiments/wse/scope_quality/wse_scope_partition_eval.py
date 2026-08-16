#!/usr/bin/env python3
"""Closed-ID schema partition runtime for consumed SCOPE development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parents[1] / "src" if len(HERE.parents) > 1 else HERE
for candidate in (HERE, HERE.parent / "triple_quality", REPO_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import wse_scope_eval as legacy  # noqa: E402
from wse_eval import (  # noqa: E402
    TransportIncomplete,
    WSEClient,
    canonical_json_bytes,
    write_checkpoint_atomic,
    write_json_atomic,
)


SCHEMA = "wse-scope-closed-id-partition-manifest-v1"
PROTOCOL = "wse-scope-closed-id-partition-development-v1"
CHECKPOINT_SCHEMA = "wse-scope-closed-id-partition-development-v1"
EXPECTED_MANIFEST_SHA256: dict[str, str] = {}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic_sha(value: dict[str, Any]) -> str:
    return _sha256(canonical_json_bytes(value))


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    observed = _sha256(raw)
    expected = EXPECTED_MANIFEST_SHA256.get(path.name)
    if EXPECTED_MANIFEST_SHA256 and observed != expected:
        raise ValueError("manifest bytes differ from the packaged binding")
    manifest = json.loads(raw.decode("utf-8"))
    semantic = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("protocol_id") != PROTOCOL
        or manifest.get("artifact_sha256") != _semantic_sha(semantic)
        or manifest.get("server_label") not in {"gpu01", "h200"}
        or not isinstance(manifest.get("model"), str)
        or not manifest["model"]
        or not isinstance(manifest.get("documents"), list)
        or not manifest["documents"]
        or manifest.get("partition_request_max_bytes") != 6000
        or manifest.get("candidate_packet_contract", {}).get(
            "maximum_prompt_utf8_bytes"
        )
        != 6000
        or manifest.get("request_parameters", {}).get("candidate", {}).get(
            "temperature"
        )
        != 0
        or manifest.get("request_parameters", {}).get("candidate", {}).get(
            "chat_template_kwargs"
        )
        != {"enable_thinking": False}
    ):
        raise ValueError("closed partition manifest contract mismatch")
    return manifest, observed


def _validate_live_model(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    available = [
        row
        for row in rows
        if (row.get("name") or row.get("id")) == manifest["model"]
    ]
    if len(available) != 1:
        raise ValueError("manifest model is not uniquely available")
    expected = manifest.get("catalog_binding", {}).get("selected_record_sha256")
    if expected is not None and expected != _sha256(
        canonical_json_bytes({"name": manifest["model"]})
    ):
        raise ValueError("live WSE selected model differs from frozen binding")
    manager_context = available[0].get(
        "context_length", available[0].get("max_model_len")
    )
    if manager_context is not None and int(manager_context) < int(
        manifest.get("context_length", 32768)
    ):
        raise ValueError("manager model context contradicts the frozen manifest")


def raw_vocabularies(
    baseline: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    edges = baseline.get("edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError("bound baseline edges are missing")
    types: set[str] = set()
    relations: set[str] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("bound baseline edge is invalid")
        for key in ("head_type", "tail_type"):
            label = edge.get(key)
            if not isinstance(label, str) or not label:
                raise ValueError("bound baseline type label is invalid")
            types.add(label)
        relation = edge.get("relation_type")
        if not isinstance(relation, str) or not relation:
            raise ValueError("bound baseline relation label is invalid")
        relations.add(relation)
    return tuple(sorted(types)), tuple(sorted(relations))


def _closed_partition_messages(payload: dict[str, Any]) -> tuple[str, str]:
    system = (
        "Partition the supplied raw schema labels into a smaller closed vocabulary. "
        "Each supplied raw_types or raw_relations item is [source_id,label]. "
        "Return exactly one JSON object with keys type_assignments and "
        "relation_assignments. Each value is a list with exactly one object per "
        "source_id and exactly the integer keys source_id and representative_id. "
        "Every representative_id must be one of the supplied IDs, must map to "
        "itself, and every source ID must occur exactly once. Use no label strings, "
        "no explanations and no extra keys. Both assignments must be strictly "
        "contractive when more than one raw label is supplied. Output JSON only."
    )
    user = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return system, user


def build_partition_request(
    manifest: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    types, relations = raw_vocabularies(baseline)
    payload = {
        "raw_types": [[index, label] for index, label in enumerate(types)],
        "raw_relations": [
            [index, label] for index, label in enumerate(relations)
        ],
    }
    system, user = _closed_partition_messages(payload)
    prompt_bytes = len(system.encode("utf-8")) + len(user.encode("utf-8"))
    if prompt_bytes > manifest["partition_request_max_bytes"]:
        raise TransportIncomplete("closed partition request exceeds byte limit")
    return {
        "system_prompt": system,
        "user_prompt": user,
        "prompt_utf8_bytes": prompt_bytes,
        "prompt_sha256": _sha256(
            canonical_json_bytes({"system_prompt": system, "user_prompt": user})
        ),
    }


def _validate_assignments(
    value: Any, *, expected_count: int
) -> list[dict[str, int]]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError("closed partition coverage is invalid")
    observed: dict[int, int] = {}
    for row in value:
        if (
            not isinstance(row, dict)
            or set(row) != {"source_id", "representative_id"}
            or type(row.get("source_id")) is not int
            or type(row.get("representative_id")) is not int
            or not 0 <= row["source_id"] < expected_count
            or not 0 <= row["representative_id"] < expected_count
            or row["source_id"] in observed
        ):
            raise ValueError("closed partition assignment is invalid")
        observed[row["source_id"]] = row["representative_id"]
    if set(observed) != set(range(expected_count)):
        raise ValueError("closed partition coverage is invalid")
    representatives = set(observed.values())
    if any(observed[representative] != representative for representative in representatives):
        raise ValueError("closed partition representative must map to itself")
    if any(observed[observed[source_id]] != observed[source_id] for source_id in observed):
        raise ValueError("closed partition is not idempotent")
    if expected_count > 1 and len(representatives) >= expected_count:
        raise ValueError("closed partition is not contractive")
    return [
        {"source_id": source_id, "representative_id": observed[source_id]}
        for source_id in sorted(observed)
    ]


def parse_closed_partition(
    response: dict[str, Any], baseline: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    value = json.loads(legacy._finished_content(response))
    if not isinstance(value, dict) or set(value) != {
        "type_assignments",
        "relation_assignments",
    }:
        raise ValueError("closed partition response schema is invalid")
    types, relations = raw_vocabularies(baseline)
    type_rows = _validate_assignments(
        value["type_assignments"], expected_count=len(types)
    )
    relation_rows = _validate_assignments(
        value["relation_assignments"], expected_count=len(relations)
    )
    type_ids = tuple(sorted({row["representative_id"] for row in type_rows}))
    relation_ids = tuple(
        sorted({row["representative_id"] for row in relation_rows})
    )
    audit = {
        "raw_types": list(types),
        "raw_relations": list(relations),
        "type_assignments": type_rows,
        "relation_assignments": relation_rows,
        "closed_vocabulary": True,
        "raw_type_count": len(types),
        "representative_type_count": len(type_ids),
        "raw_relation_count": len(relations),
        "representative_relation_count": len(relation_ids),
    }
    return (
        tuple(types[index] for index in type_ids),
        tuple(relations[index] for index in relation_ids),
        audit,
    )


def _validate_baseline_contract(
    baseline: dict[str, Any], manifest: dict[str, Any], binding: dict[str, Any]
) -> None:
    allowed_documents = {row["doc_id"] for row in manifest["documents"]}
    if (
        baseline.get("schema") != "wse-scope-checkpoint-v1"
        or baseline.get("arm") != "autoschemakg_baseline"
        or baseline.get("status") != "ok"
        or baseline.get("manifest_sha256") != binding["source_manifest_sha256"]
        or baseline.get("protocol_id") != binding["protocol_id"]
        or baseline.get("server_label") != binding["server_label"]
        or baseline.get("model") != binding["model"]
        or baseline.get("server_label") != manifest["server_label"]
        or baseline.get("model") != manifest["model"]
        or type(baseline.get("actual_calls")) is not int
        or baseline["actual_calls"] < 1
        or not isinstance(baseline.get("usage"), dict)
        or type(baseline["usage"].get("prompt_tokens")) is not int
        or type(baseline["usage"].get("completion_tokens")) is not int
        or type(baseline.get("client_elapsed_seconds")) not in (int, float)
        or type(baseline.get("retry_wait_seconds")) not in (int, float)
    ):
        raise ValueError("development baseline contract mismatch")
    for edge in baseline.get("edges", []):
        if (
            not isinstance(edge, dict)
            or set(edge)
            != {"head_type", "relation_type", "tail_type", "evidence_doc_ids"}
            or not isinstance(edge["evidence_doc_ids"], list)
            or not edge["evidence_doc_ids"]
            or any(doc_id not in allowed_documents for doc_id in edge["evidence_doc_ids"])
        ):
            raise ValueError("development baseline edge is invalid")
    raw_vocabularies(baseline)


def load_bound_baseline(
    path: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    observed = _sha256(raw)
    binding = manifest.get("source_baseline_binding")
    if not isinstance(binding, dict) or observed != binding.get("sha256"):
        raise ValueError("development baseline bytes differ from binding")
    baseline = json.loads(raw.decode("utf-8"))
    _validate_baseline_contract(baseline, manifest, binding)
    return baseline, observed


def _costs(
    baseline: dict[str, Any], call_records: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    marginal = {
        "actual_calls": len(call_records),
        "usage": {
            "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in call_records),
            "completion_tokens": sum(
                row["usage"]["completion_tokens"] for row in call_records
            ),
        },
        "client_elapsed_seconds": sum(
            row["client_elapsed_seconds"] for row in call_records
        ),
        "retry_wait_seconds": sum(row["retry_wait_seconds"] for row in call_records),
    }
    total = {
        "actual_calls": baseline["actual_calls"] + marginal["actual_calls"],
        "usage": {
            key: baseline["usage"][key] + marginal["usage"][key]
            for key in ("prompt_tokens", "completion_tokens")
        },
        "client_elapsed_seconds": baseline["client_elapsed_seconds"]
        + marginal["client_elapsed_seconds"],
        "retry_wait_seconds": baseline["retry_wait_seconds"]
        + marginal["retry_wait_seconds"],
    }
    return marginal, total


def _execute_with_lease(
    *,
    manifest: dict[str, Any],
    manifest_sha: str,
    baseline: dict[str, Any],
    baseline_sha: str,
    results_root: Path,
    client: Any,
    lease: Any,
) -> dict[str, Any]:
    call_records: list[dict[str, Any]] = []

    def call(request: dict[str, Any], stage: str) -> Any:
        lease.assert_healthy()
        result = client.chat(
            model=manifest["model"],
            messages=[
                {"role": "system", "content": request["system_prompt"]},
                {"role": "user", "content": request["user_prompt"]},
            ],
            parameters=manifest["request_parameters"]["candidate"],
            lease=lease,
        )
        lease.assert_healthy()
        call_records.append(
            {
                "stage": stage,
                "prompt_sha256": request["prompt_sha256"],
                "prompt_utf8_bytes": request["prompt_utf8_bytes"],
                "response_sha256": _sha256(canonical_json_bytes(result.response)),
                "response": result.response,
                "usage": legacy._usage(result.response),
                "client_elapsed_seconds": result.client_elapsed_seconds,
                "retry_wait_seconds": result.retry_wait_seconds,
            }
        )
        return result

    try:
        partition_request = build_partition_request(manifest, baseline)
        partition_result = call(partition_request, "closed-partition")
        canonical_types, canonical_relations, partition_audit = (
            parse_closed_partition(partition_result.response, baseline)
        )
        mappings: list[dict[str, Any]] = []
        packets = legacy._mapping_packets(
            manifest, baseline, canonical_types, canonical_relations
        )
        for index, packet in enumerate(packets, start=1):
            result = call(packet["request"], f"edge-mapping-{index:04d}")
            mappings.extend(
                legacy._parse_edge_mappings(
                    result.response,
                    packet["source_edges"],
                    canonical_types,
                    canonical_relations,
                )
            )
    except TransportIncomplete as error:
        failure = {
            "schema": "wse-scope-closed-id-partition-technical-failure-v1",
            "manifest_sha256": manifest_sha,
            "source_baseline_sha256": baseline_sha,
            "server_label": manifest["server_label"],
            "model": manifest["model"],
            "status": "technical_failure",
            "completed_call_records": call_records,
            "technical_error": str(error),
        }
        write_json_atomic(results_root / "technical_failure.json", failure)
        summary = {
            "schema": "wse-scope-closed-id-partition-summary-v1",
            "status": "technical_failure",
            "runtime_complete": False,
            "quality_evaluated": False,
            "actual_calls": len(call_records),
        }
        write_json_atomic(results_root / "summary.json", summary)
        return summary
    except (json.JSONDecodeError, ValueError) as error:
        status = "parse_error"
        parser_error = str(error)
        canonical_types = ()
        canonical_relations = ()
        partition_audit = None
        edges: list[dict[str, Any]] = []
    else:
        status = "ok"
        parser_error = None
        edges = legacy._project_global_mapping(baseline, tuple(mappings))

    marginal, total = _costs(baseline, call_records)
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
        "method": "closed ID partition plus evidence-bound edge projection",
        "status": status,
        "raw_edge_count": len(baseline["edges"]),
        "canonical_edge_count": len(edges),
        "canonical_vocabulary": {
            "types": list(canonical_types),
            "relations": list(canonical_relations),
        },
        "partition_audit": partition_audit,
        "edges": edges,
        "call_records": call_records,
        "parser_error": parser_error,
        "marginal_cost": marginal,
        "total_method_cost": total,
        "actual_calls": len(call_records),
    }
    write_checkpoint_atomic(results_root / "closed_partition.json", checkpoint)
    summary = {
        "schema": "wse-scope-closed-id-partition-summary-v1",
        "status": "development_complete" if status == "ok" else "development_parse_failure",
        "runtime_complete": status == "ok",
        "quality_evaluated": False,
        "actual_calls": len(call_records),
        "raw_edge_count": len(baseline["edges"]),
        "canonical_edge_count": len(edges),
        "marginal_cost": marginal,
        "total_method_cost": total,
        "claim_guard": manifest["claim_guard"],
    }
    write_json_atomic(results_root / "summary.json", summary)
    return summary


def run_closed_partition(
    manifest_path: str | Path,
    baseline_path: str | Path,
    results_root: str | Path,
    client: Any,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(Path(manifest_path))
    if manifest.get("baseline_mode") not in (None, "bound_existing"):
        raise ValueError("manifest baseline mode is not bound_existing")
    baseline, baseline_sha = load_bound_baseline(Path(baseline_path), manifest)
    _validate_live_model(manifest, list(client.get_models()))
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        return _execute_with_lease(
            manifest=manifest,
            manifest_sha=manifest_sha,
            baseline=baseline,
            baseline_sha=baseline_sha,
            results_root=Path(results_root),
            client=client,
            lease=lease,
        )


def run_paired_closed_partition(
    manifest_path: str | Path,
    results_root: str | Path,
    client: Any,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(Path(manifest_path))
    if manifest.get("baseline_mode") != "fresh_same_job":
        raise ValueError("manifest baseline mode is not fresh_same_job")
    _validate_live_model(manifest, list(client.get_models()))
    root = Path(results_root)
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        try:
            baseline, _ = legacy._run_autoschemakg_baseline(
                client, lease, manifest, manifest_sha
            )
        except TransportIncomplete as error:
            summary = {
                "schema": "wse-scope-closed-id-partition-summary-v1",
                "status": "technical_failure",
                "runtime_complete": False,
                "quality_evaluated": False,
                "actual_calls": 0,
                "technical_error": str(error),
            }
            write_json_atomic(root / "summary.json", summary)
            return summary
        baseline_path = root / "autoschemakg_baseline.json"
        write_checkpoint_atomic(baseline_path, baseline)
        if baseline["status"] != "ok":
            summary = {
                "schema": "wse-scope-closed-id-partition-summary-v1",
                "status": "development_baseline_failure",
                "runtime_complete": False,
                "quality_evaluated": False,
                "actual_calls": baseline["actual_calls"],
                "baseline_status": baseline["status"],
            }
            write_json_atomic(root / "summary.json", summary)
            return summary
        baseline_sha = _sha256(baseline_path.read_bytes())
        return _execute_with_lease(
            manifest=manifest,
            manifest_sha=manifest_sha,
            baseline=baseline,
            baseline_sha=baseline_sha,
            results_root=root,
            client=client,
            lease=lease,
        )


def _main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--baseline", required=True)
    run.add_argument("--results-root", default="/results")
    paired = commands.add_parser("run-paired")
    paired.add_argument("--manifest", required=True)
    paired.add_argument("--results-root", default="/results")
    args = parser.parse_args()
    client = WSEClient()
    if args.command == "run":
        value = run_closed_partition(
            args.manifest, args.baseline, args.results_root, client
        )
    else:
        value = run_paired_closed_partition(
            args.manifest, args.results_root, client
        )
    print(json.dumps(value, sort_keys=True))
    return 0 if value.get("status") == "development_complete" else 4


if __name__ == "__main__":
    raise SystemExit(_main())
