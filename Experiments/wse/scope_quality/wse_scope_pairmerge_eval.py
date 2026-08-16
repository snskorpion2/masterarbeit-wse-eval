"""Evidence-bound pair-merge development runtime for consumed REDFM Stage A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wse_eval import (
    TransportIncomplete,
    WSEClient,
    canonical_json_bytes,
    write_checkpoint_atomic,
    write_json_atomic,
)
import wse_scope_eval as legacy
import wse_scope_partition_eval as partition


CHECKPOINT_SCHEMA = "wse-scope-evidence-pairmerge-development-v1"
MANIFEST_SCHEMA = "wse-scope-evidence-pairmerge-manifest-v1"
PROTOCOL = "wse-scope-evidence-pairmerge-development-v1"
EXPECTED_MANIFEST_SHA256: dict[str, str] = {}


def _sha256(value: bytes) -> str:
    return partition._sha256(value)


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    observed = _sha256(raw)
    expected = EXPECTED_MANIFEST_SHA256.get(path.name)
    if EXPECTED_MANIFEST_SHA256 and observed != expected:
        raise ValueError("pair-merge manifest bytes differ from the packaged binding")
    manifest = json.loads(raw.decode("utf-8"))
    semantic = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("protocol_id") != PROTOCOL
        or manifest.get("artifact_sha256") != partition._semantic_sha(semantic)
        or manifest.get("server_label") != "gpu01"
        or manifest.get("pair_request_max_bytes") != 6000
        or not isinstance(manifest.get("candidate_pairs"), list)
        or not manifest["candidate_pairs"]
        or manifest.get("request_parameters", {}).get("candidate", {}).get(
            "temperature"
        )
        != 0
        or manifest.get("request_parameters", {}).get("candidate", {}).get(
            "chat_template_kwargs"
        )
        != {"enable_thinking": False}
    ):
        raise ValueError("pair-merge manifest contract mismatch")
    return manifest, observed


def _contexts(baseline: dict[str, Any], kind: str, label: str) -> list[list[str]]:
    contexts: list[list[str]] = []
    for edge in baseline["edges"]:
        matches = (
            edge["relation_type"] == label
            if kind == "relation"
            else edge["head_type"] == label or edge["tail_type"] == label
        )
        if matches:
            contexts.append(
                [edge["head_type"], edge["relation_type"], edge["tail_type"]]
            )
        if len(contexts) == 2:
            break
    return contexts


def _pair_payload(pair: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": pair["pair_id"],
        "kind": pair["kind"],
        "left": [pair["left_id"], pair["left_label"]],
        "right": [pair["right_id"], pair["right_label"]],
        "left_contexts": _contexts(baseline, pair["kind"], pair["left_label"]),
        "right_contexts": _contexts(baseline, pair["kind"], pair["right_label"]),
    }


def _request(rows: list[dict[str, Any]]) -> dict[str, Any]:
    system = (
        "Decide whether each same-kind schema-label pair is semantically "
        "interchangeable in every supplied source-edge context. Merge spelling, "
        "inflection and true synonyms. Reject antonyms, inverse relations, sibling "
        "concepts and broader/narrower or merely related meanings. Return exactly "
        "one JSON object with key decisions. Its value must contain exactly one "
        "object per pair with keys pair_id and merge, where merge is a JSON boolean. "
        "Use no explanations and no extra keys. Output JSON only."
    )
    user = json.dumps(
        {"pairs": rows},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    size = len(system.encode("utf-8")) + len(user.encode("utf-8"))
    return {
        "system_prompt": system,
        "user_prompt": user,
        "prompt_utf8_bytes": size,
        "prompt_sha256": _sha256(
            canonical_json_bytes({"system_prompt": system, "user_prompt": user})
        ),
    }


def build_pair_packets(
    manifest: dict[str, Any],
    baseline: dict[str, Any],
    *,
    request_builder: Any = _request,
) -> list[dict[str, Any]]:
    types, relations = partition.raw_vocabularies(baseline)
    labels = {"type": types, "relation": relations}
    pairs = manifest.get("candidate_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pair-merge manifest candidates are missing")
    expected_ids: set[str] = set()
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pair in pairs:
        if (
            not isinstance(pair, dict)
            or set(pair)
            != {
                "pair_id",
                "kind",
                "left_id",
                "right_id",
                "left_label",
                "right_label",
                "cosine",
            }
            or pair["kind"] not in labels
            or type(pair["left_id"]) is not int
            or type(pair["right_id"]) is not int
            or pair["left_id"] >= pair["right_id"]
            or pair["left_id"] < 0
            or pair["right_id"] >= len(labels[pair["kind"]])
            or labels[pair["kind"]][pair["left_id"]] != pair["left_label"]
            or labels[pair["kind"]][pair["right_id"]] != pair["right_label"]
            or pair["pair_id"] in expected_ids
        ):
            raise ValueError("pair-merge candidate binding is invalid")
        expected_ids.add(pair["pair_id"])
        rows.append((pair, _pair_payload(pair, baseline)))
    maximum = manifest["pair_request_max_bytes"]
    packets: list[dict[str, Any]] = []
    current: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pair, row in rows:
        trial = current + [(pair, row)]
        request = request_builder([item[1] for item in trial])
        if request["prompt_utf8_bytes"] <= maximum:
            current = trial
            continue
        if not current:
            raise TransportIncomplete("one pair-merge request exceeds byte limit")
        packets.append(
            {
                "pairs": [item[0] for item in current],
                "request": request_builder([item[1] for item in current]),
            }
        )
        current = [(pair, row)]
        if request_builder([row])["prompt_utf8_bytes"] > maximum:
            raise TransportIncomplete("one pair-merge request exceeds byte limit")
    if current:
        packets.append(
            {
                "pairs": [item[0] for item in current],
                "request": request_builder([item[1] for item in current]),
            }
        )
    return packets


def parse_pair_decisions(
    response: dict[str, Any], expected_pairs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    value = json.loads(legacy._finished_content(response))
    if not isinstance(value, dict) or set(value) != {"decisions"}:
        raise ValueError("pair-merge response schema is invalid")
    decisions = value["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(expected_pairs):
        raise ValueError("pair-merge decision coverage is invalid")
    observed: dict[str, bool] = {}
    for row in decisions:
        if (
            not isinstance(row, dict)
            or set(row) != {"pair_id", "merge"}
            or not isinstance(row["pair_id"], str)
            or type(row["merge"]) is not bool
            or row["pair_id"] in observed
        ):
            raise ValueError("pair-merge decision is invalid")
        observed[row["pair_id"]] = row["merge"]
    expected = [pair["pair_id"] for pair in expected_pairs]
    if set(observed) != set(expected):
        raise ValueError("pair-merge decision IDs differ from the denominator")
    return [{"pair_id": pair_id, "merge": observed[pair_id]} for pair_id in expected]


def close_pair_decisions(
    baseline: dict[str, Any],
    pairs: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    types, relations = partition.raw_vocabularies(baseline)
    values = {"type": types, "relation": relations}
    parents = {kind: list(range(len(labels))) for kind, labels in values.items()}

    def find(kind: str, value: int) -> int:
        while parents[kind][value] != value:
            parents[kind][value] = parents[kind][parents[kind][value]]
            value = parents[kind][value]
        return value

    def union(kind: str, left: int, right: int) -> None:
        left_root, right_root = find(kind, left), find(kind, right)
        if left_root != right_root:
            representative = min(left_root, right_root)
            parents[kind][max(left_root, right_root)] = representative

    by_id = {pair["pair_id"]: pair for pair in pairs}
    if len(by_id) != len(pairs) or [row["pair_id"] for row in decisions] != [
        pair["pair_id"] for pair in pairs
    ]:
        raise ValueError("pair-merge decisions differ from the frozen candidates")
    accepted: list[str] = []
    for decision in decisions:
        pair = by_id[decision["pair_id"]]
        if decision["merge"]:
            union(pair["kind"], pair["left_id"], pair["right_id"])
            accepted.append(pair["pair_id"])
    assignments = {
        kind: [
            {"source_id": index, "representative_id": find(kind, index)}
            for index in range(len(labels))
        ]
        for kind, labels in values.items()
    }
    type_ids = sorted({row["representative_id"] for row in assignments["type"]})
    relation_ids = sorted(
        {row["representative_id"] for row in assignments["relation"]}
    )
    audit = {
        "raw_types": list(types),
        "raw_relations": list(relations),
        "type_assignments": assignments["type"],
        "relation_assignments": assignments["relation"],
        "closed_vocabulary": True,
        "candidate_pair_count": len(pairs),
        "accepted_pair_count": len(accepted),
        "accepted_pair_ids": accepted,
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


def _costs(
    baseline: dict[str, Any], call_records: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return partition._costs(baseline, call_records)


def _execute(
    *,
    manifest: dict[str, Any],
    manifest_sha: str,
    baseline: dict[str, Any],
    baseline_sha: str,
    results_root: Path,
    client: Any,
    lease: Any,
    packet_builder: Any = build_pair_packets,
    decision_parser: Any = parse_pair_decisions,
    deterministic_projector: Any = None,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
    failure_schema: str = "wse-scope-evidence-pairmerge-technical-failure-v1",
    summary_schema: str = "wse-scope-evidence-pairmerge-summary-v1",
    method: str = "semantic blocking plus evidence-bound pair decisions",
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

    decisions: list[dict[str, Any]] = []
    try:
        packets = packet_builder(manifest, baseline)
        for index, packet in enumerate(packets, start=1):
            result = call(packet["request"], f"pair-merge-{index:04d}")
            decisions.extend(decision_parser(result.response, packet["pairs"]))
        canonical_types, canonical_relations, audit = close_pair_decisions(
            baseline, manifest["candidate_pairs"], decisions
        )
        if deterministic_projector is not None:
            edges = deterministic_projector(baseline, audit)
        else:
            mappings: list[dict[str, Any]] = []
            for index, packet in enumerate(
                legacy._mapping_packets(
                    manifest, baseline, canonical_types, canonical_relations
                ),
                start=1,
            ):
                result = call(packet["request"], f"edge-mapping-{index:04d}")
                mappings.extend(
                    legacy._parse_edge_mappings(
                        result.response,
                        packet["source_edges"],
                        canonical_types,
                        canonical_relations,
                    )
                )
            edges = legacy._project_global_mapping(baseline, tuple(mappings))
    except TransportIncomplete as error:
        failure = {
            "schema": failure_schema,
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
            "schema": summary_schema,
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
        audit = None
        edges = []
    else:
        status = "ok"
        parser_error = None

    marginal, total = _costs(baseline, call_records)
    checkpoint = {
        "schema": checkpoint_schema,
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
        "method": method,
        "status": status,
        "raw_edge_count": len(baseline["edges"]),
        "canonical_edge_count": len(edges),
        "canonical_vocabulary": {
            "types": list(canonical_types),
            "relations": list(canonical_relations),
        },
        "partition_audit": audit,
        "pair_decisions": decisions,
        "edges": edges,
        "call_records": call_records,
        "parser_error": parser_error,
        "marginal_cost": marginal,
        "total_method_cost": total,
        "actual_calls": len(call_records),
    }
    write_checkpoint_atomic(results_root / "pairmerge.json", checkpoint)
    summary = {
        "schema": summary_schema,
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


def run_pair_merge(
    manifest_path: str | Path,
    baseline_path: str | Path,
    results_root: str | Path,
    client: Any,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(Path(manifest_path))
    if manifest.get("baseline_mode") not in (None, "bound_existing"):
        raise ValueError("pair-merge manifest baseline mode is invalid")
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
        )


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--results-root", default="/results")
    args = parser.parse_args()
    value = run_pair_merge(
        args.manifest, args.baseline, args.results_root, WSEClient()
    )
    print(json.dumps(value, sort_keys=True))
    return 0 if value.get("status") == "development_complete" else 4


if __name__ == "__main__":
    raise SystemExit(_main())
