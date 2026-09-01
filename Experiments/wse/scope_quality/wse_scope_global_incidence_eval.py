"""Global incidence-guided schema induction on a fresh AutoSchemaKG baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import wse_scope_eval as legacy  # noqa: E402
import wse_scope_partition_eval as partition  # noqa: E402


CHECKPOINT_SCHEMA = "wse-scope-global-incidence-checkpoint-v1"
MANIFEST_SCHEMA = "wse-scope-global-incidence-manifest-v1"
PROTOCOL = "wse-scope-global-incidence-development-v1"
SUMMARY_SCHEMA = "wse-scope-global-incidence-summary-v1"
FAILURE_SCHEMA = "wse-scope-global-incidence-technical-failure-v1"
SMOKE_SCHEMA = "wse-scope-global-incidence-smoke-v1"
MAX_PROMPT_BYTES = 6000
MAX_DEFINITION_CHARS = 240
MAX_LABEL_CHARS = 80
MAX_INCIDENCE_ITEMS = 24
MAX_EVIDENCE_DOCUMENTS = 2
MAX_EVIDENCE_CHARS = 360
MAX_PEERS = 6
MDL_CONTRACT: dict[str, int | str] = {
    "schema": "deterministic-local-cluster-mdl-v1",
    "label_overhead_bits": 64,
    "assignment_bits_per_member": 8,
    "exception_bits_per_distinct_incidence": 8,
    "decision": "accept_multi_member_cluster_only_when_candidate_bits_lt_singleton_bits",
}
SUPPORTED_MODEL_ASSIGNMENTS = {
    ("gpu01", "Qwen/Qwen3-14B"),
    ("h200", "Qwen/Qwen3.6-35B-A3B"),
}
DEFINITION_SYSTEM_PROMPT = (
    "Induce globally reusable schema labels from the supplied raw labels, corpus-level "
    "directed incidence signatures, peer labels and evidence. Return exactly one JSON "
    "object with key definitions. Include exactly one row per label_id with label_id, "
    "canonical_label, definition and reverse. canonical_label must be concise lowercase "
    "English. Equivalent labels should use exactly the same canonical_label. Preserve "
    "head-tail meaning; reverse may be true only for relation labels when the chosen "
    "canonical predicate has the opposite direction. A canonical label may be new only "
    "when the supplied incidence and evidence support it. Do not invent facts, omit "
    "source labels or use evaluator knowledge. Output JSON only."
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _label_id(kind: str, raw_label: str) -> str:
    return _sha256_bytes(canonical_json_bytes({"kind": kind, "raw_label": raw_label}))


def source_edges_by_id(baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [legacy._source_edge(edge) for edge in baseline["edges"]]
    observed = {row["source_edge_id"]: row for row in rows}
    if len(observed) != len(rows):
        raise ValueError("global-incidence baseline source-edge IDs are not unique")
    return observed


def _normal_label(value: Any) -> str:
    if type(value) is not str:
        raise ValueError("global-incidence label is not a string")
    normal = " ".join(value.split())
    if (
        value != normal
        or not normal
        or normal != normal.lower()
        or len(normal) > MAX_LABEL_CHARS
        or any(ord(character) < 32 for character in normal)
    ):
        raise ValueError("global-incidence label is invalid")
    return normal


def _label_inventory(
    manifest: dict[str, Any], baseline: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    documents = {row["doc_id"]: row for row in manifest["documents"]}
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}

    def observe(
        kind: str,
        raw_label: str,
        incidence: str,
        roles: str,
        evidence_doc_ids: list[str],
    ) -> None:
        key = (kind, raw_label)
        row = aggregate.setdefault(
            key,
            {
                "kind": kind,
                "raw_label": raw_label,
                "incidence": set(),
                "roles": set(),
                "evidence_doc_ids": set(),
                "occurrence_count": 0,
            },
        )
        row["incidence"].add(incidence)
        row["roles"].add(roles)
        row["evidence_doc_ids"].update(evidence_doc_ids)
        row["occurrence_count"] += 1

    for source in source_edges_by_id(baseline).values():
        evidence_ids = source["evidence_doc_ids"]
        if any(doc_id not in documents for doc_id in evidence_ids):
            raise ValueError("global-incidence evidence lies outside the manifest")
        head = source["head_type"]
        relation = source["relation_type"]
        tail = source["tail_type"]
        observe("type", head, f"head:{relation}->{tail}", "head", evidence_ids)
        observe("type", tail, f"tail:{relation}<-{head}", "tail", evidence_ids)
        observe("relation", relation, f"edge:{head}->{tail}", "directed", evidence_ids)

    rows: list[dict[str, Any]] = []
    for (kind, raw_label), value in sorted(aggregate.items()):
        incidence = sorted(value["incidence"])
        evidence_ids = sorted(value["evidence_doc_ids"])
        rows.append(
            {
                "label_id": _label_id(kind, raw_label),
                "kind": kind,
                "raw_label": raw_label,
                "roles": sorted(value["roles"]),
                "occurrence_count": value["occurrence_count"],
                "incidence_count": len(incidence),
                "incidence_sha256": _sha256_bytes(canonical_json_bytes(incidence)),
                "incidence": incidence[:MAX_INCIDENCE_ITEMS],
                "evidence_doc_ids": evidence_ids,
                "evidence": [
                    {
                        "doc_id": doc_id,
                        "text": documents[doc_id]["text"][:MAX_EVIDENCE_CHARS],
                    }
                    for doc_id in evidence_ids[:MAX_EVIDENCE_DOCUMENTS]
                ],
            }
        )

    for row in rows:
        peers = [candidate for candidate in rows if candidate["kind"] == row["kind"] and candidate["label_id"] != row["label_id"]]
        row_tokens = set(re.findall(r"[a-z0-9]+", row["raw_label"].lower()))
        row_incidence = set(row["incidence"])
        ranked = sorted(
            peers,
            key=lambda candidate: (
                -len(row_incidence.intersection(candidate["incidence"])),
                -len(
                    row_tokens.intersection(
                        re.findall(r"[a-z0-9]+", candidate["raw_label"].lower())
                    )
                ),
                candidate["raw_label"],
            ),
        )
        row["peer_labels"] = [
            candidate["raw_label"] for candidate in ranked[:MAX_PEERS]
        ]
    return tuple(rows)


def _response_format(label_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "global_incidence_definitions",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["definitions"],
                "properties": {
                    "definitions": {
                        "type": "array",
                        "minItems": len(label_ids),
                        "maxItems": len(label_ids),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "label_id",
                                "canonical_label",
                                "definition",
                                "reverse",
                            ],
                            "properties": {
                                "label_id": {"type": "string", "enum": list(label_ids)},
                                "canonical_label": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": MAX_LABEL_CHARS,
                                },
                                "definition": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": MAX_DEFINITION_CHARS,
                                },
                                "reverse": {"type": "boolean"},
                            },
                        },
                    }
                },
            },
        },
    }


def _request(rows: list[dict[str, Any]], inventory_sha256: str) -> dict[str, Any]:
    payload = {
        "global_inventory_sha256": inventory_sha256,
        "labels": rows,
    }
    user_prompt = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    response_format = _response_format(tuple(row["label_id"] for row in rows))
    prompt_bytes = len(DEFINITION_SYSTEM_PROMPT.encode("utf-8")) + len(
        user_prompt.encode("utf-8")
    )
    binding = {
        "system_prompt": DEFINITION_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "response_format": response_format,
    }
    return {
        "system_prompt": DEFINITION_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "prompt_utf8_bytes": prompt_bytes,
        "prompt_sha256": _sha256_bytes(canonical_json_bytes(binding)),
        "response_format": response_format,
        "response_format_sha256": _sha256_bytes(canonical_json_bytes(response_format)),
    }


def build_definition_packets(
    manifest: dict[str, Any], baseline: dict[str, Any], *, label_limit: int | None = None
) -> tuple[dict[str, Any], ...]:
    inventory = list(_label_inventory(manifest, baseline))
    if label_limit is not None:
        if label_limit < 1:
            raise ValueError("global-incidence label limit is invalid")
        inventory = inventory[:label_limit]
    inventory_sha = _sha256_bytes(
        canonical_json_bytes(
            [
                {key: row[key] for key in ("label_id", "kind", "raw_label", "incidence_sha256")}
                for row in inventory
            ]
        )
    )
    maximum = manifest["definition_contract"]["maximum_prompt_utf8_bytes"]
    packets: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def packet(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {"labels": tuple(rows), "request": _request(rows, inventory_sha)}

    for row in inventory:
        trial = packet([*current, row])
        if trial["request"]["prompt_utf8_bytes"] <= maximum:
            current.append(row)
            continue
        if not current:
            raise TransportIncomplete("one global-incidence label exceeds 6 KB")
        packets.append(packet(current))
        current = [row]
        if packet(current)["request"]["prompt_utf8_bytes"] > maximum:
            raise TransportIncomplete("one global-incidence label exceeds 6 KB")
    if current:
        packets.append(packet(current))
    return tuple(packets)


def parse_definitions(
    response: dict[str, Any], labels: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    value = json.loads(legacy._finished_content(response))
    if not isinstance(value, dict) or set(value) != {"definitions"}:
        raise ValueError("global-incidence response schema is invalid")
    expected = {row["label_id"]: row for row in labels}
    observed: dict[str, dict[str, Any]] = {}
    for row in value["definitions"] if isinstance(value["definitions"], list) else ():
        if (
            not isinstance(row, dict)
            or set(row) != {"label_id", "canonical_label", "definition", "reverse"}
            or row.get("label_id") not in expected
            or row["label_id"] in observed
            or type(row.get("definition")) is not str
            or not row["definition"].strip()
            or row["definition"] != " ".join(row["definition"].split())
            or len(row["definition"]) > MAX_DEFINITION_CHARS
            or type(row.get("reverse")) is not bool
            or (expected[row["label_id"]]["kind"] == "type" and row["reverse"])
        ):
            raise ValueError("global-incidence definition binding is invalid")
        observed[row["label_id"]] = {
            "label_id": row["label_id"],
            "kind": expected[row["label_id"]]["kind"],
            "raw_label": expected[row["label_id"]]["raw_label"],
            "canonical_label": _normal_label(row["canonical_label"]),
            "definition": row["definition"],
            "reverse": row["reverse"],
            "incidence_sha256": expected[row["label_id"]]["incidence_sha256"],
        }
    if set(observed) != set(expected):
        raise ValueError("global-incidence definitions are incomplete")
    return tuple(observed[label_id] for label_id in sorted(observed))


def _mdl_select(
    inventory: tuple[dict[str, Any], ...], definitions: tuple[dict[str, Any], ...]
) -> tuple[tuple[dict[str, Any], ...], list[dict[str, Any]]]:
    inventory_by_id = {row["label_id"]: row for row in inventory}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in definitions:
        groups.setdefault((row["kind"], row["canonical_label"]), []).append(row)
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    overhead = int(MDL_CONTRACT["label_overhead_bits"])
    assignment = int(MDL_CONTRACT["assignment_bits_per_member"])
    exception_unit = int(MDL_CONTRACT["exception_bits_per_distinct_incidence"])
    for (kind, canonical_label), members in sorted(groups.items()):
        members = sorted(members, key=lambda row: row["label_id"])
        singleton_bits = sum(
            overhead + 8 * len(row["raw_label"].encode("utf-8")) for row in members
        )
        distinct_incidence = len({row["incidence_sha256"] for row in members})
        exception_count = max(0, distinct_incidence - 1)
        candidate_bits = (
            overhead
            + 8 * len(canonical_label.encode("utf-8"))
            + assignment * len(members)
            + exception_unit * exception_count
        )
        accepted = len(members) > 1 and candidate_bits < singleton_bits
        audit.append(
            {
                "kind": kind,
                "proposed_canonical_label": canonical_label,
                "member_label_ids": [row["label_id"] for row in members],
                "singleton_bits": singleton_bits,
                "candidate_bits": candidate_bits,
                "exception_count": exception_count,
                "accepted": accepted,
            }
        )
        for row in members:
            raw = inventory_by_id[row["label_id"]]["raw_label"]
            selected.append(
                {
                    "label_id": row["label_id"],
                    "kind": kind,
                    "raw_label": raw,
                    "selected_label": canonical_label if accepted else raw,
                    "reverse": row["reverse"] if accepted else False,
                }
            )
    return tuple(sorted(selected, key=lambda row: row["label_id"])), audit


def _project(
    baseline: dict[str, Any], selected: tuple[dict[str, Any], ...]
) -> list[dict[str, Any]]:
    by_id = {row["label_id"]: row for row in selected}
    mappings: list[dict[str, Any]] = []
    for source_id, source in sorted(source_edges_by_id(baseline).items()):
        head = by_id[_label_id("type", source["head_type"])]
        relation = by_id[_label_id("relation", source["relation_type"])]
        tail = by_id[_label_id("type", source["tail_type"])]
        mappings.append(
            {
                "source_edge_id": source_id,
                "head_type": head["selected_label"],
                "relation_type": relation["selected_label"],
                "tail_type": tail["selected_label"],
                "reverse": relation["reverse"],
            }
        )
    return legacy._project_global_mapping(baseline, tuple(mappings))


def _costs(
    baseline: dict[str, Any], records: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return partition._costs(baseline, records)


def _execute_candidate(
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
    inventory = _label_inventory(manifest, baseline)
    packets = build_definition_packets(
        manifest, baseline, label_limit=1 if smoke else None
    )
    records: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
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
            records.append(
                {
                    "stage": f"global-incidence-definition-{index:04d}",
                    "prompt_sha256": request["prompt_sha256"],
                    "prompt_utf8_bytes": request["prompt_utf8_bytes"],
                    "response_format_sha256": request["response_format_sha256"],
                    "response_sha256": _sha256_bytes(canonical_json_bytes(result.response)),
                    "response": result.response,
                    "usage": legacy._usage(result.response),
                    "client_elapsed_seconds": result.client_elapsed_seconds,
                    "retry_wait_seconds": result.retry_wait_seconds,
                }
            )
            definitions.extend(parse_definitions(result.response, packet["labels"]))
    except (json.JSONDecodeError, ValueError) as error:
        parser_error = str(error)

    if smoke:
        value = {
            "schema": SMOKE_SCHEMA,
            "status": "smoke_complete" if parser_error is None else "smoke_parse_failure",
            "manifest_sha256": manifest_sha,
            "source_baseline_sha256": baseline_sha,
            "fixed_denominator": 1,
            "received_calls": len(records),
            "definition": definitions[0] if len(definitions) == 1 else None,
            "call_records": records,
            "parser_error": parser_error,
            "claim_guard": "Technical one-label smoke only; no quality evidence.",
        }
        write_json_atomic(results_root / "smoke.json", value)
        return value

    status = "ok" if parser_error is None else "parse_error"
    if status == "ok":
        selected, mdl_audit = _mdl_select(inventory, tuple(definitions))
        edges = _project(baseline, selected)
    else:
        selected, mdl_audit, edges = (), [], []
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
        "status": status,
        "fixed_denominator_label_count": len(inventory),
        "fixed_denominator_source_edge_count": len(baseline["edges"]),
        "label_inventory_sha256": _sha256_bytes(canonical_json_bytes(inventory)),
        "definitions": definitions,
        "selected_label_mappings": list(selected),
        "mdl_audit": mdl_audit,
        "edges": edges,
        "call_records": records,
        "actual_remote_calls_this_run": len(records),
        "parser_error": parser_error,
        "marginal_cost": marginal,
        "total_method_cost": total,
    }
    write_checkpoint_atomic(results_root / "global_incidence.json", checkpoint)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "development_complete" if status == "ok" else "development_parse_failure",
        "runtime_complete": status == "ok",
        "quality_evaluated": False,
        "fixed_denominator_label_count": len(inventory),
        "fixed_denominator_source_edge_count": len(baseline["edges"]),
        "candidate_edge_count": len(edges),
        "actual_remote_calls_this_run": len(records),
        "marginal_cost": marginal,
        "total_method_cost": total,
        "claim_guard": manifest["claim_guard"],
    }
    write_json_atomic(results_root / "summary.json", summary)
    return summary


def _load_manifest(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    observed = _sha256_bytes(raw)
    if observed != expected_sha256:
        raise ValueError("global-incidence manifest bytes differ from binding")
    manifest = json.loads(raw.decode("utf-8"))
    semantic = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    contract = manifest.get("definition_contract", {})
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("protocol_id") != PROTOCOL
        or manifest.get("artifact_sha256") != partition._semantic_sha(semantic)
        or (manifest.get("server_label"), manifest.get("model"))
        not in SUPPORTED_MODEL_ASSIGNMENTS
        or contract.get("maximum_prompt_utf8_bytes") != MAX_PROMPT_BYTES
        or contract.get("maximum_definition_characters") != MAX_DEFINITION_CHARS
        or contract.get("gold_access") is not False
        or manifest.get("mdl_contract") != MDL_CONTRACT
        or manifest.get("request_parameters", {}).get("candidate", {}).get("temperature") != 0
        or manifest.get("request_parameters", {}).get("candidate", {}).get("chat_template_kwargs") != {"enable_thinking": False}
    ):
        raise ValueError("global-incidence manifest contract mismatch")
    return manifest, observed


def run_global_incidence(
    manifest_path: str | Path,
    results_root: str | Path,
    client: Any,
    *,
    expected_manifest_sha256: str,
    smoke: bool = False,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(Path(manifest_path), expected_manifest_sha256)
    legacy._validate_live_model(manifest, list(client.get_models()))
    root = Path(results_root)
    runtime_manifest = manifest
    if smoke:
        runtime_manifest = json.loads(json.dumps(manifest))
        runtime_manifest["documents"] = runtime_manifest["documents"][:1]
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        baseline, _ = legacy._run_autoschemakg_baseline(
            client,
            lease,
            runtime_manifest,
            manifest_sha,
        )
        baseline_path = root / ("smoke_baseline.json" if smoke else "checkpoints/autoschemakg_baseline.json")
        write_checkpoint_atomic(baseline_path, baseline)
        if baseline["status"] != "ok":
            summary = {
                "schema": SUMMARY_SCHEMA,
                "status": "baseline_incomplete",
                "runtime_complete": False,
                "quality_evaluated": False,
                "baseline_status": baseline["status"],
                "claim_guard": manifest["claim_guard"],
            }
            write_json_atomic(root / "summary.json", summary)
            return summary
        baseline_sha = _sha256_bytes(baseline_path.read_bytes())
        return _execute_candidate(
            manifest=runtime_manifest,
            manifest_sha=manifest_sha,
            baseline=baseline,
            baseline_sha=baseline_sha,
            results_root=root,
            client=client,
            lease=lease,
            smoke=smoke,
        )


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--results-root", default="/results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        value = run_global_incidence(
            args.manifest,
            args.results_root,
            WSEClient(),
            expected_manifest_sha256=args.manifest_sha256,
            smoke=args.smoke,
        )
    except TransportIncomplete as error:
        value = {
            "schema": FAILURE_SCHEMA,
            "status": "technical_failure",
            "technical_error": str(error),
            "claim_guard": "Infrastructure or request-bound failure; no quality evidence.",
        }
        write_json_atomic(Path(args.results_root) / "technical_failure.json", value)
        print(json.dumps(value, sort_keys=True))
        return 4
    print(json.dumps(value, sort_keys=True))
    return 0 if value["status"] in {"development_complete", "smoke_complete"} else 3


if __name__ == "__main__":
    raise SystemExit(_main())
