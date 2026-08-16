#!/usr/bin/env python3
"""WSE runtime for the held-out SCOPE comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parents[1] / "src" if len(HERE.parents) > 1 else HERE
for candidate in (HERE, HERE.parent / "triple_quality", REPO_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pipelines  # noqa: E402
from scope_ablation import (  # noqa: E402
    apply_frozen_candidate_miner,
    CandidateEvidence,
    CandidateMinerConfig,
    validate_candidate_edges,
)
from scope_prompts import (  # noqa: E402
    DirectEdge,
    build_direct_request,
    build_frozen_ablation_request,
    parse_direct_typed_edges,
)
from scope_protocol import TypedSchemaEdge  # noqa: E402
from scope_adapter import group_triples_with_provenance, source_global_edges  # noqa: E402
from wse_eval import (  # noqa: E402
    ChatResult,
    TransportIncomplete,
    WSEClient,
    canonical_json_bytes,
    write_checkpoint_atomic,
    write_json_atomic,
)


EXPECTED_MANIFEST_SHA256 = {}
ARMS = ("autoschemakg_baseline", "candidate_evidence_projected")
DEVELOPMENT_BASELINE_SHA256 = (
    "b21147cb7638e2409cbd2bdb818e9f72282a5aaad708a45d71f63ba3c6988fea"
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _message(response: dict[str, Any]) -> dict[str, Any] | None:
    try:
        value = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _edge_json(row: DirectEdge) -> dict[str, Any]:
    return {
        "head_type": row.edge.head_type,
        "relation_type": row.edge.relation_type,
        "tail_type": row.edge.tail_type,
        "evidence_doc_ids": list(row.evidence_doc_ids),
    }


def _candidate_json(row: CandidateEvidence) -> dict[str, Any]:
    return {
        "candidate_id": row.candidate_id,
        "edge": asdict(row.edge),
        "evidence_doc_ids": list(row.evidence_doc_ids),
        "evidence_offsets": [list(span) for span in row.evidence_offsets],
        "evidence_sha256": list(row.evidence_sha256),
    }


def _candidate_packets(
    manifest: dict[str, Any],
    documents: list[dict[str, Any]],
    candidates: tuple[CandidateEvidence, ...],
) -> tuple[dict[str, Any], ...]:
    contract = manifest["candidate_packet_contract"]
    maximum_bytes = contract["maximum_prompt_utf8_bytes"]
    document_by_id = {row["doc_id"]: row for row in documents}

    def build(rows: list[CandidateEvidence]) -> dict[str, Any]:
        evidence_ids = {
            doc_id for candidate in rows for doc_id in candidate.evidence_doc_ids
        }
        selected_documents = [
            document for document in documents if document["doc_id"] in evidence_ids
        ]
        if len(selected_documents) != len(evidence_ids):
            raise TransportIncomplete("candidate evidence document is missing")
        request = build_frozen_ablation_request(
            "candidate_evidence",
            selected_documents,
            choice_sha256=manifest["development_freeze"]["choice_sha256"],
            candidates=[_candidate_json(row) for row in rows],
        )
        prompt_bytes = len(request["system_prompt"].encode("utf-8")) + len(
            request["user_prompt"].encode("utf-8")
        )
        return {
            "request": request,
            "candidates": tuple(rows),
            "prompt_utf8_bytes": prompt_bytes,
        }

    packets: list[dict[str, Any]] = []
    current: list[CandidateEvidence] = []
    for candidate in sorted(candidates, key=lambda row: row.candidate_id):
        if any(doc_id not in document_by_id for doc_id in candidate.evidence_doc_ids):
            raise TransportIncomplete("candidate evidence lies outside the manifest")
        trial = build([*current, candidate])
        if trial["prompt_utf8_bytes"] <= maximum_bytes:
            current.append(candidate)
            continue
        if not current:
            raise TransportIncomplete("one candidate evidence packet exceeds the bound")
        packets.append(build(current))
        current = [candidate]
        if build(current)["prompt_utf8_bytes"] > maximum_bytes:
            raise TransportIncomplete("one candidate evidence packet exceeds the bound")
    if current:
        packets.append(build(current))
    return tuple(packets)


def _run_candidate_projection(
    client: Any,
    lease: Any,
    manifest: dict[str, Any],
    manifest_sha: str,
    documents: list[dict[str, Any]],
    candidates: tuple[CandidateEvidence, ...],
) -> dict[str, Any]:
    packets = _candidate_packets(manifest, documents, candidates)
    packet_records: list[dict[str, Any]] = []
    accepted_edges: set[DirectEdge] = set()
    statuses: list[str] = []
    for packet_index, packet in enumerate(packets):
        request = packet["request"]
        packet_candidates = packet["candidates"]
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
        status, edges, parser_errors, violations = _parse_received(
            result.response,
            request["document_ids"],
            candidates=packet_candidates,
        )
        statuses.append(status)
        accepted_edges.update(edges)
        packet_records.append(
            {
                "packet_index": packet_index,
                "candidate_ids": [row.candidate_id for row in packet_candidates],
                "document_ids": request["document_ids"],
                "prompt_sha256": request["prompt_sha256"],
                "prompt_utf8_bytes": packet["prompt_utf8_bytes"],
                "response_sha256": _sha256(canonical_json_bytes(result.response)),
                "response": result.response,
                "status": status,
                "parser_errors": list(parser_errors),
                "candidate_violations": list(violations),
                "usage": _usage(result.response),
                "client_elapsed_seconds": result.client_elapsed_seconds,
                "retry_wait_seconds": result.retry_wait_seconds,
            }
        )

    invalid = next(
        (status for status in statuses if status not in {"ok", "valid_empty"}),
        None,
    )
    if invalid is not None:
        accepted_edges.clear()
        status = invalid
    else:
        status = "ok" if accepted_edges else "valid_empty"
    usage = {
        "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in packet_records),
        "completion_tokens": sum(
            row["usage"]["completion_tokens"] for row in packet_records
        ),
    }
    return {
        "schema": "wse-scope-checkpoint-v1",
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": manifest_sha,
        "server_label": manifest["server_label"],
        "model": manifest["model"],
        "arm": ARMS[1],
        "status": status,
        "edges": [_edge_json(row) for row in sorted(accepted_edges)],
        "candidate_count": len(candidates),
        "packet_contract": manifest["candidate_packet_contract"],
        "packet_records": packet_records,
        "usage": usage,
        "client_elapsed_seconds": sum(
            row["client_elapsed_seconds"] for row in packet_records
        ),
        "retry_wait_seconds": sum(row["retry_wait_seconds"] for row in packet_records),
        "actual_calls": len(packet_records),
    }


def _parse_received(
    response: dict[str, Any],
    allowed_document_ids: Sequence[str],
    *,
    candidates: Sequence[CandidateEvidence] | None = None,
) -> tuple[str, tuple[DirectEdge, ...], tuple[str, ...], tuple[str, ...]]:
    message = _message(response)
    if message is None:
        return "parse_error", (), ("manager-message-invalid",), ()
    refusal = message.get("refusal")
    content = message.get("content")
    if (isinstance(refusal, str) and refusal.strip()) or not isinstance(content, str) or not content.strip():
        return "refusal", (), (), ()
    parsed = parse_direct_typed_edges(
        content, allowed_document_ids=allowed_document_ids
    )
    if not parsed.valid:
        return "parse_error", (), parsed.parser_errors, ()
    violations = validate_candidate_edges(parsed, candidates) if candidates is not None else ()
    if violations:
        return "evidence_rejected", (), (), violations
    return ("ok" if parsed.edges else "valid_empty"), parsed.edges, (), ()


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    observed = _sha256(raw)
    expected = EXPECTED_MANIFEST_SHA256.get(path.name)
    if EXPECTED_MANIFEST_SHA256 and expected != observed:
        raise ValueError("manifest bytes differ from the packaged binding")
    manifest = json.loads(raw.decode("utf-8"))
    if (
        manifest.get("schema") != "wse-scope-quality-manifest-v1"
        or manifest.get("arm_order") != list(ARMS)
        or manifest.get("method_contract", {}).get("fixed_arm_denominator") != 2
        or any(
            row.get("chat_template_kwargs") != {"enable_thinking": False}
            or row.get("temperature") != 0
            for row in manifest.get("request_parameters", {}).values()
        )
    ):
        raise ValueError("SCOPE manifest contract mismatch")
    return manifest, observed


def _validate_live_model(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    current_models = sorted(
        ({"name": row.get("name") or row.get("id")} for row in rows),
        key=lambda row: str(row["name"]),
    )
    if (
        any(
            not isinstance(row["name"], str) or not row["name"]
            for row in current_models
        )
    ):
        raise ValueError("live WSE model catalog is invalid")
    available = [
        row
        for row in rows
        if (row.get("name") or row.get("id")) == manifest["model"]
    ]
    if len(available) != 1:
        raise ValueError("manifest model is not uniquely available")
    selected = {"name": manifest["model"]}
    if _sha256(canonical_json_bytes(selected)) != manifest["catalog_binding"][
        "selected_record_sha256"
    ]:
        raise ValueError("live WSE selected model differs from frozen binding")
    manager_context = available[0].get(
        "context_length", available[0].get("max_model_len")
    )
    if manager_context is not None and int(manager_context) < int(
        manifest["context_length"]
    ):
        raise ValueError("manager model context contradicts the frozen manifest")


def _response_parts(response: dict[str, Any]) -> tuple[str, str | None, str | None]:
    message = _message(response)
    content = message.get("content") if message is not None else None
    refusal = message.get("refusal") if message is not None else None
    finish = None
    try:
        finish = response["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    return (
        content if isinstance(content, str) else "",
        refusal if isinstance(refusal, str) and refusal.strip() else None,
        finish if isinstance(finish, str) else None,
    )


def _usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("manager response has no usage evidence")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if type(prompt) is not int or prompt < 0 or type(completion) is not int or completion < 0:
        raise ValueError("manager usage evidence is invalid")
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def _concept_record(
    row: Any,
    *,
    source_dataset: str,
    concepts: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    if row.triple_type == "EE":
        subject_kind, object_kind = "entity", "entity"
    elif row.triple_type == "EV":
        subject_kind, object_kind = "event", "entity"
    else:
        subject_kind, object_kind = "event", "event"
    return {
        "source_dataset": source_dataset,
        "subject": row.subject,
        "predicate": row.predicate,
        "object": row.object,
        "subject_concept_types": list(concepts[subject_kind].get(row.subject, ())),
        "predicate_concept_types": list(concepts["relation"].get(row.predicate, ())),
        "object_concept_types": list(concepts[object_kind].get(row.object, ())),
    }


def _run_autoschemakg_baseline(
    client: Any,
    lease: Any,
    manifest: dict[str, Any],
    manifest_sha: str,
) -> tuple[dict[str, Any], tuple[DirectEdge, ...]]:
    model = manifest["model"]
    source_dataset = "instructIE_en"
    extraction_parameters = manifest["request_parameters"]["extraction"]
    schema_parameters = manifest["request_parameters"]["schema"]
    calls: list[dict[str, Any]] = []
    observed_statuses: list[str] = []
    triples_by_document: dict[str, list[dict[str, str]]] = {}

    def call(
        *,
        system_prompt: str,
        user_prompt: str,
        parameters: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[ChatResult, str, str | None, str | None]:
        lease.assert_healthy()
        result = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            parameters=parameters,
            lease=lease,
        )
        lease.assert_healthy()
        content, refusal, finish = _response_parts(result.response)
        calls.append(
            {
                **metadata,
                "request_sha256": _sha256(
                    canonical_json_bytes(
                        {"system_prompt": system_prompt, "user_prompt": user_prompt}
                    )
                ),
                "response_sha256": _sha256(canonical_json_bytes(result.response)),
                "response": result.response,
                "usage": _usage(result.response),
                "client_elapsed_seconds": result.client_elapsed_seconds,
                "retry_wait_seconds": result.retry_wait_seconds,
            }
        )
        return result, content, refusal, finish

    for document in manifest["documents"]:
        doc_id = document["doc_id"]
        text = document["text"]
        triples_by_document[doc_id] = []
        requests = pipelines.build_autoschemakg_extraction_requests(model, text)
        for stage in ("EE", "EV", "VV"):
            request = requests[stage]
            _result, content, refusal, finish = call(
                system_prompt=request["system_prompt"],
                user_prompt=request["user_prompt"],
                parameters=extraction_parameters,
                metadata={"phase": "extraction", "document_id": doc_id, "stage": stage},
            )
            parsed = pipelines.parse_autoschemakg_stage_response(
                content,
                stage,
                refusal=refusal,
                finish_reason=finish,
            )
            observed_statuses.append(parsed["status"])
            for row in parsed["triples"]:
                triples_by_document[doc_id].append(
                    {
                        "triple_type": row["triple_type"],
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "batch_id": doc_id,
                        "chunk_id": doc_id,
                        "evidence_sha256": document["text_sha256"],
                    }
                )

    all_triples = [row for rows in triples_by_document.values() for row in rows]
    entities, events, relations, contexts = pipelines._collect_autoschemakg_schema_inputs(
        all_triples
    )
    concepts: dict[str, dict[str, list[str]]] = {
        "entity": {},
        "event": {},
        "relation": {},
    }
    for element_type, elements in (
        ("entity", sorted(entities)),
        ("event", sorted(events)),
        ("relation", sorted(relations)),
    ):
        for element in elements:
            request = pipelines.build_autoschemakg_schema_request(
                element,
                element_type,
                context=contexts.get(element, "") if element_type == "entity" else "",
            )
            _result, content, refusal, _finish = call(
                system_prompt=request["system_prompt"],
                user_prompt=request["user_prompt"],
                parameters=schema_parameters,
                metadata={
                    "phase": "schema-induction",
                    "element_type": element_type,
                    "element_sha256": _sha256(element.encode("utf-8")),
                },
            )
            phrases = [] if refusal else pipelines.parse_autoschemakg_schema_phrases(content)
            observed_statuses.append("ok" if phrases else ("refusal" if refusal else "parse_error"))
            concepts[element_type][element] = phrases

    saved_documents: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for doc_id in sorted(triples_by_document):
        triples = triples_by_document[doc_id]
        rows = [
            {**row, "source_dataset": source_dataset, "document_id": doc_id}
            for row in triples
        ]
        grouped = group_triples_with_provenance(
            rows, expected_source_dataset=source_dataset
        )
        saved_documents.append(
            {
                "document_id": doc_id,
                "triples": triples,
                "concept_rows": [
                    _concept_record(row, source_dataset=source_dataset, concepts=concepts)
                    for row in grouped
                ],
            }
        )
        all_rows.extend(rows)
    source_grouped = group_triples_with_provenance(
        all_rows, expected_source_dataset=source_dataset
    )
    saved_response = {
        "source_dataset": source_dataset,
        "documents": saved_documents,
        "source_concept_rows": [
            _concept_record(row, source_dataset=source_dataset, concepts=concepts)
            for row in source_grouped
        ],
    }
    projected = source_global_edges(saved_response, source_dataset=source_dataset)
    edges = tuple(
        DirectEdge(row.edge, row.document_ids) for row in projected
    )
    if any(status == "refusal" for status in observed_statuses):
        status = "refusal"
    elif any(status == "parse_error" for status in observed_statuses):
        status = "parse_error"
    else:
        status = "ok" if edges else "valid_empty"
    usage = {
        "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in calls),
        "completion_tokens": sum(row["usage"]["completion_tokens"] for row in calls),
    }
    checkpoint = {
        "schema": "wse-scope-checkpoint-v1",
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": manifest_sha,
        "server_label": manifest["server_label"],
        "model": model,
        "arm": ARMS[0],
        "status": status,
        "edges": [_edge_json(row) for row in edges],
        "usage": usage,
        "client_elapsed_seconds": sum(row["client_elapsed_seconds"] for row in calls),
        "retry_wait_seconds": sum(row["retry_wait_seconds"] for row in calls),
        "actual_calls": len(calls),
        "call_records": calls,
    }
    return checkpoint, edges


def run_scope(
    manifest_path: str | Path,
    results_root: str | Path,
    client: Any,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(Path(manifest_path))
    _validate_live_model(manifest, client.get_models())
    root = Path(results_root)
    checkpoint_root = root / "checkpoints"
    documents = [
        {key: row[key] for key in ("doc_id", "language", "source_dataset", "text")}
        for row in manifest["documents"]
    ]
    checkpoints: list[dict[str, Any]] = []
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        baseline, baseline_edges = _run_autoschemakg_baseline(
            client, lease, manifest, manifest_sha
        )
        write_checkpoint_atomic(checkpoint_root / f"{ARMS[0]}.json", baseline)
        checkpoints.append(baseline)
        candidates = apply_frozen_candidate_miner(
            documents,
            baseline_edges,
            config=CandidateMinerConfig(),
            development_candidate_contract=manifest["candidate_miner_contract"],
        )
        candidate = _run_candidate_projection(
            client,
            lease,
            manifest,
            manifest_sha,
            documents,
            candidates,
        )
        write_checkpoint_atomic(checkpoint_root / f"{ARMS[1]}.json", candidate)
        checkpoints.append(candidate)
    summary = {
        "schema": "wse-scope-runtime-summary-v1",
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": manifest_sha,
        "server_label": manifest["server_label"],
        "model": manifest["model"],
        "scientific_complete": True,
        "fixed_denominator": 2,
        "arm_statuses": {row["arm"]: row["status"] for row in checkpoints},
        "actual_calls": sum(row["actual_calls"] for row in checkpoints),
        "total_client_elapsed_seconds": sum(
            row["client_elapsed_seconds"] for row in checkpoints
        ),
        "total_prompt_tokens": sum(
            row.get("usage", {}).get("prompt_tokens", 0) for row in checkpoints
        ),
        "total_completion_tokens": sum(
            row.get("usage", {}).get("completion_tokens", 0) for row in checkpoints
        ),
    }
    write_json_atomic(root / "summary.json", summary)
    return summary


def _load_development_baseline(
    path: Path,
    manifest: dict[str, Any],
    manifest_sha: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    observed = _sha256(raw)
    if observed != expected_sha256:
        raise ValueError("development baseline bytes differ from the frozen binding")
    baseline = json.loads(raw.decode("utf-8"))
    allowed_documents = {row["doc_id"] for row in manifest["documents"]}
    edges = baseline.get("edges")
    if (
        baseline.get("schema") != "wse-scope-checkpoint-v1"
        or baseline.get("arm") != ARMS[0]
        or baseline.get("status") != "ok"
        or baseline.get("manifest_sha256") != manifest_sha
        or baseline.get("server_label") != manifest["server_label"]
        or baseline.get("model") != manifest["model"]
        or not isinstance(edges, list)
        or not edges
        or type(baseline.get("actual_calls")) is not int
        or baseline["actual_calls"] < 1
        or not isinstance(baseline.get("usage"), dict)
        or any(
            type(baseline["usage"].get(key)) is not int
            or baseline["usage"][key] < 0
            for key in ("prompt_tokens", "completion_tokens")
        )
        or type(baseline.get("client_elapsed_seconds")) not in (int, float)
        or baseline["client_elapsed_seconds"] < 0
        or type(baseline.get("retry_wait_seconds")) not in (int, float)
        or baseline["retry_wait_seconds"] < 0
    ):
        raise ValueError("development baseline contract mismatch")
    for edge in edges:
        if (
            not isinstance(edge, dict)
            or set(edge)
            != {"head_type", "relation_type", "tail_type", "evidence_doc_ids"}
            or any(
                not isinstance(edge[key], str) or not edge[key].strip()
                for key in ("head_type", "relation_type", "tail_type")
            )
            or not isinstance(edge["evidence_doc_ids"], list)
            or not edge["evidence_doc_ids"]
            or any(
                not isinstance(doc_id, str) or doc_id not in allowed_documents
                for doc_id in edge["evidence_doc_ids"]
            )
        ):
            raise ValueError("development baseline edge is invalid")
    return baseline, observed


def _canonicalization_request(
    manifest: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    raw_types = sorted(
        {
            edge[key]
            for edge in baseline["edges"]
            for key in ("head_type", "tail_type")
        }
    )
    raw_relations = sorted(
        {edge["relation_type"] for edge in baseline["edges"]}
    )
    payload = {
        "raw_types": raw_types,
        "raw_relations": raw_relations,
    }
    system_prompt = (
        "Induce a compact reusable vocabulary for one automatically generated "
        "knowledge-graph schema. Return exactly one JSON object with keys "
        "canonical_types and canonical_relations; both values are unique lists of "
        "concise lowercase English labels. The type list must be strictly smaller "
        "than raw_types and the relation list strictly smaller than raw_relations. "
        "Retain semantically necessary distinctions but merge surface roles, aliases, "
        "verb inflections and document-specific wording. Use only the supplied raw "
        "labels. No evaluator schema or examples are "
        "available. Output JSON only."
    )
    user_prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    prompt_bytes = len(system_prompt.encode("utf-8")) + len(user_prompt.encode("utf-8"))
    maximum = manifest["candidate_packet_contract"]["maximum_prompt_utf8_bytes"]
    if prompt_bytes > maximum:
        raise TransportIncomplete("vocabulary request exceeds the proven transport bound")
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_utf8_bytes": prompt_bytes,
        "prompt_sha256": _sha256(
            canonical_json_bytes(
                {"system_prompt": system_prompt, "user_prompt": user_prompt}
            )
        ),
    }


def _finished_content(response: dict[str, Any]) -> str:
    message = _message(response)
    content = message.get("content") if message is not None else None
    refusal = message.get("refusal") if message is not None else None
    try:
        finish_reason = response["choices"][0]["finish_reason"]
    except (KeyError, IndexError, TypeError):
        finish_reason = None
    if isinstance(refusal, str) and refusal.strip():
        raise ValueError("canonicalization response is a refusal")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("canonicalization response content is missing")
    if finish_reason != "stop":
        raise ValueError("canonicalization response did not finish cleanly")
    return content


def _canonical_vocabulary(value: Any, raw_count: int) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(label, str)
            or label != label.strip().lower()
            or not label
            or len(label) > 60
            or "\n" in label
            or "\r" in label
            for label in value
        )
    ):
        raise ValueError("canonicalization vocabulary is invalid")
    unique = tuple(sorted(set(value)))
    if len(unique) >= raw_count:
        raise ValueError("canonicalization vocabulary is invalid")
    return unique


def _parse_global_vocabulary(
    response: dict[str, Any], baseline: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    value = json.loads(_finished_content(response))
    if not isinstance(value, dict) or set(value) != {
        "canonical_types",
        "canonical_relations",
    }:
        raise ValueError("canonicalization vocabulary response schema is invalid")
    raw_types = {
        edge[key]
        for edge in baseline["edges"]
        for key in ("head_type", "tail_type")
    }
    raw_relations = {edge["relation_type"] for edge in baseline["edges"]}
    return (
        _canonical_vocabulary(value["canonical_types"], len(raw_types)),
        _canonical_vocabulary(value["canonical_relations"], len(raw_relations)),
    )


def _source_edge(edge: dict[str, Any]) -> dict[str, Any]:
    value = {
        "head_type": edge["head_type"],
        "relation_type": edge["relation_type"],
        "tail_type": edge["tail_type"],
        "evidence_doc_ids": sorted(edge["evidence_doc_ids"]),
    }
    return {"source_edge_id": _sha256(canonical_json_bytes(value)), **value}


def _mapping_request(
    manifest: dict[str, Any],
    documents: list[dict[str, Any]],
    source_edges: list[dict[str, Any]],
    canonical_types: tuple[str, ...],
    canonical_relations: tuple[str, ...],
) -> dict[str, Any]:
    payload = {
        "canonical_vocabulary": {
            "types": [
                {"type_id": index, "label": label}
                for index, label in enumerate(canonical_types)
            ],
            "relations": [
                {"relation_type_id": index, "label": label}
                for index, label in enumerate(canonical_relations)
            ],
        },
        "documents": [
            {"doc_id": row["doc_id"], "text": row["text"]}
            for row in documents
        ],
        "raw_schema_edges": source_edges,
    }
    system_prompt = (
        "Map every supplied raw schema edge to the fixed global canonical "
        "vocabulary using its evidence documents. Return exactly one JSON object "
        "with key edge_mappings. It is a list containing exactly one object per "
        "source_edge_id with exactly source_edge_id, head_type_id, "
        "relation_type_id, tail_type_id and reverse. Select only integer IDs from "
        "canonical_vocabulary; never repeat or invent label strings. head_type_id "
        "and tail_type_id classify the original endpoints. Set reverse to "
        "true only when the canonical relation direction requires swapping them. "
        "Do not drop, add or merge source_edge_ids. Output JSON only."
    )
    user_prompt = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    prompt_bytes = len(system_prompt.encode("utf-8")) + len(
        user_prompt.encode("utf-8")
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_utf8_bytes": prompt_bytes,
        "prompt_sha256": _sha256(
            canonical_json_bytes(
                {"system_prompt": system_prompt, "user_prompt": user_prompt}
            )
        ),
    }


def _mapping_packets(
    manifest: dict[str, Any],
    baseline: dict[str, Any],
    canonical_types: tuple[str, ...],
    canonical_relations: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    maximum = manifest["candidate_packet_contract"]["maximum_prompt_utf8_bytes"]
    document_by_id = {row["doc_id"]: row for row in manifest["documents"]}
    source_edges = sorted(
        (_source_edge(edge) for edge in baseline["edges"]),
        key=lambda row: row["source_edge_id"],
    )
    if len({row["source_edge_id"] for row in source_edges}) != len(source_edges):
        raise ValueError("development baseline source-edge IDs are not unique")

    def build(rows: list[dict[str, Any]]) -> dict[str, Any]:
        evidence_ids = {
            doc_id for row in rows for doc_id in row["evidence_doc_ids"]
        }
        documents = [
            document_by_id[doc_id] for doc_id in sorted(evidence_ids)
        ]
        request = _mapping_request(
            manifest,
            documents,
            rows,
            canonical_types,
            canonical_relations,
        )
        return {"request": request, "source_edges": tuple(rows)}

    packets: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for source_edge in source_edges:
        trial = build([*current, source_edge])
        if trial["request"]["prompt_utf8_bytes"] <= maximum:
            current.append(source_edge)
            continue
        if not current:
            raise TransportIncomplete("one canonicalization edge exceeds 6 KB")
        packets.append(build(current))
        current = [source_edge]
        if build(current)["request"]["prompt_utf8_bytes"] > maximum:
            raise TransportIncomplete("one canonicalization edge exceeds 6 KB")
    if current:
        packets.append(build(current))
    return tuple(packets)


def _parse_edge_mappings(
    response: dict[str, Any],
    source_edges: tuple[dict[str, Any], ...],
    canonical_types: tuple[str, ...],
    canonical_relations: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    value = json.loads(_finished_content(response))
    if not isinstance(value, dict) or set(value) != {"edge_mappings"}:
        raise ValueError("edge-mapping response schema is invalid")
    rows = value["edge_mappings"]
    if not isinstance(rows, list):
        raise ValueError("edge mappings are not a list")
    expected_ids = {row["source_edge_id"] for row in source_edges}
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "source_edge_id",
                "head_type_id",
                "relation_type_id",
                "tail_type_id",
                "reverse",
            }
            or row.get("source_edge_id") not in expected_ids
            or row["source_edge_id"] in observed
            or type(row.get("head_type_id")) is not int
            or not 0 <= row["head_type_id"] < len(canonical_types)
            or type(row.get("tail_type_id")) is not int
            or not 0 <= row["tail_type_id"] < len(canonical_types)
            or type(row.get("relation_type_id")) is not int
            or not 0 <= row["relation_type_id"] < len(canonical_relations)
            or type(row.get("reverse")) is not bool
        ):
            raise ValueError("edge-mapping binding is invalid")
        observed[row["source_edge_id"]] = {
            "source_edge_id": row["source_edge_id"],
            "head_type": canonical_types[row["head_type_id"]],
            "relation_type": canonical_relations[row["relation_type_id"]],
            "tail_type": canonical_types[row["tail_type_id"]],
            "reverse": row["reverse"],
        }
    if set(observed) != expected_ids:
        raise ValueError("edge mappings are incomplete")
    return tuple(observed[source_id] for source_id in sorted(observed))


def _project_global_mapping(
    baseline: dict[str, Any],
    mappings: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    source_by_id = {
        row["source_edge_id"]: row
        for row in (_source_edge(edge) for edge in baseline["edges"])
    }
    evidence: dict[tuple[str, str, str], set[str]] = {}
    lineage: dict[tuple[str, str, str], set[str]] = {}
    for mapping in mappings:
        source = source_by_id[mapping["source_edge_id"]]
        head = mapping["head_type"]
        tail = mapping["tail_type"]
        if mapping["reverse"]:
            head, tail = tail, head
        canonical = (
            head,
            mapping["relation_type"],
            tail,
        )
        evidence.setdefault(canonical, set()).update(source["evidence_doc_ids"])
        lineage.setdefault(canonical, set()).add(mapping["source_edge_id"])
    return [
        {
            "head_type": edge[0],
            "relation_type": edge[1],
            "tail_type": edge[2],
            "evidence_doc_ids": sorted(evidence[edge]),
            "source_edge_ids": sorted(lineage[edge]),
        }
        for edge in sorted(evidence)
    ]


def run_global_canonicalization(
    manifest_path: str | Path,
    baseline_path: str | Path,
    results_root: str | Path,
    client: Any,
    *,
    expected_baseline_sha256: str = DEVELOPMENT_BASELINE_SHA256,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(Path(manifest_path))
    phase = manifest.get("phase", "consumed-development")
    is_confirmation = phase.startswith("held-out-confirmation-")
    baseline, baseline_sha = _load_development_baseline(
        Path(baseline_path), manifest, manifest_sha, expected_baseline_sha256
    )
    _validate_live_model(manifest, client.get_models())
    vocabulary_request = _canonicalization_request(manifest, baseline)
    call_records: list[dict[str, Any]] = []

    def call(request: dict[str, Any], stage: str, lease: Any) -> ChatResult:
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
                "usage": _usage(result.response),
                "client_elapsed_seconds": result.client_elapsed_seconds,
                "retry_wait_seconds": result.retry_wait_seconds,
            }
        )
        return result

    try:
        with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
            vocabulary_result = call(vocabulary_request, "vocabulary", lease)
            canonical_types, canonical_relations = _parse_global_vocabulary(
                vocabulary_result.response, baseline
            )
            mapping_packets = _mapping_packets(
                manifest, baseline, canonical_types, canonical_relations
            )
            mappings: list[dict[str, Any]] = []
            for index, packet in enumerate(mapping_packets):
                mapping_result = call(
                    packet["request"], f"edge-mapping-{index + 1:04d}", lease
                )
                mappings.extend(
                    _parse_edge_mappings(
                        mapping_result.response,
                        packet["source_edges"],
                        canonical_types,
                        canonical_relations,
                    )
                )
    except TransportIncomplete as error:
        failure = {
            "schema": "wse-scope-global-canonicalization-technical-failure-v1",
            "manifest_sha256": manifest_sha,
            "source_baseline_sha256": baseline_sha,
            "server_label": manifest["server_label"],
            "model": manifest["model"],
            "status": "technical_failure",
            "completed_call_records": call_records,
            "technical_error": str(error),
        }
        root = Path(results_root)
        write_json_atomic(root / "technical_failure.json", failure)
        summary = {
            "schema": "wse-scope-global-canonicalization-summary-v2",
            "status": "technical_failure",
            "runtime_complete": False,
            "quality_evaluated": False,
            "actual_calls": len(call_records),
        }
        write_json_atomic(root / "summary.json", summary)
        return summary
    except (json.JSONDecodeError, ValueError) as error:
        status = "parse_error"
        canonical_types = ()
        canonical_relations = ()
        mappings = []
        edges: list[dict[str, Any]] = []
        parser_error = str(error)
    else:
        status = "ok"
        edges = _project_global_mapping(baseline, tuple(mappings))
        parser_error = None

    marginal_usage = {
        "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in call_records),
        "completion_tokens": sum(
            row["usage"]["completion_tokens"] for row in call_records
        ),
    }
    marginal_elapsed = sum(row["client_elapsed_seconds"] for row in call_records)
    marginal_retry = sum(row["retry_wait_seconds"] for row in call_records)
    marginal_cost = {
        "actual_calls": len(call_records),
        "usage": marginal_usage,
        "client_elapsed_seconds": marginal_elapsed,
        "retry_wait_seconds": marginal_retry,
    }
    total_method_cost = {
        "actual_calls": baseline["actual_calls"] + marginal_cost["actual_calls"],
        "usage": {
            "prompt_tokens": baseline["usage"]["prompt_tokens"]
            + marginal_usage["prompt_tokens"],
            "completion_tokens": baseline["usage"]["completion_tokens"]
            + marginal_usage["completion_tokens"],
        },
        "client_elapsed_seconds": baseline["client_elapsed_seconds"]
        + marginal_elapsed,
        "retry_wait_seconds": baseline["retry_wait_seconds"] + marginal_retry,
    }
    checkpoint = {
        "schema": (
            "wse-scope-global-canonicalization-confirmation-v1"
            if is_confirmation
            else "wse-scope-global-canonicalization-development-v3"
        ),
        "protocol_id": (
            manifest["protocol_id"]
            if is_confirmation
            else "wse-scope-global-canonicalization-development-v3"
        ),
        "phase": phase if is_confirmation else "consumed-development",
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
        "method": (
            "evidence-grounded closed-vocabulary edge-level schema canonicalization"
        ),
        "status": status,
        "raw_edge_count": len(baseline["edges"]),
        "canonical_edge_count": len(edges),
        "raw_type_count": len(
            {
                edge[key]
                for edge in baseline["edges"]
                for key in ("head_type", "tail_type")
            }
        ),
        "canonical_type_count": len(canonical_types),
        "raw_relation_count": len(
            {edge["relation_type"] for edge in baseline["edges"]}
        ),
        "canonical_relation_count": len(canonical_relations),
        "canonical_vocabulary": {
            "types": list(canonical_types),
            "relations": list(canonical_relations),
        },
        "edges": edges,
        "call_records": call_records,
        "parser_error": parser_error,
        "marginal_cost": marginal_cost,
        "total_method_cost": total_method_cost,
        "actual_calls": len(call_records),
    }
    root = Path(results_root)
    write_checkpoint_atomic(root / "global_canonicalization.json", checkpoint)
    summary = {
        "schema": "wse-scope-global-canonicalization-summary-v2",
        "status": (
            (
                "confirmation_runtime_complete"
                if is_confirmation
                else "development_complete"
            )
            if status == "ok"
            else (
                "confirmation_parse_failure"
                if is_confirmation
                else "development_parse_failure"
            )
        ),
        "runtime_complete": status == "ok",
        "quality_evaluated": False,
        "actual_calls": len(call_records),
        "raw_edge_count": checkpoint["raw_edge_count"],
        "canonical_edge_count": checkpoint["canonical_edge_count"],
        "raw_type_count": checkpoint["raw_type_count"],
        "canonical_type_count": checkpoint["canonical_type_count"],
        "raw_relation_count": checkpoint["raw_relation_count"],
        "canonical_relation_count": checkpoint["canonical_relation_count"],
        "marginal_cost": marginal_cost,
        "total_method_cost": total_method_cost,
        "claim_guard": checkpoint["claim_guard"],
    }
    write_json_atomic(root / "summary.json", summary)
    return summary


def run_canonicalization_development(
    manifest_path: str | Path,
    results_root: str | Path,
    client: Any,
    *,
    baseline_path: str | Path | None = None,
    expected_baseline_sha256: str | None = None,
) -> dict[str, Any]:
    """Run a same-model development baseline when none is supplied, then canonicalize."""
    root = Path(results_root)
    if baseline_path is None:
        manifest, manifest_sha = _load_manifest(Path(manifest_path))
        is_confirmation = str(manifest.get("phase", "")).startswith(
            "held-out-confirmation-"
        )
        _validate_live_model(manifest, client.get_models())
        with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
            baseline, _edges = _run_autoschemakg_baseline(
                client, lease, manifest, manifest_sha
            )
        generated_path = root / "checkpoints" / f"{ARMS[0]}.json"
        write_checkpoint_atomic(generated_path, baseline)
        if baseline["status"] != "ok":
            summary = {
                "schema": "wse-scope-global-canonicalization-summary-v2",
                "status": (
                    "confirmation_baseline_failure"
                    if is_confirmation
                    else "development_baseline_failure"
                ),
                "runtime_complete": False,
                "quality_evaluated": False,
                "baseline_status": baseline["status"],
                "actual_calls": baseline["actual_calls"],
            }
            write_json_atomic(root / "summary.json", summary)
            return summary
        baseline_path = generated_path
        expected_baseline_sha256 = _sha256(generated_path.read_bytes())
    elif expected_baseline_sha256 is None:
        expected_baseline_sha256 = DEVELOPMENT_BASELINE_SHA256
    return run_global_canonicalization(
        manifest_path,
        baseline_path,
        root,
        client,
        expected_baseline_sha256=expected_baseline_sha256,
    )


def _candidate_smoke_rows(
    documents: list[dict[str, Any]],
) -> tuple[CandidateEvidence, ...]:
    rows: list[CandidateEvidence] = []
    for repetition in range(16):
        for index, document in enumerate(documents):
            text = document["text"]
            seed = f"{document['doc_id']}|{repetition}|{index}"
            rows.append(
                CandidateEvidence(
                    "candidate-" + _sha256(seed.encode("utf-8")),
                    TypedSchemaEdge(
                        f"diagnostic-head-{repetition}-{index}",
                        f"diagnostic-relation-{repetition}",
                        f"diagnostic-tail-{index}",
                    ),
                    (document["doc_id"],),
                    ((0, len(text)),),
                    (_sha256(text.encode("utf-8")),),
                )
            )
    return tuple(rows)


def run_candidate_smoke(
    manifest_path: str | Path,
    results_root: str | Path,
    client: Any,
) -> dict[str, Any]:
    """Probe only whether one near-bound candidate request is transport-valid."""
    manifest, manifest_sha = _load_manifest(Path(manifest_path))
    _validate_live_model(manifest, client.get_models())
    documents = [
        {key: row[key] for key in ("doc_id", "language", "source_dataset", "text")}
        for row in manifest["documents"]
    ]
    packets = _candidate_packets(
        manifest, documents, _candidate_smoke_rows(documents)
    )
    if not packets:
        raise TransportIncomplete("candidate smoke packet is empty")
    packet = packets[0]
    maximum = manifest["candidate_packet_contract"]["maximum_prompt_utf8_bytes"]
    if packet["prompt_utf8_bytes"] < int(maximum * 0.8):
        raise TransportIncomplete("candidate smoke packet does not exercise the bound")
    request = packet["request"]
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        lease.assert_healthy()
        try:
            result = client.chat(
                model=manifest["model"],
                messages=[
                    {"role": "system", "content": request["system_prompt"]},
                    {"role": "user", "content": request["user_prompt"]},
                ],
                parameters=manifest["request_parameters"]["candidate"],
                lease=lease,
            )
        except TransportIncomplete as error:
            value = {
                "schema": "wse-scope-candidate-transport-smoke-v1",
                "manifest_sha256": manifest_sha,
                "server_label": manifest["server_label"],
                "model": manifest["model"],
                "status": "transport_rejected",
                "prompt_sha256": request["prompt_sha256"],
                "prompt_utf8_bytes": packet["prompt_utf8_bytes"],
                "response_retained": False,
                "technical_error": str(error),
            }
            write_json_atomic(Path(results_root) / "candidate_smoke.json", value)
            return value
        lease.assert_healthy()
    value = {
        "schema": "wse-scope-candidate-transport-smoke-v1",
        "manifest_sha256": manifest_sha,
        "server_label": manifest["server_label"],
        "model": manifest["model"],
        "status": "transport_accepted",
        "prompt_sha256": request["prompt_sha256"],
        "prompt_utf8_bytes": packet["prompt_utf8_bytes"],
        "response_retained": False,
        "usage": _usage(result.response),
        "client_elapsed_seconds": result.client_elapsed_seconds,
        "retry_wait_seconds": result.retry_wait_seconds,
    }
    write_json_atomic(Path(results_root) / "candidate_smoke.json", value)
    return value


def _main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--server-label", required=True)
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--results-root", default="/results")
    candidate_smoke = commands.add_parser("candidate-smoke")
    candidate_smoke.add_argument("--manifest", required=True)
    candidate_smoke.add_argument("--results-root", default="/results")
    canonicalize = commands.add_parser("canonicalize-development")
    canonicalize.add_argument("--manifest", required=True)
    canonicalize.add_argument("--baseline")
    canonicalize.add_argument("--baseline-sha256")
    canonicalize.add_argument("--results-root", default="/results")
    canonicalize_confirmation = commands.add_parser("canonicalize")
    canonicalize_confirmation.add_argument("--manifest", required=True)
    canonicalize_confirmation.add_argument("--baseline")
    canonicalize_confirmation.add_argument("--baseline-sha256")
    canonicalize_confirmation.add_argument("--results-root", default="/results")
    args = parser.parse_args()
    client = WSEClient()
    if args.command == "smoke":
        from wse_eval import smoke as base_smoke

        value = base_smoke(client, args.server_label, args.model, args.output)
    elif args.command == "run":
        value = run_scope(args.manifest, args.results_root, client)
    elif args.command == "candidate-smoke":
        value = run_candidate_smoke(args.manifest, args.results_root, client)
    else:
        value = run_canonicalization_development(
            args.manifest,
            args.results_root,
            client,
            baseline_path=args.baseline,
            expected_baseline_sha256=args.baseline_sha256,
        )
    print(json.dumps(value, sort_keys=True))
    return (
        0
        if value.get("status")
        not in {
            "transport_rejected",
            "development_parse_failure",
            "confirmation_parse_failure",
            "development_baseline_failure",
            "confirmation_baseline_failure",
            "technical_failure",
        }
        else 4
    )


if __name__ == "__main__":
    raise SystemExit(_main())
