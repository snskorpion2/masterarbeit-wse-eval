"""Evidence-local edge mapping for consumed REDFM development evidence."""

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
import wse_scope_eval as legacy  # noqa: E402
import wse_scope_pairmerge_eval as pair  # noqa: E402
import wse_scope_pairmerge_keyed_eval as keyed  # noqa: E402
import wse_scope_partition_eval as partition  # noqa: E402


CHECKPOINT_SCHEMA = "wse-scope-evidence-local-label-checkpoint-v2"
MANIFEST_SCHEMA = "wse-scope-evidence-local-label-manifest-v2"
PROTOCOL = "wse-scope-evidence-local-label-development-v2"
SUMMARY_SCHEMA = "wse-scope-evidence-local-label-summary-v2"
FAILURE_SCHEMA = "wse-scope-evidence-local-label-technical-failure-v2"
SOURCE_FAILURE_SCHEMA = (
    "wse-scope-evidence-pairmerge-normalized-technical-failure-v1"
)
MAX_PROMPT_BYTES = 6000
MAX_EXCERPT_BYTES = 320
V1_MAPPING_SYSTEM_PROMPT = (
    "Map every raw schema edge using only its quoted source evidence and the "
    "fixed vocabularies. Return exactly {\"edge_mappings\":[...]} with one "
    "object per source_edge_id and exactly source_edge_id, head_type_id, "
    "relation_type_id, tail_type_id, reverse. IDs are integers from the given "
    "vocabularies. reverse is boolean and swaps endpoints only when required. "
    "Do not omit or invent edges. Output JSON only."
)
MAPPING_SYSTEM_PROMPT = (
    "Map every raw schema edge using only its quoted source evidence and the "
    "fixed vocabularies. Return exactly {\"edge_mappings\":[...]} with one "
    "object per source_edge_id and exactly source_edge_id, head_type, "
    "relation_type, tail_type, reverse. Type and relation values must exactly "
    "equal supplied labels. reverse is boolean and swaps endpoints only when "
    "required. Do not omit or invent edges. Output JSON only."
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_manifest(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    observed = _sha256_bytes(raw)
    if observed != expected_sha256:
        raise ValueError("evidence-local manifest bytes differ from the expected binding")
    manifest = json.loads(raw.decode("utf-8"))
    semantic = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    selection = manifest.get("evidence_selection_contract", {})
    replay = manifest.get("replay_binding", {})
    repair = manifest.get("repair_binding", {})
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("protocol_id") != PROTOCOL
        or manifest.get("artifact_sha256") != partition._semantic_sha(semantic)
        or manifest.get("response_contract")
        != "exact-closed-vocabulary-label-mapping-v2"
        or manifest.get("projection_contract")
        != "closed-vocabulary-evidence-local-label-direction-v2"
        or manifest.get("server_label") != "gpu01"
        or manifest.get("model") != "Qwen/Qwen3-14B"
        or manifest.get("pair_request_max_bytes") != MAX_PROMPT_BYTES
        or selection.get("maximum_prompt_utf8_bytes") != MAX_PROMPT_BYTES
        or selection.get("maximum_excerpt_utf8_bytes") != MAX_EXCERPT_BYTES
        or replay.get("source_failure_schema") != SOURCE_FAILURE_SCHEMA
        or replay.get("completed_pair_call_count") != 4
        or repair.get("schema")
        != "wse-scope-evidence-local-label-repair-v2"
        or repair.get("v1_edge_artifact_sha256")
        != "c3b9f0ff67a51f2093829610321a3ce4845a4b1f7b0c910081cf0c2e0a43e3ea"
        or repair.get("v1_summary_sha256")
        != "34bb19b13e9c039e783042d3bab40bd2ff9aca047353dd226cfd4dc6ae56c2b6"
        or repair.get("failed_packet_index") != 19
        or repair.get("closed_type_count") != 99
        or repair.get("valid_type_id_maximum") != 98
        or repair.get("observed_invalid_tail_type_id") != 99
        or repair.get("v1_parser_error") != "edge-mapping binding is invalid"
        or repair.get("mapping_packet_count") != 34
        or repair.get("maximum_v1_prompt_utf8_bytes") != 5970
        or repair.get("maximum_v2_prompt_utf8_bytes") != 5976
        or manifest.get("request_parameters", {}).get("candidate", {}).get(
            "temperature"
        )
        != 0
        or manifest.get("request_parameters", {}).get("candidate", {}).get(
            "chat_template_kwargs"
        )
        != {"enable_thinking": False}
    ):
        raise ValueError("evidence-local manifest contract mismatch")
    return manifest, observed


def _normal_sentences(text: str) -> list[str]:
    return [
        " ".join(row.split())
        for row in re.split(r"(?<=[.!?])\s+|\r?\n+", text)
        if row.strip()
    ]


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"[a-z0-9]+", value.lower()))))


def _bounded_excerpt(sentence: str, labels: tuple[str, ...]) -> tuple[str, int, int]:
    if len(sentence.encode("utf-8")) <= MAX_EXCERPT_BYTES:
        return sentence, 0, len(sentence)
    lowered = sentence.lower()
    positions = [lowered.find(token) for token in labels if lowered.find(token) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - MAX_EXCERPT_BYTES // 3)
    end = min(len(sentence), start + MAX_EXCERPT_BYTES)
    excerpt = sentence[start:end]
    while len(excerpt.encode("utf-8")) > MAX_EXCERPT_BYTES:
        excerpt = excerpt[:-1]
        end -= 1
    return excerpt.strip(), start, end


def _evidence_selection(
    edge: dict[str, Any], document_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    label_values = (
        edge["head_type"],
        edge["relation_type"],
        edge["tail_type"],
    )
    label_tokens = _tokens(" ".join(label_values))
    selected: list[dict[str, Any]] = []
    for doc_id in sorted(edge["evidence_doc_ids"]):
        document = document_by_id.get(doc_id)
        if not isinstance(document, dict) or not isinstance(document.get("text"), str):
            raise ValueError("source-edge evidence document is missing")
        sentences = _normal_sentences(document["text"])
        if not sentences:
            raise ValueError("source-edge evidence document has no text")
        ranked: list[tuple[tuple[int, int, int], int, str]] = []
        for index, sentence in enumerate(sentences):
            lowered = sentence.lower()
            phrase_hits = sum(value.lower() in lowered for value in label_values)
            token_hits = sum(token in _tokens(sentence) for token in label_tokens)
            ranked.append(((phrase_hits, token_hits, -index), index, sentence))
        _, sentence_index, sentence = max(ranked, key=lambda row: row[0])
        excerpt, start, end = _bounded_excerpt(sentence, label_tokens)
        if not excerpt:
            raise ValueError("evidence-local excerpt is empty")
        selected.append(
            {
                "doc_id": doc_id,
                "document_sha256": _sha256_bytes(document["text"].encode("utf-8")),
                "sentence_index": sentence_index,
                "sentence_sha256": _sha256_bytes(sentence.encode("utf-8")),
                "excerpt_start_character": start,
                "excerpt_end_character": end,
                "excerpt": excerpt,
                "excerpt_sha256": _sha256_bytes(excerpt.encode("utf-8")),
            }
        )
    return selected


def _mapping_request(
    rows: list[dict[str, Any]],
    canonical_types: tuple[str, ...],
    canonical_relations: tuple[str, ...],
    *,
    system_prompt: str = MAPPING_SYSTEM_PROMPT,
) -> dict[str, Any]:
    payload = {
        "types": list(enumerate(canonical_types)),
        "relations": list(enumerate(canonical_relations)),
        "edges": rows,
    }
    user = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    prompt_bytes = len(system_prompt.encode("utf-8")) + len(user.encode("utf-8"))
    return {
        "system_prompt": system_prompt,
        "user_prompt": user,
        "prompt_utf8_bytes": prompt_bytes,
        "prompt_sha256": _sha256_bytes(
            canonical_json_bytes(
                {"system_prompt": system_prompt, "user_prompt": user}
            )
        ),
    }


def build_edge_packets(
    manifest: dict[str, Any],
    baseline: dict[str, Any],
    canonical_types: tuple[str, ...],
    canonical_relations: tuple[str, ...],
    *,
    source_edge_limit: int | None = None,
    system_prompt: str = MAPPING_SYSTEM_PROMPT,
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
        selections = _evidence_selection(source, document_by_id)
        row = {
            "source_edge_id": source["source_edge_id"],
            "raw": [source["head_type"], source["relation_type"], source["tail_type"]],
            "evidence": [
                {
                    "doc_id": item["doc_id"],
                    "sentence_index": item["sentence_index"],
                    "quote": item["excerpt"],
                }
                for item in selections
            ],
        }
        prepared.append((source, row, selections))

    def build(
        values: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]
    ) -> dict[str, Any]:
        return {
            "request": _mapping_request(
                [row for _, row, _ in values],
                canonical_types,
                canonical_relations,
                system_prompt=system_prompt,
            ),
            "source_edges": tuple(source for source, _, _ in values),
            "evidence_selections": [
                {"source_edge_id": source["source_edge_id"], "documents": selections}
                for source, _, selections in values
            ],
        }

    packets: list[dict[str, Any]] = []
    current: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    maximum = manifest["evidence_selection_contract"]["maximum_prompt_utf8_bytes"]
    for value in prepared:
        trial = build([*current, value])
        if trial["request"]["prompt_utf8_bytes"] <= maximum:
            current.append(value)
            continue
        if not current:
            raise TransportIncomplete("one evidence-local edge exceeds 6 KB")
        packets.append(build(current))
        current = [value]
        if build(current)["request"]["prompt_utf8_bytes"] > maximum:
            raise TransportIncomplete("one evidence-local edge exceeds 6 KB")
    if current:
        packets.append(build(current))
    return tuple(packets)


def _parse_edge_label_mappings(
    response: dict[str, Any],
    source_edges: tuple[dict[str, Any], ...],
    canonical_types: tuple[str, ...],
    canonical_relations: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    value = json.loads(legacy._finished_content(response))
    if not isinstance(value, dict) or set(value) != {"edge_mappings"}:
        raise ValueError("edge-label mapping response schema is invalid")
    rows = value["edge_mappings"]
    if not isinstance(rows, list):
        raise ValueError("edge label mappings are not a list")
    expected_ids = {row["source_edge_id"] for row in source_edges}
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "source_edge_id",
                "head_type",
                "relation_type",
                "tail_type",
                "reverse",
            }
            or row.get("source_edge_id") not in expected_ids
            or row["source_edge_id"] in observed
            or type(row.get("head_type")) is not str
            or row["head_type"] not in canonical_types
            or type(row.get("tail_type")) is not str
            or row["tail_type"] not in canonical_types
            or type(row.get("relation_type")) is not str
            or row["relation_type"] not in canonical_relations
            or type(row.get("reverse")) is not bool
        ):
            raise ValueError("edge-label mapping binding is invalid")
        observed[row["source_edge_id"]] = dict(row)
    if set(observed) != expected_ids:
        raise ValueError("edge label mappings are incomplete")
    return tuple(observed[source_id] for source_id in sorted(observed))


def _validated_record(
    record: Any, stage: str, request: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "stage",
        "prompt_sha256",
        "prompt_utf8_bytes",
        "response_sha256",
        "response",
        "usage",
        "client_elapsed_seconds",
        "retry_wait_seconds",
    }
    response = record.get("response") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or set(record) != required
        or record.get("stage") != stage
        or record.get("prompt_sha256") != request["prompt_sha256"]
        or record.get("prompt_utf8_bytes") != request["prompt_utf8_bytes"]
        or not isinstance(response, dict)
        or record.get("response_sha256")
        != _sha256_bytes(canonical_json_bytes(response))
        or record.get("usage") != legacy._usage(response)
        or type(record.get("client_elapsed_seconds")) not in (int, float)
        or record["client_elapsed_seconds"] < 0
        or type(record.get("retry_wait_seconds")) not in (int, float)
        or record["retry_wait_seconds"] < 0
    ):
        raise ValueError("replayed pair call binding is invalid")
    return response


def replay_pair_evidence(
    manifest: dict[str, Any],
    baseline: dict[str, Any],
    baseline_sha: str,
    source_failure_path: str | Path,
) -> tuple[
    list[dict[str, Any]],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, Any],
    list[dict[str, Any]],
]:
    source_path = Path(source_failure_path)
    replay = manifest["replay_binding"]
    if _sha256_file(source_path) != replay["source_failure_sha256"]:
        raise ValueError("source pair evidence bytes differ from the frozen binding")
    failure = json.loads(source_path.read_text(encoding="utf-8"))
    records = failure.get("completed_call_records")
    if (
        failure.get("schema") != SOURCE_FAILURE_SCHEMA
        or failure.get("status") != "technical_failure"
        or failure.get("technical_error") != replay["source_technical_error"]
        or failure.get("manifest_sha256") != replay["source_manifest_sha256"]
        or failure.get("source_baseline_sha256") != baseline_sha
        or failure.get("server_label") != manifest["server_label"]
        or failure.get("model") != manifest["model"]
        or not isinstance(records, list)
        or len(records) != replay["completed_pair_call_count"]
    ):
        raise ValueError("source pair evidence is not replayable")
    decisions: list[dict[str, Any]] = []
    packets = keyed.build_pair_packets(manifest, baseline)
    if len(packets) != len(records):
        raise ValueError("source pair call count differs from the frozen denominator")
    for index, (packet, record) in enumerate(zip(packets, records), start=1):
        response = _validated_record(
            record, f"pair-merge-{index:04d}", packet["request"]
        )
        decisions.extend(keyed.parse_pair_decisions(response, packet["pairs"]))
    canonical_types, canonical_relations, audit = keyed.close_pair_decisions(
        baseline, manifest["candidate_pairs"], decisions
    )
    return decisions, canonical_types, canonical_relations, audit, records


def _costs(
    baseline: dict[str, Any], records: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return pair._costs(baseline, records)


def _execute(
    *,
    manifest: dict[str, Any],
    manifest_sha: str,
    baseline: dict[str, Any],
    baseline_sha: str,
    source_failure_path: Path,
    results_root: Path,
    client: Any,
    lease: Any,
    smoke: bool,
) -> dict[str, Any]:
    decisions, types, relations, audit, replayed = replay_pair_evidence(
        manifest, baseline, baseline_sha, source_failure_path
    )
    packets = build_edge_packets(
        manifest,
        baseline,
        types,
        relations,
        source_edge_limit=1 if smoke else None,
    )
    mapping_records: list[dict[str, Any]] = []
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
                parameters=manifest["request_parameters"]["candidate"],
                lease=lease,
            )
            lease.assert_healthy()
            record = {
                "stage": f"edge-local-mapping-{index:04d}",
                "prompt_sha256": request["prompt_sha256"],
                "prompt_utf8_bytes": request["prompt_utf8_bytes"],
                "response_sha256": _sha256_bytes(canonical_json_bytes(result.response)),
                "response": result.response,
                "usage": legacy._usage(result.response),
                "client_elapsed_seconds": result.client_elapsed_seconds,
                "retry_wait_seconds": result.retry_wait_seconds,
            }
            mapping_records.append(record)
            mappings.extend(
                _parse_edge_label_mappings(
                    result.response, packet["source_edges"], types, relations
                )
            )
            selections.extend(packet["evidence_selections"])
    except (json.JSONDecodeError, ValueError) as error:
        parser_error = str(error)

    if smoke:
        status = "smoke_complete" if parser_error is None else "smoke_parse_failure"
        value = {
            "schema": "wse-scope-evidence-local-label-smoke-v2",
            "status": status,
            "manifest_sha256": manifest_sha,
            "source_baseline_sha256": baseline_sha,
            "source_failure_sha256": manifest["replay_binding"][
                "source_failure_sha256"
            ],
            "server_label": manifest["server_label"],
            "model": manifest["model"],
            "fixed_denominator": 1,
            "received_calls": len(mapping_records),
            "mapping": mappings[0] if len(mappings) == 1 else None,
            "evidence_selection": selections[0] if len(selections) == 1 else None,
            "call_records": mapping_records,
            "parser_error": parser_error,
            "claim_guard": "Technical one-edge smoke only; no quality evidence.",
        }
        write_json_atomic(results_root / "smoke.json", value)
        return value

    status = "ok" if parser_error is None else "parse_error"
    edges = legacy._project_global_mapping(baseline, tuple(mappings)) if status == "ok" else []
    replay_marginal, _ = _costs(baseline, replayed)
    execution_marginal, _ = _costs(baseline, mapping_records)
    marginal, total = _costs(baseline, [*replayed, *mapping_records])
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol_id": manifest["protocol_id"],
        "phase": manifest["phase"],
        "claim_guard": manifest["claim_guard"],
        "manifest_sha256": manifest_sha,
        "source_baseline_sha256": baseline_sha,
        "source_failure_sha256": manifest["replay_binding"]["source_failure_sha256"],
        "server_label": manifest["server_label"],
        "model": manifest["model"],
        "source_baseline": {
            "manifest_sha256": baseline["manifest_sha256"],
            "server_label": baseline["server_label"],
            "model": baseline["model"],
            "protocol_id": baseline["protocol_id"],
        },
        "method": (
            "closed pair vocabulary plus evidence-local exact-label edge and "
            "direction mapping"
        ),
        "status": status,
        "fixed_denominator_source_edge_count": len(baseline["edges"]),
        "canonical_vocabulary": {"types": list(types), "relations": list(relations)},
        "partition_audit": audit,
        "pair_decisions": decisions,
        "evidence_selections": selections,
        "edges": edges,
        "replayed_pair_call_records": replayed,
        "mapping_call_records": mapping_records,
        "call_records": [*replayed, *mapping_records],
        "replayed_pair_call_count": len(replayed),
        "actual_remote_calls_this_run": len(mapping_records),
        "actual_calls": len(replayed) + len(mapping_records),
        "parser_error": parser_error,
        "replayed_pair_marginal_cost": replay_marginal,
        "new_execution_marginal_cost": execution_marginal,
        "marginal_cost": marginal,
        "total_method_cost": total,
    }
    write_checkpoint_atomic(results_root / "edge_local.json", checkpoint)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "development_complete" if status == "ok" else "development_parse_failure",
        "runtime_complete": status == "ok",
        "quality_evaluated": False,
        "fixed_denominator_source_edge_count": len(baseline["edges"]),
        "mapped_source_edge_count": len(mappings) if status == "ok" else 0,
        "actual_remote_calls_this_run": len(mapping_records),
        "replayed_pair_call_count": len(replayed),
        "marginal_cost": marginal,
        "total_method_cost": total,
        "claim_guard": manifest["claim_guard"],
    }
    write_json_atomic(results_root / "summary.json", summary)
    return summary


def run_edge_local(
    manifest_path: str | Path,
    baseline_path: str | Path,
    source_failure_path: str | Path,
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
            source_failure_path=Path(source_failure_path),
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
    parser.add_argument("--source-failure", required=True)
    parser.add_argument("--results-root", default="/results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        value = run_edge_local(
            args.manifest,
            args.baseline,
            args.source_failure,
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
