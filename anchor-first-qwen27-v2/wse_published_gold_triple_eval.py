#!/usr/bin/env python3
"""Gold-blind WSE runtime for the published EE/EV/VV pair comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from wse_eval import (  # noqa: E402
    LeaseRenewalFailed,
    LeaseUnavailable,
    RUN_CHAT_MAX_ELAPSED_SECONDS,
    SMOKE_CHAT_MAX_ELAPSED_SECONDS,
    TransportIncomplete,
    WSEClient,
    canonical_json_bytes,
    write_checkpoint_atomic,
    write_json_atomic,
)


PROTOCOL_ID = "published-gold-triple-quality-v1"
ANCHOR_FIRST_PROTOCOL_ID = "anchor-first-triple-quality-v2"
EXPECTED_MANIFEST_SHA256: dict[str, str] = {"development_h200_qwen3_6_27b.json": "643b76239a1eda562fa719a6d0cfaf02e85a42eae177b92aed4219a6ad4dfbe9"}
STATUSES = {
    "ok",
    "partial_accept",
    "valid_empty",
    "refusal",
    "parse_error",
    "evidence_rejected",
}
REFERENCE_KEYS = {
    "reference",
    "relations",
    "links",
    "triples",
    "event_span",
    "argument_span",
    "head_event_id",
    "tail_event_id",
}
BASE_RUNTIME = HERE / "wse_eval.py"
NOTICE_SOURCE = HERE / "PUBLISHED_GOLD_NOTICE.md"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    if "artifact_sha256" in sealed:
        raise ValueError("artifact_sha256 is reserved")
    sealed["artifact_sha256"] = _sha256_json(sealed)
    return sealed


def _verify_seal(value: dict[str, Any]) -> None:
    expected = value.get("artifact_sha256")
    body = dict(value)
    body.pop("artifact_sha256", None)
    if not isinstance(expected, str) or _sha256_json(body) != expected:
        raise ValueError("manifest artifact hash mismatch")


def _contains_reference_payload(value: Any) -> bool:
    if isinstance(value, dict):
        if REFERENCE_KEYS.intersection(value):
            return True
        return any(_contains_reference_payload(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_reference_payload(item) for item in value)
    return False


def bind_execution_manifest(
    inputs: dict[str, Any], *, model: str, server_label: str, context_length: int
) -> dict[str, Any]:
    _verify_seal(inputs)
    input_contracts = {
        "published-gold-triple-inputs-v1": (
            PROTOCOL_ID,
            "published-gold-triple-execution-v1",
        ),
        "anchor-first-triple-inputs-v2": (
            ANCHOR_FIRST_PROTOCOL_ID,
            "anchor-first-triple-execution-v2",
        ),
    }
    contract = input_contracts.get(inputs.get("schema"))
    if contract is None or inputs.get("protocol_id") != contract[0]:
        raise ValueError("input manifest schema/protocol mismatch")
    if inputs.get("phase") != "development":
        raise ValueError("only development inputs may be packaged by this runtime")
    if _contains_reference_payload(inputs.get("cases")) or _contains_reference_payload(
        inputs.get("cells")
    ):
        raise ValueError("reference payload must not enter the execution manifest")
    if not model or server_label not in {"gpu01", "h200"} or context_length < 32768:
        raise ValueError("invalid model assignment")
    execution = json.loads(json.dumps(inputs, ensure_ascii=False))
    input_artifact_sha256 = execution.pop("artifact_sha256")
    execution["schema"] = contract[1]
    execution["input_artifact_sha256"] = input_artifact_sha256
    execution["execution_ready"] = True
    execution["model_assignment"] = {
        "model": model,
        "server_label": server_label,
        "context_length": context_length,
    }
    return _seal(execution)


def bind_file(
    *,
    input_path: str | Path,
    output: str | Path,
    model: str,
    server_label: str,
    context_length: int,
) -> dict[str, Any]:
    inputs = json.loads(Path(input_path).read_text(encoding="utf-8"))
    execution = bind_execution_manifest(
        inputs,
        model=model,
        server_label=server_label,
        context_length=context_length,
    )
    write_checkpoint_atomic(output, execution)
    return {
        "output": str(output),
        "artifact_sha256": execution["artifact_sha256"],
        "file_sha256": _sha256_bytes(canonical_json_bytes(execution)),
    }


def _cell_id(cell: dict[str, Any]) -> str:
    return f"cell-{int(cell['order']):04d}-{cell['case_id']}-{cell['arm']}"


def _cell_payload(cell: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "pair_id": cell["pair_id"],
        "case_id": cell["case_id"],
        "axis": cell["axis"],
        "stage": cell["stage"],
        "arm": cell["arm"],
        "messages": cell["messages"],
        "request_parameters": cell["request_parameters"],
    }
    if "candidate_contract" in cell:
        payload["candidate_contract"] = cell["candidate_contract"]
    return payload


def _validate_manifest(
    path: Path, manifest: dict[str, Any], manifest_sha256: str
) -> tuple[str, int]:
    if EXPECTED_MANIFEST_SHA256.get(path.name) != manifest_sha256:
        raise ValueError("manifest hash does not match the pinned image contract")
    _verify_seal(manifest)
    protocol_id = manifest.get("protocol_id")
    expected_schema = {
        PROTOCOL_ID: "published-gold-triple-execution-v1",
        ANCHOR_FIRST_PROTOCOL_ID: "anchor-first-triple-execution-v2",
    }.get(protocol_id)
    if (
        manifest.get("schema") != expected_schema
        or manifest.get("phase") != "development"
        or manifest.get("execution_ready") is not True
    ):
        raise ValueError("execution manifest contract mismatch")
    if _contains_reference_payload(manifest.get("cases")) or _contains_reference_payload(
        manifest.get("cells")
    ):
        raise ValueError("execution manifest contains reference payload")
    assignment = manifest.get("model_assignment")
    if not isinstance(assignment, dict):
        raise ValueError("model assignment missing")
    model = assignment.get("model")
    context_length = assignment.get("context_length")
    if (
        not isinstance(model, str)
        or not model
        or assignment.get("server_label") not in {"gpu01", "h200"}
        or not isinstance(context_length, int)
        or context_length < 32768
    ):
        raise ValueError("WSE model assignment mismatch")
    cases = manifest.get("cases")
    cells = manifest.get("cells")
    if (
        not isinstance(cases, list)
        or not isinstance(cells, list)
        or len(cases) != manifest.get("case_count")
        or len(cells) != manifest.get("cell_count")
        or len(cells) != 2 * len(cases)
    ):
        raise ValueError("manifest cardinality mismatch")
    case_by_id = {
        case.get("case_id"): case for case in cases if isinstance(case, dict)
    }
    case_ids = set(case_by_id)
    if len(case_ids) != len(cases) or None in case_ids:
        raise ValueError("manifest case inventory mismatch")
    pairs: dict[str, list[dict[str, Any]]] = {}
    frozen_parameters: dict[str, Any] | None = None
    for order, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict) or cell.get("order") != order:
            raise ValueError("manifest cell order mismatch")
        if cell.get("case_id") not in case_ids or cell.get("axis") != cell.get("stage"):
            raise ValueError("manifest cell binding mismatch")
        expected_arms = (
            {"autoschemakg_baseline", "evidence_bound_candidate"}
            if protocol_id == PROTOCOL_ID
            else {"autoschemakg_baseline", "anchor_first_candidate"}
        )
        if cell.get("arm") not in expected_arms:
            raise ValueError("manifest arm mismatch")
        if _sha256_json(cell.get("messages")) != cell.get("prompt_sha256"):
            raise ValueError("manifest prompt hash mismatch")
        if _sha256_json(_cell_payload(cell)) != cell.get("cell_payload_sha256"):
            raise ValueError("manifest cell payload hash mismatch")
        if cell.get("arm") == "anchor_first_candidate":
            contract = cell.get("candidate_contract")
            case = case_by_id[cell["case_id"]]
            if (
                not isinstance(contract, dict)
                or contract.get("source_text_sha256") != case.get("text_sha256")
                or contract.get("itemwise_evidence") is not True
                or contract.get("exact_source_anchors") is not True
            ):
                raise ValueError("anchor-first candidate contract mismatch")
            allowed = contract.get("allowed_relations")
            if cell.get("axis") in {"EE", "VV"} and (
                not isinstance(allowed, list)
                or not allowed
                or not all(isinstance(label, str) and label for label in allowed)
                or len(set(allowed)) != len(allowed)
            ):
                raise ValueError("anchor-first relation contract mismatch")
        parameters = cell.get("request_parameters")
        if frozen_parameters is None:
            frozen_parameters = parameters
        if parameters != frozen_parameters:
            raise ValueError("paired request parameters differ")
        pairs.setdefault(str(cell.get("pair_id")), []).append(cell)
    if not isinstance(frozen_parameters, dict) or (
        frozen_parameters.get("temperature") != 0
        or frozen_parameters.get("n") != 1
        or frozen_parameters.get("chat_template_kwargs")
        != {"enable_thinking": False}
    ):
        raise ValueError("request parameter contract mismatch")
    if len(pairs) != len(cases) or any(
        len(pair) != 2
        or {cell["arm"] for cell in pair}
        != expected_arms
        or len({cell["case_id"] for cell in pair}) != 1
        for pair in pairs.values()
    ):
        raise ValueError("paired cell inventory mismatch")
    return model, context_length


def _validate_live_model(model: str, context_length: int, rows: list[dict]) -> None:
    matches = [row for row in rows if (row.get("name") or row.get("id")) == model]
    if len(matches) != 1:
        raise ValueError("manifest model is not uniquely available")
    live_context = matches[0].get("context_length", matches[0].get("max_model_len"))
    if live_context is not None and int(live_context) < context_length:
        raise ValueError("live model context is below the frozen contract")


def _assert_lease_healthy(lease: Any) -> None:
    if hasattr(lease, "assert_healthy"):
        lease.assert_healthy()
    elif getattr(lease, "renew_failures", 0):
        raise LeaseRenewalFailed("lease renewal failed")


def _decode_received_json(content: str) -> Any:
    candidate = content.strip()
    if candidate.startswith("```json\n") and candidate.endswith("\n```"):
        candidate = candidate[len("```json\n") : -len("\n```")]
    return json.loads(candidate)


def _source_text(cell: dict[str, Any]) -> str:
    marker = "Here is the passage:"
    content = cell["messages"][-1]["content"]
    if content.count(marker) != 1:
        raise ValueError("passage marker mismatch")
    return content.split(marker, 1)[1]


def _compile_vv(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    compiled: list[dict[str, Any]] = []
    direct_before: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        head, relation, tail = record["Head"], record["Relation"], record["Tail"]
        if relation == "AFTER":
            head, relation, tail = tail, "BEFORE", head
        elif relation == "SIMULTANEOUS" and tail.casefold() < head.casefold():
            head, tail = tail, head
        normalized = {
            "Head": head,
            "Relation": relation,
            "Tail": tail,
            "Evidence": record["Evidence"],
            "derivation": {"kind": "direct", "source_record_index": index},
        }
        key = (head, tail) if relation == "BEFORE" else None
        if key is not None:
            direct_before[key] = index
        compiled.append(normalized)

    adjacency: dict[str, set[str]] = {}
    for head, tail in direct_before:
        adjacency.setdefault(head, set()).add(tail)
        adjacency.setdefault(tail, set())
    visiting: set[str] = set()
    visited: set[str] = set()

    def has_cycle(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(has_cycle(target) for target in adjacency.get(node, set())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    if any(has_cycle(node) for node in sorted(adjacency)):
        return compiled, "cycle_detected_no_closure"

    derived = []
    for start in sorted(adjacency, key=str.casefold):
        queue: list[list[str]] = [[start]]
        seen = {start}
        while queue:
            path = queue.pop(0)
            for target in sorted(adjacency.get(path[-1], set()), key=str.casefold):
                if target in seen:
                    continue
                seen.add(target)
                next_path = [*path, target]
                queue.append(next_path)
                if len(next_path) < 3 or (start, target) in direct_before:
                    continue
                edge_indices = [
                    direct_before[(left, right)]
                    for left, right in zip(next_path, next_path[1:])
                ]
                derived.append(
                    {
                        "Head": start,
                        "Relation": "BEFORE",
                        "Tail": target,
                        "derivation": {
                            "kind": "transitive",
                            "path": next_path,
                            "direct_edge_record_indices": edge_indices,
                        },
                    }
                )
    derived.sort(key=lambda row: (row["Head"].casefold(), row["Tail"].casefold()))
    return [*compiled, *derived], "acyclic_closure_complete"


def _parse_anchor_first(cell: dict[str, Any], records: list[Any]) -> dict[str, Any]:
    if not records:
        return {
            "status": "valid_empty",
            "accepted_records": [],
            "rejected_records": [],
            "compiled_records": [],
        }
    source = _source_text(cell)
    axis = cell["axis"]
    expected = {"Event", "Entity", "Evidence"} if axis == "EV" else {
        "Head", "Relation", "Tail", "Evidence"
    }
    allowed = set(cell["candidate_contract"].get("allowed_relations", []))
    accepted = []
    rejected = []
    seen = set()
    for index, record in enumerate(records):
        reason = None
        if not isinstance(record, dict):
            reason = "not_object"
        elif set(record) != expected:
            reason = "shape_mismatch"
        elif not all(
            isinstance(record.get(key), str) and bool(record[key].strip())
            for key in expected
        ):
            reason = "non_empty_string_required"
        else:
            first = record["Event"] if axis == "EV" else record["Head"]
            second = record["Entity"] if axis == "EV" else record["Tail"]
            evidence = record["Evidence"]
            if first not in source or second not in source:
                reason = "anchor_not_exact_source_span"
            elif evidence not in source:
                reason = "evidence_not_exact_source_span"
            elif first not in evidence or second not in evidence:
                reason = "evidence_missing_anchor"
            elif axis in {"EE", "VV"} and record["Relation"] not in allowed:
                reason = "relation_outside_contract"
            else:
                key = tuple(record[field] for field in sorted(expected))
                if key in seen:
                    reason = "duplicate"
                else:
                    seen.add(key)
        if reason is None:
            accepted.append(record)
        else:
            rejected.append({"record_index": index, "reason": reason, "record": record})
    if axis == "VV":
        compiled, graph_status = _compile_vv(accepted)
    else:
        compiled, graph_status = list(accepted), "not_applicable"
    status = (
        "partial_accept"
        if accepted and rejected
        else "ok"
        if accepted
        else "evidence_rejected"
    )
    return {
        "status": status,
        "accepted_records": accepted,
        "rejected_records": rejected,
        "compiled_records": compiled,
        "graph_status": graph_status,
    }


def _parse_received(cell: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        return {"status": "parse_error"}
    if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
        return {"status": "parse_error"}
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return {"status": "refusal"}
    content = message.get("content")
    if not isinstance(content, str):
        return {"status": "parse_error"}
    try:
        records = _decode_received_json(content)
    except (json.JSONDecodeError, ValueError):
        return {"status": "parse_error"}
    if not isinstance(records, list):
        return {"status": "parse_error"}
    if cell.get("arm") == "anchor_first_candidate":
        return _parse_anchor_first(cell, records)
    if not records:
        return {"status": "valid_empty", "records": []}
    if not all(isinstance(record, dict) for record in records):
        return {"status": "parse_error"}
    stage = cell["stage"]
    evidence_bound = cell["arm"] == "evidence_bound_candidate"
    expected = (
        {"Event", "Entity"} if stage == "EV" else {"Head", "Relation", "Tail"}
    )
    if evidence_bound:
        expected.add("Evidence")
    valid_shape = all(set(record) == expected for record in records)
    if stage == "EV":
        valid_shape = valid_shape and all(
            isinstance(record.get("Event"), str)
            and bool(record["Event"].strip())
            and isinstance(record.get("Entity"), list)
            and bool(record["Entity"])
            and all(isinstance(item, str) and item.strip() for item in record["Entity"])
            and (
                not evidence_bound
                or (
                    isinstance(record.get("Evidence"), str)
                    and bool(record["Evidence"].strip())
                )
            )
            for record in records
        )
    else:
        valid_shape = valid_shape and all(
            all(isinstance(record.get(key), str) and record[key].strip() for key in expected)
            for record in records
        )
    if not valid_shape:
        return {"status": "parse_error", "records": records}
    if evidence_bound:
        marker = "Here is the passage:"
        source = cell["messages"][-1]["content"].split(marker, 1)[-1]
        if any(record["Evidence"] not in source for record in records):
            return {"status": "evidence_rejected", "records": records}
    return {"status": "ok", "records": records}


def _technical_failure(root: Path, error: BaseException) -> None:
    write_json_atomic(
        root / "technical_failure.json",
        {
            "schema": "published-gold-triple-technical-failure-v1",
            "status": "technical_failure",
            "error_class": type(error).__name__,
            "message": str(error)[:500],
            "scientific_complete": False,
        },
    )


def run(manifest_path: str | Path, results_root: str | Path, client: Any) -> dict:
    path = Path(manifest_path)
    root = Path(results_root)
    try:
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        model, context_length = _validate_manifest(path, manifest, manifest_sha256)
        _validate_live_model(model, context_length, client.get_models())
        checkpoints = []
        with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
            for cell in manifest["cells"]:
                _assert_lease_healthy(lease)
                result = client.chat(
                    model=model,
                    messages=cell["messages"],
                    parameters=cell["request_parameters"],
                    lease=lease,
                    max_elapsed_seconds=RUN_CHAT_MAX_ELAPSED_SECONDS,
                )
                parsed = _parse_received(cell, result.response)
                checkpoint = {
                    "schema": (
                        "anchor-first-triple-checkpoint-v2"
                        if manifest["protocol_id"] == ANCHOR_FIRST_PROTOCOL_ID
                        else "published-gold-triple-checkpoint-v1"
                    ),
                    "cell_id": _cell_id(cell),
                    "order": cell["order"],
                    "pair_id": cell["pair_id"],
                    "case_id": cell["case_id"],
                    "axis": cell["axis"],
                    "arm": cell["arm"],
                    "manifest_sha256": manifest_sha256,
                    "cell_payload_sha256": cell["cell_payload_sha256"],
                    "prompt_sha256": cell["prompt_sha256"],
                    "model_assignment": manifest["model_assignment"],
                    "manager_response": result.response,
                    "manager_response_sha256": _sha256_json(result.response),
                    "usage": result.response.get("usage", {}),
                    "client_elapsed_seconds": result.client_elapsed_seconds,
                    "retry_wait_seconds": result.retry_wait_seconds,
                    "response": parsed,
                }
                write_checkpoint_atomic(
                    root / "checkpoints" / f"{_cell_id(cell)}.json", checkpoint
                )
                checkpoints.append(checkpoint)
                _assert_lease_healthy(lease)
        statuses = [checkpoint["response"]["status"] for checkpoint in checkpoints]
        if len(statuses) != manifest["cell_count"] or any(
            status not in STATUSES for status in statuses
        ):
            raise ValueError("incomplete scientific matrix")
        summary = {
            "schema": (
                "anchor-first-triple-runtime-summary-v2"
                if manifest["protocol_id"] == ANCHOR_FIRST_PROTOCOL_ID
                else "published-gold-triple-runtime-summary-v1"
            ),
            "status": "development_complete",
            "scientific_complete": True,
            "manifest_sha256": manifest_sha256,
            "model_assignment": manifest["model_assignment"],
            "fixed_denominator": manifest["cell_count"],
            "received_count": len(statuses),
            "status_counts": dict(sorted(Counter(statuses).items())),
        }
        write_json_atomic(root / "summary.json", summary)
        return summary
    except (
        TransportIncomplete,
        LeaseUnavailable,
        LeaseRenewalFailed,
        ValueError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        _technical_failure(root, error)
        raise


def smoke(manifest_path: str | Path, output: str | Path, client: Any) -> dict:
    path = Path(manifest_path)
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    model, context_length = _validate_manifest(
        path, manifest, _sha256_bytes(manifest_bytes)
    )
    _validate_live_model(model, context_length, client.get_models())
    cell = manifest["cells"][0]
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        _assert_lease_healthy(lease)
        result = client.chat(
            model=model,
            messages=cell["messages"],
            parameters=cell["request_parameters"],
            lease=lease,
            max_elapsed_seconds=SMOKE_CHAT_MAX_ELAPSED_SECONDS,
        )
        _assert_lease_healthy(lease)
    value = {
        "schema": (
            "anchor-first-triple-smoke-v2"
            if manifest["protocol_id"] == ANCHOR_FIRST_PROTOCOL_ID
            else "published-gold-triple-smoke-v1"
        ),
        "status": "smoke_complete",
        "scientific": False,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "cell_id": _cell_id(cell),
        "response": _parse_received(cell, result.response),
    }
    write_json_atomic(output, value)
    return value


def _hash_directory(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def package_image(
    *, manifest_path: str | Path, output_root: str | Path, base_image: str
) -> dict[str, Any]:
    if not re.fullmatch(
        r"python:3\.12-slim(?:-[^@]+)?@sha256:[0-9a-f]{64}", base_image
    ):
        raise ValueError("base image must be a digest-pinned python:3.12-slim image")
    root = Path(output_root)
    if root.exists():
        raise FileExistsError("image context must be newly created")
    source_manifest = Path(manifest_path)
    manifest_bytes = source_manifest.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    _verify_seal(manifest)
    if manifest.get("schema") not in {
        "published-gold-triple-execution-v1",
        "anchor-first-triple-execution-v2",
    }:
        raise ValueError("only an execution manifest may be packaged")
    if _contains_reference_payload(manifest.get("cases")) or _contains_reference_payload(
        manifest.get("cells")
    ):
        raise ValueError("reference payload must not be packaged")
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    shutil.copyfile(BASE_RUNTIME, root / "wse_eval.py")
    shutil.copyfile(NOTICE_SOURCE, root / "NOTICE.md")
    packaged_manifest = bundle / source_manifest.name
    packaged_manifest.write_bytes(manifest_bytes)
    runtime_source = Path(__file__).read_text(encoding="utf-8")
    # Assemble the marker so this lookup string does not count as a second
    # occurrence of the declaration that is being sealed.
    marker = "EXPECTED_MANIFEST_SHA256" + ": dict[str, str] = {}"
    if runtime_source.count(marker) != 1:
        raise ValueError("runtime manifest-binding marker is missing or ambiguous")
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    replacement = (
        "EXPECTED_MANIFEST_SHA256: dict[str, str] = "
        + json.dumps({packaged_manifest.name: manifest_sha256}, sort_keys=True)
    )
    (root / "wse_published_gold_triple_eval.py").write_text(
        runtime_source.replace(marker, replacement), encoding="utf-8", newline="\n"
    )
    dockerfile = (
        f"FROM {base_image}\n"
        "ARG SOURCE_COMMIT\n"
        "LABEL org.opencontainers.image.source=\"https://github.com/snskorpion2/Masterarbeit\" "
        "org.opencontainers.image.revision=\"${SOURCE_COMMIT}\" "
        "org.opencontainers.image.description=\"Gold-blind paired triple-quality evaluation\"\n"
        "ENV PYTHONDONTWRITEBYTECODE=1\n"
        "ENV HOME=/results/home TMPDIR=/results/tmp XDG_CACHE_HOME=/results/cache\n"
        "WORKDIR /app\n"
        "COPY wse_eval.py /app/wse_eval.py\n"
        "COPY wse_published_gold_triple_eval.py /app/wse_published_gold_triple_eval.py\n"
        "COPY NOTICE.md /app/NOTICE.md\n"
        "COPY bundle /app/bundle\n"
        "RUN chown root:root /tmp /var/tmp && chmod a-w /tmp /var/tmp && "
        "chmod -R a-w /app && mkdir -p /results/home /results/tmp /results/cache && "
        "chown -R 65532:65532 /results && chmod 700 /results /results/home /results/tmp /results/cache\n"
        "VOLUME /results\n"
        "USER 65532:65532\n"
    )
    (root / "Dockerfile").write_text(dockerfile, encoding="utf-8", newline="\n")
    inventory = [
        path.relative_to(root).as_posix()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    (root / ".dockerignore").write_text(
        "\n".join(["**", *[f"!{name}" for name in inventory], "!.dockerignore"])
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "base_image": base_image,
        "manifest_sha256": manifest_sha256,
        "context_sha256": _hash_directory(root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--results-root", required=True)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--manifest", required=True)
    smoke_parser.add_argument("--output", required=True)
    package_parser = commands.add_parser("package")
    package_parser.add_argument("--manifest", required=True)
    package_parser.add_argument("--output-root", required=True)
    package_parser.add_argument("--base-image", required=True)
    bind_parser = commands.add_parser("bind")
    bind_parser.add_argument("--input", required=True)
    bind_parser.add_argument("--output", required=True)
    bind_parser.add_argument("--model", required=True)
    bind_parser.add_argument("--server-label", required=True, choices=("gpu01", "h200"))
    bind_parser.add_argument("--context-length", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bind":
        print(
            json.dumps(
                bind_file(
                    input_path=args.input,
                    output=args.output,
                    model=args.model,
                    server_label=args.server_label,
                    context_length=args.context_length,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "package":
        print(
            json.dumps(
                package_image(
                    manifest_path=args.manifest,
                    output_root=args.output_root,
                    base_image=args.base_image,
                ),
                sort_keys=True,
            )
        )
        return 0
    client = WSEClient()
    if args.command == "run":
        run(args.manifest, args.results_root, client)
    else:
        smoke(args.manifest, args.output, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
