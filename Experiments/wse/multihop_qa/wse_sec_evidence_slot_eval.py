#!/usr/bin/env python3
"""Run the fixed SEC-v2 evidence-slot composer development arm on WSE."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
WSE_COMMON = ROOT / "Experiments/wse/triple_quality"
if str(WSE_COMMON) not in sys.path:
    sys.path.insert(0, str(WSE_COMMON))

from wse_eval import TransportIncomplete, WSEClient  # noqa: E402


MANIFEST_SCHEMA = "wse-sec-evidence-slot-development-manifest-v1"
PROTOCOL_ID = "wse-sec-evidence-slot-composer-development-v1"
CHECKPOINT_SCHEMA = "wse-sec-evidence-slot-question-checkpoint-v1"
SUMMARY_SCHEMA = "wse-sec-evidence-slot-runtime-summary-v1"
MAX_PROMPT_BYTES = 6000
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _sha256_bytes(canonical_json_bytes(result))
    return result


def _load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != expected_sha256:
        raise ValueError("SEC evidence-slot manifest raw SHA differs")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SEC evidence-slot manifest is invalid JSON") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError("SEC evidence-slot manifest is not canonical")
    artifact = value.get("artifact_sha256")
    unsealed = dict(value)
    unsealed.pop("artifact_sha256", None)
    if artifact != _sha256_bytes(canonical_json_bytes(unsealed)):
        raise ValueError("SEC evidence-slot manifest artifact binding differs")
    if (
        value.get("schema") != MANIFEST_SCHEMA
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("server_label") != "gpu01"
        or value.get("model") != "Qwen/Qwen3-14B"
        or value.get("context_length") != 32768
    ):
        raise ValueError("SEC evidence-slot manifest identity differs")
    budget = value.get("budget", {})
    if (
        budget.get("maximum_prompt_utf8_bytes") != MAX_PROMPT_BYTES
        or not isinstance(budget.get("evidence_context_utf8_bytes"), int)
        or not 0 < budget["evidence_context_utf8_bytes"] < MAX_PROMPT_BYTES
    ):
        raise ValueError("SEC evidence-slot byte budget differs")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("SEC evidence-slot case inventory is empty")
    if len({case.get("question_id") for case in cases if isinstance(case, dict)}) != len(cases):
        raise ValueError("SEC evidence-slot question IDs are not unique")
    for case in cases:
        pool = case.get("candidate_pool") if isinstance(case, dict) else None
        if not isinstance(pool, list) or len(pool) != budget["candidate_pool_size"]:
            raise ValueError("SEC evidence-slot candidate denominator differs")
        for passage in pool:
            text = passage.get("text") if isinstance(passage, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("SEC evidence-slot passage is empty")
            if passage.get("text_sha256") != _sha256_bytes(text.encode("utf-8")):
                raise ValueError("SEC evidence-slot passage hash differs")
    return value


def _validate_live_model(manifest: dict[str, Any], models: Iterable[dict[str, Any]]) -> None:
    expected = manifest["catalog_binding"]["selected_record_sha256"]
    selected = None
    available = None
    for row in models:
        if isinstance(row, dict) and (row.get("name") or row.get("id")) == manifest["model"]:
            selected = {"name": manifest["model"]}
            available = row
            break
    if selected is None or _sha256_bytes(canonical_json_bytes(selected)) != expected:
        raise ValueError("live WSE SEC model differs from frozen binding")
    manager_context = available.get(
        "context_length", available.get("max_model_len")
    )
    if manager_context is not None and int(manager_context) < int(manifest["context_length"]):
        raise ValueError("live WSE SEC context is below frozen requirement")


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _bm25_scores(query: str, passages: list[dict[str, Any]]) -> list[float]:
    query_tokens = list(dict.fromkeys(_tokens(query)))
    documents = [_tokens(row["text"]) for row in passages]
    lengths = [len(tokens) for tokens in documents]
    average = sum(lengths) / len(lengths) if lengths else 1.0
    scores: list[float] = []
    for tokens in documents:
        counts = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            document_frequency = sum(term in set(other) for other in documents)
            inverse = math.log(1.0 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
            frequency = counts[term]
            if frequency:
                score += inverse * frequency * 2.2 / (
                    frequency + 1.2 * (0.25 + 0.75 * len(tokens) / max(average, 1.0))
                )
        scores.append(score)
    return scores


def _truncate_utf8(text: str, maximum: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= maximum:
        return text
    return raw[:maximum].decode("utf-8", errors="ignore").rstrip()


def _sentences(text: str) -> list[str]:
    values = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", text)]
    return [item for item in values if item]


def _focused_excerpt(text: str, query: str, maximum: int) -> str:
    sentences = _sentences(text)
    if not sentences:
        return _truncate_utf8(text, maximum)
    pseudo = [{"text": sentence} for sentence in sentences]
    scores = _bm25_scores(query, pseudo)
    order = sorted(range(len(sentences)), key=lambda index: (-scores[index], index))
    selected: list[int] = []
    used = 0
    for index in order:
        candidate = sentences[index]
        size = len(candidate.encode("utf-8")) + (1 if selected else 0)
        if used + size <= maximum:
            selected.append(index)
            used += size
        if len(selected) == 3:
            break
    if not selected:
        return _truncate_utf8(sentences[order[0]], maximum)
    return " ".join(sentences[index] for index in sorted(selected))


def _label(passage: dict[str, Any]) -> str:
    return f"[{passage['article_id']} c{passage['chunk_id']}]"


def _baseline_context(case: dict[str, Any], maximum: int) -> dict[str, Any]:
    parts: list[str] = []
    source_ids: list[str] = []
    remaining = maximum
    for passage in case["candidate_pool"]:
        prefix = f"{_label(passage)} "
        separator = "\n" if parts else ""
        overhead = len((separator + prefix).encode("utf-8"))
        if remaining <= overhead:
            break
        text = _truncate_utf8(passage["text"], remaining - overhead)
        if not text:
            break
        parts.append(prefix + text)
        source_ids.append(passage["passage_id"])
        remaining -= len((separator + parts[-1]).encode("utf-8"))
        if text != passage["text"]:
            break
    context = "\n".join(parts)
    return {"context": context, "source_passage_ids": source_ids}


def _candidate_context(
    case: dict[str, Any], subquestions: tuple[str, str], maximum: int
) -> dict[str, Any]:
    pool = case["candidate_pool"]
    chosen: list[int] = []
    per_slot = max(256, maximum // 2)
    parts: list[str] = []
    selections: list[dict[str, Any]] = []
    for slot, subquestion in enumerate(subquestions, start=1):
        scores = _bm25_scores(subquestion, pool)
        order = sorted(
            range(len(pool)),
            key=lambda index: (-scores[index], pool[index]["dense_rank"]),
        )
        index = next((value for value in order if value not in chosen), order[0])
        chosen.append(index)
        passage = pool[index]
        prefix = f"{_label(passage)} "
        excerpt = _focused_excerpt(
            passage["text"], subquestion, max(1, per_slot - len(prefix.encode("utf-8")))
        )
        parts.append(prefix + excerpt)
        selections.append(
            {
                "slot": slot,
                "subquestion": subquestion,
                "passage_id": passage["passage_id"],
                "dense_rank": passage["dense_rank"],
            }
        )
    context = _truncate_utf8("\n".join(parts), maximum)
    return {
        "context": context,
        "source_passage_ids": [pool[index]["passage_id"] for index in chosen],
        "selections": selections,
    }


def _decomposition_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Decompose the financial multi-hop question into exactly two independent "
                "fact-seeking subquestions. Return strict JSON only: "
                '{"subquestions":["first","second"]}.'
            ),
        },
        {"role": "user", "content": case["question"]},
    ]


def _reader_messages(case: dict[str, Any], context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer both parts of the question using only the supplied SEC evidence. "
                "Return strict JSON only: "
                '{"answer":{"slot_1":"...","slot_2":"..."}}.'
            ),
        },
        {
            "role": "user",
            "content": f"Evidence:\n{context}\n\nQuestion: {case['question']}",
        },
    ]


def _content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("WSE SEC response choices differ")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise ValueError("WSE SEC response did not finish cleanly")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("WSE SEC response content is missing")
    return content


def parse_decomposition(response: dict[str, Any]) -> tuple[str, str]:
    content = _content(response)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("WSE SEC decomposition is invalid JSON") from exc
    subquestions = value.get("subquestions") if isinstance(value, dict) else None
    if (
        not isinstance(subquestions, list)
        or len(subquestions) != 2
        or any(not isinstance(item, str) or not item.strip() for item in subquestions)
    ):
        raise ValueError("WSE SEC decomposition schema differs")
    return subquestions[0].strip(), subquestions[1].strip()


def parse_answer(response: dict[str, Any]) -> dict[str, str]:
    content = _content(response)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("WSE SEC answer is invalid JSON") from exc
    answer = value.get("answer") if isinstance(value, dict) else None
    if (
        not isinstance(answer, dict)
        or set(answer) != {"slot_1", "slot_2"}
        or any(not isinstance(answer[key], str) for key in answer)
    ):
        raise ValueError("WSE SEC answer schema differs")
    return {key: answer[key].strip() for key in ("slot_1", "slot_2")}


def _usage(response: dict[str, Any]) -> dict[str, int]:
    value = response.get("usage")
    if not isinstance(value, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        key: int(value.get(key, 0))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _call(
    client: Any,
    lease: Any,
    manifest: dict[str, Any],
    stage: str,
    messages: list[dict[str, str]],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    prompt_bytes = len(canonical_json_bytes(messages))
    if prompt_bytes > manifest["budget"]["maximum_prompt_utf8_bytes"]:
        raise TransportIncomplete(f"WSE SEC {stage} prompt exceeds byte limit")
    lease.assert_healthy()
    result = client.chat(
        model=manifest["model"],
        messages=messages,
        parameters=parameters,
        lease=lease,
    )
    lease.assert_healthy()
    return {
        "stage": stage,
        "messages_sha256": _sha256_bytes(canonical_json_bytes(messages)),
        "prompt_utf8_bytes": prompt_bytes,
        "response": result.response,
        "response_sha256": _sha256_bytes(canonical_json_bytes(result.response)),
        "usage": _usage(result.response),
        "client_elapsed_seconds": float(result.client_elapsed_seconds),
        "retry_wait_seconds": float(result.retry_wait_seconds),
    }


def _write_immutable(path: Path, value: dict[str, Any]) -> None:
    content = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"different WSE SEC artifact already exists: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _case_binding(case: dict[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(case))


def _build_checkpoint(
    manifest: dict[str, Any],
    manifest_sha: str,
    case: dict[str, Any],
    client: Any,
    lease: Any,
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    decomposition = _call(
        client,
        lease,
        manifest,
        "decomposition",
        _decomposition_messages(case),
        manifest["request_parameters"]["decomposition"],
    )
    calls.append(decomposition)
    decomposition_error = None
    try:
        subquestions = parse_decomposition(decomposition["response"])
    except ValueError as exc:
        decomposition_error = str(exc)
        subquestions = (case["question"], case["retrieval_query"])

    evidence_bytes = manifest["budget"]["evidence_context_utf8_bytes"]
    baseline = _baseline_context(case, evidence_bytes)
    candidate = _candidate_context(case, subquestions, evidence_bytes)
    for name, value in (("baseline", baseline), ("candidate", candidate)):
        record = _call(
            client,
            lease,
            manifest,
            f"reader_{name}",
            _reader_messages(case, value["context"]),
            manifest["request_parameters"]["reader"],
        )
        calls.append(record)
        answer_error = None
        try:
            answer = parse_answer(record["response"])
        except ValueError as exc:
            answer_error = str(exc)
            answer = {"slot_1": "", "slot_2": ""}
        value.update(
            {
                "answer": answer,
                "answer_parse_error": answer_error,
                "reader_call": record,
                "context_sha256": _sha256_bytes(value["context"].encode("utf-8")),
                "context_utf8_bytes": len(value["context"].encode("utf-8")),
            }
        )
    return _seal(
        {
            "schema": CHECKPOINT_SCHEMA,
            "manifest_sha256": manifest_sha,
            "question_id": case["question_id"],
            "case_sha256": _case_binding(case),
            "decomposition": {
                "subquestions": list(subquestions),
                "parse_error": decomposition_error,
                "call": decomposition,
            },
            "baseline": baseline,
            "candidate": candidate,
            "call_inventory": calls,
        }
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("WSE SEC checkpoint is invalid") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError("WSE SEC checkpoint is not canonical")
    artifact = value.get("artifact_sha256")
    unsealed = dict(value)
    unsealed.pop("artifact_sha256", None)
    if artifact != _sha256_bytes(canonical_json_bytes(unsealed)):
        raise ValueError("WSE SEC checkpoint binding differs")
    return value


def recompute_runtime_summary(
    manifest: dict[str, Any], manifest_sha: str, results_root: Path
) -> dict[str, Any]:
    checkpoints = sorted((results_root / "checkpoints").glob("*.json"))
    cases = {case["question_id"]: case for case in manifest["cases"]}
    if len(checkpoints) != len(cases):
        raise ValueError("WSE SEC checkpoint denominator differs")
    rows = [_load_checkpoint(path) for path in checkpoints]
    if {row.get("question_id") for row in rows} != set(cases):
        raise ValueError("WSE SEC checkpoint question inventory differs")
    calls: list[dict[str, Any]] = []
    for row in rows:
        case = cases[row["question_id"]]
        if (
            row.get("schema") != CHECKPOINT_SCHEMA
            or row.get("manifest_sha256") != manifest_sha
            or row.get("case_sha256") != _case_binding(case)
            or len(row.get("call_inventory", ())) != 3
        ):
            raise ValueError("WSE SEC checkpoint manifest binding differs")
        calls.extend(row["call_inventory"])
    expected_calls = len(cases) * 3
    parse_failures = sum(
        row["decomposition"]["parse_error"] is not None for row in rows
    )
    answer_failures = {
        arm: sum(row[arm]["answer_parse_error"] is not None for row in rows)
        for arm in ("baseline", "candidate")
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "runtime_complete",
        "manifest_sha256": manifest_sha,
        "fixed_denominator": {"questions": len(cases), "remote_calls": expected_calls},
        "received_calls": len(calls),
        "decomposition_parse_failures": parse_failures,
        "answer_parse_failures": answer_failures,
        "quality_gate_passed": False if parse_failures or any(answer_failures.values()) else None,
        "cost": {
            "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in calls),
            "completion_tokens": sum(row["usage"]["completion_tokens"] for row in calls),
            "total_tokens": sum(row["usage"]["total_tokens"] for row in calls),
            "client_elapsed_seconds": sum(row["client_elapsed_seconds"] for row in calls),
            "retry_wait_seconds": sum(row["retry_wait_seconds"] for row in calls),
        },
        "claim_guard": manifest["claim_guard"],
    }
    if len(calls) != expected_calls:
        raise ValueError("WSE SEC received-call denominator differs")
    return _seal(summary)


def run(
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    results_root: str | Path,
    client: Any,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = _load_manifest(path, expected_manifest_sha256)
    _validate_live_model(manifest, client.get_models())
    if smoke:
        manifest = json.loads(json.dumps(manifest))
        manifest["cases"] = manifest["cases"][:1]
    root = Path(results_root)
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        for case in manifest["cases"]:
            checkpoint = _build_checkpoint(
                manifest, expected_manifest_sha256, case, client, lease
            )
            _write_immutable(
                root / "checkpoints" / f"{case['question_id']}.json", checkpoint
            )
    summary = recompute_runtime_summary(manifest, expected_manifest_sha256, root)
    _write_immutable(root / ("smoke.json" if smoke else "summary.json"), summary)
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--results-root", default="/results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        summary = run(
            args.manifest,
            args.manifest_sha256,
            args.results_root,
            WSEClient(),
            smoke=args.smoke,
        )
    except TransportIncomplete as exc:
        root = Path(args.results_root)
        root.mkdir(parents=True, exist_ok=True)
        _write_immutable(
            root / "technical_failure.json",
            {
                "schema": "wse-sec-evidence-slot-technical-failure-v1",
                "status": "technical_incomplete",
                "sanitized_error": str(exc),
            },
        )
        return 4
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
