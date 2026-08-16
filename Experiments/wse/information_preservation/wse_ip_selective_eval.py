#!/usr/bin/env python3
"""Run the frozen selective-context information-preservation confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


try:
    from wse_client import WSEClient, canonical_json_bytes, write_checkpoint_atomic, write_json_atomic
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "triple_quality"))
    from wse_eval import WSEClient, canonical_json_bytes, write_checkpoint_atomic, write_json_atomic


EXPECTED_MANIFEST_SHA256_BY_FILENAME: dict[str, str] = {
    "confirmation_gpu01.json": "60872c44b2bb2229a535cd6e1a5be6683509ecce67dace2ba5fc67230c21b1c5",
}
ARMS = ("event", "selective")
_ANSWER_PATTERNS = (
    re.compile(r"^\s*\**\(?([A-D])\)?[.):\s]*\**\s*$", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\**\(?([A-D])\)?[.):\s]*\**\s*$", re.IGNORECASE),
)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if EXPECTED_MANIFEST_SHA256_BY_FILENAME.get(path.name) != digest:
        raise ValueError("manifest hash does not match the pinned image contract")
    value = json.loads(raw.decode("utf-8"))
    if value.get("schema") != "ip-selective-confirmation-manifest-v2":
        raise ValueError("manifest schema mismatch")
    if (
        value.get("role") != "confirmation"
        or value.get("case_count") != 40
        or value.get("cell_count") != 80
        or tuple(value.get("arms", ())) != ARMS
    ):
        raise ValueError("manifest role or cardinality mismatch")
    assignment = value.get("model_assignment", {})
    if (
        assignment.get("server_label") != "gpu01"
        or assignment.get("model") != "Qwen/Qwen3-14B"
        or assignment.get("context_length") != 32768
    ):
        raise ValueError("manifest model binding mismatch")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != 80:
        raise ValueError("manifest cell inventory mismatch")
    if [cell.get("order") for cell in cells] != list(range(80)):
        raise ValueError("manifest cell order mismatch")
    if Counter(cell.get("arm") for cell in cells) != Counter({arm: 40 for arm in ARMS}):
        raise ValueError("manifest arm denominator mismatch")
    pairs: dict[Any, list[Any]] = {}
    for cell in cells:
        pairs.setdefault(cell.get("case_id"), []).append(cell.get("arm"))
        claimed = cell.get("cell_payload_sha256")
        payload = dict(cell)
        payload.pop("cell_payload_sha256", None)
        if claimed != _sha_json(payload):
            raise ValueError("manifest cell payload hash mismatch")
        if cell.get("request_parameters") != value.get("request_parameters"):
            raise ValueError("manifest request parameters mismatch")
    if len(pairs) != 40 or any(Counter(arms) != Counter(ARMS) for arms in pairs.values()):
        raise ValueError("manifest pair inventory mismatch")
    return value, digest


def _parse_response(response: Any) -> tuple[str, str | None]:
    if not isinstance(response, dict):
        return "invalid_answer", None
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return "invalid_answer", None
    choice = choices[0]
    message = choice.get("message")
    if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
        return "invalid_answer", None
    if message.get("refusal") not in (None, "") or not isinstance(message.get("content"), str):
        return "invalid_answer", None
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(message["content"])
        if match:
            return "accepted", match.group(1).upper()
    return "invalid_answer", None


def _verify_checkpoint(cell: dict[str, Any], manifest_sha256: str, row: dict[str, Any]) -> None:
    if row.get("schema") != "ip-selective-confirmation-checkpoint-v2":
        raise ValueError("checkpoint schema mismatch")
    if (
        row.get("manifest_sha256") != manifest_sha256
        or row.get("cell_id") != cell["cell_id"]
        or row.get("cell_payload_sha256") != cell["cell_payload_sha256"]
    ):
        raise ValueError("checkpoint binding mismatch")


def recompute_summary(manifest: dict[str, Any], manifest_sha256: str, results_root: Path) -> dict[str, Any]:
    checkpoint_dir = results_root / "checkpoints"
    expected = {f"{cell['cell_id']}.json" for cell in manifest["cells"]}
    actual = {path.name for path in checkpoint_dir.glob("*.json")} if checkpoint_dir.exists() else set()
    if actual != expected:
        raise ValueError("checkpoint inventory mismatch")
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    rows: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        row = json.loads(path.read_text("utf-8"))
        cell = cells.get(row.get("cell_id"))
        if cell is None:
            raise ValueError("unknown checkpoint cell")
        _verify_checkpoint(cell, manifest_sha256, row)
        rows.append(row)

    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        correct = sum(bool(row["correct"]) for row in selected)
        arms[arm] = {
            "total": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected),
            "invalid_answers": sum(row["status"] != "accepted" for row in selected),
            "prompt_tokens": sum(int(row["usage"].get("prompt_tokens", 0) or 0) for row in selected),
            "output_tokens": sum(int(row["usage"].get("completion_tokens", 0) or 0) for row in selected),
            "wall_seconds": sum(float(row["client_elapsed_seconds"]) for row in selected),
        }
    event = arms["event"]
    selective = arms["selective"]
    accuracy_delta = selective["accuracy"] - event["accuracy"]
    prompt_reduction = (
        (event["prompt_tokens"] - selective["prompt_tokens"]) / event["prompt_tokens"]
        if event["prompt_tokens"]
        else None
    )
    frozen_gate = manifest["quality_gate"]
    gate = {
        "accuracy_delta": accuracy_delta,
        "minimum_accuracy_delta": float(frozen_gate["minimum_accuracy_delta"]),
        "prompt_token_reduction": prompt_reduction,
        "minimum_prompt_token_reduction": float(frozen_gate["minimum_prompt_token_reduction"]),
        "passed": (
            accuracy_delta >= float(frozen_gate["minimum_accuracy_delta"])
            and prompt_reduction is not None
            and prompt_reduction >= float(frozen_gate["minimum_prompt_token_reduction"])
        ),
    }
    return {
        "schema": "ip-selective-confirmation-summary-v2",
        "status": "confirmation_complete",
        "manifest_sha256": manifest_sha256,
        "model_assignment": manifest["model_assignment"],
        "runtime_sha256": _sha_file(Path(__file__)),
        "fixed_denominator": 80,
        "received_responses": len(rows),
        "invalid_answers": sum(row["status"] != "accepted" for row in rows),
        "arms": arms,
        "quality_gate": gate,
        "claim_guard": manifest["claim_guard"],
    }


def run(manifest_path: Path, results_root: Path, client: Any) -> dict[str, Any]:
    manifest, digest = _manifest(manifest_path)
    models = client.get_models()
    exact = [row for row in models if (row.get("name") or row.get("id")) == "Qwen/Qwen3-14B"]
    if len(exact) != 1:
        raise ValueError("manifest model is not uniquely available")
    live_context = exact[0].get("context_length", exact[0].get("max_model_len"))
    if live_context is not None and int(live_context) < int(
        manifest["model_assignment"]["context_length"]
    ):
        raise ValueError("manager model context contradicts the frozen manifest")
    checkpoint_dir = results_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        for cell in manifest["cells"]:
            path = checkpoint_dir / f"{cell['cell_id']}.json"
            if path.exists():
                _verify_checkpoint(cell, digest, json.loads(path.read_text("utf-8")))
                continue
            lease.assert_healthy()
            result = client.chat(
                model=manifest["model_assignment"]["model"],
                messages=cell["messages"],
                parameters=cell["request_parameters"],
                lease=lease,
            )
            status, letter = _parse_response(result.response)
            checkpoint = {
                "schema": "ip-selective-confirmation-checkpoint-v2",
                "manifest_sha256": digest,
                "cell_id": cell["cell_id"],
                "cell_payload_sha256": cell["cell_payload_sha256"],
                "arm": cell["arm"],
                "status": status,
                "parsed_letter": letter,
                "correct": letter == cell["gold_letter"],
                "prompt_sha256": cell["prompt_sha256"],
                "manager_response": result.response,
                "manager_response_sha256": _sha_json(result.response),
                "usage": result.response.get("usage", {}) if isinstance(result.response, dict) else {},
                "client_elapsed_seconds": result.client_elapsed_seconds,
                "retry_wait_seconds": result.retry_wait_seconds,
            }
            write_checkpoint_atomic(path, checkpoint)
            lease.assert_healthy()
    summary = recompute_summary(manifest, digest, results_root)
    write_json_atomic(results_root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    manifest, digest = _manifest(args.manifest)
    summary = (
        recompute_summary(manifest, digest, args.results_root)
        if args.check
        else run(args.manifest, args.results_root, WSEClient())
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0 if summary["quality_gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
