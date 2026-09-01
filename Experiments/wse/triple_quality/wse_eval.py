#!/usr/bin/env python3
"""Secrets-free WSE runtime for the frozen triple-quality protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple


PROTOCOL_ID = "wse-triple-quality-v1"
V2_PROTOCOL_ID = "wse-triple-quality-v2"
ALLOWED_MANIFESTS = {
    "development_gpu01.json",
    "confirmation_gpu01.json",
    "confirmation_h200.json",
    "confirmation_qwen3_14b_gpu01.json",
    "confirmation_v2_qwen3_14b_gpu01.json",
}
STATUSES = {"ok", "valid_empty", "refusal", "parse_error", "evidence_rejected"}
EXPECTED_MANIFEST_SHA256 = {}
FROZEN_REQUEST_PARAMETERS = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 4096,
    "seed": 20260811,
    "n": 1,
    "chat_template_kwargs": {"enable_thinking": False},
}
V2_FROZEN_REQUEST_PARAMETERS = {
    **FROZEN_REQUEST_PARAMETERS,
    "seed": 20260813,
}


class LeaseUnavailable(RuntimeError):
    """The requested model slot could not be leased."""


class LeaseRenewalFailed(RuntimeError):
    """Automatic renewal failed; no further chat may be sent."""


class TransportIncomplete(RuntimeError):
    """No complete manager response was received."""


class RetryPolicy(NamedTuple):
    max_total_wait_seconds: float = 8 * 3600
    max_single_sleep_seconds: float = 65 * 60
    backoff_cap_seconds: float = 120


class ChatResult(NamedTuple):
    response: dict[str, Any]
    client_elapsed_seconds: float
    retry_wait_seconds: float


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_checkpoint_atomic(path: str | Path, checkpoint: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        if target.read_bytes() == canonical_json_bytes(checkpoint):
            return
        raise ValueError(f"different checkpoint already exists: {target.name}")
    write_json_atomic(target, checkpoint)


def _retry_after_seconds(headers: Any) -> float | None:
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return None


class ModelLease:
    def __init__(
        self,
        client: "WSEClient",
        lease_id: str,
        ttl_seconds: int,
        auto_renew: bool,
    ) -> None:
        self.client = client
        self.lease_id = lease_id
        self.ttl_seconds = ttl_seconds
        self.auto_renew = auto_renew
        self.renew_failures = 0
        self._stop = threading.Event()
        self._renew_in_progress = threading.Event()
        self._thread: threading.Thread | None = None
        self._expires_at = time.monotonic() + ttl_seconds

    def __enter__(self) -> "ModelLease":
        if self.auto_renew:
            self._thread = threading.Thread(target=self._renew_loop, daemon=True)
            self._thread.start()
        return self

    def _renew_loop(self) -> None:
        interval = max(min(self.ttl_seconds / 3, 300), 0.05)
        while not self._stop.wait(interval):
            self._renew_in_progress.set()
            try:
                self.client.renew_lease(self.lease_id, self.ttl_seconds)
                self._expires_at = time.monotonic() + self.ttl_seconds
            except Exception:
                self.renew_failures += 1
                self._stop.set()
            finally:
                self._renew_in_progress.clear()

    def assert_healthy(self) -> None:
        if self.renew_failures:
            raise LeaseRenewalFailed("lease renewal failed")
        if self._renew_in_progress.is_set():
            raise LeaseRenewalFailed("lease renewal is in progress")
        if time.monotonic() >= self._expires_at:
            raise LeaseRenewalFailed("lease expiry could not be ruled out")
        if self.auto_renew and (
            self._thread is None or (not self._thread.is_alive() and not self._stop.is_set())
        ):
            raise LeaseRenewalFailed("lease renewal thread is not healthy")

    def __exit__(self, *_args: Any) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=50)
            if self._thread.is_alive():
                self.renew_failures += 1
        try:
            self.client.release_lease(self.lease_id)
        finally:
            if self.renew_failures and not any(_args):
                raise LeaseRenewalFailed("lease renewal did not terminate safely")
        return False


class WSEClient:
    """Independent standard-library implementation of the documented API."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        retry_policy: RetryPolicy | None = None,
        on_retry: Callable[[float], None] | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ["VLLM_MANAGER_URL"]).rstrip("/")
        self._token = token or os.environ["VLLM_MANAGER_TOKEN"]
        self.retry_policy = retry_policy or RetryPolicy()
        self.on_retry = on_retry

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
        request_timeout_seconds: float = 300,
        max_elapsed_seconds: float | None = None,
        nonretryable_http_statuses: frozenset[int] = frozenset(),
    ) -> ChatResult:
        started = time.monotonic()
        policy = retry_policy or self.retry_policy
        retry_wait = 0.0
        attempt = 0
        last_error: BaseException | None = None
        while True:
            elapsed = time.monotonic() - started
            if max_elapsed_seconds is not None and elapsed >= max_elapsed_seconds:
                raise TransportIncomplete("manager request deadline exhausted")
            request_timeout = request_timeout_seconds
            if max_elapsed_seconds is not None:
                request_timeout = min(request_timeout, max_elapsed_seconds - elapsed)
            body = canonical_json_bytes(payload) if payload is not None else None
            request = urllib.request.Request(
                f"{self.base_url}{path}",
                data=body,
                method=method,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=request_timeout) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                return ChatResult(decoded, time.monotonic() - started, retry_wait)
            except urllib.error.HTTPError as error:
                last_error = error
                error_body = ""
                try:
                    error_body = " ".join(
                        error.read(2048).decode("utf-8", errors="replace").split()
                    )[:500]
                except (OSError, ValueError):
                    pass
                if "Bearer " in error_body or "Authorization" in error_body:
                    error_body = "<redacted>"
                detail = f": {error_body}" if error_body else ""
                if error.code in nonretryable_http_statuses:
                    raise TransportIncomplete(
                        f"manager HTTP {error.code}{detail}"
                    ) from None
                retryable = error.code == 429 or 500 <= error.code < 600
                retry_after = _retry_after_seconds(error.headers)
                if not retryable:
                    raise TransportIncomplete(
                        f"manager HTTP {error.code}{detail}"
                    ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
                retry_after = None
            attempt += 1
            backoff = min(
                policy.backoff_cap_seconds,
                2 ** min(attempt - 1, 16) + random.random(),
            )
            wait = retry_after if retry_after is not None else backoff
            wait = min(wait, policy.max_single_sleep_seconds)
            elapsed = time.monotonic() - started
            if retry_wait + wait > policy.max_total_wait_seconds or (
                max_elapsed_seconds is not None and elapsed + wait >= max_elapsed_seconds
            ):
                raise TransportIncomplete(
                    "manager retry budget exhausted"
                ) from last_error
            retry_wait += wait
            if self.on_retry is not None:
                self.on_retry(wait)
            time.sleep(wait)

    def get_models(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/models").response
        models = response.get("data", response.get("models", response))
        if not isinstance(models, list):
            raise TransportIncomplete("invalid models response")
        return models

    @contextmanager
    def model_lease(self, *, ttl_seconds: int, auto_renew: bool):
        try:
            response = self._request(
                "POST",
                "/lock/acquire",
                {"ttl_seconds": ttl_seconds},
                nonretryable_http_statuses=frozenset({409, 423, 503}),
            ).response
        except TransportIncomplete as error:
            raise LeaseUnavailable("manager did not grant a lease") from error
        lease_id = response.get("lease_id") or response.get("id")
        if not lease_id:
            raise LeaseUnavailable("manager did not grant a lease")
        with ModelLease(self, str(lease_id), ttl_seconds, auto_renew) as lease:
            yield lease

    def renew_lease(self, lease_id: str, ttl_seconds: int) -> None:
        self._request(
            "POST",
            "/lock/renew",
            {"lease_id": lease_id, "ttl_seconds": ttl_seconds},
            retry_policy=RetryPolicy(30, 10, 5),
            request_timeout_seconds=10,
            max_elapsed_seconds=45,
        )

    def release_lease(self, lease_id: str) -> None:
        self._request(
            "POST",
            "/lock/release",
            {"lease_id": lease_id},
            retry_policy=RetryPolicy(10, 5, 2),
            request_timeout_seconds=5,
            max_elapsed_seconds=15,
        )

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        parameters: dict[str, Any],
        lease: ModelLease,
    ) -> ChatResult:
        payload = {"model": model, "messages": messages, **parameters}
        payload["lease_id"] = lease.lease_id
        return self._request(
            "POST",
            "/v1/chat/completions",
            payload,
            max_elapsed_seconds=self.retry_policy.max_total_wait_seconds,
        )


def _runtime_hash() -> str:
    return _sha256_file(Path(__file__))


def catalog(client: Any, server_label: str, output: str | Path) -> dict[str, Any]:
    allowed = {
        "architecture": ("architecture", "architectures"),
        "parameter_count": ("parameter_count", "parameters"),
        "quantization": ("quantization",),
        "context_length": ("context_length", "max_model_len"),
    }
    projected = []
    for model in client.get_models():
        name = model.get("name") or model.get("id")
        if not name:
            continue
        row: dict[str, Any] = {"name": name}
        for output_name, source_names in allowed.items():
            for source_name in source_names:
                if source_name in model:
                    row[output_name] = model[source_name]
                    break
        projected.append(row)
    result = {
        "schema": "wse-model-catalog-v1",
        "status": "catalog_complete",
        "server_label": server_label,
        "created_at_utc": _utc_now(),
        "runtime_sha256": _runtime_hash(),
        "chat_requests": 0,
        "models": projected,
    }
    write_json_atomic(output, result)
    return result


def _message(response: dict[str, Any]) -> dict[str, Any]:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise TransportIncomplete("manager response has no message") from error
    if not isinstance(message, dict):
        raise TransportIncomplete("manager message is invalid")
    return message


def smoke(
    client: Any, server_label: str, model: str, output: str | Path
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": 'Return exactly this JSON object: {"echo":"wse-smoke-v1"}',
        }
    ]
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        _assert_lease_healthy(lease)
        result = client.chat(
            model=model,
            messages=messages,
            parameters={
                "temperature": 0,
                "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            lease=lease,
        )
        _assert_lease_healthy(lease)
    try:
        echo = json.loads(_message(result.response).get("content", ""))
    except json.JSONDecodeError as error:
        raise ValueError("smoke response is not JSON") from error
    if echo != {"echo": "wse-smoke-v1"}:
        raise ValueError("smoke echo mismatch")
    output_value = {
        "schema": "wse-smoke-v1",
        "status": "smoke_complete",
        "scientific": False,
        "server_label": server_label,
        "model": model,
        "runtime_sha256": _runtime_hash(),
        "chat_requests": 1,
        "echo": echo,
        "usage": result.response.get("usage", {}),
        "client_elapsed_seconds": result.client_elapsed_seconds,
        "retry_wait_seconds": result.retry_wait_seconds,
        "request_elapsed_excluding_retry_sleep_seconds": max(
            result.client_elapsed_seconds - result.retry_wait_seconds, 0
        ),
    }
    write_json_atomic(output, output_value)
    return output_value


def smoke_cell(
    client: Any, manifest_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    model, server_label = _validate_manifest(path, manifest, manifest_sha256)
    cells = [
        cell
        for cell in manifest["cells"]
        if cell.get("stage") == "EV"
        and cell.get("arm") == "evidence_bound_candidate"
    ]
    if not cells:
        raise ValueError("manifest has no EV candidate smoke cell")
    cell = cells[0]
    _validate_live_model(
        model,
        int(manifest["model_assignment"]["context_length"]),
        client.get_models(),
    )
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        _assert_lease_healthy(lease)
        result = client.chat(
            model=model,
            messages=cell["messages"],
            parameters=cell["request_parameters"],
            lease=lease,
        )
        _assert_lease_healthy(lease)
    response = _parse_received(cell, result.response)
    if response.get("status") != "ok":
        raise ValueError(f"EV candidate smoke failed: {response.get('status')}")
    output_value = {
        "schema": "wse-smoke-cell-v1",
        "status": "smoke_complete",
        "scientific": False,
        "server_label": server_label,
        "model": model,
        "runtime_sha256": _runtime_hash(),
        "chat_requests": 1,
        "cell_id": cell["cell_id"],
        "stage": cell["stage"],
        "arm": cell["arm"],
        "response": response,
        "usage": result.response.get("usage", {}),
        "client_elapsed_seconds": result.client_elapsed_seconds,
        "retry_wait_seconds": result.retry_wait_seconds,
        "request_elapsed_excluding_retry_sleep_seconds": max(
            result.client_elapsed_seconds - result.retry_wait_seconds, 0
        ),
    }
    write_json_atomic(output, output_value)
    return output_value


def _assert_lease_healthy(lease: Any) -> None:
    if hasattr(lease, "assert_healthy"):
        lease.assert_healthy()
    elif lease.renew_failures:
        raise LeaseRenewalFailed("lease renewal failed")


def _validate_manifest(
    path: Path, manifest: dict[str, Any], manifest_sha256: str
) -> tuple[str, str]:
    if path.name not in ALLOWED_MANIFESTS:
        raise ValueError("unsupported manifest name")
    if EXPECTED_MANIFEST_SHA256.get(path.name) != manifest_sha256:
        raise ValueError("manifest hash does not match the pinned image contract")
    if manifest.get("schema") != "wse-triple-quality-manifest-v1":
        raise ValueError("manifest schema mismatch")
    protocol_id = manifest.get("protocol_id")
    if protocol_id not in {PROTOCOL_ID, V2_PROTOCOL_ID} or manifest.get("execution_ready") is not True:
        raise ValueError("manifest is not execution-ready")
    assignment = manifest.get("model_assignment")
    if not isinstance(assignment, dict) or not assignment.get("model"):
        raise ValueError("manifest does not bind a model")
    if assignment.get("context_length", 0) < 32768:
        raise ValueError("model does not satisfy the 32K context budget")
    expected_server = path.stem.rsplit("_", 1)[-1]
    if assignment.get("server_label") != expected_server:
        raise ValueError("manifest server/model binding mismatch")
    confirmation = path.name.startswith("confirmation_")
    expected_phase = (
        "confirmation-v2-cases"
        if protocol_id == V2_PROTOCOL_ID
        else ("confirmation-cases" if confirmation else "development")
    )
    expected_cases = 20 if confirmation else 11
    expected_cells = 120 if confirmation else 66
    if (
        manifest.get("phase") != expected_phase
        or manifest.get("case_count") != expected_cases
        or manifest.get("cell_count") != expected_cells
        or manifest.get("stages") != ["EE", "EV", "VV"]
        or manifest.get("arms")
        != ["autoschemakg_baseline", "evidence_bound_candidate"]
    ):
        raise ValueError("manifest frozen phase/cardinality contract mismatch")
    cases = manifest.get("cases")
    cells = manifest.get("cells")
    if (
        not isinstance(cases, list)
        or len(cases) != expected_cases
        or len({case.get("case_id") for case in cases if isinstance(case, dict)})
        != expected_cases
        or not isinstance(cells, list)
        or len(cells) != expected_cells
    ):
        raise ValueError("manifest cell count mismatch")
    if len({cell.get("cell_id") for cell in cells if isinstance(cell, dict)}) != expected_cells:
        raise ValueError("manifest cell IDs are not unique")
    if [cell.get("order") for cell in cells] != list(range(len(cells))):
        raise ValueError("manifest frozen order mismatch")
    for index, cell in enumerate(cells):
        pair_index = index // 2
        arm_order = (
            ("autoschemakg_baseline", "evidence_bound_candidate")
            if pair_index % 2 == 0
            else ("evidence_bound_candidate", "autoschemakg_baseline")
        )
        if (
            cell.get("protocol_id") != protocol_id
            or cell.get("phase") != expected_phase
            or cell.get("pair_index") != pair_index
            or cell.get("stage") != ("EE", "EV", "VV")[pair_index % 3]
            or cell.get("arm") != arm_order[index % 2]
            or cell.get("request_parameters")
            != (
                V2_FROZEN_REQUEST_PARAMETERS
                if protocol_id == V2_PROTOCOL_ID
                else FROZEN_REQUEST_PARAMETERS
            )
        ):
            raise ValueError("manifest frozen cell contract mismatch")
        wire = {"messages": cell.get("messages"), **cell.get("request_parameters", {})}
        if cell.get("prompt_sha256") != sha256_json(cell.get("messages")):
            raise ValueError("manifest prompt hash mismatch")
        if cell.get("cell_payload_sha256") != sha256_json(wire):
            raise ValueError("manifest request hash mismatch")
    return assignment["model"], assignment["server_label"]


def _parse_received(cell: dict[str, Any], manager_response: dict[str, Any]) -> dict[str, Any]:
    try:
        message = _message(manager_response)
    except TransportIncomplete:
        return {"status": "parse_error"}
    refusal = message.get("refusal")
    content = message.get("content") or ""
    if refusal:
        return {"status": "refusal", "refusal_present": True}
    try:
        triples = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"status": "parse_error"}
    if not isinstance(triples, list):
        return {"status": "parse_error"}
    if not triples:
        return {"status": "valid_empty", "triples": []}
    if not all(isinstance(row, dict) for row in triples):
        return {"status": "parse_error"}
    if cell.get("arm") == "evidence_bound_candidate":
        marker = "Here is the passage:"
        user_content = cell["messages"][-1]["content"]
        source = user_content.split(marker, 1)[1] if marker in user_content else ""
        if any(
            not isinstance(row.get("Evidence"), str)
            or not row["Evidence"]
            or row["Evidence"] not in source
            for row in triples
        ):
            return {"status": "evidence_rejected", "triples": triples}
        if cell.get("protocol_id") == V2_PROTOCOL_ID:
            stage = cell.get("stage")
            if stage == "EV":
                valid = all(
                    set(row) == {"Event", "Entity", "Evidence"}
                    and isinstance(row["Event"], str)
                    and bool(row["Event"].strip())
                    and isinstance(row["Entity"], list)
                    and bool(row["Entity"])
                    and all(
                        isinstance(entity, str) and bool(entity.strip())
                        for entity in row["Entity"]
                    )
                    for row in triples
                )
            else:
                valid = all(
                    set(row) == {"Head", "Relation", "Tail", "Evidence"}
                    and all(
                        isinstance(row[field], str) and bool(row[field].strip())
                        for field in row
                    )
                    for row in triples
                )
            if not valid:
                return {"status": "parse_error", "triples": triples}
    return {"status": "ok", "triples": triples}


def _summary(manifest: dict[str, Any], checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    if len(checkpoints) != manifest["cell_count"]:
        raise ValueError("incomplete checkpoint matrix")
    expected = [cell["cell_id"] for cell in manifest["cells"]]
    if [checkpoint.get("cell_id") for checkpoint in checkpoints] != expected:
        raise ValueError("incomplete checkpoint matrix")
    statuses = [checkpoint["response"]["status"] for checkpoint in checkpoints]
    if any(status not in STATUSES for status in statuses):
        raise ValueError("invalid response status")
    return {
        "schema": "wse-triple-quality-summary-v1",
        "fixed_denominator": len(expected),
        "received_count": len(statuses),
        "status_counts": dict(sorted(Counter(statuses).items())),
        "complete": True,
    }


def _validate_live_model(model: str, manifest_context: int, rows: list[dict[str, Any]]) -> None:
    available = [row for row in rows if (row.get("name") or row.get("id")) == model]
    if len(available) != 1:
        raise ValueError("manifest model is not uniquely available")
    manager_context = available[0].get(
        "context_length", available[0].get("max_model_len")
    )
    if manager_context is not None and int(manager_context) < manifest_context:
        raise ValueError("manager model context contradicts the frozen manifest")


def run(
    manifest_path: str | Path, results_root: str | Path, client: Any
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    model, _server = _validate_manifest(path, manifest, manifest_sha256)
    _validate_live_model(
        model,
        int(manifest["model_assignment"]["context_length"]),
        client.get_models(),
    )
    root = Path(results_root)
    checkpoints: list[dict[str, Any]] = []
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        for cell in manifest["cells"]:
            _assert_lease_healthy(lease)
            result = client.chat(
                model=model,
                messages=cell["messages"],
                parameters=cell["request_parameters"],
                lease=lease,
            )
            response = _parse_received(cell, result.response)
            checkpoint = {
                "schema": "wse-triple-quality-checkpoint-v1",
                "cell_id": cell["cell_id"],
                "order": cell["order"],
                "manifest_sha256": manifest_sha256,
                "cell_payload_sha256": cell["cell_payload_sha256"],
                "prompt_sha256": cell["prompt_sha256"],
                "request_parameters": cell["request_parameters"],
                "model_assignment": manifest["model_assignment"],
                "manager_response": result.response,
                "usage": (
                    result.response.get("usage", {})
                    if isinstance(result.response, dict)
                    else {}
                ),
                "client_elapsed_seconds": result.client_elapsed_seconds,
                "retry_wait_seconds": result.retry_wait_seconds,
                "request_elapsed_excluding_retry_sleep_seconds": max(
                    result.client_elapsed_seconds - result.retry_wait_seconds, 0
                ),
                "response": response,
            }
            write_checkpoint_atomic(
                root / "checkpoints" / f"{cell['cell_id']}.json", checkpoint
            )
            checkpoints.append(checkpoint)
            _assert_lease_healthy(lease)
    summary = _summary(manifest, checkpoints)
    summary.update(
        {
            "manifest_sha256": manifest_sha256,
            "model_assignment": manifest["model_assignment"],
            "runtime_sha256": _runtime_hash(),
        }
    )
    write_json_atomic(root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog_parser = commands.add_parser("catalog")
    catalog_parser.add_argument("--server-label", required=True, choices=("gpu01", "h200"))
    catalog_parser.add_argument("--output", required=True)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--server-label", required=True, choices=("gpu01", "h200"))
    smoke_parser.add_argument("--model", required=True)
    smoke_parser.add_argument("--output", required=True)
    smoke_cell_parser = commands.add_parser("smoke-cell")
    smoke_cell_parser.add_argument("--manifest", required=True)
    smoke_cell_parser.add_argument("--output", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--results-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = WSEClient()
    if args.command == "catalog":
        catalog(client, args.server_label, args.output)
    elif args.command == "smoke":
        smoke(client, args.server_label, args.model, args.output)
    elif args.command == "smoke-cell":
        smoke_cell(client, args.manifest, args.output)
    else:
        run(args.manifest, args.results_root, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
