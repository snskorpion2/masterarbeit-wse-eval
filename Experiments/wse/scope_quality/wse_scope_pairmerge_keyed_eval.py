"""Wire-normalized pair decisions with deterministic closed projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wse_eval import WSEClient, canonical_json_bytes
import wse_scope_pairmerge_eval as prior


CHECKPOINT_SCHEMA = "wse-scope-evidence-pairmerge-deterministic-development-v1"
MANIFEST_SCHEMA = "wse-scope-evidence-pairmerge-deterministic-manifest-v1"
PROTOCOL = "wse-scope-evidence-pairmerge-deterministic-development-v1"
SUMMARY_SCHEMA = "wse-scope-evidence-pairmerge-deterministic-summary-v1"
FAILURE_SCHEMA = "wse-scope-evidence-pairmerge-deterministic-technical-failure-v1"
EXPECTED_MANIFEST_SHA256: dict[str, str] = {}

legacy = prior.legacy
partition = prior.partition
_sha256 = prior._sha256
_costs = prior._costs
close_pair_decisions = prior.close_pair_decisions


def project_closed_partition(
    baseline: dict[str, Any], audit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply accepted label unions without another model-dependent mapping stage."""
    raw_types = audit.get("raw_types")
    raw_relations = audit.get("raw_relations")
    type_assignments = audit.get("type_assignments")
    relation_assignments = audit.get("relation_assignments")
    if not all(
        isinstance(value, list)
        for value in (raw_types, raw_relations, type_assignments, relation_assignments)
    ):
        raise ValueError("closed pair-partition audit is incomplete")

    def assignment_map(rows: list[dict[str, Any]], size: int) -> dict[int, int]:
        observed: dict[int, int] = {}
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"source_id", "representative_id"}
                or type(row["source_id"]) is not int
                or type(row["representative_id"]) is not int
                or row["source_id"] in observed
                or not 0 <= row["source_id"] < size
                or not 0 <= row["representative_id"] < size
            ):
                raise ValueError("closed pair-partition assignment is invalid")
            observed[row["source_id"]] = row["representative_id"]
        if set(observed) != set(range(size)):
            raise ValueError("closed pair-partition assignment is incomplete")
        return observed

    type_ids = {label: index for index, label in enumerate(raw_types)}
    relation_ids = {label: index for index, label in enumerate(raw_relations)}
    if len(type_ids) != len(raw_types) or len(relation_ids) != len(raw_relations):
        raise ValueError("closed pair-partition vocabulary is invalid")
    type_map = assignment_map(type_assignments, len(raw_types))
    relation_map = assignment_map(relation_assignments, len(raw_relations))
    mappings: list[dict[str, Any]] = []
    for edge in baseline["edges"]:
        source = legacy._source_edge(edge)
        try:
            head = raw_types[type_map[type_ids[source["head_type"]]]]
            relation = raw_relations[
                relation_map[relation_ids[source["relation_type"]]]
            ]
            tail = raw_types[type_map[type_ids[source["tail_type"]]]]
        except (KeyError, IndexError) as error:
            raise ValueError("baseline edge is outside the closed vocabulary") from error
        mappings.append(
            {
                "source_edge_id": source["source_edge_id"],
                "head_type": head,
                "relation_type": relation,
                "tail_type": tail,
                "reverse": False,
            }
        )
    return legacy._project_global_mapping(baseline, tuple(mappings))


def _request(rows: list[dict[str, Any]]) -> dict[str, Any]:
    system = (
        "Judge whether each same-kind schema-label pair is interchangeable in every "
        "source-edge context. Merge spelling variants, inflections and true "
        "synonyms. Reject antonyms, inverse relations, siblings, broader/narrower "
        "and merely related meanings. Return JSON only as one object mapping each "
        "supplied pair_id to Boolean true or false. Include every pair_id exactly "
        "once. No explanation or extra keys."
    )
    reference_size = len(prior._request([])["system_prompt"].encode("utf-8"))
    encoded_size = len(system.encode("utf-8"))
    if encoded_size > reference_size:
        raise ValueError("keyed response contract exceeds V2 prompt boundary")
    system += " " * (reference_size - encoded_size)
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
    manifest: dict[str, Any], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    return prior.build_pair_packets(
        manifest, baseline, request_builder=_request
    )


def parse_pair_decisions(
    response: dict[str, Any], expected_pairs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    value = json.loads(legacy._finished_content(response))
    expected = [pair["pair_id"] for pair in expected_pairs]
    if not isinstance(value, dict):
        raise ValueError("normalized pair-merge response schema is invalid")

    observed: dict[str, bool] = {}
    if set(value) == set(expected):
        if any(type(value[pair_id]) is not bool for pair_id in expected):
            raise ValueError("normalized pair-merge decision is invalid")
        observed = {pair_id: value[pair_id] for pair_id in expected}
    elif set(value) == {"decisions"}:
        decisions = value["decisions"]
        if isinstance(decisions, list):
            rows = [(None, row) for row in decisions]
        elif isinstance(decisions, dict) and set(decisions) == set(expected):
            rows = [(pair_id, decisions[pair_id]) for pair_id in expected]
        else:
            raise ValueError("normalized pair-merge decision coverage is invalid")
        for container_id, row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"pair_id", "merge"}
                or not isinstance(row.get("pair_id"), str)
                or (container_id is not None and row["pair_id"] != container_id)
                or row["pair_id"] in observed
                or type(row.get("merge")) is not bool
            ):
                raise ValueError("normalized pair-merge decision is invalid")
            observed[row["pair_id"]] = row["merge"]
    else:
        raise ValueError("normalized pair-merge decision coverage is invalid")
    if set(observed) != set(expected):
        raise ValueError("normalized pair-merge decision coverage is invalid")
    return [{"pair_id": pair_id, "merge": observed[pair_id]} for pair_id in expected]


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    observed = _sha256(raw)
    expected = EXPECTED_MANIFEST_SHA256.get(path.name)
    if EXPECTED_MANIFEST_SHA256 and observed != expected:
        raise ValueError("keyed pair-merge manifest bytes differ from binding")
    manifest = json.loads(raw.decode("utf-8"))
    semantic = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("protocol_id") != PROTOCOL
        or manifest.get("response_contract") != "decision-map-normalized-v1"
        or manifest.get("projection_contract")
        != "deterministic-closed-union-substitution-v1"
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
        raise ValueError("normalized pair-merge manifest contract mismatch")
    return manifest, observed


def run_pair_merge(
    manifest_path: str | Path,
    baseline_path: str | Path,
    results_root: str | Path,
    client: Any,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(Path(manifest_path))
    if manifest.get("baseline_mode") not in (None, "bound_existing"):
        raise ValueError("keyed pair-merge baseline mode is invalid")
    baseline, baseline_sha = partition.load_bound_baseline(
        Path(baseline_path), manifest
    )
    partition._validate_live_model(manifest, list(client.get_models()))
    with client.model_lease(ttl_seconds=7200, auto_renew=True) as lease:
        return prior._execute(
            manifest=manifest,
            manifest_sha=manifest_sha,
            baseline=baseline,
            baseline_sha=baseline_sha,
            results_root=Path(results_root),
            client=client,
            lease=lease,
            packet_builder=build_pair_packets,
            decision_parser=parse_pair_decisions,
            deterministic_projector=project_closed_partition,
            checkpoint_schema=CHECKPOINT_SCHEMA,
            failure_schema=FAILURE_SCHEMA,
            summary_schema=SUMMARY_SCHEMA,
            method=(
                "semantic blocking plus evidence-bound wire-normalized pair decisions "
                "and deterministic closed-union projection"
            ),
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
