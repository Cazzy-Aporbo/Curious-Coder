"""Resolve bounded factual requests before choosing how to explain them."""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re

from studies.data import ROOT


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class Source:
    source_id: str
    revision: str
    context: str
    payload: bytes
    expected_sha256: str
    units: dict
    allowed_uses: tuple
    limits: tuple
    valid_from: int = 0
    valid_until: int = 10
    disposition: str = "approved"


@dataclass(frozen=True)
class Fact:
    key: str
    pointer: str
    unit: str


@dataclass(frozen=True)
class Request:
    context: str
    purpose: str
    facts: tuple


@dataclass(frozen=True)
class Policy:
    revision: str
    clock: int
    approved_revisions: dict
    max_source_bytes: int = 2000000


def pointer_value(document, pointer):
    if not pointer.startswith("/"):
        raise ValueError("Use a non-root JSON pointer for an explicit scalar fact.")
    current = document
    for token in pointer.split("/")[1:]:
        if re.search(r"~(?![01])", token):
            raise ValueError("Invalid JSON-pointer escape.")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current[token]
        elif isinstance(current, list) and re.fullmatch(r"0|[1-9][0-9]*", token):
            current = current[int(token)]
        else:
            raise KeyError(token)
    if not isinstance(current, (str, int, float, bool)) or current is None:
        raise ValueError("A fact must resolve to a scalar, not a partial document.")
    encoded(current)
    return current


def validate_contract(request, sources, policy):
    if not request.context or not request.purpose or not request.facts or len(request.facts) > 32 or len(sources) > 32:
        raise ValueError("Require a bounded request with explicit context, purpose, and facts.")
    if any(not all(isinstance(value, str) and value.strip() for value in (fact.key, fact.pointer, fact.unit)) for fact in request.facts):
        raise ValueError("Fact identity, pointer, and unit must be explicit strings.")
    if len({fact.key for fact in request.facts}) != len(request.facts):
        raise ValueError("Fact keys must be unique.")
    if len({(source.source_id, source.revision) for source in sources}) != len(sources):
        raise ValueError("Duplicate source identity/revision requires reconciliation.")
    if type(policy.clock) is not int or policy.clock < 0 or not 1 <= policy.max_source_bytes <= 5000000:
        raise ValueError("Invalid bounded policy clock or payload budget.")
    for source in sources:
        if not source.source_id or not source.revision or not source.context or type(source.valid_from) is not int or type(source.valid_until) is not int or source.valid_from < 0 or source.valid_until <= source.valid_from:
            raise ValueError("Source identity, revision, context, and integer validity interval must be explicit.")
        if not isinstance(source.payload, bytes) or len(source.payload) > policy.max_source_bytes:
            raise ValueError("Source bytes exceed the input contract; reject before parsing or cache lookup.")


def snapshot_key(request, sources, policy):
    metadata = [{**{key: value for key, value in asdict(source).items() if key != "payload"}, "actual_sha256": digest(source.payload)}
                for source in sorted(sources, key=lambda item: (item.source_id, item.revision))]
    return digest(encoded({"request": asdict(request), "policy": asdict(policy), "sources": metadata}))


def evaluate(request, sources, policy):
    validate_contract(request, sources, policy)
    eligible, trace = [], []
    for source in sorted(sources, key=lambda item: (item.source_id, item.revision)):
        reasons = []
        if source.disposition != "approved":
            reasons.append("SOURCE_NOT_APPROVED")
        if policy.approved_revisions.get(source.source_id) != source.revision:
            reasons.append("REVISION_NOT_APPROVED")
        if not source.valid_from <= policy.clock < source.valid_until:
            reasons.append("OUTSIDE_VALIDITY_WINDOW")
        if source.context != request.context:
            reasons.append("CONTEXT_MISMATCH")
        if request.purpose not in source.allowed_uses:
            reasons.append("USE_NOT_ALLOWED")
        if digest(source.payload) != source.expected_sha256:
            reasons.append("DIGEST_MISMATCH")
        if not reasons:
            try:
                document = json.loads(source.payload)
                encoded(document)
            except (ValueError, UnicodeError):
                reasons.append("INVALID_SOURCE_DOCUMENT")
            else:
                eligible.append((source, document))
        trace.append({"source_id": source.source_id, "revision": source.revision, "eligible": not reasons, "reasons": reasons})
    facts, missing, conflicts = [], [], []
    for fact in request.facts:
        candidates = []
        for source, document in eligible:
            if source.units.get(fact.pointer) != fact.unit:
                continue
            try:
                value = pointer_value(document, fact.pointer)
            except (KeyError, IndexError, ValueError):
                continue
            candidates.append({"value": value, "source_id": source.source_id, "revision": source.revision,
                               "sha256": source.expected_sha256, "pointer": fact.pointer})
        if not candidates:
            missing.append(fact.key)
        elif len({encoded(candidate["value"]) for candidate in candidates}) > 1:
            conflicts.append(fact.key)
        else:
            facts.append({"key": fact.key, "value": candidates[0]["value"], "unit": fact.unit,
                          "context": request.context, "support": candidates})
    decision = "needs_review" if conflicts else "refused" if missing else "supported_within_contract"
    return {"decision": decision, "facts": facts if decision == "supported_within_contract" else [],
            "missing_facts": missing, "conflicting_facts": conflicts, "trace": trace,
            "purpose": request.purpose, "context": request.context, "policy_revision": policy.revision,
            "snapshot_sha256": snapshot_key(request, sources, policy),
            "limits": sorted({limit for source in sources for limit in source.limits}),
            "assurance_scope": "Eligibility and scalar provenance under a supplied registry; not proof that a source is true, a use is lawful, or an open-ended generated answer is correct."}


class DecisionCache:
    def __init__(self):
        self._entries = {}

    def resolve(self, request, sources, policy):
        validate_contract(request, sources, policy)
        key = snapshot_key(request, sources, policy)
        hit = key in self._entries
        if not hit:
            self._entries[key] = evaluate(request, sources, policy)
        return deepcopy(self._entries[key]), hit


def explain(decision, audience):
    introductions = {"learner": "Start with what the records can support, then read the limits beside it.",
                     "researcher": "Inspect the values, context, and source pointers before extending the interpretation.",
                     "reviewer": "Check the approved revision, eligibility trace, and preserved limits against the intended use."}
    if audience not in introductions:
        raise ValueError("Choose an explicit reader view; a role label does not grant access.")
    statements = [f"{fact['key']}: {json.dumps(fact['value'])} {fact['unit']}" for fact in decision["facts"]]
    return {**deepcopy(decision), "audience": audience, "introduction": introductions[audience], "statements": statements,
            "evidence_snapshot_sha256": decision["snapshot_sha256"]}


def run(output=ROOT / "studies/results/evidence_contracts.json"):
    payload = (ROOT / "studies/results/facility.json").read_bytes()
    source = Source("facility-record", "fixture-v1", "synthetic-facility-run", payload, digest(payload),
                    {"/counts/accepted": "records", "/counts/quarantined": "records"}, ("software_review",),
                    ("Synthetic fixture, not a manufacturing batch.", "Input acceptance does not mean the process is safe or approved for release."))
    request = Request(source.context, "software_review", (Fact("accepted_observations", "/counts/accepted", "records"), Fact("quarantined_observations", "/counts/quarantined", "records")))
    policy = Policy("demo-policy-v1", 3, {source.source_id: source.revision})
    altered = json.loads(payload)
    altered["counts"]["accepted"] += 1
    other_payload = encoded(altered)
    other = replace(source, source_id="second-fixture-record", payload=other_payload, expected_sha256=digest(other_payload))
    cases = {
        "supported": evaluate(request, [source], policy),
        "expired": evaluate(request, [source], replace(policy, clock=10)),
        "wrong_context": evaluate(replace(request, context="another-facility"), [source], policy),
        "unsupported_use": evaluate(replace(request, purpose="clinical_release"), [source], policy),
        "tampered": evaluate(request, [replace(source, payload=other_payload)], policy),
        "conflict": evaluate(request, [source, other], replace(policy, approved_revisions={source.source_id: source.revision, other.source_id: other.revision})),
    }
    cache = DecisionCache()
    cache.resolve(request, [source], policy)
    _, same_hit = cache.resolve(request, [source], policy)
    expired, expired_hit = cache.resolve(request, [source], replace(policy, clock=10))
    report = {"evidence_type": "Synthetic policy exercise over a recorded software fixture; approval labels and logical clock are not real QA approvals or dates.",
              "source_file": "studies/results/facility.json", "source_file_sha256": digest(payload),
              "cases": cases, "reader_views": {audience: explain(cases["supported"], audience) for audience in ("learner", "researcher", "reviewer")},
              "cache_checks": {"identical_snapshot_hit": same_hit, "expired_snapshot_hit": expired_hit, "expired_decision": expired["decision"]},
              "implementation_sha256": digest(Path(__file__).read_bytes())}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: decision["decision"] for name, decision in cases.items()}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "studies/results/evidence_contracts.json")
    run(parser.parse_args().output)
