from dataclasses import replace

import pytest

from studies.evidence_contracts import DecisionCache, Fact, Policy, Request, Source, digest, encoded, evaluate, explain, pointer_value


@pytest.fixture
def contract():
    payload = encoded({"counts": {"accepted": 480, "quarantined": 4}})
    source = Source("source-a", "v1", "fixture-run", payload, digest(payload), {"/counts/accepted": "records", "/counts/quarantined": "records"},
                    ("software_review",), ("Synthetic fixture only.", "Acceptance is not batch release."))
    request = Request("fixture-run", "software_review", (Fact("accepted", "/counts/accepted", "records"),))
    return source, request, Policy("policy-v1", 3, {"source-a": "v1"})


def test_supported_fact_is_bound_to_value_unit_context_revision_and_pointer(contract):
    source, request, policy = contract
    result = evaluate(request, [source], policy)
    assert result["decision"] == "supported_within_contract"
    fact = result["facts"][0]
    assert (fact["value"], fact["unit"], fact["context"]) == (480, "records", "fixture-run")
    assert fact["support"][0]["sha256"] == digest(source.payload)
    assert fact["support"][0]["pointer"] == "/counts/accepted"


@pytest.mark.parametrize("change, reason", [
    ({"disposition": "withdrawn"}, "SOURCE_NOT_APPROVED"),
    ({"revision": "v2"}, "REVISION_NOT_APPROVED"),
    ({"context": "another-run"}, "CONTEXT_MISMATCH"),
    ({"allowed_uses": ("other_use",)}, "USE_NOT_ALLOWED"),
    ({"payload": b'{"counts":{"accepted":999}}'}, "DIGEST_MISMATCH"),
])
def test_invalid_sources_cannot_supply_a_fact(contract, change, reason):
    source, request, policy = contract
    result = evaluate(request, [replace(source, **change)], policy)
    assert result["decision"] == "refused" and result["facts"] == []
    assert reason in result["trace"][0]["reasons"]


def test_validity_uses_an_exclusive_end_boundary(contract):
    source, request, policy = contract
    assert evaluate(request, [source], replace(policy, clock=9))["decision"] == "supported_within_contract"
    assert evaluate(request, [source], replace(policy, clock=10))["decision"] == "refused"


def test_conflict_requires_review_instead_of_ranking_away_disagreement(contract):
    source, request, policy = contract
    payload = encoded({"counts": {"accepted": 481}})
    other = replace(source, source_id="source-b", payload=payload, expected_sha256=digest(payload))
    policy = replace(policy, approved_revisions={"source-a": "v1", "source-b": "v1"})
    first = evaluate(request, [source, other], policy)
    second = evaluate(request, [other, source], policy)
    assert first == second
    assert first["decision"] == "needs_review" and first["facts"] == []
    assert first["conflicting_facts"] == ["accepted"]


def test_missing_or_wrong_unit_does_not_produce_partial_success(contract):
    source, request, policy = contract
    extra = Fact("missing", "/counts/not_present", "records")
    result = evaluate(replace(request, facts=(*request.facts, extra)), [source], policy)
    assert result["decision"] == "refused" and result["facts"] == []
    wrong_unit = replace(request, facts=(Fact("accepted", "/counts/accepted", "patients"),))
    assert evaluate(wrong_unit, [source], policy)["decision"] == "refused"


def test_cache_invalidates_for_expiry_revocation_and_revision(contract):
    source, request, policy = contract
    cache = DecisionCache()
    _, hit = cache.resolve(request, [source], policy)
    assert not hit
    result, hit = cache.resolve(request, [source], policy)
    assert hit
    result["facts"][0]["value"] = -1
    result, _ = cache.resolve(request, [source], policy)
    assert result["facts"][0]["value"] == 480
    for changed_source, changed_policy in ((source, replace(policy, clock=10)),
                                            (replace(source, disposition="withdrawn"), policy),
                                            (source, replace(policy, approved_revisions={"source-a": "v2"}))):
        result, hit = cache.resolve(request, [changed_source], changed_policy)
        assert not hit and result["decision"] == "refused"


def test_reader_views_preserve_every_decision_field_and_cannot_authorize_use(contract):
    source, request, policy = contract
    decision = evaluate(replace(request, purpose="clinical_release"), [source], policy)
    for audience in ("learner", "researcher", "reviewer"):
        view = explain(decision, audience)
        for key, value in decision.items():
            assert view[key] == value
        assert view["decision"] == "refused"
    with pytest.raises(ValueError):
        explain(decision, "administrator")


def test_duplicate_source_identity_and_oversized_input_fail_closed(contract):
    source, request, policy = contract
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(request, [source, source], policy)
    with pytest.raises(ValueError, match="Source bytes"):
        DecisionCache().resolve(request, [source], replace(policy, max_source_bytes=1))


def test_json_pointer_escaping_and_scalar_boundary():
    assert pointer_value({"a/b": {"~key": [7]}}, "/a~1b/~0key/0") == 7
    with pytest.raises(ValueError):
        pointer_value({"a": {}}, "/a")
    with pytest.raises(ValueError):
        pointer_value({"a~2": 1}, "/a~2")
