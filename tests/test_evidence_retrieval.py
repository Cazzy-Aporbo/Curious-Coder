import numpy as np
import pytest

from studies.evidence_retrieval import LexicalIndex, known_item_metrics, ranked_indices, reciprocal_rank_fusion
from studies.literature_data import audit_records, load_snapshot, merge_record, normalize_record


def record(license_name="cc by", abstract="A <b>licensed</b> abstract."):
    return {"source": "MED", "id": "123", "title": "A useful method", "doi": "10.1/TEST", "license": license_name,
            "abstractText": abstract, "authorString": "Author A", "affiliation": "private@example.test",
            "commentCorrectionList": {"commentCorrection": [{"type": "Erratum in", "id": "456", "source": "MED"}]}}


def test_reuse_policy_is_not_inferred_from_free_access():
    allowed = normalize_record(record(), "query")
    withheld = normalize_record(record("cc by-nc"), "query")
    unknown = normalize_record(record(""), "query")
    assert allowed["abstract"] == "A licensed abstract."
    assert withheld["abstract_present_upstream"] and not withheld["abstract_included"]
    assert withheld["abstract"] == unknown["abstract"] == ""
    assert "affiliation" not in allowed
    assert allowed["notice_links"][0]["id"] == "456"


def test_duplicate_doi_preserves_query_provenance():
    records = {}
    first = normalize_record(record(), "one")
    second = normalize_record({**record(), "id": "999"}, "two")
    assert merge_record(records, first)
    assert not merge_record(records, second)
    assert len(records) == 1
    assert records["10.1/test"]["queries"] == ["one", "two"]


def test_missing_abstract_and_withheld_abstract_are_distinct():
    records = [normalize_record(record(abstract=""), "q"), normalize_record(record("cc by-nc"), "q"), normalize_record(record(), "q")]
    audit = audit_records(records)
    assert audit["abstract_missing_upstream"] == 1
    assert audit["abstract_withheld_by_reuse_policy"] == 1
    assert audit["abstract_included"] == 1


def test_rank_ties_are_deterministic_and_fusion_uses_ranks():
    np.testing.assert_array_equal(ranked_indices([.2, .2, .9], ["b", "a", "c"]), [2, 1, 0])
    fusion = reciprocal_rank_fusion([np.array([0, 1, 2]), np.array([1, 0, 2])])
    assert fusion[0] == fusion[1] and fusion[0] > fusion[2]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[0, 0]])


def test_known_item_metrics_do_not_pretend_to_be_full_recall():
    result = known_item_metrics([["a", "b"], ["b", "a"]], ["a", "a"])
    assert result["known_target_hit_at_5"] == 1
    assert result["mean_anchor_reciprocal_rank"] == .75
    assert result["target_ranks"] == [1, 2]


def test_lexical_index_is_operational_and_reports_numeric_storage():
    index = LexicalIndex(["nested validation prevents optimistic selection", "protein sequences and structure"], ["a", "b"])
    ranking, scores = index.search("optimistic validation")
    assert ranking[0] == 0 and scores[0] > scores[1]
    assert index.numeric_bytes > 0


def test_retained_snapshot_has_unique_keys_and_explicit_anchor_targets():
    corpus, tasks, manifest = load_snapshot()
    assert len(corpus) > 50
    assert len({entry["key"] for entry in corpus}) == len(corpus)
    assert all(task["target"] in {entry["key"] for entry in corpus} for task in tasks)
    assert len(tasks) == 6
    assert manifest["audit"] == audit_records(corpus)
    for entry in corpus:
        if entry["abstract_included"]:
            assert entry["license_metadata"] in {"cc by", "cc0"}
        else:
            assert not entry["abstract"]


def test_retry_after_is_honored_and_long_waits_fail_closed(monkeypatch):
    import io
    from email.message import Message
    from urllib.error import HTTPError
    import studies.literature_data as module

    waits, calls = [], []
    headers = Message()
    headers["Retry-After"] = "2"

    def respond(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 429, "rate limit", headers, None)
        return io.BytesIO(b'{"resultList":{"result":[]},"hitCount":0}')

    monkeypatch.setattr(module, "urlopen", respond)
    page, _ = module.fetch_page("safe query", wait=waits.append)
    assert page["hitCount"] == 0 and waits == [1., 2., 1.]
    assert len(calls) == 2
    headers.replace_header("Retry-After", "120")
    calls.clear()
    with pytest.raises(HTTPError):
        module.fetch_page("safe query", wait=lambda _: None)
    assert len(calls) == 1


def test_access_denial_is_not_retried(monkeypatch):
    from urllib.error import HTTPError
    import studies.literature_data as module

    attempts = []
    def denied(request, timeout):
        attempts.append(request.full_url)
        raise HTTPError(request.full_url, 403, "forbidden", {}, None)
    monkeypatch.setattr(module, "urlopen", denied)
    with pytest.raises(HTTPError):
        module.fetch_page("query", wait=lambda _: None)
    assert len(attempts) == 1


def test_presentation_markup_is_not_indexed_as_words():
    from studies.literature_data import plain_text
    assert plain_text("A &lt;i&gt;protein&lt;/i&gt; study") == "A protein study"


def test_recorded_ranks_and_percentiles_match_the_raw_evidence():
    import json
    from studies.data import ROOT

    report = json.loads((ROOT / "studies/results/evidence_retrieval.json").read_text())
    for method, result in report["methods"].items():
        ranks = [row["target_rank"] for row in report["query_results"] if row["method"] == method]
        assert result["known_target_hit_at_5"] == pytest.approx(np.mean(np.array(ranks) <= 5))
        assert result["mean_anchor_reciprocal_rank"] == pytest.approx(np.mean(1 / np.array(ranks)))
        assert result["warm_query_p95_ms"] == pytest.approx(np.quantile(result["raw_warm_query_ms"], .95))
        assert result["timed_requests"] == len(result["raw_warm_query_ms"])
