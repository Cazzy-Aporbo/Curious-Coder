import json

import numpy as np
import pytest
import torch
from sklearn.datasets import load_breast_cancer

from studies.data import clinical_data, climate_data, parse_daily, verify_snapshot
from studies.modeling import TrainingConfig, bootstrap_intervals, fit_network, partitions, predict_network
from studies.systems import Box, MutationJournal, heat_scenario, possible_clash, spatial_cells


def test_measured_data_matches_published_reference():
    X, y = clinical_data()
    reference = load_breast_cancer()
    np.testing.assert_allclose(X, reference.data)
    np.testing.assert_array_equal(y, 1 - reference.target)
    assert X.shape == (569, 30)
    assert verify_snapshot()["sources"]["wdbc"]["license"] == "CC BY 4.0"


def test_climate_snapshot_has_real_calendar_and_valid_ordering():
    frame, station = climate_data()
    assert len(frame) == 731
    assert station["id"] == "USW00023183"
    complete = frame.dropna()
    assert len(complete) > 650
    assert (complete.TMAX >= complete.TMIN).all()


def test_quality_flag_is_not_treated_as_a_good_temperature():
    fields = ["  250   "] * 31
    fields[0] = "  250 X "
    fields[1] = "-9999   "
    line = "USW00023183202301TMAX" + "".join(fields)
    parsed = parse_daily(line)
    assert np.isnan(parsed.loc[0, "value_c"])
    assert np.isnan(parsed.loc[1, "value_c"])
    assert parsed.loc[2, "value_c"] == 25


def test_partition_contract():
    _, y = clinical_data()
    splits = partitions(y)
    assert sum(map(len, splits.values())) == len(y)
    for first in splits:
        for second in splits:
            if first != second:
                assert not set(splits[first]) & set(splits[second])
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [341, 114, 114]


def test_pytorch_training_is_seeded_and_restores_best_validation_state():
    X = np.random.default_rng(4).normal(size=(48, 5)).astype("float32")
    y = (X[:, 0] > 0).astype("float32")
    config = TrainingConfig(epochs=6, patience=3, width=8)
    model, history, best = fit_network(X[:32], y[:32], X[32:], y[32:], config)
    other, repeated, _ = fit_network(X[:32], y[:32], X[32:], y[32:], config)
    assert history == repeated
    assert best >= 1
    np.testing.assert_allclose(predict_network(model, X), predict_network(other, X))
    with torch.inference_mode():
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(torch.tensor(X[32:])), torch.tensor(y[32:])).item()
    assert loss == pytest.approx(history[best - 1]["validation_log_loss"])


def test_bootstrap_handles_class_imbalance_without_one_class_resamples():
    result = bootstrap_intervals(np.array([0, 0, 0, 1]), np.array([.1, .2, .3, .9]), repeats=20)
    assert result["roc_auc"] == [1, 1]
    assert result["brier"][0] <= result["brier"][1]


def test_retry_changes_state_exactly_once_and_conflicting_intent_is_rejected(tmp_path):
    journal = MutationJournal(tmp_path / "events.db")
    try:
        first = journal.apply("city-a", "request-1", "tree-7", {"canopy": .3})
        second = journal.apply("city-a", "request-1", "tree-7", {"canopy": .3})
        assert first["event_hash"] == second["event_hash"] and second["replayed"]
        assert journal.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        with pytest.raises(ValueError, match="different intent"):
            journal.apply("city-a", "request-1", "tree-7", {"canopy": .8})
        assert journal.verify(first["event_hash"])
    finally:
        journal.close()


def test_failure_rolls_back_mutation_and_journal_together(tmp_path):
    journal = MutationJournal(tmp_path / "events.db")
    try:
        with pytest.raises(RuntimeError, match="Injected"):
            journal.apply("city", "request", "tree", .4, fail_before_commit=True)
        assert journal.state() == {}
        assert journal.reconstructed_state() == {}
        assert not journal.apply("city", "request", "tree", .4)["replayed"]
        assert journal.state() == journal.reconstructed_state()
    finally:
        journal.close()


def test_journal_detects_payload_modification(tmp_path):
    journal = MutationJournal(tmp_path / "events.db")
    try:
        journal.apply("city", "key", "asset", 1)
        journal.connection.execute("UPDATE events SET payload=?", (json.dumps({"asset": "asset", "value": 9}),))
        assert not journal.verify()
    finally:
        journal.close()


def test_unknown_geometry_uncertainty_expands_possible_conflict():
    asset = Box((0, 0, -2), (1, 1, -1))
    pipe = Box((1.2, 0, -2), (2, 1, -1))
    uncertain_pipe = Box(pipe.lower, pipe.upper, .3)
    assert not possible_clash(asset, pipe)
    assert possible_clash(asset, uncertain_pipe)


def test_h3_and_heat_scenario_contracts():
    assert len(spatial_cells(33.4, -112, rings=1)) == 7
    assert heat_scenario(35, .6, .3) < heat_scenario(35, .1, .3)
    assert heat_scenario(35, 1, .3) == 35
    with pytest.raises(ValueError):
        heat_scenario(35, 1.1, .3)


def test_concurrent_duplicate_requests_share_one_commit(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "concurrent.db"
    MutationJournal(path).close()

    def attempt(_):
        journal = MutationJournal(path)
        try:
            return journal.apply("city", "same-key", "asset", {"version": 1})
        finally:
            journal.close()

    with ThreadPoolExecutor(max_workers=4) as workers:
        responses = list(workers.map(attempt, range(8)))
    assert sum(not response["replayed"] for response in responses) == 1
    assert len({response["event_hash"] for response in responses}) == 1


def test_reopening_journal_preserves_state_and_replay(tmp_path):
    path = tmp_path / "restart.db"
    journal = MutationJournal(path)
    response = journal.apply("city", "key", "asset", {"height_m": 4})
    journal.close()
    recovered = MutationJournal(path)
    try:
        assert recovered.verify(response["event_hash"])
        assert recovered.state() == recovered.reconstructed_state()
        assert recovered.apply("city", "key", "asset", {"height_m": 4})["replayed"]
    finally:
        recovered.close()


def test_stored_results_match_exported_predictions():
    from studies.data import ROOT
    from studies.modeling import scores
    import pandas as pd

    report = json.loads((ROOT / "studies/results/benchmark.json").read_text())
    predictions = pd.read_csv(ROOT / "studies/results/test_predictions.csv")
    _, y = clinical_data()
    np.testing.assert_array_equal(predictions.malignant, y[predictions.source_row])
    assert predictions.source_row.tolist() == report["split_indices"]["test"]
    for name, result in report["test_evaluation"].items():
        actual = scores(predictions.malignant, predictions[name])
        for metric, value in actual.items():
            assert value == pytest.approx(result["test"][metric], abs=1e-7)
    assert min(report["validation"], key=lambda key: report["validation"][key]["log_loss"]) == report["selected_by_validation_log_loss"]


def test_snapshot_tampering_fails_closed(tmp_path):
    from studies.data import DATA
    import shutil

    snapshot = tmp_path / "snapshot"
    shutil.copytree(DATA, snapshot)
    (snapshot / "wdbc.data").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_snapshot(snapshot)


def test_figure_exports_are_labeled_and_repeatable(tmp_path):
    import xml.etree.ElementTree as ET
    import pandas as pd
    from studies.data import ROOT
    from studies.figures import cohort_figure, render_all

    report = json.loads((ROOT / "studies/results/benchmark.json").read_text())
    predictions = pd.read_csv(ROOT / "studies/results/test_predictions.csv")
    render_all(report, predictions, tmp_path)
    for name in ("cohort", "diagnostics", "training", "climate", "spatial", "architecture"):
        svg = tmp_path / f"{name}.svg"
        tree = ET.parse(svg)
        assert tree.getroot().tag == "{http://www.w3.org/2000/svg}svg"
        assert len(tree.findall(".//{http://www.w3.org/2000/svg}text")) > 10
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
    assert "synthetic" in (tmp_path / "spatial.svg").read_text()
    assert "95% Wilson" in (tmp_path / "diagnostics.svg").read_text()
    first = (tmp_path / "cohort.svg").read_bytes()
    cohort_figure(report, tmp_path)
    assert (tmp_path / "cohort.svg").read_bytes() == first
