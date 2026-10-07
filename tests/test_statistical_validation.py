import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from studies.statistical_validation import (
    net_benefit, nested_selection, observation_losses, paired_bootstrap, predictive_value,
)


def test_proper_scoring_rules_match_reference_implementations():
    y = np.array([0, 1, 1, 0])
    p = np.array([0, 1, .2, .7])
    losses = observation_losses(y, p)
    assert losses["brier"].mean() == pytest.approx(brier_score_loss(y, p))
    assert losses["log_loss"].mean() == pytest.approx(log_loss(y, p))


def test_paired_bootstrap_identity_and_reversal():
    y = np.array([0, 0, 1, 1, 1])
    first = np.array([.1, .4, .7, .8, .6])
    second = np.array([.2, .2, .9, .8, .7])
    identical = paired_bootstrap(y, first, first, repeats=50)
    forward = paired_bootstrap(y, first, second, repeats=50)
    backward = paired_bootstrap(y, second, first, repeats=50)
    for metric in identical:
        assert identical[metric]["pointwise_percentile_95"] == [0, 0]
        np.testing.assert_allclose(forward[metric]["pointwise_percentile_95"], -np.array(backward[metric]["pointwise_percentile_95"])[::-1])


def test_nested_partitions_preserve_outer_isolation_and_predict_each_row_once():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(60, 4))
    y = np.tile([0, 1], 30)
    ids = np.arange(100, 160)
    candidates = {"small": make_pipeline(StandardScaler(), LogisticRegression(C=.1)),
                  "large": make_pipeline(StandardScaler(), LogisticRegression(C=1))}
    result, predictions = nested_selection(X, y, ids, outer_folds=3, inner_folds=2, seeds=(2, 3), candidates=candidates)
    for _, rows in predictions.groupby("repeat"):
        assert sorted(rows.source_row) == ids.tolist()
        assert len(rows) == 60
    for audit in result["fold_audits"]:
        outer_train, outer_test = set(audit["outer_training_rows"]), set(audit["outer_evaluation_rows"])
        assert not outer_train & outer_test
        for inner in audit["inner_partitions"]:
            fitting, validation = set(inner["fit_rows"]), set(inner["validation_rows"])
            assert not fitting & validation
            assert fitting | validation == outer_train
            assert not (fitting | validation) & outer_test
        assert audit["chosen_candidate"] == min(audit["inner_oof_log_loss"], key=audit["inner_oof_log_loss"].get)


def test_net_benefit_matches_hand_calculation_and_default_strategies():
    y = np.array([1, 1, 0, 0])
    assert net_benefit(y, np.array([.9, .6, .7, .1]), .5) == pytest.approx(.25)
    assert net_benefit(y, np.ones(4), .2) == pytest.approx(.5 - .5 * .2 / .8)
    assert net_benefit(y, np.zeros(4), .2) == 0


def test_predictive_value_exposes_the_base_rate_effect():
    assert predictive_value(.9, .9, .5) == pytest.approx(.9)
    assert predictive_value(.9, .9, .01) == pytest.approx(.009 / (.009 + .099))
    with pytest.raises(ValueError):
        predictive_value(0, 1, .5)


@pytest.mark.parametrize("p", [[.1, np.nan], [.1, 1.1], [.1]])
def test_invalid_probabilities_fail_before_resampling(p):
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([0, 1]), np.array([.1, .9]), p, repeats=10)


def test_scalers_fit_only_on_authorized_inner_or_outer_training_rows(monkeypatch):
    fitted_rows = []
    original = StandardScaler.fit

    def record_fit(self, values, labels=None, **kwargs):
        fitted_rows.append(frozenset(np.asarray(values)[:, 0].astype(int)))
        return original(self, values, labels, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    X = np.column_stack([np.arange(60), np.sin(np.arange(60))])
    y = np.tile([0, 1], 30)
    candidates = {"linear": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))}
    report, _ = nested_selection(X, y, np.arange(60), outer_folds=3, inner_folds=2, seeds=(7,), candidates=candidates)
    allowed = set()
    for fold in report["fold_audits"]:
        allowed.add(frozenset(fold["outer_training_rows"]))
        allowed.update(frozenset(inner["fit_rows"]) for inner in fold["inner_partitions"])
    assert fitted_rows
    assert all(rows in allowed for rows in fitted_rows)
    assert all(len(rows) < len(X) for rows in fitted_rows)


def test_recorded_nested_results_preserve_the_exclusion_and_metrics():
    import json
    import pandas as pd
    from studies.data import ROOT, clinical_data
    from studies.modeling import scores

    report = json.loads((ROOT / "studies/results/statistical_validation.json").read_text())
    predictions = pd.read_csv(ROOT / "studies/results/nested_predictions.csv")
    _, y = clinical_data()
    excluded = set(report["excluded_test_rows"])
    for repeat, rows in predictions.groupby("repeat"):
        assert set(rows.source_row) == set(report["development_rows"])
        assert not set(rows.source_row) & excluded
        assert not rows.source_row.duplicated().any()
        np.testing.assert_array_equal(rows.malignant, y[rows.source_row])
        expected = report["nested_selection"]["repeat_metrics"][repeat]["selected_procedure"]
        for metric, value in scores(rows.malignant, rows.selected_probability).items():
            assert value == pytest.approx(expected[metric], abs=1e-8)


def test_statistical_figures_and_finite_animation(tmp_path):
    import json
    import xml.etree.ElementTree as ET
    from PIL import Image
    from studies.data import ROOT
    from studies.statistical_figures import render_statistics

    report = json.loads((ROOT / "studies/results/statistical_validation.json").read_text())
    render_statistics(report, tmp_path)
    for name in ("paired_comparison", "nested_validation", "decision_context"):
        tree = ET.parse(tmp_path / f"{name}.svg")
        assert len(tree.findall(".//{http://www.w3.org/2000/svg}text")) > 10
    with Image.open(tmp_path / "statistical_flow.gif") as animation:
        assert "loop" not in animation.info
        assert animation.n_frames == 8
        duration = 0
        for frame in range(animation.n_frames):
            animation.seek(frame)
            duration += animation.info["duration"]
        assert duration <= 5000
    assert (tmp_path / "statistical_flow.png").exists()
