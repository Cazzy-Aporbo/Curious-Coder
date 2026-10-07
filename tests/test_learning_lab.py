import numpy as np
import pytest

from conftest import load_module


lab = load_module("Core-algorithms/patient_leakage_lab.py")


def test_patient_split_has_no_patient_overlap():
    X, y, groups = lab.make_cohort()
    train, test = lab.split_cohort(y, groups)
    assert len(np.intersect1d(groups[train], groups[test])) == 0
    assert len(np.intersect1d(train, test)) == 0
    assert len(train) + len(test) == len(X)
    assert np.isnan(X).any()


@pytest.mark.parametrize("seed", [7, 42, 123])
def test_leakage_counterexample(seed):
    report = lab.run_experiment(seed)
    row = report["row_split"]
    patient = report["patient_split"]
    assert row["overlapping_patients"] > 0
    assert patient["overlapping_patients"] == 0
    assert row["models"]["nearest_neighbor"]["accuracy"] > 0.9
    assert patient["models"]["nearest_neighbor"]["accuracy"] < 0.75
    assert report == lab.run_experiment(seed)


def test_preprocessing_fits_only_training_rows(monkeypatch):
    X, y, groups = lab.make_cohort()
    train, test = lab.split_cohort(y, groups)
    original_fit = lab.SimpleImputer.fit
    seen = []

    def record_fit(self, features, target=None):
        seen.append(features.copy())
        return original_fit(self, features, target)

    monkeypatch.setattr(lab.SimpleImputer, "fit", record_fit)
    lab.evaluate_split(X, y, groups, train, test)
    assert len(seen) == 2
    for fitted in seen:
        np.testing.assert_allclose(fitted, X[train], equal_nan=True)


@pytest.mark.parametrize("kwargs", [{"n_patients": 10}, {"visits": 1}])
def test_rejects_cohorts_too_small_for_demonstration(kwargs):
    with pytest.raises(ValueError, match="at least"):
        lab.make_cohort(**kwargs)


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_rejects_unusable_training_features(value):
    X, y, groups = lab.make_cohort()
    train, test = lab.split_cohort(y, groups)
    X[train, 0] = value
    with pytest.raises(ValueError, match="Features"):
        lab.evaluate_split(X, y, groups, train, test)


def test_rejects_single_class_evaluation():
    X, y, groups = lab.make_cohort()
    train, test = lab.split_cohort(y, groups)
    y[test] = 0
    with pytest.raises(ValueError, match="Both classes"):
        lab.evaluate_split(X, y, groups, train, test)


def test_cli_outputs_reproducible_json(monkeypatch, capsys):
    import json
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["patient_leakage_lab.py", "--seed", "7"])
    runpy.run_path(lab.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out)
    assert report == lab.run_experiment(7)
