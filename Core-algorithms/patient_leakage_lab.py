"""A synthetic counterexample: recognizing patients is not predicting new patients.

Run: python Core-algorithms/patient_leakage_lab.py --seed 42
Predict first: will a row split or a patient split score higher, and why?
No patient records, clinical parameters, or externally downloaded data are used.
"""

import argparse
import json

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_cohort(seed=42, n_patients=120, visits=4):
    if n_patients < 20 or visits < 2:
        raise ValueError("Use at least 20 patients and two visits per patient.")
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_patients), visits)
    fingerprints = rng.normal(size=(n_patients, 12))
    labels = rng.integers(0, 2, size=n_patients)
    X = fingerprints[groups] + rng.normal(scale=0.01, size=(len(groups), 12))
    X[rng.random(X.shape) < 0.02] = np.nan
    return X, labels[groups], groups


def split_cohort(y, groups, seed=42, grouped=True):
    if grouped:
        return next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed).split(y, y, groups))
    return train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)


def evaluate_split(X, y, groups, train, test):
    if np.isinf(X).any() or np.isnan(X[train]).all(axis=0).any():
        raise ValueError("Features must not contain infinity or entirely missing training columns.")
    if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
        raise ValueError("Both classes must be present in each split; try another seed.")
    results = {}
    for name, estimator in (("nearest_neighbor", KNeighborsClassifier(n_neighbors=1)),
                            ("prior_baseline", DummyClassifier(strategy="prior"))):
        model = make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler(), estimator)
        model.fit(X[train], y[train])
        predictions = model.predict(X[test])
        probabilities = model.predict_proba(X[test])[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y[test], predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], predictions)),
            "brier_score": float(brier_score_loss(y[test], probabilities)),
        }
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "overlapping_patients": len(np.intersect1d(groups[train], groups[test])),
        "missing_fraction_train": float(np.isnan(X[train]).mean()),
        "missing_fraction_test": float(np.isnan(X[test]).mean()),
        "models": results,
    }


def run_experiment(seed=42):
    X, y, groups = make_cohort(seed)
    report = {"seed": seed, "synthetic": True, "label_mechanism": "random per patient"}
    for name, grouped in (("row_split", False), ("patient_split", True)):
        train, test = split_cohort(y, groups, seed, grouped)
        report[name] = evaluate_split(X, y, groups, train, test)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
