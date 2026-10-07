"""Separate fixed-prediction comparisons from development-only selection audits."""

import argparse
from collections import Counter
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from studies.data import ROOT, clinical_data, digest, verify_snapshot
from studies.modeling import scores


SEEDS = (42, 1729)


def checked_predictions(y, probability):
    y, probability = np.asarray(y), np.asarray(probability, dtype=float)
    if y.ndim != 1 or len(y) == 0 or probability.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Require nonempty aligned binary labels and one probability per observation.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    return y.astype(int), probability


def observation_losses(y, probability):
    y, probability = checked_predictions(y, probability)
    clipped = np.clip(probability, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {"brier": (probability - y) ** 2,
            "log_loss": -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))}


def paired_bootstrap(y, reference, candidate, repeats=2000, seed=42):
    y, reference = checked_predictions(y, reference)
    _, candidate = checked_predictions(y, candidate)
    if len(np.unique(y)) != 2 or repeats < 2:
        raise ValueError("Stratified bootstrap requires both classes and at least two resamples.")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == value) for value in (0, 1)]
    indices = np.concatenate([rng.choice(group, size=(repeats, len(group)), replace=True) for group in groups], axis=1)
    reference_loss, candidate_loss = observation_losses(y, reference), observation_losses(y, candidate)
    result = {}
    for metric in reference_loss:
        differences = candidate_loss[metric] - reference_loss[metric]
        bootstrap = differences[indices].mean(axis=1)
        result[metric] = {"candidate_minus_reference": float(differences.mean()),
                          "pointwise_percentile_95": np.quantile(bootstrap, [.025, .975]).tolist()}
    return result


def candidate_models(seed):
    models = {f"Logistic C={value:g}": make_pipeline(StandardScaler(), LogisticRegression(C=value, max_iter=2000, random_state=seed))
              for value in (.1, 1., 10.)}
    models.update({f"Boosting leaves={leaves}": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=leaves,
                    l2_regularization=1, early_stopping=False, random_state=seed) for leaves in (3, 7)})
    return models


def nested_selection(X, y, source_rows, *, outer_folds=5, inner_folds=3, seeds=SEEDS, candidates=None):
    X, y, source_rows = np.asarray(X, dtype=float), np.asarray(y), np.asarray(source_rows)
    if X.ndim != 2 or y.shape != (len(X),) or source_rows.shape != y.shape or not np.isfinite(X).all():
        raise ValueError("Finite features, labels, and source rows must be aligned.")
    if len(set(source_rows.tolist())) != len(source_rows) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Source rows must be unique and both binary classes present.")
    if min(outer_folds, inner_folds) < 2 or not seeds or min(np.bincount(y.astype(int))) < outer_folds:
        raise ValueError("Insufficient classes/folds or no repeat seeds.")
    predictions, audits, repeat_metrics = [], [], []
    for repeat, seed in enumerate(seeds):
        models = candidates if candidates is not None else candidate_models(seed)
        if not models:
            raise ValueError("Declare at least one candidate model.")
        outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
        adaptive_probabilities = np.full(len(y), np.nan)
        fixed_probabilities = np.full(len(y), np.nan)
        for fold, (training, evaluation) in enumerate(outer.split(X, y)):
            if min(np.bincount(y[training].astype(int))) < inner_folds:
                raise ValueError("Insufficient training observations per class for inner folds.")
            inner = list(StratifiedKFold(inner_folds, shuffle=True, random_state=seed + fold + 1).split(X[training], y[training]))
            inner_scores = {}
            for name, estimator in models.items():
                probabilities = np.full(len(training), np.nan)
                for fitting, validation in inner:
                    fitted = clone(estimator).fit(X[training[fitting]], y[training[fitting]])
                    probabilities[validation] = fitted.predict_proba(X[training[validation]])[:, 1]
                inner_scores[name] = float(observation_losses(y[training], probabilities)["log_loss"].mean())
            choice = min(inner_scores, key=inner_scores.get)
            selected = clone(models[choice]).fit(X[training], y[training])
            fixed = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=seed)).fit(X[training], y[training])
            selected_probability = selected.predict_proba(X[evaluation])[:, 1]
            fixed_probability = fixed.predict_proba(X[evaluation])[:, 1]
            adaptive_probabilities[evaluation], fixed_probabilities[evaluation] = selected_probability, fixed_probability
            audit = {"repeat": repeat, "seed": seed, "fold": fold, "chosen_candidate": choice,
                     "outer_training_rows": source_rows[training].tolist(), "outer_evaluation_rows": source_rows[evaluation].tolist(),
                     "inner_partitions": [{"fit_rows": source_rows[training[a]].tolist(), "validation_rows": source_rows[training[b]].tolist()} for a, b in inner],
                     "inner_oof_log_loss": inner_scores, "selected_outer_scores": scores(y[evaluation], selected_probability),
                     "fixed_outer_scores": scores(y[evaluation], fixed_probability)}
            audits.append(audit)
            predictions.extend({"repeat": repeat, "fold": fold, "source_row": int(row), "malignant": int(label),
                                "selected_probability": float(selected_p), "fixed_logistic_probability": float(fixed_p)}
                               for row, label, selected_p, fixed_p in zip(source_rows[evaluation], y[evaluation], selected_probability, fixed_probability))
        repeat_metrics.append({"seed": seed, "selected_procedure": scores(y, adaptive_probabilities),
                               "fixed_logistic": scores(y, fixed_probabilities)})
    return {"outer_folds": outer_folds, "inner_folds": inner_folds, "seeds": list(seeds),
            "candidate_names": list(models), "fold_audits": audits, "repeat_metrics": repeat_metrics,
            "selection_counts": dict(Counter(audit["chosen_candidate"] for audit in audits)),
            "estimand": "Out-of-fold performance of the declared selection procedure on development records; folds and repeats are not independent studies."}, pd.DataFrame(predictions).sort_values(["repeat", "source_row"])


def net_benefit(y, probability, threshold):
    y, probability = checked_predictions(y, probability)
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Decision threshold must be strictly between zero and one.")
    positive = probability >= threshold
    tp = np.sum(positive & (y == 1))
    fp = np.sum(positive & (y == 0))
    return float((tp - fp * threshold / (1 - threshold)) / len(y))


def predictive_value(sensitivity, specificity, prevalence):
    values = np.asarray([sensitivity, specificity, prevalence], dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Sensitivity, specificity, and prevalence must lie in [0, 1].")
    numerator = sensitivity * prevalence
    denominator = numerator + (1 - specificity) * (1 - prevalence)
    if denominator == 0:
        raise ValueError("Predictive value is undefined when no positive classifications are expected.")
    return float(numerator / denominator)


def run(output=ROOT / "studies/results/statistical_validation.json"):
    benchmark_path = ROOT / "studies/results/benchmark.json"
    predictions_path = ROOT / "studies/results/test_predictions.csv"
    benchmark = json.loads(benchmark_path.read_text())
    predictions = pd.read_csv(predictions_path)
    X, y = clinical_data()
    if benchmark["dataset"]["sha256"] != verify_snapshot()["files"]["wdbc.data"]["sha256"]:
        raise ValueError("Benchmark and source-data digests differ.")
    if predictions.source_row.tolist() != benchmark["split_indices"]["test"] or not np.array_equal(predictions.malignant, y[predictions.source_row]):
        raise ValueError("Stored test predictions do not align with source observations.")
    development = np.sort(np.concatenate([benchmark["split_indices"]["train"], benchmark["split_indices"]["validation"]]))
    if set(development) & set(predictions.source_row) or len(set(development)) != len(development) or len(development) + len(predictions) != len(X):
        raise ValueError("Development and recorded test partitions must be disjoint and exhaustive.")
    nested, oof = nested_selection(X.iloc[development].to_numpy(), y[development], development)
    reference = "Logistic regression"
    paired = {name: paired_bootstrap(predictions.malignant.to_numpy(), predictions[reference].to_numpy(), predictions[name].to_numpy())
              for name in ("Gradient boosting", "Residual MLP")}
    thresholds = np.linspace(.05, .8, 31)
    decisions = []
    for repeat, rows in oof.groupby("repeat", sort=True):
        labels = rows.malignant.to_numpy()
        for threshold in thresholds:
            decisions.append({"repeat": int(repeat), "threshold": float(threshold),
                              "selected": net_benefit(labels, rows.selected_probability, threshold),
                              "fixed_logistic": net_benefit(labels, rows.fixed_logistic_probability, threshold),
                              "all_positive": float(labels.mean() - (1 - labels.mean()) * threshold / (1 - threshold)),
                              "all_negative": 0.0})
    tn, fp, fn, tp = np.asarray(benchmark["confusion_at_fixed_0_5"]).ravel()
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    prevalence = np.geomspace(.001, .5, 100)
    report = {"schema_version": 1, "package_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")}, "source_data": verify_snapshot()["files"]["wdbc.data"],
              "benchmark_sha256": digest(benchmark_path.read_bytes()), "test_predictions_sha256": digest(predictions_path.read_bytes()),
              "implementation_sha256": digest(Path(__file__).read_bytes()), "development_rows": development.tolist(),
              "excluded_test_rows": predictions.source_row.tolist(), "nested_selection": nested,
              "paired_test_comparisons": {"reference": reference, "bootstrap_repeats": 2000, "seed": 42,
                 "direction": "candidate minus reference; negative favors candidate for these losses",
                 "scope": "Post-hoc pointwise intervals, conditional on fixed models and observed class counts; no multiplicity adjustment or retraining uncertainty.", "comparisons": paired},
              "decision_curves": decisions,
              "prevalence_sensitivity": {"assumed_sensitivity": float(sensitivity), "assumed_specificity": float(specificity),
                 "prevalence": prevalence.tolist(), "ppv": [predictive_value(sensitivity, specificity, value) for value in prevalence]},
              "clinical_scope": "Educational post-FNA feature analysis. No pre-biopsy triage claim, validated action costs, target-population prevalence, or deployment recommendation."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    oof.to_csv(output.with_name("nested_predictions.csv"), index=False, lineterminator="\n")
    print(json.dumps({"selection_counts": nested["selection_counts"], "repeat_metrics": nested["repeat_metrics"], "paired_comparisons": paired}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "studies/results/statistical_validation.json")
    args = parser.parse_args()
    run(args.output)
