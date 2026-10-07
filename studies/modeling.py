"""A bounded comparison: measured diagnostic features, not clinical deployment."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from studies.data import clinical_data, verify_snapshot


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 180
    patience: int = 25
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    width: int = 32


class ResidualTabularNetwork(nn.Module):
    def __init__(self, features, width=32):
        super().__init__()
        self.input = nn.Linear(features, width)
        self.block = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width),
                                   nn.GELU(), nn.Dropout(0.1), nn.Linear(width, width))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, x):
        hidden = self.input(x)
        return self.head(hidden + self.block(hidden)).squeeze(-1)


def partitions(y, seed=42):
    development, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    train, validation = train_test_split(development, test_size=0.25, stratify=y[development], random_state=seed)
    return {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}


def fit_network(X_train, y_train, X_validation, y_validation, config=TrainingConfig()):
    if min(config.epochs, config.patience, config.batch_size, config.width) < 1:
        raise ValueError("Training sizes and patience must be positive.")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("Require a positive learning rate and nonnegative weight decay.")
    if not all(np.isfinite(x).all() for x in (X_train, X_validation)):
        raise ValueError("Network inputs must be finite.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = ResidualTabularNetwork(X_train.shape[1], config.width)
        generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                                         torch.as_tensor(y_train, dtype=torch.float32)),
                            batch_size=config.batch_size, shuffle=True, generator=generator)
        validation = torch.as_tensor(X_validation, dtype=torch.float32)
        targets = torch.as_tensor(y_validation, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        criterion = nn.BCEWithLogitsLoss()
        best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
        history = []
        for epoch in range(config.epochs):
            model.train()
            total = 0.0
            for features, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                total += loss.item() * len(labels)
            model.eval()
            with torch.inference_mode():
                validation_loss = criterion(model(validation), targets).item()
            history.append({"epoch": epoch + 1, "train_log_loss": total / len(y_train),
                            "validation_log_loss": validation_loss, "learning_rate": optimizer.param_groups[0]["lr"]})
            if validation_loss < best_loss - 1e-5:
                best_loss, best_epoch, best_state, stale = validation_loss, epoch + 1, deepcopy(model.state_dict()), 0
            else:
                stale += 1
            scheduler.step()
            if stale >= config.patience:
                break
        model.load_state_dict(best_state)
        model.eval()
    return model, history, best_epoch


def predict_network(model, X):
    model.eval()
    with torch.inference_mode():
        return model(torch.as_tensor(X, dtype=torch.float32)).sigmoid().numpy()


def scores(y, probability):
    return {"roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "brier": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1]))}


def bootstrap_intervals(y, probability, repeats=500, seed=42):
    if repeats < 2 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Bootstrap needs both classes and at least two repeats.")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    samples = []
    for _ in range(repeats):
        rows = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples.append(scores(y[rows], probability[rows]))
    return {key: [float(x) for x in np.quantile([sample[key] for sample in samples], [.025, .975])]
            for key in samples[0]}


def run_benchmark(config=TrainingConfig(), bootstrap_repeats=500):
    X, y = clinical_data()
    split = partitions(y, config.seed)
    train, validation, test = (split[name] for name in ("train", "validation", "test"))
    estimators = {
        "Prior baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=config.seed)),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=1,
                                                             early_stopping=False, random_state=config.seed),
    }
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed).split(X.iloc[train], y[train]))
    validation_predictions, test_predictions, cross_validation = {}, {}, {}
    for name, estimator in estimators.items():
        cv = cross_val_score(clone(estimator), X.iloc[train], y[train], cv=folds, scoring="neg_log_loss", n_jobs=1)
        cross_validation[name] = (-cv).tolist()
        estimator.fit(X.iloc[train], y[train])
        validation_predictions[name] = estimator.predict_proba(X.iloc[validation])[:, 1]
        test_predictions[name] = estimator.predict_proba(X.iloc[test])[:, 1]
    scaler = StandardScaler().fit(X.iloc[train])
    scaled_train, scaled_validation, scaled_test = (scaler.transform(X.iloc[rows]) for rows in (train, validation, test))
    model, history, best_epoch = fit_network(scaled_train, y[train], scaled_validation, y[validation], config)
    validation_predictions["Residual MLP"] = predict_network(model, scaled_validation)
    test_predictions["Residual MLP"] = predict_network(model, scaled_test)
    validation_scores = {name: scores(y[validation], prediction) for name, prediction in validation_predictions.items()}
    selected = min(validation_scores, key=lambda name: validation_scores[name]["log_loss"])
    evaluation = {name: {"test": scores(y[test], prediction),
                         "bootstrap_95": bootstrap_intervals(y[test], prediction, bootstrap_repeats, config.seed)}
                  for name, prediction in test_predictions.items()}
    importance = permutation_importance(estimators["Logistic regression"], X.iloc[validation], y[validation],
                                       scoring="neg_log_loss", n_repeats=15, random_state=config.seed)
    selected_probability = test_predictions[selected]
    fraction_positive, mean_probability = calibration_curve(y[test], selected_probability, n_bins=5, strategy="uniform")
    size_cut = float(X.iloc[train]["mean radius"].median())
    slices = {}
    for label, mask in (("radius below training median", X.iloc[test]["mean radius"].to_numpy() < size_cut),
                        ("radius at or above training median", X.iloc[test]["mean radius"].to_numpy() >= size_cut)):
        slices[label] = {"n": int(mask.sum()), "malignant": int(y[test][mask].sum()),
                         "brier": float(brier_score_loss(y[test][mask], selected_probability[mask]))}
    report = {"config": asdict(config), "dataset": verify_snapshot()["files"]["wdbc.data"],
              "positive_class": "malignant", "split_indices": {name: rows.tolist() for name, rows in split.items()},
              "split_sha256": hashlib.sha256(json.dumps({k: v.tolist() for k, v in split.items()}, sort_keys=True).encode()).hexdigest(),
              "audit": {"rows": len(X), "features": X.shape[1], "missing_values": int(X.isna().sum().sum()),
                        "malignant": int(y.sum()), "benign": int((1 - y).sum()), "duplicate_feature_rows": int(X.duplicated().sum())},
              "validation": validation_scores, "selected_by_validation_log_loss": selected,
              "test_evaluation": evaluation, "baseline_train_cv_log_loss": cross_validation,
              "mlp_best_epoch": best_epoch, "mlp_parameter_count": sum(p.numel() for p in model.parameters()),
              "training_history": history, "test_slices": slices, "radius_slice_cut_training_median": size_cut,
              "confusion_at_fixed_0_5": confusion_matrix(y[test], selected_probability >= .5, labels=[0, 1]).tolist(),
              "calibration": {"mean_probability": mean_probability.tolist(), "fraction_malignant": fraction_positive.tolist()},
              "validation_permutation_importance": {"features": X.columns.tolist(), "mean": importance.importances_mean.tolist(),
                                                     "std": importance.importances_std.tolist()},
              "uncertainty_scope": "Stratified row bootstrap of fixed-model test predictions; not training, site, or population uncertainty.",
              "selection_policy": "Minimum validation log loss; threshold fixed at 0.5; no test-set tuning; no refit on held-out data."}
    prediction_table = pd.DataFrame({"source_row": test, "malignant": y[test], **test_predictions})
    checkpoint = {"state_dict": model.state_dict(), "config": asdict(config), "input_features": [str(name) for name in X.columns],
                  "scaler_mean": torch.tensor(scaler.mean_), "scaler_scale": torch.tensor(scaler.scale_),
                  "best_epoch": best_epoch, "data_sha256": report["dataset"]["sha256"]}
    return report, prediction_table, checkpoint
