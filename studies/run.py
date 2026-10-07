"""Reproduce the measured-data study and its figures without network access."""

import argparse
import json
from pathlib import Path
import platform

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from studies.data import ROOT, digest
from studies.figures import render_all
from studies.modeling import TrainingConfig, run_benchmark


def run(output=ROOT / "studies" / "results", figures=ROOT / "assets" / "figures", epochs=180, bootstrap=500):
    torch.set_num_threads(1)
    config = TrainingConfig(epochs=epochs)
    report, predictions, checkpoint = run_benchmark(config, bootstrap)
    report["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__,
                             "matplotlib": matplotlib.__version__, "device": "cpu", "threads": 1}
    report["bootstrap_repeats"] = bootstrap
    report["source_code_sha256"] = {path.name: digest(path.read_bytes()) for path in sorted((ROOT / "studies").glob("*.py"))}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    predictions.to_csv(output / "test_predictions.csv", index=False, lineterminator="\n")
    render_all(report, predictions, figures)
    checkpoint_dir = ROOT / "artifacts"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoint_dir / "wdbc_residual_mlp.pt")
    print(json.dumps({"selected_by_validation": report["selected_by_validation_log_loss"],
                      "audit": report["audit"], "results": report["test_evaluation"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "studies" / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "assets" / "figures")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    run(args.output, args.figures, args.epochs, args.bootstrap)
