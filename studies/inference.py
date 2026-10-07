"""Export a trusted local research checkpoint with its preprocessing contract."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from studies.data import ROOT, clinical_data
from studies.modeling import ResidualTabularNetwork


class InferenceGraph(nn.Module):
    def __init__(self, model, mean, scale):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean.to(torch.float32))
        self.register_buffer("scale", scale.to(torch.float32))

    def forward(self, raw_features):
        return self.model((raw_features - self.mean) / self.scale).sigmoid()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frame(frame, feature_names, max_batch=4096):
    if list(frame.columns) != feature_names:
        raise ValueError("Input columns must match the manifest's exact feature names and order.")
    if not 1 <= len(frame) <= max_batch:
        raise ValueError(f"Batch must contain 1–{max_batch} rows.")
    if not all(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("Input features must be numeric, not strings or booleans.")
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Input features must remain finite after float32 conversion.")
    return torch.from_numpy(values)


def export_bundle(checkpoint_path, destination, example):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Use a new bundle directory; existing artifacts are not overwritten.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    features = checkpoint["input_features"]
    if len(features) != len(set(features)) or len(features) == 0:
        raise ValueError("Feature names must be nonempty and unique.")
    inputs = validate_frame(example, features)
    if len(inputs) < 2:
        raise ValueError("Export needs at least two example rows to infer a dynamic batch.")
    mean, scale = checkpoint["scaler_mean"], checkpoint["scaler_scale"]
    if mean.shape != (len(features),) or scale.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Scaler state must be finite, positive-scale, and aligned with the feature schema.")
    model = ResidualTabularNetwork(len(features), checkpoint["config"]["width"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    graph = InferenceGraph(model, mean, scale).eval()
    batch = torch.export.Dim("batch", min=1, max=4096)
    program = torch.export.export(graph, (inputs,), dynamic_shapes={"raw_features": {0: batch}})
    with torch.inference_mode():
        original_inputs = torch.from_numpy(example.to_numpy(dtype=np.float64))
        reference = model(((original_inputs - mean.double()) / scale.double()).float()).sigmoid()
        exported = program.module()(inputs)
    torch.testing.assert_close(exported, reference, rtol=1e-5, atol=1e-6)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "model.pt2"
    torch.export.save(program, artifact)
    manifest = {"schema_version": 1, "artifact": "model.pt2", "artifact_sha256": sha256(artifact),
                "checkpoint_sha256": sha256(checkpoint_path), "training_data_sha256": checkpoint["data_sha256"],
                "torch_version": torch.__version__, "feature_names": features, "dtype": "float32",
                "max_batch": 4096, "output": "experimental P(malignant), not a clinical decision",
                "max_export_error": float((exported - reference).abs().max())}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    reloaded = predict_bundle(destination, example)
    np.testing.assert_allclose(reloaded.probability_malignant, reference.numpy(), rtol=1e-5, atol=1e-6)
    return manifest


def predict_bundle(bundle, frame):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["artifact"] != "model.pt2" or manifest["max_batch"] != 4096:
        raise ValueError("Unsupported bundle schema.")
    artifact = bundle / "model.pt2"
    if sha256(artifact) != manifest["artifact_sha256"]:
        raise ValueError("Artifact digest mismatch; refusing inference.")
    inputs = validate_frame(frame, manifest["feature_names"], manifest["max_batch"])
    program = torch.export.load(artifact)
    with torch.inference_mode():
        probability = program.module()(inputs).numpy()
    if probability.shape != (len(frame),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Exported model violated its probability-output contract.")
    return pd.DataFrame({"input_row": np.arange(len(frame)), "probability_malignant": probability})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/wdbc_residual_mlp.pt")
    export.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    sample = commands.add_parser("example-input")
    sample.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        X, _ = clinical_data()
        print(json.dumps(export_bundle(args.checkpoint, args.output, X.iloc[:8]), indent=2))
    elif args.command == "predict":
        print(predict_bundle(args.bundle, pd.read_csv(args.input)).to_csv(index=False), end="")
    else:
        X, _ = clinical_data()
        with args.output.open("x") as output:
            X.iloc[:8].to_csv(output, index=False)
        print(f"Wrote eight public benchmark records to {args.output}")


if __name__ == "__main__":
    main()
