"""Audit low-rank adaptation of a pinned ESM-2 checkpoint on public human proteins."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from studies.adapters import frozen_digest, install_attention_adapters
from studies.data import ROOT, digest, download


MODEL_ID = "facebook/esm2_t6_8M_UR50D"
REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
WEIGHTS_SHA256 = "24c5fa474c48f3b754b86efe752d5f189d2bcd88190fa2270fc92b2ef3034189"
MODEL_DIR = ROOT / "artifacts/esm2-base"
SEQUENCE_DIR = ROOT / "data/protein_fixture"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(text):
    lines = text.strip().splitlines()
    if not lines or not lines[0].startswith(">") or any(line.startswith(">") for line in lines[1:]):
        raise ValueError("Expect exactly one FASTA record.")
    sequence = "".join(line.strip() for line in lines[1:])
    if not sequence or len(sequence) > 1022 or set(sequence) - AMINO_ACIDS:
        raise ValueError("This fixture accepts 1–1022 standard amino-acid residues, not DNA or ambiguous symbols.")
    return sequence


def acquire_resources():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    model_files = {}
    for filename in ("config.json", "model.safetensors", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md"):
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{filename}"
        path = MODEL_DIR / filename
        payload = path.read_bytes() if path.exists() else download(url, limit=40_000_000)
        if filename == "model.safetensors" and digest(payload) != WEIGHTS_SHA256:
            raise ValueError("Pretrained weight digest does not match the pinned publisher metadata.")
        if not path.exists():
            path.write_bytes(payload)
        model_files[filename] = {"source": url, "sha256": digest(payload)}
    if not (MODEL_DIR / "manifest.json").exists():
        (MODEL_DIR / "manifest.json").write_text(json.dumps({"model_id": MODEL_ID, "revision": REVISION,
            "license": "MIT", "retrieved_utc": datetime.now(timezone.utc).isoformat(), "files": model_files}, indent=2) + "\n")
    if not (SEQUENCE_DIR / "manifest.json").exists():
        records = {}
        for accession in ("P69905", "P68871"):
            url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
            payload = download(url)
            sequence = parse_fasta(payload.decode("utf-8"))
            path = SEQUENCE_DIR / f"{accession}.fasta"
            if path.exists() and path.read_bytes() != payload:
                raise ValueError("Existing sequence differs from upstream; do not silently replace it.")
            path.write_bytes(payload)
            records[accession] = {"source": url, "sha256": digest(payload), "residues": len(sequence)}
        (SEQUENCE_DIR / "manifest.json").write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "attribution": "Copyright UniProt Consortium; CC BY 4.0; https://www.uniprot.org/help/license",
            "purpose": "Human hemoglobin mechanics fixtures; not independent evaluation samples.", "records": records}, indent=2) + "\n")


def load_resources():
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("Checkpoint revision mismatch.")
    for name, entry in manifest["files"].items():
        if Path(name).name != name or digest((MODEL_DIR / name).read_bytes()) != entry["sha256"]:
            raise ValueError("Model resource integrity mismatch.")
    if digest((MODEL_DIR / "model.safetensors").read_bytes()) != WEIGHTS_SHA256:
        raise ValueError("Pinned pretrained weights have changed.")
    sequence_manifest = json.loads((SEQUENCE_DIR / "manifest.json").read_text())
    sequences = []
    for accession in ("P69905", "P68871"):
        payload = (SEQUENCE_DIR / f"{accession}.fasta").read_bytes()
        if digest(payload) != sequence_manifest["records"][accession]["sha256"]:
            raise ValueError("Protein sequence integrity mismatch.")
        sequences.append(parse_fasta(payload.decode()))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False, use_safetensors=True)
    return model, tokenizer, sequences


def run(steps=3, output=ROOT / "studies/results/protein_transfer.json"):
    from safetensors.torch import save_file

    if not 1 <= steps <= 20:
        raise ValueError("This small mechanics fixture permits 1–20 adaptation steps.")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, tokenizer, sequences = load_resources()
    batch = tokenizer(sequences, padding=True, return_tensors="pt", return_special_tokens_mask=True)
    valid = batch["attention_mask"].bool() & ~batch.pop("special_tokens_mask").bool()
    labels = torch.full_like(batch["input_ids"], -100)
    for row in range(len(sequences)):
        selected = torch.nonzero(valid[row], as_tuple=True)[0][::11]
        labels[row, selected] = batch["input_ids"][row, selected]
        batch["input_ids"][row, selected] = tokenizer.mask_token_id
    model.eval()
    with torch.no_grad():
        original = model(**batch).logits.clone()
    projections = install_attention_adapters(model, rank=4, alpha=8)
    before = frozen_digest(model)
    with torch.no_grad():
        initial = model(**batch).logits
    torch.testing.assert_close(initial, original, atol=0, rtol=0)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, labels=labels).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite adaptation loss.")
        losses.append(float(loss.detach()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
    after = frozen_digest(model)
    if before != after:
        raise AssertionError("A frozen pretrained parameter changed.")
    with torch.no_grad():
        final = model(**batch, labels=labels)
    adapter_state = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}
    destination = ROOT / "artifacts/esm2-adapter.safetensors"
    save_file(adapter_state, destination)
    report = {"model_id": MODEL_ID, "revision": REVISION, "base_weights_sha256": WEIGHTS_SHA256,
              "sequence_accessions": ["P69905", "P68871"], "residue_counts": list(map(len, sequences)),
              "masked_residue_count": int((labels != -100).sum()), "adapted_projections": projections,
              "rank": 4, "alpha": 8, "steps": steps, "seed": 42, "learning_rate": .001,
              "trainable_parameters": sum(p.numel() for p in trainable), "total_parameters_with_adapters": sum(p.numel() for p in model.parameters()),
              "training_loss_before_each_step": losses, "same_fixture_loss_after": float(final.loss),
              "initial_adapter_output_max_error": float((initial - original).abs().max()),
              "changed_output_max_difference": float((final.logits - original).abs().max()),
              "frozen_parameters_unchanged": before == after, "frozen_parameter_digest": after,
              "adapter_sha256": digest(destination.read_bytes()), "torch_version": torch.__version__,
              "source_code_sha256": {name: digest((ROOT / "studies" / name).read_bytes()) for name in ("adapters.py", "protein_transfer.py")},
              "interpretation": "Training-mechanics audit only. Homologous proteins; possible pretraining overlap; no held-out biological evaluation or efficacy claim."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "studies/results/protein_transfer.json")
    args = parser.parse_args()
    if args.download:
        acquire_resources()
    run(args.steps, args.output)
