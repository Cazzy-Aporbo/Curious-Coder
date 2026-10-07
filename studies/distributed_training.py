"""Show why averaging rank-local means can change the training objective."""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    features = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., -1.]], dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    return features, targets


def classifier():
    model = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.2, -.1], [.4, .3], [-.2, .5]], dtype=torch.float64))
    return model


def token_loss_sum(logits, targets):
    if logits.ndim != 2 or targets.shape != (logits.shape[0],) or targets.dtype != torch.long:
        raise ValueError("Expected (tokens, classes) logits and one int64 target per token.")
    count = (targets != -100).sum()
    return F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum"), count


def accumulated_backward(model, microbatches, *, distributed=False):
    local_count = sum(int((target != -100).sum()) for _, target in microbatches)
    global_count = torch.tensor(local_count, dtype=torch.long)
    world_size = dist.get_world_size() if distributed else 1
    if distributed:
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if global_count.item() == 0:
        raise ValueError("At least one supervised token is required globally.")
    model.zero_grad(set_to_none=True)
    if not microbatches:
        raise ValueError("Use a dummy masked microbatch for an empty rank so it participates in synchronization.")
    for index, (features, targets) in enumerate(microbatches):
        context = model.no_sync() if distributed and index < len(microbatches) - 1 else nullcontext()
        with context:
            loss, _ = token_loss_sum(model(features), targets)
            (loss * world_size / global_count).backward()
    return int(global_count)


def reference_experiment():
    features, targets = fixture()
    reference = classifier()
    F.cross_entropy(reference(features), targets).backward()
    expected = reference.weight.grad.clone()
    local_gradients = []
    for rows in (slice(0, 1), slice(1, 4)):
        model = classifier()
        F.cross_entropy(model(features[rows]), targets[rows]).backward()
        local_gradients.append(model.weight.grad.clone())
    naive = (local_gradients[0] + local_gradients[1]) / 2
    weighted = (local_gradients[0] + 3 * local_gradients[1]) / 4
    accumulated = classifier()
    accumulated_backward(accumulated, [(features[:1], targets[:1]), (features[1:], targets[1:])])
    return {"fixture": "four synthetic token representations; shard sizes 1 and 3",
            "reference_gradient": expected.tolist(), "naive_rank_mean_gradient": naive.tolist(),
            "naive_max_error": float((naive - expected).abs().max()),
            "weighted_max_error": float((weighted - expected).abs().max()),
            "accumulated_max_error": float((accumulated.weight.grad - expected).abs().max()),
            "scope": "Numerical objective-equivalence fixture; not protein-model accuracy."}


def ddp_experiment(output):
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This fixture requires exactly two ranks.")
        features, targets = fixture()
        reference = classifier()
        F.cross_entropy(reference(features), targets).backward()
        model = DistributedDataParallel(classifier())
        rank = dist.get_rank()
        batches = [(features[:1], targets[:1])] if rank == 0 else [(features[1:2], targets[1:2]), (features[2:], targets[2:])]
        count = accumulated_backward(model, batches, distributed=True)
        error = float((model.module.weight.grad - reference.weight.grad).abs().max())
        torch.testing.assert_close(model.module.weight.grad, reference.weight.grad, atol=1e-12, rtol=1e-12)
        errors = [None, None]
        dist.all_gather_object(errors, error)
        if rank == 0:
            report = {**reference_experiment(), "backend": "gloo", "world_size": 2,
                      "global_valid_tokens": count, "ddp_max_error_by_rank": errors}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "studies/results/distributed.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.ddp:
        if "RANK" not in os.environ:
            parser.error("Use torchrun with --nproc-per-node=2 for --ddp.")
        ddp_experiment(args.output)
    else:
        print(json.dumps(reference_experiment(), indent=2))
