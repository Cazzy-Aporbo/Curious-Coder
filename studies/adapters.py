"""Small, inspectable low-rank updates with explicit frozen-weight invariants."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError("Require a linear layer and a positive rank within both dimensions.")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive.")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)

    def merged(self):
        result = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                           device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            result.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
            if self.base.bias is not None:
                result.bias.copy_(self.base.bias)
        return result.eval()


def install_attention_adapters(model, rank=4, alpha=8):
    targets = [(name, layer) for name, layer in model.named_modules()
               if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in {"query", "value"}]
    if not targets:
        raise ValueError("No query/value projections found; inspect the model architecture before adapting it.")
    model.requires_grad_(False)
    for name, layer in targets:
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, attribute, LowRankLinear(layer, rank, alpha))
    return [name for name, _ in targets]


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            tensor = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def masked_mean(hidden, attention_mask, special_tokens_mask):
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2] or special_tokens_mask.shape != attention_mask.shape:
        raise ValueError("Hidden states and token masks must agree on batch and sequence dimensions.")
    valid = attention_mask.bool() & ~special_tokens_mask.bool()
    counts = valid.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("Each sequence needs at least one biological residue.")
    safe = hidden.masked_fill(~valid.unsqueeze(-1), 0)
    if not torch.isfinite(safe).all():
        raise ValueError("Residue representations must be finite.")
    return safe.sum(dim=1) / counts
