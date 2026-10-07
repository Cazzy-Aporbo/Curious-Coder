import pytest
import torch

from studies.adapters import LowRankLinear, frozen_digest, masked_mean
from studies.distributed_training import accumulated_backward, classifier, fixture, reference_experiment


def test_adapter_starts_as_exact_identity_update_and_freezes_base():
    torch.manual_seed(3)
    layer = LowRankLinear(torch.nn.Linear(5, 4), rank=2)
    inputs = torch.randn(7, 5)
    torch.testing.assert_close(layer(inputs), layer.base(inputs), rtol=0, atol=0)
    before = frozen_digest(layer)
    loss = layer(inputs).square().mean()
    loss.backward()
    assert layer.A.grad.count_nonzero() == 0
    assert layer.B.grad.count_nonzero() > 0
    assert layer.base.weight.grad is None
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=.01)
    optimizer.step()
    assert frozen_digest(layer) == before
    assert layer.B.count_nonzero() > 0
    torch.testing.assert_close(layer(inputs), layer.merged()(inputs), atol=1e-6, rtol=1e-5)


def test_protein_pooling_excludes_padding_and_special_tokens():
    hidden = torch.tensor([[[999., 999.], [2., 4.], [4., 8.], [float('nan'), 0.]]])
    pooled = masked_mean(hidden, torch.tensor([[1, 1, 1, 0]]), torch.tensor([[1, 0, 0, 1]]))
    torch.testing.assert_close(pooled, torch.tensor([[3., 6.]]))
    with pytest.raises(ValueError):
        masked_mean(hidden, torch.zeros(1, 4), torch.ones(1, 4))


def test_unequal_rank_means_differ_from_global_token_mean():
    report = reference_experiment()
    assert report["naive_max_error"] > .01
    assert report["weighted_max_error"] < 1e-12
    assert report["accumulated_max_error"] < 1e-12


def test_masked_tokens_do_not_change_accumulation_denominator():
    features, targets = fixture()
    extra = torch.cat([features, torch.zeros(2, 2, dtype=features.dtype)])
    labels = torch.cat([targets, torch.tensor([-100, -100])])
    model = classifier()
    count = accumulated_backward(model, [(extra[:2], labels[:2]), (extra[2:], labels[2:])])
    reference = classifier()
    torch.nn.functional.cross_entropy(reference(features), targets).backward()
    assert count == 4
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)


def test_all_masked_batch_requires_an_explicit_decision():
    with pytest.raises(ValueError, match="supervised token"):
        accumulated_backward(classifier(), [(torch.zeros(2, 2, dtype=torch.float64), torch.tensor([-100, -100]))])
