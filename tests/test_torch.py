import numpy as np
import pytest
import torch

from adaptive_gradient_scheduler import (
    AdaptiveGradientScheduler,
    FeatureSpaceRegularizer,
    ImbalancedDatasetSampler,
)
from medical_timeseries_implementation import (
    MedicalDataAugmentation,
    MedicalTimeSeriesEncoder,
    PatientRiskStratificationModel,
    TrainingPipeline,
    generate_synthetic_medical_data,
)


@pytest.fixture(autouse=True)
def seed_torch():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def test_warmup_updates_optimizer_and_diagnostics():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = AdaptiveGradientScheduler(optimizer, base_lr=0.01, warmup_steps=10)
    lr = scheduler.compute_adaptive_lr(1.0, model)
    assert lr == pytest.approx(0.001)
    assert optimizer.param_groups[0]["lr"] == lr
    assert scheduler.lr_history == [lr]
    assert scheduler.get_diagnostics()["current_lr"] == lr


def test_time_warp_is_not_a_noop_and_preserves_input():
    signal = torch.arange(40, dtype=torch.float32).reshape(1, 1, 40)
    original = signal.clone()
    warped = MedicalDataAugmentation(time_warp_factor=2).apply_time_warping(signal)
    assert warped.shape == signal.shape
    assert not torch.equal(warped, signal)
    assert torch.equal(signal, original)
    assert torch.isfinite(warped).all()
    assert warped.min() >= signal.min()
    assert warped.max() <= signal.max()


def test_zero_time_warp_is_identity():
    signal = torch.randn(2, 3, 40)
    torch.testing.assert_close(MedicalDataAugmentation(time_warp_factor=0).apply_time_warping(signal), signal)


def test_motion_artifacts_do_not_mutate_training_data():
    signal = torch.zeros(4, 3, 40)
    MedicalDataAugmentation().add_motion_artifacts(signal)
    assert torch.count_nonzero(signal) == 0


def test_patient_model_forward_backward_cpu():
    model = PatientRiskStratificationModel(sequence_length=16, embed_dim=16, num_layers=1)
    signals, labels, vitals = generate_synthetic_medical_data(4, sequence_length=16)
    outputs = model(signals)
    assert outputs["risk_logits"].shape == (4, 4)
    assert outputs["vital_predictions"].shape == (4, 5)
    pipeline = TrainingPipeline(model, device="cpu")
    loss = pipeline.train_step((signals, labels, vitals))
    assert np.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_validation_weights_examples_not_batches():
    class FixedModel(torch.nn.Module):
        feature_dim = 2

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, signals):
            return {"risk_logits": signals * self.scale}

    signals = torch.tensor([[5., 0.], [5., 0.], [0., 5.]])
    labels = torch.zeros(3, dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(signals, labels, signals), batch_size=2)
    pipeline = TrainingPipeline(FixedModel(), device="cpu", num_classes=2)
    loss, accuracy = pipeline.validate(loader)
    assert loss == pytest.approx(torch.nn.functional.cross_entropy(signals, labels).item())
    assert accuracy == pytest.approx(2 / 3)


@pytest.mark.parametrize("num_channels", [1, 4, 6])
def test_synthetic_data_rejects_unsupported_channel_count(num_channels):
    with pytest.raises(ValueError, match="five"):
        generate_synthetic_medical_data(2, sequence_length=16, num_channels=num_channels)


def test_encoder_ignores_values_in_masked_positions_and_pools_valid_steps():
    encoder = MedicalTimeSeriesEncoder(3, 16, embed_dim=16, num_layers=1, dropout=0).eval()
    x = torch.randn(2, 3, 16)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[0, 8:] = True
    mask[1, 12:] = True
    changed = x.masked_fill(mask.unsqueeze(1), float('nan'))
    pooled, temporal = encoder(x, mask)
    changed_pooled, changed_temporal = encoder(changed, mask)
    torch.testing.assert_close(pooled, changed_pooled)
    torch.testing.assert_close(temporal, changed_temporal)
    for row in range(2):
        valid = temporal[row, ~mask[row]]
        torch.testing.assert_close(pooled[row], torch.cat([valid.max(dim=0).values, valid.mean(dim=0)]))
    assert torch.count_nonzero(temporal[mask]) == 0


def test_encoder_accepts_sequences_shorter_than_maximum():
    encoder = MedicalTimeSeriesEncoder(3, 16, embed_dim=16, num_layers=1).eval()
    pooled, temporal = encoder(torch.randn(2, 3, 9))
    assert pooled.shape == (2, 32)
    assert temporal.shape == (2, 9, 16)


@pytest.mark.parametrize("mask", [torch.ones(2, 16, dtype=torch.bool), torch.zeros(2, 15, dtype=torch.bool), torch.zeros(2, 16)])
def test_encoder_rejects_invalid_or_entirely_masked_sequences(mask):
    encoder = MedicalTimeSeriesEncoder(3, 16, embed_dim=16, num_layers=1)
    with pytest.raises(ValueError):
        encoder(torch.randn(2, 3, 16), mask)


def test_encoder_rejects_missing_values_at_valid_positions():
    encoder = MedicalTimeSeriesEncoder(3, 16, embed_dim=16, num_layers=1)
    x = torch.randn(2, 3, 16)
    x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match="finite"):
        encoder(x)


def test_sampler_assigns_equal_probability_mass_to_observed_classes():
    labels = torch.tensor([0] * 90 + [2] * 10)
    sampler = ImbalancedDatasetSampler(labels)
    weights = sampler.sample_weights / sampler.sample_weights.sum()
    assert weights[labels == 0].sum().item() == pytest.approx(0.5)
    assert weights[labels == 2].sum().item() == pytest.approx(0.5)
    indices = list(sampler)
    assert len(indices) == len(labels)
    assert all(0 <= index < len(labels) for index in indices)


def test_sampler_adapts_for_noncontiguous_label_indices():
    labels = torch.tensor([2] * 121 + [5] * 121)
    sampler = ImbalancedDatasetSampler(labels, adaptation_rate=0.5)
    logits = torch.zeros(len(labels), 6)
    logits[:, 2] = 1
    sampler.update_weights(logits, labels)
    assert sampler.sample_weights[labels == 5].sum() > sampler.sample_weights[labels == 2].sum()
    assert torch.isfinite(sampler.sample_weights).all()


@pytest.mark.parametrize("labels", [torch.tensor([]), torch.tensor([-1, 0]), torch.tensor([0., 1.]), torch.tensor([[0, 1]])])
def test_sampler_rejects_invalid_labels(labels):
    with pytest.raises(ValueError):
        ImbalancedDatasetSampler(labels)


@pytest.mark.parametrize("weights", [torch.tensor([0., 0.]), torch.tensor([-1., 2.]), torch.tensor([1.]), torch.tensor([float('nan'), 1.])])
def test_sampler_rejects_invalid_initial_weights(weights):
    with pytest.raises(ValueError):
        ImbalancedDatasetSampler(torch.tensor([0, 1]), initial_weights=weights)


def test_spectral_diversity_prefers_spread_to_collapsed_features():
    labels = torch.zeros(4, dtype=torch.long)
    collapsed = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
    spread = torch.eye(4)
    regularizer = FeatureSpaceRegularizer(4, 1).eval()
    assert regularizer(spread, labels) < regularizer(collapsed, labels)


def test_class_separation_contributes_a_gradient():
    features = torch.tensor([[1., 0., 0.], [.8, .2, 0.], [.9, .2, .1], [.8, .3, .1]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    regularizer = FeatureSpaceRegularizer(3, 2).eval()
    total = regularizer(features, labels)
    diversity_only = regularizer(features, torch.zeros_like(labels))
    total_gradient = torch.autograd.grad(total, features, retain_graph=True)[0]
    diversity_gradient = torch.autograd.grad(diversity_only, features)[0]
    assert total > diversity_only
    assert torch.isfinite(total_gradient).all()
    assert not torch.allclose(total_gradient, diversity_gradient)


def test_regularizer_evaluation_does_not_update_prototypes():
    regularizer = FeatureSpaceRegularizer(3, 2).eval()
    regularizer(torch.randn(4, 3), torch.tensor([0, 0, 1, 1]))
    assert torch.count_nonzero(regularizer.class_counts) == 0
    assert torch.count_nonzero(regularizer.class_prototypes) == 0


def test_regularizer_handles_single_example_with_finite_backward():
    features = torch.randn(1, 3, requires_grad=True)
    loss = FeatureSpaceRegularizer(3, 2)(features, torch.tensor([1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()


def test_regularizer_does_not_reward_zero_features_as_diverse():
    regularizer = FeatureSpaceRegularizer(4, 1).eval()
    labels = torch.zeros(4, dtype=torch.long)
    assert regularizer(torch.eye(4), labels) < regularizer(torch.zeros(4, 4), labels)


def test_regularizer_prototype_updates_support_repeated_backward():
    regularizer = FeatureSpaceRegularizer(3, 2)
    for _ in range(3):
        features = torch.randn(4, 3, requires_grad=True)
        loss = regularizer(features, torch.tensor([0, 0, 1, 1]))
        loss.backward()
        assert torch.isfinite(features.grad).all()
    torch.testing.assert_close(regularizer.class_counts, torch.tensor([3., 3.]))
    assert not regularizer.class_prototypes.requires_grad


def test_encoder_zeroes_gradients_for_padding():
    encoder = MedicalTimeSeriesEncoder(3, 16, embed_dim=16, num_layers=1, dropout=0).eval()
    inputs = torch.randn(2, 3, 16, requires_grad=True)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, 8:] = True
    pooled, _ = encoder(inputs, mask)
    pooled.square().sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad[:, :, 8:]) == 0
    assert torch.count_nonzero(inputs.grad[:, :, :8]) > 0


def test_sampler_adaptation_keeps_class_probability_floor():
    labels = torch.tensor([0] * 1000 + [1] * 121)
    sampler = ImbalancedDatasetSampler(labels, adaptation_rate=1, min_sample_rate=0.2)
    logits = torch.tensor([[1., 0.]]).repeat(len(labels), 1)
    sampler.update_weights(logits, labels)
    assert sampler.sample_weights[labels == 0].sum().item() == pytest.approx(0.1)
    assert sampler.sample_weights[labels == 1].sum().item() == pytest.approx(0.9)


@pytest.mark.parametrize("features, labels", [
    (torch.empty(0, 3), torch.empty(0, dtype=torch.long)),
    (torch.ones(2, 3), torch.tensor([0, 2])),
    (torch.ones(2, 3), torch.tensor([0., 1.])),
    (torch.full((2, 3), float('nan')), torch.tensor([0, 1])),
])
def test_regularizer_rejects_invalid_features_and_labels(features, labels):
    with pytest.raises(ValueError):
        FeatureSpaceRegularizer(3, 2)(features, labels)


@pytest.mark.parametrize("logits, targets", [
    (torch.zeros(2, 2), torch.tensor([2, 5])),
    (torch.zeros(2, 6), torch.tensor([2, 4])),
    (torch.full((2, 6), float('nan')), torch.tensor([2, 5])),
])
def test_sampler_rejects_invalid_update_inputs(logits, targets):
    sampler = ImbalancedDatasetSampler(torch.tensor([2, 5]))
    with pytest.raises(ValueError):
        sampler.update_weights(logits, targets)


def test_sampler_preserves_caller_owned_initial_weights():
    weights = torch.tensor([1., 3.])
    sampler = ImbalancedDatasetSampler(torch.tensor([0, 0, 1]), initial_weights=weights)
    weights.zero_()
    torch.testing.assert_close(sampler.class_weights, torch.tensor([.25, .75]))
