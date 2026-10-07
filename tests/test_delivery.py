import asyncio
from builtins import ExceptionGroup
import json

import numpy as np
import pandas as pd
import pytest
import torch

from studies.async_pipeline import Job, PermanentFailure, TransientFailure, process_jobs, request_id
from studies.inference import export_bundle, predict_bundle, validate_frame
from studies.modeling import ResidualTabularNetwork


@pytest.fixture
def bundle(tmp_path):
    torch.manual_seed(4)
    model = ResidualTabularNetwork(3, 8).eval()
    checkpoint = {"input_features": ["a", "b", "c"], "config": {"width": 8},
                  "scaler_mean": torch.tensor([1., 2., 3.]), "scaler_scale": torch.tensor([2., 3., 4.]),
                  "state_dict": model.state_dict(), "data_sha256": "fixture"}
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    examples = pd.DataFrame(np.arange(24).reshape(8, 3), columns=["a", "b", "c"])
    destination = tmp_path / "bundle"
    export_bundle(path, destination, examples)
    return destination, examples, model, checkpoint


def test_export_reload_parity_and_dynamic_batches(bundle):
    destination, examples, model, checkpoint = bundle
    for count in (1, 3, 8):
        frame = examples.iloc[:count]
        with torch.inference_mode():
            normalized = (torch.tensor(frame.to_numpy(), dtype=torch.float32) - checkpoint["scaler_mean"]) / checkpoint["scaler_scale"]
            expected = model(normalized).sigmoid().numpy()
        actual = predict_bundle(destination, frame).probability_malignant.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_artifact_digest_mismatch_rejected_before_loading(bundle):
    destination, examples, _, _ = bundle
    (destination / "model.pt2").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        predict_bundle(destination, examples)


@pytest.mark.parametrize("frame", [pd.DataFrame([[1, 2]], columns=["b", "a"]),
                                   pd.DataFrame([[1, np.nan]], columns=["a", "b"]),
                                   pd.DataFrame([[True, 2]], columns=["a", "b"]),
                                   pd.DataFrame(columns=["a", "b"])])
def test_inference_schema_fails_closed(frame):
    with pytest.raises(ValueError):
        validate_frame(frame, ["a", "b"])


def test_bundle_manifest_records_preprocessing_contract(bundle):
    destination, _, _, _ = bundle
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["feature_names"] == ["a", "b", "c"]
    assert manifest["max_export_error"] <= 1e-6


def test_pipeline_retries_expected_failures_and_preserves_context():
    async def scenario():
        calls = {}

        async def handler(job):
            await asyncio.sleep(0)
            assert request_id.get() == job.key
            calls[job.key] = calls.get(job.key, 0) + 1
            if calls[job.key] == 1:
                raise TransientFailure()
            return job.payload["value"]

        outcomes, stats = await process_jobs((Job(str(i), {"value": i}) for i in range(12)), handler,
                                             workers=2, capacity=3, base_delay=0)
        assert [result.value for result in outcomes] == list(range(12))
        assert all(result.attempts == 2 for result in outcomes)
        assert stats.retries == 12 and stats.completed == 12
        assert stats.max_active <= 2 and stats.max_queued <= 3
        assert request_id.get() is None
    asyncio.run(scenario())


def test_permanent_errors_are_not_retried():
    async def handler(job):
        raise PermanentFailure()

    outcomes, stats = asyncio.run(process_jobs([Job("bad", {})], handler))
    assert outcomes[0].attempts == 1 and stats.failed == 1 and stats.retries == 0


def test_timeout_has_a_bounded_attempt_budget():
    async def handler(job):
        await asyncio.sleep(10)

    outcomes, stats = asyncio.run(process_jobs([Job("slow", {})], handler, attempts=2, timeout=.005, base_delay=0))
    assert outcomes[0].attempts == 2 and outcomes[0].error == "TimeoutError"
    assert stats.retries == 1


def test_unexpected_error_cancels_blocked_producer_instead_of_hanging():
    async def handler(job):
        raise RuntimeError("programming bug")

    async def scenario():
        async with asyncio.timeout(2):
            with pytest.raises(ExceptionGroup):
                await process_jobs((Job(str(i), {}) for i in range(100)), handler, capacity=1)
    asyncio.run(scenario())


def test_measured_study_checkpoint_loads_without_unsafe_deserialization(tmp_path):
    from studies.modeling import TrainingConfig, run_benchmark

    _, _, checkpoint = run_benchmark(TrainingConfig(epochs=1), bootstrap_repeats=2)
    path = tmp_path / "measured.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, weights_only=True)
    assert all(type(name) is str for name in loaded["input_features"])
    assert len(loaded["input_features"]) == 30


def test_cancellation_reaches_active_handlers():
    async def scenario():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def handler(job):
            started.set()
            try:
                await asyncio.sleep(100)
            finally:
                cancelled.set()

        task = asyncio.create_task(process_jobs([Job("active", {})], handler))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
    asyncio.run(scenario())
