import numpy as np
import pytest

from studies.learning_signals import categorical_belief, curiosity_trial, proxy_overoptimization, td_prediction_errors


def test_td_error_transfers_from_reward_to_cue_and_turns_negative_on_omission():
    history = td_prediction_errors(trials=200)["history"]
    assert history[0]["reward_error"] == pytest.approx(1) and history[0]["cue_error"] == 0
    assert history[-1]["cue_error"] == pytest.approx(1, abs=1e-6)
    assert history[-1]["reward_error"] == pytest.approx(0, abs=1e-6)
    omitted = td_prediction_errors(trials=201, omission_trial=200)["history"][-1]
    assert omitted["reward_error"] == pytest.approx(-1, abs=1e-6)


def test_belief_is_a_proper_distribution():
    belief = categorical_belief(np.array([3., 0., 1.]), 1.)
    assert belief.sum() == pytest.approx(1) and (belief > 0).all()


def test_surprise_seeking_is_trapped_while_belief_change_and_progress_escape():
    trapped = [curiosity_trial("prediction_error", seed=seed) for seed in range(3)]
    for signal in ("belief_change", "learning_progress"):
        escaped = [curiosity_trial(signal, seed=seed) for seed in range(3)]
        assert np.mean([t["screen_fraction_second_half"] for t in escaped]) < .1
        assert np.mean([t["extrinsic_return"] for t in escaped]) > 5 * np.mean([t["extrinsic_return"] for t in trapped])
    assert np.mean([t["screen_fraction_second_half"] for t in trapped]) > .8


def test_trial_is_reproducible_and_rejects_unknown_signals():
    assert curiosity_trial("belief_change", seed=4) == curiosity_trial("belief_change", seed=4)
    with pytest.raises(ValueError):
        curiosity_trial("novelty_maximizer")


def test_proxy_keeps_rising_after_true_utility_turns_down():
    result = proxy_overoptimization(repeats=2000)
    rows = result["rows"]
    proxies = [row["mean_proxy"] for row in rows]
    assert all(later > earlier for earlier, later in zip(proxies, proxies[1:]))
    best = max(row["mean_true"] for row in rows)
    assert rows[-1]["mean_true"] < best - 5 * rows[-1]["true_standard_error"]
    assert 1 < result["true_optimum_candidates"] < rows[-1]["candidates"]
    with pytest.raises(ValueError):
        proxy_overoptimization(pressures=(0,))
