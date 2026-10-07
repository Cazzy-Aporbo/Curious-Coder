"""Three objective failures: prediction-error timing, stochastic curiosity traps, and proxy overoptimization."""

import argparse
import json
from pathlib import Path

import numpy as np

from studies.data import ROOT


def td_prediction_errors(trials=150, steps=5, alpha=.2, omission_trial=None):
    if trials < 1 or steps < 2 or not 0 < alpha <= 1:
        raise ValueError("Require positive trials, at least two post-cue steps, and alpha in (0, 1].")
    values = np.zeros(steps + 1)
    history = []
    for trial in range(trials):
        reward = 0. if trial == omission_trial else 1.
        cue_error = values[0]
        errors = [cue_error]
        for step in range(steps):
            outcome = reward if step == steps - 1 else 0.
            delta = outcome + values[step + 1] - values[step]
            values[step] += alpha * delta
            errors.append(delta)
        history.append({"trial": trial, "cue_error": float(cue_error), "reward_error": float(errors[-1]), "omitted": reward == 0.})
    return {"history": history, "final_values": values[:steps].tolist(),
            "model": "Tabular TD(0), gamma=1. Cue timing is unpredictable, so the pre-cue state value is fixed at zero."}


def categorical_belief(counts, prior):
    return (counts + prior) / (counts.sum() + prior * len(counts))


def curiosity_trial(signal, steps=2000, symbols=16, task_reward=.5, intrinsic_weight=1., epsilon=.05, seed=0, window=20):
    if signal not in {"prediction_error", "belief_change", "learning_progress"}:
        raise ValueError("Unknown intrinsic signal.")
    if steps < 2 * window or symbols < 2 or window < 2:
        raise ValueError("Require enough steps, symbols, and window length.")
    rng = np.random.default_rng(seed)
    arms = ["task", "landmark-A", "landmark-B", "noisy-screen"]
    fixed_symbol = {"task": 0, "landmark-A": 1, "landmark-B": 2}
    counts = np.zeros((len(arms), symbols))
    surprises = [[] for _ in arms]
    estimate = np.full(len(arms), 10.)
    choices, extrinsic = [], 0.
    for _ in range(steps):
        arm = int(rng.integers(len(arms))) if rng.random() < epsilon else int(np.argmax(estimate + np.arange(len(arms)) * 1e-9))
        observation = int(rng.integers(symbols)) if arms[arm] == "noisy-screen" else fixed_symbol[arms[arm]]
        before = categorical_belief(counts[arm], 1.)
        surprise = float(-np.log(before[observation]))
        counts[arm, observation] += 1
        after = categorical_belief(counts[arm], 1.)
        surprises[arm].append(surprise)
        if signal == "prediction_error":
            intrinsic = surprise
        elif signal == "belief_change":
            intrinsic = float(np.sum(after * np.log(after / before)))
        else:
            history = surprises[arm]
            intrinsic = abs(float(np.mean(history[-2 * window:-window])) - float(np.mean(history[-window:]))) if len(history) >= 2 * window else 10.
        reward = task_reward if arms[arm] == "task" else 0.
        extrinsic += reward
        estimate[arm] = reward + intrinsic_weight * intrinsic
        choices.append(arms[arm])
    tail = choices[steps // 2:]
    return {"signal": signal, "seed": seed, "screen_fraction_second_half": tail.count("noisy-screen") / len(tail),
            "task_fraction_second_half": tail.count("task") / len(tail), "extrinsic_return": extrinsic,
            "screen_fraction_by_block": [choices[start:start + 100].count("noisy-screen") / 100 for start in range(0, steps, 100)]}


def proxy_overoptimization(pressures=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), repeats=2000, seed=0, style_weight=1., style_penalty=.5):
    rng = np.random.default_rng(seed)
    rows = []
    for count in pressures:
        if type(count) is not int or count < 1:
            raise ValueError("Optimization pressure must be a positive integer candidate count.")
        quality = rng.standard_normal((repeats, count))
        style = rng.standard_normal((repeats, count))
        proxy = quality + style_weight * style
        true = quality - style_penalty * style ** 2
        chosen = np.argmax(proxy, axis=1)
        selected_true = true[np.arange(repeats), chosen]
        selected_proxy = proxy[np.arange(repeats), chosen]
        rows.append({"candidates": count, "mean_proxy": float(selected_proxy.mean()), "mean_true": float(selected_true.mean()),
                     "true_standard_error": float(selected_true.std(ddof=1) / np.sqrt(repeats)),
                     "mean_selected_style": float(style[np.arange(repeats), chosen].mean())})
    best = max(rows, key=lambda row: row["mean_true"])
    return {"rows": rows, "true_optimum_candidates": best["candidates"],
            "model": f"proxy = quality + {style_weight}·style; true utility = quality − {style_penalty}·style². Best-of-n selection on proxy."}


def run(output=ROOT / "studies/results/learning_signals.json", seeds=tuple(range(10))):
    curiosity = {signal: [curiosity_trial(signal, seed=seed) for seed in seeds] for signal in ("prediction_error", "belief_change", "learning_progress")}
    summary = {signal: {key: float(np.mean([trial[key] for trial in trials])) for key in ("screen_fraction_second_half", "task_fraction_second_half", "extrinsic_return")}
               for signal, trials in curiosity.items()}
    report = {"evidence_type": "Seeded computational demonstrations; not neural recordings, human-behaviour data, or evaluations of a deployed model.",
              "td": td_prediction_errors(), "td_omission": td_prediction_errors(trials=151, omission_trial=150)["history"][-1],
              "curiosity": {"trials": curiosity, "summary": summary, "seeds": list(seeds)},
              "proxy": proxy_overoptimization(),
              "implementation_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"curiosity": summary, "td_omission": report["td_omission"], "proxy_optimum": report["proxy"]["true_optimum_candidates"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "studies/results/learning_signals.json")
    run(parser.parse_args().output)
