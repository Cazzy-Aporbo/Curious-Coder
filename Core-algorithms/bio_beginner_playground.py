#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bio Beginner Playground — Think Like a Computational Biologist 

How to run
  python bio_beginner_playground.py           # interactive menu
  python bio_beginner_playground.py --demo all
  python bio_beginner_playground.py --demo growth
  python bio_beginner_playground.py --demo virus
  python bio_beginner_playground.py --demo feedback
  python bio_beginner_playground.py --demo dice
  python bio_beginner_playground.py --demo cell

- How to turn a biology story into code with “state”, “rules”, and “time steps”
- Growth as repeated multiplication (like interest) and crowding (logistic)
- Virus spread like rumors in a group (S, I, R counts)
- Feedback loops like a thermostat (aim for a setpoint)
- Randomness with dice to make simulations feel real

"""

from __future__ import annotations

import argparse
import math
import random
import sys
from typing import List, Tuple

# Optional plotting — will be used if available
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


def maybe_plot(xs, ys_list, labels, title, xlab, ylab) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7,4), dpi=130)
    for y, lab in zip(ys_list, labels):
        ax.plot(xs, y, marker='o', lw=1.8, label=lab)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


# 1) Cells as simple "state machines"
def demo_cell_state_machine(steps: int = 12) -> None:
    """
    A tiny cell with states: Resting -> Growing -> Dividing -> Resting ...
    Rule: after a few steps in Growing, it divides into two cells.
    """
    print("\nCELLS AS STATE MACHINES")
    print("========================")
    state = "Resting"
    cells = 1
    grow_timer = 0
    history = [cells]
    states = [state]

    for t in range(1, steps+1):
        if state == "Resting":
            # cell decides to start growing with some chance
            if random.random() < 0.6:
                state = "Growing"
                grow_timer = 0
        elif state == "Growing":
            grow_timer += 1
            if grow_timer >= 3:
                state = "Dividing"
        elif state == "Dividing":
            cells += 1  # one cell -> two cells
            state = "Resting"

        history.append(cells)
        states.append(state)

    print("Time:", list(range(steps+1)))
    print("Cells:", history)
    print("State:", states)
    maybe_plot(list(range(steps+1)), [history], ["cells"], "Cell State Machine", "time", "cells")


# 2) Population growth (exponential then logistic with crowding)
def demo_growth(steps: int = 20, r: float = 0.4, K: int = 100) -> None:
    """
    Exponential: N[t+1] = N[t] * (1 + r)
    Logistic:    N[t+1] = N[t] + r * N[t] * (1 - N[t]/K)
    """
    print("\nPOPULATION GROWTH")
    print("==================")
    N_exp = 5.0
    N_log = 5.0
    xs = [0]
    exp_hist = [N_exp]
    log_hist = [N_log]

    for t in range(1, steps+1):
        N_exp = N_exp * (1 + r)
        N_log = N_log + r * N_log * (1 - N_log / K)
        xs.append(t)
        exp_hist.append(N_exp)
        log_hist.append(N_log)

    print(f"After {steps} steps: exponential≈{exp_hist[-1]:.1f}, logistic≈{log_hist[-1]:.1f} (K={K})")
    maybe_plot(xs, [exp_hist, log_hist], ["exponential", "logistic"], "Population Growth", "time", "N")


# 3) Virus spread like rumors (SIR-style counts)
def demo_virus(n_people: int = 120, p_infect: float = 0.08, p_recover: float = 0.05, steps: int = 80) -> None:
    """
    S (susceptible), I (infected), R (recovered). At each step:
      - Each infected person tries to infect a few random contacts
      - Each infected person may recover
    """
    print("\nVIRUS SPREAD LIKE RUMORS")
    print("=========================")
    rng = random.Random(2)
    S, I, R = n_people - 1, 1, 0  # one starter
    S_hist, I_hist, R_hist = [S], [I], [R]

    for t in range(steps):
        # everyone meets 'm' others randomly
        m = 6
        new_inf = 0
        for _ in range(I * m):
            # randomly pick a person; if S, they might be infected
            if S > 0 and rng.random() < p_infect * (S / n_people):
                new_inf += 1
        new_inf = min(new_inf, S)
        # recoveries
        recov = 0
        for _ in range(I):
            if rng.random() < p_recover:
                recov += 1

        S -= new_inf
        I += new_inf - recov
        R += recov
        S_hist.append(S); I_hist.append(I); R_hist.append(R)

    peak_I = max(I_hist)
    t_peak = I_hist.index(peak_I)
    print(f"Peak infected={peak_I} at time {t_peak}, final recovered={R_hist[-1]} of {n_people}")
    maybe_plot(list(range(len(S_hist))), [S_hist, I_hist, R_hist], ["S","I","R"], "Rumor-like Virus Spread", "time", "people")


# 4) Feedback loop like a thermostat (simple control to reach a target)
def demo_feedback(setpoint: float = 37.0, steps: int = 60) -> None:
    """
    We want to keep 'temperature' near 37. Each step:
      - Measure error (target - current)
      - Adjust heater by a small amount proportional to error
      - Environment also nudges temperature up/down a bit
    """
    print("\nFEEDBACK LOOP (THERMOSTAT)")
    print("===========================")
    temp = 30.0
    heater = 0.0
    alpha = 0.2  # how strongly the heater reacts to error
    env_wobble = 0.6

    temps = [temp]
    heaters = [heater]

    for t in range(steps):
        error = setpoint - temp
        heater += alpha * error
        # system dynamics: temp moves a bit toward heater + random environment
        env = (random.random() - 0.5) * env_wobble
        temp += 0.1 * heater + env
        temps.append(temp); heaters.append(heater)

    print(f"Final temperature≈{temps[-1]:.2f} (target {setpoint})")
    maybe_plot(list(range(len(temps))), [temps], ["temperature"], "Thermostat Feedback", "time", "temp")
    maybe_plot(list(range(len(heaters))), [heaters], ["heater"], "Heater Signal", "time", "heater")


# 5) Randomness with dice (why simulations vary)
def demo_dice(trials: int = 20) -> None:
    """
    We roll dice to show randomness. Imagine each roll is a molecule bumping into another.
    """
    print("\nRANDOMNESS WITH DICE")
    print("====================")
    rng = random.Random(5)
    rolls = [rng.randint(1, 6) for _ in range(trials)]
    avg = sum(rolls) / len(rolls)
    print("Rolls:", rolls)
    print(f"Average≈{avg:.2f} (should be near 3.5 with many trials)")
    if plt is not None:
        xs = list(range(1, 7))
        counts = [rolls.count(x) for x in xs]
        maybe_plot(xs, [counts], ["count"], "Dice Rolls", "face", "count")


def run_all() -> None:
    demo_cell_state_machine()
    demo_growth()
    demo_virus()
    demo_feedback()
    demo_dice()


def main(argv=None):
    p = argparse.ArgumentParser(description="Bio Beginner Playground — single-file teaching script")
    p.add_argument("--demo", default=None, help="cell|growth|virus|feedback|dice|all")
    args = p.parse_args(argv)

    if args.demo in (None, ""):
        options = ["cell","growth","virus","feedback","dice","all","quit"]
        while True:
            print("\nChoose a demo:", ", ".join(options))
            choice = input("> ").strip().lower()
            if choice in ("quit","q","exit"):
                break
            try:
                if choice == "cell": demo_cell_state_machine()
                elif choice == "growth": demo_growth()
                elif choice == "virus": demo_virus()
                elif choice == "feedback": demo_feedback()
                elif choice == "dice": demo_dice()
                elif choice == "all": run_all()
                else: print("Unknown choice")
            except Exception as e:
                print("[error]", type(e).__name__, e)
    else:
        mapping = {
            "cell": demo_cell_state_machine,
            "growth": demo_growth,
            "virus": demo_virus,
            "feedback": demo_feedback,
            "dice": demo_dice,
            "all": run_all,
        }
        fn = mapping.get(args.demo.lower())
        if fn is None:
            raise SystemExit("Unknown demo. Use --demo cell|growth|virus|feedback|dice|all")
        fn()


if __name__ == "__main__":
    main()
