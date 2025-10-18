#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biological Systems Thinking with Python — Advanced, Hands-on Guide

Run
  python bio_systems_thinking.py                 # interactive menu
  python bio_systems_thinking.py --demo all      # run everything non-interactively
  python bio_systems_thinking.py --demo ssa      # run a specific demo

What you will practice
- Translate a biological story into state variables, parameters, and processes
- Deterministic dynamics (ODE) vs stochastic dynamics (Gillespie SSA)
- Feedback, nonlinearity, and bifurcation intuition
- Parameter sensitivity and simple fitting
- Evolutionary and ecological reasoning with minimal math

Dependencies
- Python 3.9+
- Numpy is strongly recommended
- Scipy and Matplotlib are optional; the script will fall back to pure-Python RK4 and text outputs
"""

from __future__ import annotations

import argparse
import math
import random
import textwrap
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

# Optional SciPy for ODEs
try:
    from scipy.integrate import solve_ivp  # type: ignore
except Exception:  # pragma: no cover
    solve_ivp = None  # type: ignore

# Optional Matplotlib for plots
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


def banner(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def explain(txt: str) -> None:
    print(textwrap.dedent(txt).strip() + "\n")


def have_num() -> bool:
    return np is not None


def rk4(f: Callable[[float, List[float], Dict[str, float]], List[float]], y0: List[float], t: List[float], pars: Dict[str, float]) -> Tuple[List[float], List[List[float]]]:
    """Minimal RK4 if SciPy is unavailable. f(t, y, pars) -> dy/dt"""
    y = np.array(y0, dtype=float) if have_num() else list(map(float, y0))
    traj = [y.copy() if have_num() else y0[:]]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if have_num():
            k1 = np.array(f(t[i], y, pars))
            k2 = np.array(f(t[i] + dt / 2, y + dt * k1 / 2, pars))
            k3 = np.array(f(t[i] + dt / 2, y + dt * k2 / 2, pars))
            k4 = np.array(f(t[i] + dt, y + dt * k3, pars))
            y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(y.copy())
        else:
            k1 = f(t[i], y, pars)
            k2 = f(t[i] + dt / 2, [y[j] + dt * k1[j] / 2 for j in range(len(y))], pars)
            k3 = f(t[i] + dt / 2, [y[j] + dt * k2[j] / 2 for j in range(len(y))], pars)
            k4 = f(t[i] + dt, [y[j] + dt * k3[j] for j in range(len(y))], pars)
            y = [y[j] + (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) for j in range(len(y))]
            traj.append(y[:])
    return t, traj


def maybe_plot(x, ys, labels, title, xlabel, ylabel) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    for y, lab in zip(ys, labels):
        ax.plot(x, y, label=lab)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def logistic_growth_demo() -> None:
    banner("Logistic growth and carrying capacity")
    explain("""
    Stocks and flows: one population N, with per-capita growth r and crowding 1 - N/K.
    dN/dt = r N (1 - N/K). Thinking move: identify feedback sign: positive when N small (growth), negative near K.
    """)
    r, K, N0 = 0.8, 1000.0, 10.0
    t_eval = list(np.linspace(0, 10, 200)) if have_num() else [i * 0.05 for i in range(200)]

    def f(t, y, p):
        N = y[0]
        return [p["r"] * N * (1.0 - N / p["K"])]

    if solve_ivp is not None and have_num():
        sol = solve_ivp(lambda t, y: f(t, y, {"r": r, "K": K}), [t_eval[0], t_eval[-1]], [N0], t_eval=t_eval, rtol=1e-6, atol=1e-9)
        N = sol.y[0]
    else:
        _, traj = rk4(f, [N0], t_eval, {"r": r, "K": K})
        N = [row[0] for row in traj]
    print(f"Final size ≈ {N[-1]:.1f} (should approach K={K})")
    maybe_plot(t_eval, [N], ["N"], "Logistic Growth", "time", "population")

    explain("Sensitivity thought experiment: double r versus halve K. Stronger r gets you to K faster; smaller K lowers the ceiling.")

def predator_prey_demo() -> None:
    banner("Lotka–Volterra predator–prey and orbits")
    explain("""
    Two stocks: prey X and predators Y.
    dX/dt = aX - bXY
    dY/dt = cXY - dY
    Nonlinear interaction bXY and cXY couples the species; orbits emerge instead of fixed points.
    """)
    pars = {"a": 1.0, "b": 0.1, "c": 0.075, "d": 1.5}
    X0, Y0 = 10.0, 5.0
    t_eval = list(np.linspace(0, 60, 800)) if have_num() else [i * 0.075 for i in range(800)]

    def f(t, y, p):
        X, Y = y
        return [p["a"] * X - p["b"] * X * Y, p["c"] * X * Y - p["d"] * Y]

    if solve_ivp is not None and have_num():
        sol = solve_ivp(lambda t, y: f(t, y, pars), [t_eval[0], t_eval[-1]], [X0, Y0], t_eval=t_eval, rtol=1e-6, atol=1e-9)
        X, Y = sol.y
    else:
        _, traj = rk4(f, [X0, Y0], t_eval, pars)
        X, Y = [row[0] for row in traj], [row[1] for row in traj]
    print(f"Extrema examples X_max≈{max(X):.1f}, Y_max≈{max(Y):.1f}")
    maybe_plot(t_eval, [X, Y], ["prey X", "predator Y"], "Predator–Prey", "time", "abundance")

def enzyme_kinetics_demo() -> None:
    banner("Michaelis–Menten vs Hill kinetics as input–output maps")
    explain("""
    Many biological subsystems behave like saturating transducers. For substrate S -> rate v:
    Michaelis–Menten: v = Vmax S / (Km + S)
    Hill (cooperativity n): v = Vmax S^n / (K^n + S^n)
    Thinking move: treat components as transfer functions and study gain, saturation, and thresholds.
    """)
    if not have_num():
        print("NumPy not available; printing a small table instead.")
        def mm(S, Vmax=1.0, Km=1.0): return Vmax * S / (Km + S)
        for S in [0.0, 0.5, 1.0, 2.0, 5.0]:
            print(f"S={S:>4}: v={mm(S):.3f}")
        return

    S = np.logspace(-2, 2, 100)
    Vmax, Km, K, n = 1.0, 1.0, 1.0, 3.0
    v_mm = Vmax * S / (Km + S)
    v_hill = Vmax * (S**n) / (K**n + S**n)
    maybe_plot(S, [v_mm, v_hill], ["Michaelis–Menten", "Hill n=3"], "Enzyme-like Transfer", "substrate S", "rate v")
    print("Half-max at S ≈ Km for MM; Hill adds a threshold-like steepness around K.")

def ssa_birth_death_demo() -> None:
    banner("Stochastic gene expression with Gillespie SSA (birth–death)")
    explain("""
    When counts are low, molecule numbers jump discretely and randomness matters.
    Reactions:
      ∅ -> X at rate k_prod
      X -> ∅ at rate k_deg * X
    Gillespie SSA samples next reaction time and which reaction fires from propensities.
    """)
    k_prod, k_deg = 5.0, 0.4
    T, X0 = 20.0, 0
    rng = random.Random(7)
    t, x = 0.0, X0
    ts, xs = [t], [x]
    while t < T:
        a1 = k_prod
        a2 = k_deg * x
        a0 = a1 + a2
        if a0 <= 0:
            break
        r1, r2 = rng.random(), rng.random()
        dt = -math.log(r1) / a0
        if r2 < a1 / a0:
            x = x + 1
        else:
            x = max(0, x - 1)
        t = t + dt
        ts.append(t); xs.append(x)
    print(f"Final copy number after T={T} is {xs[-1]} with {len(xs)} jumps")
    maybe_plot(ts, [xs], ["X"], "Gillespie SSA Birth–Death", "time", "molecule count")

def toggle_switch_demo() -> None:
    banner("Bistability in a genetic toggle switch")
    explain("""
    Mutual repression can create two stable states (bistability), a memory element.
    dA/dt = alpha/(1 + B^n) - delta*A
    dB/dt = alpha/(1 + A^n) - delta*B
    Thinking move: feedback and nonlinearity shape landscape with multiple attractors.
    """)
    pars = {"alpha": 5.0, "delta": 1.0, "n": 3.0}
    A0, B0 = 0.1, 4.0
    t_eval = list(np.linspace(0, 20, 600)) if have_num() else [i * 0.033 for i in range(600)]

    def f(t, y, p):
        A, B = y
        n = p["n"]
        dA = p["alpha"] / (1.0 + (B**n)) - p["delta"] * A
        dB = p["alpha"] / (1.0 + (A**n)) - p["delta"] * B
        return [dA, dB]

    if solve_ivp is not None and have_num():
        sol = solve_ivp(lambda t, y: f(t, y, pars), [t_eval[0], t_eval[-1]], [A0, B0], t_eval=t_eval, rtol=1e-6, atol=1e-9)
        A, B = sol.y
    else:
        _, traj = rk4(f, [A0, B0], t_eval, pars)
        A, B = [row[0] for row in traj], [row[1] for row in traj]
    print(f"Reached state A≈{A[-1]:.2f}, B≈{B[-1]:.2f}")
    maybe_plot(t_eval, [A, B], ["A", "B"], "Genetic Toggle Switch", "time", "expression")

def replicator_demo() -> None:
    banner("Replicator dynamics for evolutionary strategies")
    explain("""
    Fractions of strategies x_i change according to payoff advantage over the mean.
    dx_i/dt = x_i ( (A x)_i - x^T A x )
    Thinking move: evolution as hill-climbing on payoff landscape with simplex constraints.
    """)
    if not have_num():
        print("NumPy not available; skipping replicator dynamics.")
        return
    A = np.array([[0, 3, -1],
                  [-1, 0, 3],
                  [3, -1, 0]], dtype=float)
    x = np.array([0.34, 0.33, 0.33], dtype=float)
    dt = 0.05
    xs = [x.copy()]
    for _ in range(200):
        Ax = A @ x
        phi = float(x @ Ax)
        dx = x * (Ax - phi)
        x = x + dt * dx
        x = np.clip(x, 1e-9, 1.0); x = x / x.sum()
        xs.append(x.copy())
    xs = np.array(xs)
    maybe_plot(list(range(xs.shape[0])), [xs[:,0], xs[:,1], xs[:,2]], ["x1", "x2", "x3"], "Replicator Dynamics", "step", "fraction")
    print("Final mix:", np.round(xs[-1], 3).tolist())

def diffusion_1d_demo() -> None:
    banner("Reaction–diffusion intuition in 1D")
    explain("""
    Spatial coupling spreads concentration by diffusion while reactions add sources/sinks.
    u_t = D u_xx - k u + s(x)
    Thinking move: local rules plus spatial coupling create global patterns.
    """)
    if not have_num():
        print("NumPy not available; skipping diffusion demo.")
        return
    D, k = 0.6, 0.1
    L, nx = 50.0, 200
    dx = L / nx
    dt = 0.2 * dx * dx / D
    steps = 400
    u = np.zeros(nx, dtype=float)
    source = np.exp(-((np.linspace(0, L, nx) - L/2.0) ** 2) / 20.0)
    for _ in range(steps):
        lap = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx * dx)
        u = u + dt * (D * lap - k * u + source * 0.1)
    maybe_plot(np.linspace(0, L, nx), [u], ["u(x)"], "1D Reaction–Diffusion", "position", "concentration")
    print(f"Max concentration ≈ {float(u.max()):.3f}")

def fit_logistic_demo() -> None:
    banner("Parameter fitting on logistic data with noise")
    explain("""
    Data rarely follows equations perfectly. We simulate noisy observations and fit r, K by least squares.
    Thinking move: tie parameters to observables and quantify uncertainty with simple resampling.
    """)
    if not have_num():
        print("NumPy not available; skipping fit demo.")
        return

    true_r, true_K = 0.7, 800.0
    t = np.linspace(0, 8, 40)
    N0 = 12.0
    N_true = true_K / (1 + ((true_K - N0) / N0) * np.exp(-true_r * t))
    rng = np.random.default_rng(3)
    y = N_true + rng.normal(0, 15.0, size=t.size)

    def model(t, r, K):
        return K / (1 + ((K - N0) / N0) * np.exp(-r * t))

    r, K = 0.5, 900.0
    for _ in range(2000):
        # simple gradient-free tweak
        r_try = r + rng.normal(0, 0.01)
        K_try = max(100.0, K + rng.normal(0, 3.0))
        err_old = float(np.mean((y - model(t, r, K)) ** 2))
        err_new = float(np.mean((y - model(t, r_try, K_try)) ** 2))
        if err_new < err_old:
            r, K = r_try, K_try
    print(f"Recovered r≈{r:.3f}, K≈{K:.1f} vs truth r={true_r}, K={true_K}")
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7,4), dpi=130)
        ax.scatter(t, y, s=12, label="data")
        ax.plot(t, model(t, r, K), label="fit")
        ax.plot(t, N_true, label="truth", linestyle="--")
        ax.set_title("Logistic fit via simple search")
        ax.set_xlabel("time")
        ax.set_ylabel("population")
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()

def run_all() -> None:
    logistic_growth_demo()
    predator_prey_demo()
    enzyme_kinetics_demo()
    ssa_birth_death_demo()
    toggle_switch_demo()
    replicator_demo()
    diffusion_1d_demo()
    fit_logistic_demo()

def main():
    p = argparse.ArgumentParser(description="Biological Systems Thinking with Python")
    p.add_argument("--demo", default=None, help="choose: logistic, predator, enzyme, ssa, toggle, replicator, diffusion, fit, all")
    args = p.parse_args()

    if args.demo in (None, ""):
        options = ["logistic","predator","enzyme","ssa","toggle","replicator","diffusion","fit","all","quit"]
        while True:
            print("\nChoose a demo:", ", ".join(options))
            choice = input("> ").strip().lower()
            if choice in ("quit","q","exit"):
                break
            try:
                if choice == "logistic": logistic_growth_demo()
                elif choice == "predator": predator_prey_demo()
                elif choice == "enzyme": enzyme_kinetics_demo()
                elif choice == "ssa": ssa_birth_death_demo()
                elif choice == "toggle": toggle_switch_demo()
                elif choice == "replicator": replicator_demo()
                elif choice == "diffusion": diffusion_1d_demo()
                elif choice == "fit": fit_logistic_demo()
                elif choice == "all": run_all()
                else:
                    print("Unknown choice")
            except Exception as e:
                print("[error]", type(e).__name__, e)
    else:
        mapping = {
            "logistic": logistic_growth_demo,
            "predator": predator_prey_demo,
            "enzyme": enzyme_kinetics_demo,
            "ssa": ssa_birth_death_demo,
            "toggle": toggle_switch_demo,
            "replicator": replicator_demo,
            "diffusion": diffusion_1d_demo,
            "fit": fit_logistic_demo,
            "all": run_all,
        }
        fn = mapping.get(args.demo.lower())
        if fn is None:
            raise SystemExit("Unknown demo. Use --demo logistic|predator|enzyme|ssa|toggle|replicator|diffusion|fit|all")
        fn()

if __name__ == "__main__":
    main()
