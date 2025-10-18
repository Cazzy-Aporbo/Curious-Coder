#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biological Systems Lab 2 — Agents, Noise, Inference, and Control in Python

Run
  python bio_systems_lab2.py
  python bio_systems_lab2.py --demo all
  python bio_systems_lab2.py --demo abm

Modules and ideas
1) Agent-based epidemic on a contact network (SIR on graph)
2) Spatial evolutionary game on a 2D grid (local imitation dynamics)
3) Gene expression noise via SDE (Ornstein–Uhlenbeck) with Euler–Maruyama
4) Bayesian parameter inference (Metropolis) for logistic growth data
5) Homeostasis with PID control (temperature-like regulation)

Dependencies
- NumPy (optional but recommended)
- Matplotlib (optional; prints tables if missing)
"""

from __future__ import annotations

import argparse
import math
import random
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


def have_num() -> bool:
    return np is not None


def explain(msg: str) -> None:
    print(textwrap.dedent(msg).strip() + "\n")


def maybe_plot(x, ys, labels, title, xlabel, ylabel) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7,4), dpi=130)
    for y, lab in zip(ys, labels):
        ax.plot(x, y, label=lab, lw=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def abm_epidemic_demo(n: int = 200, p_edge: float = 0.03, beta: float = 0.12, gamma: float = 0.04, steps: int = 200) -> None:
    print("\nAgent-based epidemic on a random contact network")
    print("=" * 49)
    explain("""
    We model SIR at the individual level on a random graph.
    Each time step, each infected agent transmits to susceptible neighbors with probability beta and recovers with probability gamma.
    Thinking move: zoom into agents and contacts to see stochastic waves and the role of degree.
    """)
    rng = random.Random(1)
    # Build Erdos–Renyi graph as adjacency list
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < p_edge:
                adj[i].append(j); adj[j].append(i)

    # States: 0=S, 1=I, 2=R
    state = [0]*n
    patient_zero = rng.randrange(n)
    state[patient_zero] = 1

    S, I, R = [n-1], [1], [0]
    for t in range(1, steps+1):
        new_state = state[:]
        for i in range(n):
            if state[i] == 1:
                # infect neighbors
                for nb in adj[i]:
                    if state[nb] == 0 and rng.random() < beta:
                        new_state[nb] = 1
                # recover
                if rng.random() < gamma:
                    new_state[i] = 2
        state = new_state
        S.append(state.count(0)); I.append(state.count(1)); R.append(state.count(2))
    if plt is None:
        print(f"Peak infected: {max(I)} at t={I.index(max(I))}, final recovered={R[-1]}")
    else:
        maybe_plot(list(range(len(S))), [S,I,R], ["S","I","R"], "ABM Epidemic (SIR)", "time", "count")


def spatial_evolution_demo(L: int = 40, steps: int = 150, w: float = 0.9) -> None:
    print("\nSpatial evolutionary game on a grid")
    print("=" * 39)
    explain("""
    Spatial structure matters. On a torus grid, each cell plays Prisoner's Dilemma with neighbors.
    Payoffs: R=1, T=1.4, S=0, P=0. Update: imitate the most successful neighbor with probability w, else explore.
    Thinking move: local imitation can stabilize cooperation pockets even when global incentives discourage it.
    """)
    if not have_num():
        print("NumPy not available; printing a small run with text output.")
        return
    R, T, S, P = 1.0, 1.4, 0.0, 0.0
    rng = np.random.default_rng(2)
    # 1 = Cooperator, 0 = Defector
    grid = rng.integers(0, 2, size=(L, L))
    def payoff(a, b):
        if a == 1 and b == 1: return R
        if a == 1 and b == 0: return S
        if a == 0 and b == 1: return T
        return P
    def neighborhood(i, j):
        # Moore neighborhood (8 neighbors) with wrap-around
        for di in (-1,0,1):
            for dj in (-1,0,1):
                if di == 0 and dj == 0: continue
                yield (i+di) % L, (j+dj) % L
    coop_frac = []
    for t in range(steps):
        pay = np.zeros_like(grid, dtype=float)
        for i in range(L):
            for j in range(L):
                s = 0.0
                for u,v in neighborhood(i,j):
                    s += payoff(grid[i,j], grid[u,v])
                pay[i,j] = s
        # imitation dynamics
        new_grid = grid.copy()
        for i in range(L):
            for j in range(L):
                # best neighbor
                best_val = pay[i,j]
                best_strat = grid[i,j]
                for u,v in neighborhood(i,j):
                    if pay[u,v] > best_val:
                        best_val = pay[u,v]; best_strat = grid[u,v]
                if rng.random() < w:
                    new_grid[i,j] = best_strat
                else:
                    new_grid[i,j] = 1 - grid[i,j]  # explore
        grid = new_grid
        coop_frac.append(grid.mean())
    if plt is None:
        print(f"Final cooperation fraction ≈ {coop_frac[-1]:.3f}")
    else:
        maybe_plot(list(range(len(coop_frac))), [coop_frac], ["cooperation"], "Spatial Evolution (PD)", "step", "fraction")


def sde_gene_noise_demo(T: float = 10.0, dt: float = 0.01, theta: float = 1.2, mu: float = 5.0, sigma: float = 1.0) -> None:
    print("\nGene expression noise as an SDE (Ornstein–Uhlenbeck)")
    print("=" * 47)
    explain("""
    Stochastic differential equation dX = theta*(mu - X) dt + sigma dW models mean-reverting expression with white noise.
    Euler–Maruyama discretization shows noisy trajectories and stationary variance sigma^2/(2*theta).
    Thinking move: use SDEs to reason about fluctuations around set points.
    """)
    n = int(T/dt)
    if have_num():
        rng = np.random.default_rng(0)
        X = np.zeros(n); X[0] = mu
        for t in range(1, n):
            dW = math.sqrt(dt) * rng.normal()
            X[t] = X[t-1] + theta*(mu - X[t-1])*dt + sigma*dW
        mean = float(X.mean()); var = float(X.var())
        target_var = sigma**2/(2*theta)
        print(f"Empirical mean≈{mean:.2f} var≈{var:.2f} vs stationary var≈{target_var:.2f}")
        if plt is not None:
            maybe_plot([i*dt for i in range(n)], [X], ["X(t)"], "OU Gene Noise", "time", "expression")
    else:
        rng = random.Random(0)
        X = mu
        mean_acc = 0.0
        for _ in range(n):
            dW = math.sqrt(dt) * (rng.random()*2-1)
            X = X + theta*(mu - X)*dt + sigma*dW
            mean_acc += X
        print(f"Approx mean≈{mean_acc/n:.2f} (no NumPy available)")


def bayes_fit_logistic_demo() -> None:
    print("\nBayesian inference for logistic growth via Metropolis")
    print("=" * 52)
    explain("""
    We simulate noisy logistic observations and infer posterior over r and K using a simple Metropolis sampler.
    Thinking move: uncertainty lives on parameters, not just on predictions; quantify it with samples.
    """)
    if not have_num():
        print("NumPy not available; skipping Bayesian demo.")
        return
    rng = np.random.default_rng(4)
    true_r, true_K = 0.6, 900.0
    N0 = 15.0
    t = np.linspace(0, 10, 30)
    def model(t, r, K): return K/(1+((K-N0)/N0)*np.exp(-r*t))
    y = model(t, true_r, true_K) + rng.normal(0, 20.0, size=t.size)

    def loglike(r, K, sigma=20.0):
        pred = model(t, r, K)
        return -0.5*np.sum(((y - pred)/sigma)**2) - t.size*np.log(sigma)

    # Priors: r ~ N(0.5, 0.3^2), K ~ N(800, 300^2), truncated to sensible ranges
    def logprior(r, K):
        if r <= 0 or K <= 100: return -np.inf
        lp_r = -0.5*((r-0.5)/0.3)**2
        lp_K = -0.5*((K-800.0)/300.0)**2
        return lp_r + lp_K

    def logpost(r, K):
        return logprior(r,K) + loglike(r,K)

    # Metropolis
    cur_r, cur_K = 0.5, 800.0
    cur_lp = logpost(cur_r, cur_K)
    samples = []
    acc = 0
    for i in range(6000):
        cand_r = abs(cur_r + rng.normal(0, 0.03))
        cand_K = max(100.0, cur_K + rng.normal(0, 8.0))
        cand_lp = logpost(cand_r, cand_K)
        if np.log(rng.random()) < cand_lp - cur_lp:
            cur_r, cur_K, cur_lp = cand_r, cand_K, cand_lp
            acc += 1
        if i > 1000:
            samples.append((cur_r, cur_K))
    samples = np.array(samples)
    r_mean, K_mean = float(samples[:,0].mean()), float(samples[:,1].mean())
    print(f"Posterior means r≈{r_mean:.3f}, K≈{K_mean:.1f}; acceptance={acc/6000:.2f}")
    if plt is not None:
        fig, ax = plt.subplots(1,2, figsize=(9,3.6), dpi=130)
        ax[0].plot(samples[:,0], lw=0.6); ax[0].set_title("r trace")
        ax[1].plot(samples[:,1], lw=0.6); ax[1].set_title("K trace")
        fig.tight_layout(); plt.show()


def pid_homeostasis_demo(setpoint: float = 37.0, steps: int = 200, dt: float = 0.1) -> None:
    print("\nHomeostasis with PID control")
    print("=" * 27)
    explain("""
    A controller senses error = setpoint - state and acts via P (proportional), I (integral), and D (derivative) terms.
    Thinking move: organisms maintain internal variables against disturbances through feedback control.
    """)
    Kp, Ki, Kd = 0.9, 0.2, 0.05
    env_noise = 0.2
    T = 30.0
    e_prev, i_term = 0.0, 0.0
    xs, us, ts = [], [], []
    x = 33.0
    for k in range(steps):
        t = k*dt
        e = setpoint - x
        i_term += e*dt
        d_term = (e - e_prev)/dt
        u = Kp*e + Ki*i_term + Kd*d_term
        # plant: x' = -a(x - env) + b u + disturbance
        env = 35.0 + (1.5*math.sin(0.2*t))
        a, b = 0.6, 0.8
        x = x + dt*( -a*(x - env) + b*u ) + env_noise*(random.random()-0.5)
        e_prev = e
        xs.append(x); us.append(u); ts.append(t)
    if plt is None:
        print(f"Final temperature≈{xs[-1]:.2f} target={setpoint}")
    else:
        maybe_plot(ts, [xs], ["state"], "PID Homeostasis", "time", "state")
        maybe_plot(ts, [us], ["control u"], "PID Control Signal", "time", "u")

def run_all() -> None:
    abm_epidemic_demo()
    spatial_evolution_demo()
    sde_gene_noise_demo()
    bayes_fit_logistic_demo()
    pid_homeostasis_demo()

def main():
    p = argparse.ArgumentParser(description="Biological Systems Lab 2 — Agents, Noise, Inference, and Control")
    p.add_argument("--demo", default=None, help="abm|spatial|sde|bayes|pid|all")
    args = p.parse_args()
    if args.demo in (None, ""):
        options = ["abm","spatial","sde","bayes","pid","all","quit"]
        while True:
            print("\nChoose a demo:", ", ".join(options))
            choice = input("> ").strip().lower()
            if choice in ("quit","q","exit"):
                break
            try:
                if choice == "abm": abm_epidemic_demo()
                elif choice == "spatial": spatial_evolution_demo()
                elif choice == "sde": sde_gene_noise_demo()
                elif choice == "bayes": bayes_fit_logistic_demo()
                elif choice == "pid": pid_homeostasis_demo()
                elif choice == "all": run_all()
                else:
                    print("Unknown choice")
            except Exception as e:
                print("[error]", type(e).__name__, e)
    else:
        mapping = {
            "abm": abm_epidemic_demo,
            "spatial": spatial_evolution_demo,
            "sde": sde_gene_noise_demo,
            "bayes": bayes_fit_logistic_demo,
            "pid": pid_homeostasis_demo,
            "all": run_all,
        }
        fn = mapping.get(args.demo.lower())
        if fn is None:
            raise SystemExit("Unknown demo. Use --demo abm|spatial|sde|bayes|pid|all")
        fn()

if __name__ == "__main__":
    main()
