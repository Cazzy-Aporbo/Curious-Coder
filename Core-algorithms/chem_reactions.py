#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chemistry Reactions in Python — Stoichiometry, Kinetics, and Equilibrium

Run
  python chem_reactions.py                         # interactive menu
  python chem_reactions.py --balance "H2 + O2 -> H2O"
  python chem_reactions.py --kinetics "2 H2 + O2 -> 2 H2O" --k 0.1 --t 40
  python chem_reactions.py --equilibrium "A + B <-> C" --Ka 50 --A0 1 --B0 2 --C0 0

Features
- Equation balancer (integer coefficients; understands parentheses and hydrates like CuSO4·5H2O)
- Mass‑action kinetics simulator using RK4; optional Matplotlib plot if installed
- Equilibrium calculator for a single reversible reaction with given equilibrium constant

No external libraries required. If matplotlib is installed, plots will be shown.
"""

from __future__ import annotations

import argparse
import math
import re
from fractions import Fraction
from typing import Dict, List, Tuple

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


# ----------------------------- Formula parsing -----------------------------
# Supports parentheses (...) and hydrate dot "·".
TOKEN = re.compile(r"([A-Z][a-z]?|\(|\)|·|\d+)")

def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a chemical formula into a dict of element -> count.
    Examples: H2O, Ca(OH)2, CuSO4·5H2O
    """
    def parse_tokens(tokens: List[str], i: int = 0) -> Tuple[Dict[str, int], int]:
        counts: Dict[str, int] = {}
        while i < len(tokens):
            tok = tokens[i]
            if tok == ")":
                break
            elif tok == "(":
                inner, j = parse_tokens(tokens, i + 1)
                i = j
                # parse multiplier after closing )
                mult = 1
                if i < len(tokens) and tokens[i].isdigit():
                    mult = int(tokens[i]); i += 1
                for k, v in inner.items():
                    counts[k] = counts.get(k, 0) + v * mult
                continue
            elif tok == "·":  # hydrate separator
                # split: left counts already captured; parse right side and merge
                right, j = parse_tokens(tokens, i + 1)
                i = j
                for k, v in right.items():
                    counts[k] = counts.get(k, 0) + v
                continue
            elif tok.isdigit():
                # a number should only follow element or ')', handled above
                raise ValueError("Unexpected number position in formula.")
            else:
                # element symbol
                elem = tok
                i += 1
                mult = 1
                if i < len(tokens) and tokens[i].isdigit():
                    mult = int(tokens[i]); i += 1
                counts[elem] = counts.get(elem, 0) + mult
                continue
            i += 1
        # consume closing ')'
        if i < len(tokens) and tokens[i] == ")":
            i += 1
        return counts, i

    # Preprocess: expand hydrates like "CuSO4·5H2O" into left + 5*(H2O) by faking parentheses
    # Our parser already handles '·' by parsing the right side, but we need to respect leading coefficient after dot.
    # We'll tokenize and during parsing when we see digits right after '·' we effectively multiply the subsequent group.
    # To keep parser simple, wrap number+formula after dot as "(formula)*number" by duplicating tokens.
    # Instead, we handle it inline inside parse_tokens above by merging counts; we need to expand multiplier.
    # We'll do a lightweight pre-pass to convert '·5H2O' into '·(H2O)5'
    formula = re.sub(r"·\s*(\d+)\s*([A-Z][a-z]?|\()", lambda m: "·(" + (m.group(2)) + ")" + m.group(1), formula)

    toks = TOKEN.findall(formula.replace(" ", ""))
    counts, idx = parse_tokens(toks, 0)
    if idx != len(toks):
        # there may be trailing multiplier; parse once more
        pass
    return counts


# ----------------------------- Equation parsing ----------------------------
ARROW = re.compile(r"(->|→|<-+>|<->|⇌|⥨|⥦)")

def parse_equation(eq: str) -> Tuple[List[Tuple[str,int]], List[Tuple[str,int]]]:
    """
    Parse 'aA + bB -> cC + dD' into lists of (species, coefficient_hint).
    Coefficients hints (optional) are used if provided; otherwise default 1.
    """
    if ARROW.search(eq) is None:
        raise ValueError("Equation must contain an arrow like '->' or '<->'")
    left, right = ARROW.split(eq)[0], ARROW.split(eq)[-1]
    def side(s: str) -> List[Tuple[str,int]]:
        terms = []
        for part in s.split("+"):
            part = part.strip()
            m = re.match(r"^(\d+)\s+(.+)$", part)
            if m:
                terms.append((m.group(2).strip(), int(m.group(1))))
            else:
                # also support coefficient stuck to formula, e.g., 2H2O
                m2 = re.match(r"^(\d+)([A-Za-z(].*)$", part)
                if m2:
                    terms.append((m2.group(2).strip(), int(m2.group(1))))
                else:
                    terms.append((part, 1))
        return terms
    return side(left), side(right)


# ----------------------------- Linear algebra (exact) ----------------------------
def build_matrix(lhs: List[Tuple[str,int]], rhs: List[Tuple[str,int]]) -> Tuple[List[str], List[List[Fraction]]]:
    """
    Build element balance matrix A such that A * x = 0, where x are integer coeffs for species.
    """
    species = [s for s,_ in lhs] + [s for s,_ in rhs]
    # Unique elements
    elems = {}
    for sp in species:
        for e,c in parse_formula(sp).items():
            elems[e] = True
    elems = sorted(elems.keys())
    A: List[List[Fraction]] = [[Fraction(0) for _ in species] for _ in elems]
    for j, sp in enumerate(species):
        counts = parse_formula(sp)
        for i, e in enumerate(elems):
            if j < len(lhs):
                A[i][j] = Fraction(counts.get(e,0))
            else:
                A[i][j] = Fraction(-counts.get(e,0))  # move RHS to LHS
    return species, A

def nullspace_integer(A: List[List[Fraction]]) -> List[int]:
    """
    Compute an integer basis vector for the nullspace of A (A x = 0) using Fraction arithmetic (Gauss-Jordan).
    Returns the smallest positive integer solution.
    """
    m = len(A); n = len(A[0]) if m else 0
    # Augment with zeros (homogeneous), we just row-reduce A
    M = [row[:] for row in A]
    row = 0
    pivots = []
    for col in range(n):
        # find non-zero pivot
        pivot = None
        for r in range(row, m):
            if M[r][col] != 0:
                pivot = r; break
        if pivot is None:
            continue
        # swap
        M[row], M[pivot] = M[pivot], M[row]
        # normalize
        piv = M[row][col]
        M[row] = [v / piv for v in M[row]]
        # eliminate others
        for r in range(m):
            if r != row and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [M[r][c] - factor * M[row][c] for c in range(n)]
        pivots.append((row, col))
        row += 1
        if row == m:
            break
    # free variables = columns without pivots -> set one free var = 1 and backsolve
    pivot_cols = {c for _,c in pivots}
    free_cols = [c for c in range(n) if c not in pivot_cols]
    if not free_cols:
        # trivial or fully determined; set the last variable as free
        free_cols = [n-1]
    x = [Fraction(0) for _ in range(n)]
    free = free_cols[0]
    x[free] = Fraction(1)
    # backsolve
    for r, c in reversed(pivots):
        s = sum(M[r][j]*x[j] for j in range(n))
        x[c] = -s
    # scale to integers
    dens = [v.denominator for v in x]
    L = 1
    for d in dens:
        L = L * d // math.gcd(L, d)
    xi = [int(v * L) for v in x]
    # remove common gcd
    g = 0
    for v in xi:
        g = math.gcd(g, abs(v))
    if g > 1:
        xi = [v // g for v in xi]
    # make all positive (flip sign if needed)
    if any(v < 0 for v in xi):
        xi = [-v for v in xi]
    return xi

def balance_equation(eq: str) -> str:
    lhs, rhs = parse_equation(eq)
    species, A = build_matrix(lhs, rhs)
    coeffs = nullspace_integer(A)
    # apply to species; maintain left/right split
    L = len(lhs)
    left = " + ".join(f"{coeffs[i]} {species[i]}" for i in range(L))
    right = " + ".join(f"{coeffs[L+i]} {species[L+i]}" for i in range(len(rhs)))
    # fold any coefficient 1 to '1 X' for clarity (keep explicit for beginners)
    return f"{left} -> {right}"


# ----------------------------- Kinetics (mass‑action) ----------------------------
def kinetics_sim(eq: str, k: float = 0.1, t_end: float = 50.0, dt: float = 0.1, init: Dict[str,float] | None = None) -> Tuple[List[float], Dict[str,List[float]]]:
    """
    Simulate irreversible reaction under mass-action: aA + bB -> ... with rate v = k * [A]^a * [B]^b ...
    Uses RK4 integrator with fixed dt.
    init: optional initial concentrations dict; if not provided, start reactants at 1.0 and products at 0.0
    """
    lhs, rhs = parse_equation(eq)
    # If unbalanced, auto-balance just for stoichiometry tracking
    try:
        species, A = build_matrix(lhs, rhs)
        coeffs = nullspace_integer(A)
    except Exception:
        # fallback to coefficient hints
        species = [s for s,_ in lhs] + [s for s,_ in rhs]
        coeffs = [c for _,c in lhs] + [c for _,c in rhs]

    L = len(lhs)
    stoich_lhs = {species[i]: coeffs[i] for i in range(L)}
    stoich_rhs = {species[L+i]: coeffs[L+i] for i in range(len(rhs))}
    all_species = [*stoich_lhs.keys(), *stoich_rhs.keys()]

    # initial concentrations
    conc = {s: (1.0 if s in stoich_lhs else 0.0) for s in all_species}
    if init:
        conc.update(init)

    def rate(c: Dict[str,float]) -> float:
        # v = k * Π [S_i]^{nu_i} over LHS only
        v = k
        for s, nu in stoich_lhs.items():
            v *= max(c[s], 0.0) ** nu
        return v

    def deriv(c: Dict[str,float]) -> Dict[str,float]:
        v = rate(c)
        dc = {s: 0.0 for s in all_species}
        for s, nu in stoich_lhs.items():
            dc[s] -= nu * v
        for s, nu in stoich_rhs.items():
            dc[s] += nu * v
        return dc

    times = [0.0]
    traces = {s: [conc[s]] for s in all_species}

    steps = int(t_end / dt)
    for _ in range(steps):
        # RK4 on dict state
        def add(c, k1, a=1.0):
            return {s: c[s] + a*dt*k1[s] for s in all_species}
        k1 = deriv(conc)
        k2 = deriv(add(conc, k1, 0.5))
        k3 = deriv(add(conc, k2, 0.5))
        k4 = deriv(add(conc, k3, 1.0))
        for s in all_species:
            conc[s] += (dt/6.0)*(k1[s] + 2*k2[s] + 2*k3[s] + k4[s])
            conc[s] = max(conc[s], 0.0)
        t = times[-1] + dt
        times.append(t)
        for s in all_species:
            traces[s].append(conc[s])
    return times, traces


# ----------------------------- Equilibrium (single reaction) ----------------------------
def equilibrium_single(eq: str, Ka: float, A0: float, B0: float, C0: float = 0.0) -> Dict[str,float]:
    """
    Solve equilibrium for A + B <-> C with association constant Ka = [C]/([A][B]) at equilibrium.
    Supports general stoichiometry aA + bB <-> cC (but solved under single-product assumption).

    Returns equilibrium concentrations via ICE table logic.
    """
    lhs, rhs = parse_equation(eq)
    if len(rhs) != 1:
        raise ValueError("This simple solver supports a single product on the right side.")
    # Pull stoichiometric coefficients (auto-balance if needed)
    species, A = build_matrix(lhs, rhs)
    coeffs = nullspace_integer(A)
    L = len(lhs)
    a = sum(coeffs[i] for i in range(L))  # total reactant order (for simple K expression)
    c = coeffs[L]  # product stoich (assumes single product)

    # For beginners, we handle only A + B <-> C cases robustly
    if L != 2 or c <= 0:
        raise ValueError("Use a two-reactant to one-product reaction like 'A + B <-> C'.")

    # Names
    A_name, B_name = species[0], species[1]
    C_name = species[L]

    # Let x be the extent that goes to products: Aeq = A0 - x, Beq = B0 - x, Ceq = C0 + x
    # Equilibrium: Ka = Ceq / (Aeq * Beq)
    # Solve Ka = (C0 + x)/((A0 - x)(B0 - x))
    # Rearranged: Ka(A0 - x)(B0 - x) - (C0 + x) = 0 -> quadratic in x
    A0f, B0f, C0f = A0, B0, C0
    # Expand: Ka[(A0B0 - A0 x - B0 x + x^2)] - C0 - x = 0
    # => Ka x^2 + (-Ka A0 - Ka B0 - 1) x + (Ka A0 B0 - C0) = 0
    qa = Ka
    qb = -(Ka*A0f + Ka*B0f + 1.0)
    qc = (Ka*A0f*B0f - C0f)
    disc = qb*qb - 4*qa*qc
    if disc < 0:
        raise ValueError("No real equilibrium solution for given parameters.")
    x1 = (-qb + math.sqrt(disc)) / (2*qa)
    x2 = (-qb - math.sqrt(disc)) / (2*qa)
    # Choose physically valid root (within feasible range)
    candidates = [x for x in (x1, x2) if 0 <= x <= min(A0f, B0f)]
    if not candidates:
        raise ValueError("No physically valid equilibrium extent (check Ka and initial amounts).")
    x = max(candidates)  # typically the larger root is valid for association
    return {
        A_name: A0f - x,
        B_name: B0f - x,
        C_name: C0f + x,
    }


# ----------------------------- CLI & Interactive ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Chemistry reactions: balance, kinetics, and equilibrium")
    ap.add_argument("--balance", type=str, help='Equation to balance, e.g. "C3H8 + O2 -> CO2 + H2O"')
    ap.add_argument("--kinetics", type=str, help='Irreversible equation for kinetics, e.g. "2 H2 + O2 -> 2 H2O"')
    ap.add_argument("--k", type=float, default=0.1, help="Rate constant for kinetics (default 0.1)")
    ap.add_argument("--t", type=float, default=50.0, help="Simulation time for kinetics (default 50)")
    ap.add_argument("--dt", type=float, default=0.1, help="Time step for kinetics (default 0.1)")
    ap.add_argument("--init", type=str, default=None, help='Initial concentrations as CSV "H2=2,O2=1,H2O=0"')
    ap.add_argument("--equilibrium", type=str, help='Reversible equation like "A + B <-> C"')
    ap.add_argument("--Ka", type=float, help="Equilibrium constant (association)")
    ap.add_argument("--A0", type=float, help="Initial A")
    ap.add_argument("--B0", type=float, help="Initial B")
    ap.add_argument("--C0", type=float, default=0.0, help="Initial C (default 0)")
    ap.add_argument("--plot", action="store_true", help="Show plots (requires matplotlib)")
    args = ap.parse_args()

    if args.balance:
        print("Balanced equation:")
        print(balance_equation(args.balance))
        return

    if args.kinetics:
        init = None
        if args.init:
            init = {}
            for pair in args.init.split(","):
                k, v = pair.split("="); init[k.strip()] = float(v)
        times, traces = kinetics_sim(args.kinetics, k=args.k, t_end=args.t, dt=args.dt, init=init)
        print("Final concentrations:")
        for s in traces:
            print(f"  {s}: {traces[s][-1]:.4f}")
        if args.plot and plt is not None:
            ys = [traces[s] for s in traces]
            maybe_plot(times, ys, list(traces.keys()), "Kinetics Simulation", "time", "concentration")
        return

    if args.equilibrium:
        if args.Ka is None or args.A0 is None or args.B0 is None:
            raise SystemExit("Provide --Ka, --A0, --B0 (and optional --C0) for equilibrium.")
        eq = equilibrium_single(args.equilibrium, Ka=args.Ka, A0=args.A0, B0=args.B0, C0=args.C0)
        print("Equilibrium concentrations:")
        for k,v in eq.items():
            print(f"  {k}: {v:.4f}")
        return

    # Interactive menu
    while True:
        print("\nChoose an option: balance | kinetics | equilibrium | quit")
        choice = input("> ").strip().lower()
        if choice in ("quit", "q", "exit"):
            break
        try:
            if choice == "balance":
                eq = input("Enter equation: ").strip()
                print("Balanced:", balance_equation(eq))
            elif choice == "kinetics":
                eq = input("Equation (irreversible): ").strip()
                k = float(input("k (rate constant): ").strip() or "0.1")
                t = float(input("total time: ").strip() or "40")
                dt = float(input("dt: ").strip() or "0.1")
                times, traces = kinetics_sim(eq, k=k, t_end=t, dt=dt, init=None)
                print("Final concentrations:")
                for s in traces:
                    print(f"  {s}: {traces[s][-1]:.4f}")
                if plt is not None:
                    ys = [traces[s] for s in traces]
                    maybe_plot(times, ys, list(traces.keys()), "Kinetics Simulation", "time", "concentration")
            elif choice == "equilibrium":
                eq = input("Equation (A + B <-> C): ").strip()
                Ka = float(input("Ka (association constant): ").strip())
                A0 = float(input("A0: ").strip())
                B0 = float(input("B0: ").strip())
                C0 = float(input("C0 (0 if none): ").strip() or "0")
                out = equilibrium_single(eq, Ka=Ka, A0=A0, B0=B0, C0=C0)
                print("Equilibrium concentrations:")
                for k,v in out.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print("Unknown choice")
        except Exception as e:
            print("[error]", type(e).__name__, e)


if __name__ == "__main__":
    main()
