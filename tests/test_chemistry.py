import math

import pytest


@pytest.mark.parametrize("formula, expected", [
    ("H2O", {"H": 2, "O": 1}),
    ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}),
    ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}),
    ("Al2(SO4)3·18H2O", {"Al": 2, "S": 3, "O": 30, "H": 36}),
])
def test_formula_counts(chemistry, formula, expected):
    assert chemistry.parse_formula(formula) == expected


@pytest.mark.parametrize("formula", ["", "H2O!", "Ca(OH", "H2O)", "()", "H0", "·H2O", "H2O·", "CuSO4·0H2O"])
def test_formula_rejects_malformed_input(chemistry, formula):
    with pytest.raises(ValueError):
        chemistry.parse_formula(formula)


@pytest.mark.parametrize("equation, expected", [
    ("H2 + O2 -> H2O", "2 H2 + 1 O2 -> 2 H2O"),
    ("C3H8 + O2 -> CO2 + H2O", "1 C3H8 + 5 O2 -> 3 CO2 + 4 H2O"),
    ("CuSO4·5H2O -> CuSO4 + H2O", "1 CuSO4·5H2O -> 1 CuSO4 + 5 H2O"),
])
def test_balance_conserves_atoms(chemistry, equation, expected):
    assert chemistry.balance_equation(equation) == expected


@pytest.mark.parametrize("equation", ["H2 -> O2", "H2 + O2 -> H2O + H2O2", "H2 -> H2O -> O2", "H2 + -> H2", "0 H2 -> H2"])
def test_balance_rejects_impossible_ambiguous_or_malformed_reactions(chemistry, equation):
    with pytest.raises(ValueError):
        chemistry.balance_equation(equation)


def test_equilibrium_handles_dissociation_from_product(chemistry):
    result = chemistry.equilibrium_single("A + B <-> C", Ka=4, A0=0, B0=0, C0=1)
    assert result["A"] > 0
    assert result["A"] == pytest.approx(result["B"])
    assert result["C"] / (result["A"] * result["B"]) == pytest.approx(4)
    assert result["A"] + result["C"] == pytest.approx(1)


def test_equilibrium_handles_association(chemistry):
    result = chemistry.equilibrium_single("A + B <-> C", Ka=50, A0=1, B0=2)
    assert result["C"] / (result["A"] * result["B"]) == pytest.approx(50)
    assert result["A"] + result["C"] == pytest.approx(1)
    assert result["B"] + result["C"] == pytest.approx(2)


@pytest.mark.parametrize("kwargs", [{"Ka": 0}, {"Ka": -1}, {"A0": -1}, {"C0": math.nan}])
def test_equilibrium_rejects_invalid_parameters(chemistry, kwargs):
    parameters = dict(Ka=4, A0=1, B0=1, C0=0)
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        chemistry.equilibrium_single("A + B <-> C", **parameters)


def test_equilibrium_rejects_unsupported_stoichiometry(chemistry):
    with pytest.raises(ValueError, match="1:1:1"):
        chemistry.equilibrium_single("2 H2 + O2 <-> 2 H2O", Ka=4, A0=1, B0=1)


def test_kinetics_conserves_atoms(chemistry):
    times, traces = chemistry.kinetics_sim("2 H2 + O2 -> 2 H2O", t_end=1, dt=0.01)
    assert times[-1] == pytest.approx(1)
    for h2, o2, water in zip(traces["H2"], traces["O2"], traces["H2O"]):
        assert h2 + water == pytest.approx(1)
        assert 2 * o2 + water == pytest.approx(2)
        assert min(h2, o2, water) >= 0


@pytest.mark.parametrize("kwargs", [{"dt": 0}, {"dt": -1}, {"k": -1}, {"t_end": -1}, {"dt": math.nan}])
def test_kinetics_rejects_invalid_parameters(chemistry, kwargs):
    with pytest.raises(ValueError):
        chemistry.kinetics_sim("H2 + O2 -> H2O", **kwargs)


def test_kinetics_reaches_requested_end_time(chemistry):
    times, traces = chemistry.kinetics_sim("H2 + O2 -> H2O", t_end=1, dt=0.3)
    assert times[-1] == pytest.approx(1)
    assert len(times) == len(traces["H2"])


@pytest.mark.parametrize("initial", [{"H2": -1}, {"H2": math.nan}, {"CO2": 1}])
def test_kinetics_rejects_invalid_initial_state(chemistry, initial):
    with pytest.raises(ValueError):
        chemistry.kinetics_sim("H2 + O2 -> H2O", init=initial)


def test_kinetics_rejects_nonconserving_reaction(chemistry):
    with pytest.raises(ValueError):
        chemistry.kinetics_sim("H2 -> O2")


def test_kinetics_rejects_unstable_step_instead_of_clipping_mass(chemistry):
    with pytest.raises(ValueError, match="dt"):
        chemistry.kinetics_sim("H2 + O2 -> H2O", k=100, dt=1, t_end=1)
