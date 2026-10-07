import numpy as np

from conftest import load_module


def test_tcell_proliferation_uses_defined_cell_class():
    biology = load_module("Biological-Systems/bio-model-1.py")
    cell = biology.TCell("parent", np.zeros(3))
    assert cell.proliferate() is None
    cell.proliferation_counter = 1
    daughter = cell.proliferate()
    assert isinstance(daughter, biology.TCell)
    assert daughter.cell_id == "parent_d0"
    assert cell.proliferation_counter == 0


def test_advanced_sepsis_model_resolves_shared_components():
    sepsis = load_module("Biological-Systems/advanced_sepsis_model_v2.py")
    model = sepsis.AdvancedSepsisRiskModel()
    assert isinstance(model.cardiovascular, sepsis.CardiovascularSystem)
    assert isinstance(model.organ_scoring, sepsis.OrganDysfunctionScoring)


def test_environmental_forecast_returns_requested_days():
    environment = load_module("Environmental/environmental_health_risk_model.py")
    forecast = environment.EnvironmentalHealthAPI().get_forecast(0, 0, days=3)
    assert len(forecast) == 3
    assert len({day["date"] for day in forecast}) == 3
    assert all(np.isfinite(day["conditions"].pm25) for day in forecast)
