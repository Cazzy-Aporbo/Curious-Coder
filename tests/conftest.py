import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Explore-PyTorch"))
sys.path.insert(0, str(ROOT))


def load_module(relative_path):
    path = ROOT / relative_path
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def core():
    return load_module("Core-algorithms/core-algorithms-tutorial.py")


@pytest.fixture(scope="session")
def chemistry():
    return load_module("Core-algorithms/chem_reactions.py")
