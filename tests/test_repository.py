import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("Core-algorithms", "Biological-Systems", "Environmental", "Explore-PyTorch")
SOURCES = sorted(path for folder in SOURCE_DIRS for path in (ROOT / folder).glob("*.py"))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_all_examples_parse(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_all_examples_import_without_running_demos(path):
    code = (
        "import importlib.util, sys; "
        "from pathlib import Path; "
        "p = Path(sys.argv[1]); sys.path.insert(0, str(p.parent)); "
        "spec = importlib.util.spec_from_file_location(p.stem.replace('-', '_'), p); "
        "m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; "
        "spec.loader.exec_module(m)"
    )
    result = subprocess.run([sys.executable, "-c", code, str(path)],
                            capture_output=True, text=True, timeout=60,
                            env={**os.environ, "MPLBACKEND": "Agg"})
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path, args, expected", [
    ("Core-algorithms/chem_reactions.py", ["--balance", "H2 + O2 -> H2O"], "2 H2 + 1 O2 -> 2 H2O"),
    ("Core-algorithms/chem_reactions.py", ["--equilibrium", "A + B <-> C", "--Ka", "50", "--A0", "1", "--B0", "2"], "Equilibrium concentrations"),
    ("Core-algorithms/bio_beginner_playground.py", ["--demo", "growth"], "POPULATION GROWTH"),
])
def test_documented_cli_commands(path, args, expected):
    result = subprocess.run([sys.executable, str(ROOT / path), *args],
                            capture_output=True, text=True, timeout=30,
                            env={**os.environ, "MPLBACKEND": "Agg"})
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
