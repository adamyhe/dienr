"""The version is written once, in pyproject.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

import dienr

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_package_version_comes_from_pyproject():
    assert dienr.__version__ == _pyproject_version()


def test_citation_version_matches_pyproject():
    lines = (ROOT / "CITATION.cff").read_text().splitlines()
    cited = [ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("version:")]
    assert cited == [_pyproject_version()]
