"""Shared fixtures for geometry tests."""

from pathlib import Path

import pytest

from autogeom import Geometry


@pytest.fixture
def water() -> Geometry:
    """Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        charge=0,
        spin=0,
    )


@pytest.fixture
def peroxide() -> Geometry:
    """Peroxide geometry fixture."""
    return Geometry(
        symbols=["H", "O", "O", "H"],
        coordinates=[[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        charge=0,
        spin=0,
    )


@pytest.fixture
def propyl_oxirane() -> Geometry:
    """Asymmetric 15-atom radical geometry fixture (no symmetry ambiguities)."""
    path = Path(__file__).parent / "data" / "propyl_oxirane.xyz"
    return Geometry.from_xyz_file(path, charge=0, spin=1)
