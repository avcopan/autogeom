"""rdkit mol tests."""

import math

import numpy as np
import pytest
from rdkit.Chem import Mol

from autogeom import GeometryConversionError, rd


@pytest.fixture
def water() -> Mol:
    """Water fixture."""
    return rd.from_smiles("O", with_coords=True)


def test__smiles_rt() -> None:
    """Test SMILES roundtrip."""
    smiles = "[H]O[H]"
    water = rd.from_smiles(smiles, with_coords=True)

    assert rd.smiles(water) == smiles


def test__inchi_rt() -> None:
    """Test InChI roundtrip."""
    smiles = "InChI=1S/H2O/h1H2"
    water = rd.from_inchi(smiles, with_coords=True)

    assert rd.inchi(water) == smiles


def test__xyz_rt() -> None:
    """Test XYZ roundtrip."""
    xyz = "1\n\nO      0.000000    0.000000    0.000000\n"
    water = rd.from_xyz_block(xyz)

    assert rd.xyz_block(water) == xyz


def test__add_atom_numbers(water: Mol) -> None:
    """Test adding atom numbers."""
    mapping = {0: 1, 1: 2, 2: 0}
    rd.add_atom_numbers(water, to_number=mapping, in_place=True)

    for a in water.GetAtoms():
        assert a.GetProp("atomLabel")


def test__coordinates_without_coordinates_raises() -> None:
    """Test that reading coordinates from a Mol without them is rejected."""
    no_coords = rd.from_smiles("O", with_coords=False)
    with pytest.raises(GeometryConversionError, match="no coordinates"):
        rd.coordinates(no_coords)


def test__from_connectivity_singlet_carbene() -> None:
    """Test that a closed-shell lone pair is not turned into radicals."""
    rdm = rd.from_connectivity(["C", "H", "H"], [(0, 1), (0, 2)], spin=0)

    assert rd.spin(rdm) == 0


@pytest.mark.parametrize(
    ("symbols", "bonds", "charge", "spin", "match"),
    [
        (["O", "H", "H"], [(0, 1), (0, 2)], 0, 1, "inconsistent"),
        (["H", "H", "H"], [(0, 1), (1, 2)], 0, 1, "cannot accommodate"),
        (["O", "H"], [(0, 0)], 0, 1, "Invalid bonds"),
        (["H", "H"], [(0, 1)], 0, 2, "No Lewis structure"),
    ],
)
def test__from_connectivity_raises(
    symbols: list[str],
    bonds: list[tuple[int, int]],
    charge: int,
    spin: int,
    match: str,
) -> None:
    """Test that impossible Lewis structures are rejected."""
    with pytest.raises(GeometryConversionError, match=match):
        rd.from_connectivity(symbols, bonds, charge=charge, spin=spin)


def test__from_connectivity_metal_raises() -> None:
    """Test that metals are not supported."""
    with pytest.raises(NotImplementedError, match="metals"):
        rd.from_connectivity(["Fe", "O"], [(0, 1)])


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("H", 1), ("C", 4), ("N", 4), ("O", 3), ("F", 2), ("S", 6), ("Fe", math.inf)],
)
def test__max_bond_count(symbol: str, expected: float) -> None:
    """Test maximum bond counts, allowing for formal charges of +/-1."""
    assert rd.max_bond_count(symbol) == expected


@pytest.mark.parametrize(
    ("parser", "value"),
    [
        (rd.from_smiles, "C1CC"),
        (rd.from_inchi, "InChI=garbage"),
        (rd.from_xyz_block, "not xyz"),
    ],
)
def test__parse_invalid_raises(parser: object, value: str) -> None:
    """Test that invalid identifiers raise a clear error."""
    with pytest.raises(ValueError, match="Invalid"):
        parser(value)  # ty: ignore[call-non-callable]


def test__add_coordinates_reproducible() -> None:
    """Test that embedding is seeded by default, and can be randomized."""
    coords1 = rd.coordinates(rd.from_smiles("CCO", with_coords=True))
    coords2 = rd.coordinates(rd.from_smiles("CCO", with_coords=True))
    assert np.allclose(coords1, coords2)

    no_coords = rd.from_smiles("CCO")
    coords3 = rd.coordinates(rd.add_coordinates(no_coords, seed=None))
    assert coords3.shape == coords1.shape


def test__set_coordinates_wrong_shape_raises(water: Mol) -> None:
    """Test that coordinates of the wrong shape are rejected."""
    with pytest.raises(ValueError, match="shape"):
        rd.set_coordinates(water, np.zeros((2, 3)))
