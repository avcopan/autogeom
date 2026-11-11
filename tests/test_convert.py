"""InChI, SMILES, and formula conversion tests."""

from collections.abc import Callable

import numpy as np
import pytest

import autogeom
from autogeom import Geometry


def test__inchi_roundtrip() -> None:
    """Test InChI -> Geometry -> InChI."""
    inchi = "InChI=1S/H2O/h1H2"
    assert autogeom.inchi(autogeom.from_inchi(inchi)) == inchi


def test__smiles_roundtrip() -> None:
    """Test SMILES -> Geometry -> SMILES."""
    assert autogeom.smiles(autogeom.from_smiles("O")) == "O"


def test__smiles_is_canonical() -> None:
    """Test that equivalent SMILES strings give one canonical SMILES."""
    geo = autogeom.from_smiles("C(C)CCC")
    assert autogeom.smiles(geo) == "CCCCC"


def test__smiles_includes_stereochemistry() -> None:
    """Test that stereochemistry is perceived from the coordinates."""
    smiles = autogeom.smiles(autogeom.from_smiles("C/C=C/C"))
    assert smiles == "C/C=C/C"


@pytest.mark.parametrize(
    ("from_string", "value"),
    [(autogeom.from_smiles, "C1CC"), (autogeom.from_inchi, "not an inchi")],
)
def test__from_string_invalid_value_raises(
    from_string: Callable[[str], Geometry], value: str
) -> None:
    """Test that invalid identifiers raise a clear error."""
    with pytest.raises(ValueError, match="Invalid"):
        from_string(value)


def test__from_smiles_reproducible() -> None:
    """Test that SMILES -> Geometry is reproducible."""
    assert autogeom.from_smiles("CCO") == autogeom.from_smiles("CCO")


def test__formula(water: Geometry) -> None:
    """Test Geometry to Hill-ordered formula."""
    assert autogeom.formula(water) == "H2O"


@pytest.mark.parametrize(
    ("symbols", "expected"),
    [
        (["C", "H", "H", "H", "H"], "CH4"),
        (["Cl", "Cl"], "Cl2"),
        # Without carbon, H is ordered alphabetically (strict Hill order)
        (["H", "Cl"], "ClH"),
        (["N", "H", "H", "H"], "H3N"),
        # With carbon, C and H come first, then the rest alphabetically
        (["Cl", "C", "H", "H", "H", "Br", "N"], "CH3BrClN"),
    ],
)
def test__formula_hill_order(symbols: list[str], expected: str) -> None:
    """Test that formulas follow Hill order."""
    natms = len(symbols)
    nelec = sum(autogeom.element.number(s) for s in symbols)
    geo = Geometry(
        symbols=symbols, coordinates=np.eye(natms, 3), charge=0, spin=nelec % 2
    )
    assert autogeom.formula(geo) == expected
