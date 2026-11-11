"""Utility tests: element data and exceptions."""

import pytest

from autogeom import ElementNotFoundError, element


@pytest.mark.parametrize(("key", "mass_number"), [("H", 1), (115, 287)])
def test__mass_number(key: str | int, mass_number: int) -> None:
    """Test retrieval of element mass number."""
    assert element.mass_number(key) == mass_number


@pytest.mark.parametrize(("key", "symbol"), [("H", "H"), (115, "Mc")])
def test__symbol(key: str | int, symbol: str) -> None:
    """Test retrieval of element symbol."""
    assert element.symbol(key) == symbol


@pytest.mark.parametrize(("key", "mass"), [("H", 1.008), (115, 287.191)])
def test__mass(key: str | int, mass: float) -> None:
    """Test retrieval of element mass."""
    assert round(element.mass(key), 3) == mass


@pytest.mark.parametrize(("key", "radius"), [("H", 0.32), (115, 1.62)])
def test__covalent_radius(key: str | int, radius: float) -> None:
    """Test retrieval of element covalent radius."""
    assert element.covalent_radius(key) == radius


@pytest.mark.parametrize(("key", "group"), [("H", 1), (115, 15)])
def test__group(key: str | int, group: int) -> None:
    """Test retrieval of element group."""
    assert element.group(key) == group


@pytest.mark.parametrize(("key", "en"), [("H", 2.2), ("O", 3.44), ("He", None)])
def test__electronegativity(key: str | int, en: float | None) -> None:
    """Test retrieval of element Pauling electronegativity."""
    assert element.electronegativity(key) == en


@pytest.mark.parametrize(
    ("key", "metal"), [("H", False), ("Si", False), ("Na", True), (26, True)]
)
def test__is_metal(key: str | int, *, metal: bool) -> None:
    """Test element metal classification."""
    assert element.is_metal(key) is metal


@pytest.mark.parametrize(("key", "period"), [("H", 1), (115, 7)])
def test__period(key: str | int, period: int) -> None:
    """Test retrieval of element period."""
    assert element.period(key) == period


@pytest.mark.parametrize(("key", "capacity"), [("H", 2), (115, 32)])
def test__shell_capacity(key: str | int, capacity: int) -> None:
    """Test retrieval of element shell capacity."""
    assert element.shell_capacity(key) == capacity


@pytest.mark.parametrize(
    ("key", "override", "valence"), [("H", None, 1), (115, {"Mc": 4}, 4)]
)
def test__valence(
    key: str | int, override: dict[str, int] | None, valence: int
) -> None:
    """Test retrieval of element valence."""
    assert element.valence(key, override=override) == valence


@pytest.mark.parametrize(
    ("key", "override", "capacity"), [("H", None, 1), (115, {"Mc": 6}, 6)]
)
def test__bonding_capacity(
    key: str | int, override: dict[str, int] | None, capacity: int
) -> None:
    """Test retrieval of element bonding capacity."""
    assert element.bonding_capacity(key, override=override) == capacity


@pytest.mark.parametrize("key", [999, "Xx"])
def test__from_key_not_found_raises(key: str | int) -> None:
    """Test that an unknown atomic number or symbol is rejected."""
    with pytest.raises(ElementNotFoundError):
        element.from_key(key)


def test__from_key_bad_type_raises() -> None:
    """Test that a non-int, non-str key is rejected."""
    with pytest.raises(TypeError, match="must be int or str"):
        element.from_key(1.5)  # ty: ignore[invalid-argument-type]
