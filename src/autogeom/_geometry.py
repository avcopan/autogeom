"""The `Geometry` model, single-frame xyz I/O, and atom selection."""

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import pyparsing as pp
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator
from pyparsing import pyparsing_common as ppc

from . import element
from .exc import ElementNotFoundError, XYZFormatError
from .types import CoordinatesField, FloatArray

__all__ = [
    "Geometry",
    "from_xyz_block",
    "from_xyz_file",
    "join",
    "remove_atoms",
    "select_atoms",
    "xyz_block",
    "xyz_file",
]


class Geometry(BaseModel):
    """Molecular geometry.

    Attributes:
        symbols: Atomic symbols, e.g. ``["H", "O", "H"]``. Validated against the
            periodic table and canonicalized (``"cl"`` -> ``"Cl"``).
        coordinates: Cartesian coordinates in Angstroms, shape ``(N, 3)`` and
            ordered like `symbols`. A `pint.Quantity` with length units is
            converted to Angstroms; plain values are assumed to be Angstroms.
        charge: Total molecular charge.
        spin: Number of unpaired electrons (``2S``). Must be non-negative and
            have the same parity as the electron count.

    Example:
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["H", "O", "H"],
        ...     coordinates=[[0.0, 0.0, -0.74], [0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> geo.atom_count
        3
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    symbols: list[str]
    coordinates: CoordinatesField
    charge: int
    spin: int

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, v: list[str]) -> list[str]:
        """Validate and canonicalize atomic symbols.

        Args:
            v: Atomic symbols.

        Returns:
            Canonicalized atomic symbols.

        Raises:
            ValueError: If any symbol is not a known element.
        """
        try:
            return [element.symbol(s) for s in v]
        except ElementNotFoundError as err:
            raise ValueError(str(err)) from err

    # A field validator (not a model validator), so that a failed assignment
    # under `validate_assignment` leaves the model unchanged
    @field_validator("symbols", "coordinates", "charge", "spin")
    @classmethod
    def _validate_consistency(cls, v: Any, info: ValidationInfo) -> Any:  # noqa: ANN401
        """Check that symbols, coordinates, charge, and spin are consistent.

        Args:
            v: Candidate field value.
            info: Validation context.

        Returns:
            The unchanged value.
        """
        data = {**info.data, str(info.field_name): v}
        symbols = data.get("symbols")
        coordinates = data.get("coordinates")
        if symbols is not None and coordinates is not None:
            _check_atom_count(symbols, coordinates)
        if symbols is not None and "charge" in data and "spin" in data:
            _check_spin(symbols, charge=data["charge"], spin=data["spin"])
        return v

    def __eq__(self, other: object) -> bool:
        """Compare geometries field by field, including exact coordinates."""
        if type(other) is not type(self):
            return NotImplemented
        return all(
            _values_equal(getattr(self, name), getattr(other, name))
            for name in type(self).model_fields
        )

    # Mutable model with array data, so not hashable
    __hash__ = None

    def __repr__(self) -> str:
        """Render the geometry as an xyz block."""
        return self.xyz_block()

    __str__ = __repr__

    def _repr_html_(self) -> str | None:
        """Render the geometry inline in Jupyter."""
        # Imported here, since `_view` builds on this module
        from ._view import view  # noqa: PLC0415

        return view(self, label=True)._repr_html_()

    @property
    def atom_count(self) -> int:
        """Number of atoms."""
        return len(self.symbols)

    @property
    def masses(self) -> list[float]:
        """Atomic masses."""
        return list(map(element.mass, self.symbols))

    @property
    def atomic_numbers(self) -> list[int]:
        """Atomic numbers."""
        return list(map(element.number, self.symbols))

    @property
    def covalent_radii(self) -> list[float]:
        """Pyykko covalent radii, in Angstroms."""
        return list(map(element.covalent_radius, self.symbols))

    @property
    def valences(self) -> list[int]:
        """Numbers of valence electrons."""
        return list(map(element.valence, self.symbols))

    def xyz_block(self, *, comment: str | None = None) -> str:
        """Format the geometry as an xyz block (see `xyz_block`)."""
        return xyz_block(self, comment=comment)

    def xyz_file(self, *, path: str | Path, comment: str | None = None) -> None:
        """Write the geometry to an xyz file (see `xyz_file`)."""
        xyz_file(self, path=path, comment=comment)

    @classmethod
    def from_xyz_block(cls, xyz_block: str, *, charge: int, spin: int) -> Self:
        """Parse a geometry from an xyz block (see `from_xyz_block`)."""
        symbols, coordinates = _parse_xyz_block(xyz_block)
        return cls(symbols=symbols, coordinates=coordinates, charge=charge, spin=spin)

    @classmethod
    def from_xyz_file(cls, path: str | Path, *, charge: int, spin: int) -> Self:
        """Read a geometry from an xyz file (see `from_xyz_file`)."""
        return cls.from_xyz_block(Path(path).read_text(), charge=charge, spin=spin)

    def relabel_atoms(self, indices: Sequence[int]) -> Self:
        """Reorder the atoms.

        Args:
            indices: Permutation of ``range(atom_count)`` giving the new order,
                e.g. ``[2, 0, 1]`` moves atom 2 to position 0, atom 0 to
                position 1, and atom 1 to position 2.

        Returns:
            A reordered copy. Other fields, including those of subclasses, are
            kept.

        Raises:
            ValueError: If `indices` is not a permutation of ``range(atom_count)``.
        """
        indices = [int(i) for i in indices]
        if sorted(indices) != list(range(self.atom_count)):
            msg = f"{indices} is not a permutation of range({self.atom_count})."
            raise ValueError(msg)

        return self.model_copy(
            deep=True,
            update={
                "symbols": [self.symbols[i] for i in indices],
                "coordinates": self.coordinates[indices],
            },
        )


def _check_atom_count(symbols: Sequence[str], coordinates: np.ndarray) -> None:
    """Raise `ValueError` unless there is one row of coordinates per symbol."""
    if len(symbols) != coordinates.shape[0]:
        msg = (
            f"Number of symbols ({len(symbols)}) does not match number of "
            f"coordinates ({coordinates.shape[0]})."
        )
        raise ValueError(msg)


def _check_spin(symbols: Sequence[str], *, charge: int, spin: int) -> None:
    """Raise `ValueError` unless `spin` is consistent with the electron count."""
    nelec = sum(map(element.number, symbols)) - charge
    if spin < 0 or nelec < 0 or spin > nelec or (nelec - spin) % 2:
        msg = f"Spin {spin} is inconsistent with {nelec} electrons (charge {charge})."
        raise ValueError(msg)


def _values_equal(val1: Any, val2: Any) -> bool:  # noqa: ANN401
    """Compare two field values, handling numpy arrays."""
    if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
        return bool(np.array_equal(val1, val2))
    return bool(val1 == val2)


# XYZ I/O
# Each token must be followed by whitespace or the end of the line, so that
# e.g. "H1 0 0 0" is rejected rather than parsed as H at (1, 0, 0)
_END = r"(?=\s|$)"
_SYMBOL = pp.Regex(rf"(?:[A-Za-z]{{1,2}}|\d{{1,3}}){_END}")
_FLOAT = pp.Regex(rf"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?{_END}")
_FLOAT.set_parse_action(ppc.convert_to_float)
_XYZ_LINE = _SYMBOL + pp.Group(_FLOAT * 3) + pp.Suppress(... + pp.LineEnd())


def xyz_block(geo: Geometry, *, comment: str | None = None) -> str:
    """Format a geometry as an xyz block.

    Args:
        geo: Geometry.
        comment: Comment line. Defaults to the charge and spin, e.g.
            ``"Geometry(q=0, s=0)"``.

    Returns:
        The xyz block.

    Raises:
        ValueError: If `comment` contains a newline.
    """
    if comment is None:
        comment = f"Geometry(q={geo.charge}, s={geo.spin})"
    if "\n" in comment or "\r" in comment:
        msg = f"The xyz comment must be a single line, got {comment!r}."
        raise ValueError(msg)
    lines = [str(geo.atom_count), comment]
    for sym, (x, y, z) in zip(geo.symbols, geo.coordinates.tolist(), strict=True):
        lines.append(f"{sym:<4} {x:12.8f} {y:12.8f} {z:12.8f}")

    return "\n".join(lines)


def from_xyz_block(xyz_block: str, *, charge: int, spin: int) -> Geometry:
    """Parse a geometry from a single-frame xyz block.

    Atoms may be given by symbol (case-insensitive) or atomic number.

    Args:
        xyz_block: The xyz block.
        charge: Total molecular charge.
        spin: Number of unpaired electrons.

    Returns:
        The parsed geometry.

    Raises:
        XYZFormatError: If the block is empty, the atom count line is missing or
            does not match the number of atom lines, or an atom line cannot be
            parsed.
    """
    return Geometry.from_xyz_block(xyz_block, charge=charge, spin=spin)


def xyz_file(geo: Geometry, *, path: str | Path, comment: str | None = None) -> None:
    """Write a geometry to an xyz file.

    Args:
        geo: Geometry.
        path: Output path.
        comment: Comment line. Defaults to the charge and spin, e.g.
            ``"Geometry(q=0, s=0)"``.
    """
    Path(path).write_text(xyz_block(geo, comment=comment) + "\n")


def from_xyz_file(path: str | Path, *, charge: int, spin: int) -> Geometry:
    """Read a geometry from an xyz file (see `from_xyz_block`).

    Args:
        path: Path to the xyz file.
        charge: Total molecular charge.
        spin: Number of unpaired electrons.

    Returns:
        The parsed geometry.
    """
    return Geometry.from_xyz_file(path, charge=charge, spin=spin)


def _parse_xyz_block(xyz_block: str) -> tuple[list[str], FloatArray]:
    """Parse symbols and coordinates from an xyz block (see `from_xyz_block`)."""
    lines = xyz_block.strip().splitlines()
    if not lines:
        msg = "The provided xyz block is empty."
        raise XYZFormatError(msg)

    try:
        natms = int(lines[0])
    except ValueError as exc:
        msg = f"Expected an atom count on the first line, got {lines[0]!r}."
        raise XYZFormatError(msg) from exc

    atom_lines = lines[2:]
    if not atom_lines:
        msg = "The provided xyz block contains no atoms."
        raise XYZFormatError(msg)

    if len(atom_lines) != natms:
        msg = (
            f"Atom count ({natms}) does not match the number of atom lines "
            f"({len(atom_lines)}). Multi-frame xyz blocks are not supported."
        )
        raise XYZFormatError(msg)

    try:
        symbs, coords = zip(
            *[_XYZ_LINE.parse_string(line).as_list() for line in atom_lines],
            strict=True,
        )
    except pp.ParseException as exc:
        msg = f"Failed to parse xyz line: {exc.line!r}"
        raise XYZFormatError(msg) from exc

    return [_parse_symbol(s) for s in symbs], np.array(coords, dtype=np.float64)


def _parse_symbol(token: str) -> str:
    """Convert an xyz atom token (symbol or atomic number) to a symbol."""
    try:
        return element.symbol(int(token) if token.isdigit() else token)
    except ElementNotFoundError as exc:
        msg = f"Unknown element in xyz block: {token!r}"
        raise XYZFormatError(msg) from exc


# Selecting and joining atoms
def select_atoms(
    geo: Geometry,
    keys: Sequence[int],
    *,
    charge: int | None = None,
    spin: int | None = None,
) -> Geometry:
    """Select a subset of atoms.

    Args:
        geo: Geometry.
        keys: Indices of the atoms to keep, in the order to keep them.
        charge: Charge of the selection. Defaults to the charge of `geo`.
        spin: Spin of the selection. Defaults to the spin of `geo` if that is
            consistent with the electron count, and otherwise to the lowest
            consistent spin (e.g. 1 for a methyl radical selected from methane).

    Returns:
        The selected atoms.

    Example:
        >>> import numpy as np
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["C", "H", "H", "H", "H"],
        ...     coordinates=np.zeros((5, 3)),
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> methyl = autogeom.select_atoms(geo, [0, 1, 2, 3])
        >>> methyl.symbols, methyl.spin
        (['C', 'H', 'H', 'H'], 1)
    """
    keys = [int(k) for k in keys]
    symbols = [geo.symbols[k] for k in keys]
    charge = geo.charge if charge is None else charge
    if spin is None:
        nelec = sum(map(element.number, symbols)) - charge
        spin = geo.spin if (nelec - geo.spin) % 2 == 0 else nelec % 2
    return Geometry(
        symbols=symbols,
        coordinates=geo.coordinates[keys],
        charge=charge,
        spin=spin,
    )


def remove_atoms(
    geo: Geometry,
    keys: Collection[int],
    *,
    charge: int | None = None,
    spin: int | None = None,
) -> Geometry:
    """Remove a subset of atoms.

    Args:
        geo: Geometry.
        keys: Indices of the atoms to remove.
        charge: Charge of the remaining atoms (see `select_atoms`).
        spin: Spin of the remaining atoms (see `select_atoms`).

    Returns:
        The remaining atoms, in their original order.
    """
    remove = set(map(int, keys))
    keep = [k for k in range(geo.atom_count) if k not in remove]
    return select_atoms(geo, keep, charge=charge, spin=spin)


def join(geos: Sequence[Geometry]) -> Geometry:
    """Join geometries into one.

    Args:
        geos: Geometries, in order.

    Returns:
        A geometry with the atoms of each geometry in turn, and the summed
        charges and spins.
    """
    return Geometry(
        symbols=[s for geo in geos for s in geo.symbols],
        coordinates=np.vstack([geo.coordinates for geo in geos]).reshape(-1, 3),
        charge=sum(geo.charge for geo in geos),
        spin=sum(geo.spin for geo in geos),
    )
