"""Conversions between geometries and RDKit molecules, InChI, SMILES, and formulas."""

from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from . import rd
from ._geometry import Geometry
from ._measure import adjacency_matrix

__all__ = [
    "formula",
    "from_inchi",
    "from_rdkit_mol",
    "from_smiles",
    "inchi",
    "rdkit_mol",
    "smiles",
]


def rdkit_mol(geo: Geometry) -> Mol:
    """Convert a geometry to an RDKit molecule.

    Connectivity comes from `adjacency_matrix` (with valence caps). Bond orders,
    formal charges, and radicals are perceived to match the geometry's charge
    and spin (see `autogeom.rd.from_connectivity`), and stereochemistry is
    assigned from the 3D coordinates.

    Args:
        geo: Geometry.

    Returns:
        RDKit molecule with perceived bonding and stereochemistry.

    Raises:
        NotImplementedError: If the geometry contains a metal.
        GeometryConversionError: If no Lewis structure is consistent with the
            connectivity, charge, and spin.
    """
    amat = adjacency_matrix(geo, enforce_valence=True)
    bonds = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(amat)), strict=True)]
    mol = rd.from_connectivity(
        geo.symbols,
        bonds,
        charge=geo.charge,
        spin=geo.spin,
        coords=geo.coordinates,
    )
    return rd.assign_stereochemistry(mol, in_place=True)


def from_rdkit_mol(mol: Mol) -> Geometry:
    """Convert an RDKit molecule to a geometry, embedding coordinates if missing.

    Args:
        mol: RDKit molecule.

    Returns:
        The geometry.
    """
    if not rd.has_coordinates(mol):
        mol = rd.add_coordinates(mol)

    return Geometry(
        symbols=rd.symbols(mol),
        coordinates=rd.coordinates(mol),
        charge=rd.charge(mol),
        spin=rd.spin(mol),
    )


def inchi(geo: Geometry, *, stereo: bool = True) -> str:
    """Generate the standard InChI string of a geometry with RDKit.

    Args:
        geo: Geometry.
        stereo: Whether to include the stereo layers.

    Returns:
        InChI string.

    Raises:
        NotImplementedError: If the geometry contains a metal.
        GeometryConversionError: If no Lewis structure is consistent with the
            connectivity, charge, and spin.
        ValueError: If RDKit fails to generate an InChI.

    Example:
        >>> import autogeom
        >>> geo = autogeom.from_smiles("O")
        >>> autogeom.inchi(geo)
        'InChI=1S/H2O/h1H2'

    %seealso from_inchi, smiles, formula
    """
    return rd.inchi(rdkit_mol(geo), stereo=stereo)


def from_inchi(inchi: str) -> Geometry:
    """Embed a geometry for an InChI string with RDKit.

    Coordinates are embedded with a fixed random seed, so they are reproducible.

    Args:
        inchi: InChI string.

    Returns:
        The embedded geometry.

    Raises:
        ValueError: If the InChI string is invalid.

    Example:
        >>> import autogeom
        >>> geo = autogeom.from_inchi("InChI=1S/H2O/h1H2")
        >>> geo.symbols
        ['O', 'H', 'H']

    %seealso inchi, from_smiles
    """
    return from_rdkit_mol(rd.from_inchi(inchi, with_coords=True))


def smiles(geo: Geometry) -> str:
    """Generate the canonical SMILES string of a geometry with RDKit.

    Stereochemistry perceived from the 3D coordinates is included, and hydrogens
    are implicit.

    Args:
        geo: Geometry.

    Returns:
        SMILES string.

    Raises:
        NotImplementedError: If the geometry contains a metal.
        GeometryConversionError: If no Lewis structure is consistent with the
            connectivity, charge, and spin.

    Example:
        >>> import autogeom
        >>> geo = autogeom.from_smiles("C(O)C")
        >>> autogeom.smiles(geo)
        'CCO'

    %seealso from_smiles, inchi
    """
    return Chem.MolToSmiles(Chem.RemoveAllHs(rdkit_mol(geo)))


def from_smiles(smiles: str) -> Geometry:
    """Embed a geometry for a SMILES string with RDKit.

    Coordinates are embedded with a fixed random seed, so they are reproducible.

    Args:
        smiles: SMILES string.

    Returns:
        The embedded geometry.

    Raises:
        ValueError: If the SMILES string is invalid.

    Example:
        >>> import autogeom
        >>> geo = autogeom.from_smiles("[OH]")
        >>> geo.symbols, geo.spin
        (['O', 'H'], 1)

    %seealso smiles, from_inchi
    """
    return from_rdkit_mol(rd.from_smiles(smiles, with_coords=True))


def formula(geo: Geometry) -> str:
    """Generate the molecular formula of a geometry in Hill order.

    With carbon present, C comes first, then H, then the remaining elements
    alphabetically. Without carbon, all elements (including H) are alphabetical.

    Args:
        geo: Geometry.

    Returns:
        Molecular formula.

    Example:
        >>> import autogeom
        >>> autogeom.formula(autogeom.from_smiles("CO"))
        'CH4O'
        >>> autogeom.formula(autogeom.from_smiles("Cl"))
        'ClH'
    """
    counts = Counter(geo.symbols)

    ordered = []
    if "C" in counts:
        ordered.append(("C", counts.pop("C")))
        if "H" in counts:
            ordered.append(("H", counts.pop("H")))
    ordered.extend(sorted(counts.items()))

    return "".join(s if n == 1 else f"{s}{n}" for s, n in ordered)
