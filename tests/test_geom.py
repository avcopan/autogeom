"""Geometry tests: model, I/O, rdkit conversion, analysis, and transforms."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from rdkit import Chem
from scipy.spatial.transform import Rotation

import autogeom
from autogeom import Geometry, XYZFormatError, rd, ureg

# Model


def test__masses_atomic_numbers_valences(water: Geometry) -> None:
    """Test scalar per-atom properties."""
    expected_masses = [
        pytest.approx(15.9949, abs=1e-4),
        pytest.approx(1.00783, abs=1e-4),
        pytest.approx(1.00783, abs=1e-4),
    ]
    assert water.masses == expected_masses
    assert water.atomic_numbers == [8, 1, 1]
    assert water.valences == [6, 1, 1]


def test__coordinates_symbols_mismatch_raises() -> None:
    """Test that mismatched symbols/coordinates lengths are rejected."""
    with pytest.raises(ValueError, match="does not match"):
        Geometry(
            symbols=["O", "H", "H"],
            coordinates=[[0, 0, 0], [1, 0, 0]],
            charge=0,
            spin=0,
        )


@pytest.mark.parametrize(
    ("unit", "factor"), [("angstrom", 1.0), ("bohr", 0.529177), ("nm", 10.0)]
)
def test__coordinates_quantity_converted_to_angstrom(unit: str, factor: float) -> None:
    """Test that pint quantity coordinates are converted to Angstrom."""
    coords = np.eye(3)
    geo = Geometry(
        symbols=["O", "H", "H"], coordinates=coords * ureg(unit), charge=0, spin=0
    )
    assert isinstance(geo.coordinates, np.ndarray)
    np.testing.assert_allclose(geo.coordinates, coords * factor, rtol=1e-6)


def test__coordinates_quantity_bad_units_raises() -> None:
    """Test that quantity coordinates without length units are rejected."""
    with pytest.raises(ValidationError, match="length units"):
        Geometry(
            symbols=["O", "H", "H"],
            coordinates=np.eye(3) * ureg.second,
            charge=0,
            spin=0,
        )


def test__symbols_canonicalized() -> None:
    """Test that symbols are validated case-insensitively and canonicalized."""
    geo = Geometry(symbols=["cl", "CL"], coordinates=np.eye(2, 3), charge=0, spin=0)
    assert geo.symbols == ["Cl", "Cl"]


def test__unknown_symbol_raises() -> None:
    """Test that unknown element symbols are rejected."""
    with pytest.raises(ValidationError, match="No element"):
        Geometry(symbols=["Xx"], coordinates=[[0, 0, 0]], charge=0, spin=0)


@pytest.mark.parametrize(("charge", "spin"), [(0, 1), (0, -2), (1, 0), (11, 0)])
def test__inconsistent_spin_raises(charge: int, spin: int) -> None:
    """Test that spin inconsistent with the electron count is rejected."""
    with pytest.raises(ValidationError, match="inconsistent"):
        Geometry(
            symbols=["O", "H", "H"], coordinates=np.eye(3), charge=charge, spin=spin
        )


def test__invalid_assignment_is_rejected(water: Geometry) -> None:
    """Test that invalid assignments raise and leave the geometry unchanged."""
    with pytest.raises(ValidationError, match="does not match"):
        water.symbols = ["O", "H"]
    with pytest.raises(ValidationError, match="inconsistent"):
        water.spin = 1
    assert water.symbols == ["O", "H", "H"]
    assert water.spin == 0


def test__equality(water: Geometry, peroxide: Geometry) -> None:
    """Test that geometries compare by value, including coordinates."""
    assert water == water.model_copy(deep=True)
    assert water != peroxide
    assert water != autogeom.translate(water, [1e-8, 0, 0])
    assert water != "water"


def test__relabel_atoms(water: Geometry) -> None:
    """Test atom reordering."""
    relabeled = water.relabel_atoms([1, 0, 2])
    assert relabeled.symbols == ["H", "O", "H"]
    assert np.array_equal(relabeled.coordinates, water.coordinates[[1, 0, 2]])


@pytest.mark.parametrize("indices", [[0, 0, 0], [0, 1], [0, 1, 3]])
def test__relabel_atoms_non_permutation_raises(
    water: Geometry, indices: list[int]
) -> None:
    """Test that non-permutation indices are rejected."""
    with pytest.raises(ValueError, match="not a permutation"):
        water.relabel_atoms(indices)


# XYZ I/O


def test__xyz_file_roundtrip(water: Geometry, tmp_path: Path) -> None:
    """Test xyz file roundtrip through both the method and module-level API."""
    method_path = tmp_path / "method.xyz"
    water.xyz_file(path=method_path)
    function_path = tmp_path / "function.xyz"
    autogeom.xyz_file(water, path=function_path)

    for geo_rt in (
        Geometry.from_xyz_file(method_path, charge=0, spin=0),
        autogeom.from_xyz_file(function_path, charge=0, spin=0),
    ):
        assert water.symbols == geo_rt.symbols
        assert np.allclose(water.coordinates, geo_rt.coordinates)


def test__xyz_file_trailing_newline(water: Geometry, tmp_path: Path) -> None:
    """Test that xyz files end with a newline."""
    path = tmp_path / "water.xyz"
    water.xyz_file(path=path)
    assert path.read_text().endswith("\n")
    assert not path.read_text().endswith("\n\n")


def test__xyz_block_multiline_comment_raises(water: Geometry) -> None:
    """Test that a multi-line comment is rejected."""
    with pytest.raises(ValueError, match="single line"):
        water.xyz_block(comment="line 1\nline 2")


def test__from_xyz_block_flexible_formats() -> None:
    """Test leading-dot/exponent floats, lowercase symbols, and atomic numbers."""
    xyz = "3\n\no .5 -.5 1e-1\n1 0 0 0\nH\t+1.0E+0 2 3.  \n"
    geo = autogeom.from_xyz_block(xyz, charge=0, spin=0)
    assert geo.symbols == ["O", "H", "H"]
    assert np.allclose(geo.coordinates, [[0.5, -0.5, 0.1], [0, 0, 0], [1, 2, 3]])


@pytest.mark.parametrize(
    ("xyz_block", "match"),
    [
        ("", "empty"),
        ("1\n\nnot a valid xyz line\n", "Failed to parse"),
        ("1\n\nH1 0 0 0\n", "Failed to parse"),
        ("H 0 0 0\n", "atom count"),
        ("1\ncomment\n", "no atoms"),
        ("3\n\nH 0 0 0\nH 0 0 1\n", "does not match"),
        ("1\na\nH 0 0 0\n1\nb\nH 0 0 1\n", "Multi-frame"),
        ("1\n\n999 0 0 0\n", "Unknown element"),
        ("1\n\nXx 0 0 0\n", "Unknown element"),
    ],
)
def test__from_xyz_block_raises(xyz_block: str, match: str) -> None:
    """Test that empty and malformed xyz blocks are rejected."""
    with pytest.raises(XYZFormatError, match=match):
        autogeom.from_xyz_block(xyz_block, charge=0, spin=0)


def test__xyz_trajectory_file_roundtrip(
    water: Geometry, peroxide: Geometry, tmp_path: Path
) -> None:
    """Test multi-frame xyz file roundtrip, with frames of different sizes."""
    geos = [water, peroxide, autogeom.translate(water, [1.0, 2.0, 3.0])]
    comments = ["-15.2", "second", ""]
    path = tmp_path / "frames.xyz"
    autogeom.xyz_trajectory_file(geos, path=path, comments=comments)

    geos_rt = autogeom.from_xyz_trajectory_file(path, charge=0, spin=0)
    assert len(geos_rt) == len(geos)
    for geo, geo_rt in zip(geos, geos_rt, strict=True):
        assert geo.symbols == geo_rt.symbols
        assert np.allclose(geo.coordinates, geo_rt.coordinates)
    assert autogeom.xyz_trajectory_comments(path.read_text()) == comments


def test__xyz_trajectory_single_frame(water: Geometry) -> None:
    """Test that single-frame and multi-frame I/O agree on a single frame."""
    block = autogeom.xyz_trajectory_block([water])
    assert block == autogeom.xyz_block(water)
    (geo_rt,) = autogeom.from_xyz_trajectory_block(block, charge=0, spin=0)
    assert geo_rt == autogeom.from_xyz_block(block, charge=0, spin=0)


@pytest.mark.parametrize(
    ("xyz_block", "match"),
    [
        ("", "empty"),
        ("1\na\nH 0 0 0\nH 0 0 1\n", "atom count on line 4"),
        ("1\na\nH 0 0 0\n2\nb\nH 0 0 1\n", "only 1 atom lines"),
    ],
)
def test__from_xyz_trajectory_block_raises(xyz_block: str, match: str) -> None:
    """Test that empty and malformed multi-frame xyz blocks are rejected."""
    with pytest.raises(XYZFormatError, match=match):
        autogeom.from_xyz_trajectory_block(xyz_block, charge=0, spin=0)


def test__xyz_trajectory_block_comment_count_raises(water: Geometry) -> None:
    """Test that the number of comments must match the number of geometries."""
    with pytest.raises(ValueError, match="2 comments for 1 geometries"):
        autogeom.xyz_trajectory_block([water], comments=["a", "b"])


# RDKit conversion (C/H/O perception is covered in test_geom_to_mol.py)


@pytest.mark.parametrize(
    "smiles",
    [
        "C[N+](=O)[O-]",
        "[NH4+]",
        "CS(C)=O",
        "O=[N][O]",
        "[NH3+]CC(=O)[O-]",
        "F[C@H](Cl)Br",
    ],
)
def test__rdkit_mol_perceives_structure(smiles: str) -> None:
    """Test that heteroatom bond orders, charges, radicals, and stereo roundtrip."""
    geo = autogeom.from_rdkit_mol(rd.from_smiles(smiles, with_coords=True))
    mol = autogeom.rdkit_mol(geo)

    expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == expected
    assert rd.charge(mol) == geo.charge
    assert rd.spin(mol) == geo.spin


def test__from_rdkit_mol_without_coordinates() -> None:
    """Test that Geometry is built from an rdkit Mol missing coordinates."""
    mol = rd.from_smiles("O", with_coords=False)
    assert not rd.has_coordinates(mol)

    geo = autogeom.from_rdkit_mol(mol)
    assert sorted(geo.symbols) == ["H", "H", "O"]
    assert geo.coordinates.shape == (3, 3)


def test__from_rdkit_mol_with_coordinates() -> None:
    """Test that Geometry is built from an rdkit Mol that already has coordinates."""
    mol = rd.from_smiles("O", with_coords=True)
    assert rd.has_coordinates(mol)
    original_coords = rd.coordinates(mol)

    geo = autogeom.from_rdkit_mol(mol)
    assert sorted(geo.symbols) == ["H", "H", "O"]
    sorted_geo_coords = np.sort(geo.coordinates, axis=0)
    sorted_original_coords = np.sort(original_coords, axis=0)
    assert np.allclose(sorted_geo_coords, sorted_original_coords)


# Analysis


def test__center_of_mass(water: Geometry) -> None:
    """Test center of mass."""
    assert np.allclose(autogeom.center_of_mass(water), [0.05595744, 0.05595744, 0.0])


def test__empty_geometry_properties() -> None:
    """Test mass properties of an empty geometry."""
    empty = Geometry(symbols=[], coordinates=np.zeros((0, 3)), charge=0, spin=0)
    assert np.array_equal(autogeom.center_of_mass(empty), np.zeros(3))
    assert np.array_equal(autogeom.inertia_tensor(empty), np.zeros((3, 3)))


def test__inertia_tensor_values() -> None:
    """Test the inertia tensor of a linear molecule along z."""
    h2 = Geometry(
        symbols=["H", "H"], coordinates=[[0, 0, -1], [0, 0, 1]], charge=0, spin=0
    )
    mass = h2.masses[0]
    expected = np.diag([2 * mass, 2 * mass, 0.0])
    assert np.allclose(autogeom.inertia_tensor(h2), expected)


def test__inertia_tensor(water: Geometry) -> None:
    """Test that the inertia tensor is a symmetric (3, 3) matrix."""
    tensor = autogeom.inertia_tensor(water)
    assert tensor.shape == (3, 3)
    assert np.allclose(tensor, tensor.T)


def test__distance_matrix(water: Geometry) -> None:
    """Test distance matrix calculation."""
    dist_mat = autogeom.distance_matrix(water)
    expected = np.array([[0, 1, 1], [1, 0, np.sqrt(2)], [1, np.sqrt(2), 0]])
    assert np.allclose(dist_mat, expected)


def test__adjacency_matrix(water: Geometry) -> None:
    """Test adjacency matrix."""
    amat = autogeom.adjacency_matrix(geo=water)
    assert np.array_equal(amat, [[0, 1, 1], [1, 0, 0], [1, 0, 0]])


def test__adjacency_matrix_flood_fill() -> None:
    """Test that flood fill connects otherwise disconnected fragments."""
    disconnected = Geometry(
        symbols=["H", "H"],
        coordinates=[[0, 0, 0], [2, 0, 0]],
        charge=0,
        spin=0,
    )
    amat = autogeom.adjacency_matrix(geo=disconnected, flood_fill=True)
    assert np.array_equal(amat, [[0, 1], [1, 0]])


def test__adjacency_matrix_flood_fill_unconnectable() -> None:
    """Test that flood fill terminates when valence caps prevent connection."""
    two_h2 = Geometry(
        symbols=["H", "H", "H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.74], [0, 0, 5], [0, 0, 5.74]],
        charge=0,
        spin=0,
    )
    amat = autogeom.adjacency_matrix(two_h2, flood_fill=True, enforce_valence=True)
    expected = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    assert np.array_equal(amat, expected)


def test__adjacency_matrix_enforce_valence_order_independent() -> None:
    """Test that valence enforcement keeps the closest bonds for any atom order."""
    # H3 chain with a short and a long H-H contact: only the short bond survives
    h3 = Geometry(
        symbols=["H", "H", "H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.74], [0, 0, 1.6], [0, 0, 2.34]],
        charge=0,
        spin=0,
    )
    expected = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    for perm in ([0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]):
        geo = h3.relabel_atoms(perm)
        amat = autogeom.adjacency_matrix(geo, enforce_valence=True)
        inv = np.argsort(perm)
        assert np.array_equal(amat[np.ix_(inv, inv)], expected)


def test__distance_keys(water: Geometry) -> None:
    """Test distance descriptor generation."""
    keys = autogeom.distance_keys(water)
    assert keys.shape == (3, 3)
    # Pairs must be sorted (z1 <= z2), then by z1, z2, distance.
    assert np.all(keys[:, 0] <= keys[:, 1])
    assert np.array_equal(keys[:, :2], np.sort(keys[:, :2], axis=1))


def test__bonds(water: Geometry) -> None:
    """Test bond distance extraction for water."""
    amat = autogeom.adjacency_matrix(water)
    b = autogeom.bonds(water, amat)
    assert b.shape == (2, 3)
    assert np.array_equal(b, [[1.0, 8.0, 1.0], [1.0, 8.0, 1.0]])


def test__angles(water: Geometry) -> None:
    """Test bond angle extraction for water (exact 90 degree H-O-H angle)."""
    amat = autogeom.adjacency_matrix(water)
    a = autogeom.angles(water, amat)
    assert a.shape == (1, 4)
    assert np.array_equal(a[0, :3], [1.0, 8.0, 1.0])
    assert np.isclose(a[0, 3], np.pi / 2)


def test__dihedrals(peroxide: Geometry) -> None:
    """Test dihedral extraction for an H-O-O-H chain."""
    amat = autogeom.adjacency_matrix(peroxide)
    d = autogeom.dihedrals(peroxide, amat)
    assert d.shape == (1, 5)
    assert np.array_equal(d[0, :4], [1.0, 8.0, 8.0, 1.0])
    assert np.isclose(d[0, 4], np.pi / 2)


def test__internal_coordinates_counts() -> None:
    """Test bond/angle/dihedral counts for propane, including the empty case."""
    propane = autogeom.from_rdkit_mol(rd.from_smiles("CCC", with_coords=True))
    amat = autogeom.adjacency_matrix(propane)
    assert autogeom.bonds(propane, amat).shape == (10, 3)
    # Angles: 6 at each of the three carbons (4 neighbors each, choose 2)
    assert autogeom.angles(propane, amat).shape == (18, 4)
    # Dihedrals: 3 x 3 = 9 torsions about each of the two C-C bonds
    assert autogeom.dihedrals(propane, amat).shape == (18, 5)

    he = Geometry(symbols=["He"], coordinates=[[0, 0, 0]], charge=0, spin=0)
    he_amat = autogeom.adjacency_matrix(he)
    assert autogeom.bonds(he, he_amat).shape == (0, 3)
    assert autogeom.angles(he, he_amat).shape == (0, 4)
    assert autogeom.dihedrals(he, he_amat).shape == (0, 5)


def test__dihedrals_orientation_invariant(peroxide: Geometry) -> None:
    """Test that reversing atom order does not change dihedral rows."""
    amat = autogeom.adjacency_matrix(peroxide)
    reversed_peroxide = peroxide.relabel_atoms([3, 2, 1, 0])
    reversed_amat = autogeom.adjacency_matrix(reversed_peroxide)
    assert np.allclose(
        autogeom.dihedrals(peroxide, amat),
        autogeom.dihedrals(reversed_peroxide, reversed_amat),
    )


def test__internal_coordinates_bad_amat_raises(water: Geometry) -> None:
    """Test that a wrongly shaped adjacency matrix is rejected."""
    with pytest.raises(ValueError, match="shape"):
        autogeom.bonds(water, np.zeros((2, 2)))


# Transforms


def test__reflection(peroxide: Geometry) -> None:
    """Test reflection."""
    normal = np.random.rand(3)  # noqa: NPY002
    refl_peroxide = autogeom.reflect(peroxide, normal)
    double_refl_peroxide = autogeom.reflect(refl_peroxide, normal)
    assert not np.allclose(peroxide.coordinates, refl_peroxide.coordinates)
    assert np.allclose(
        peroxide.coordinates, double_refl_peroxide.coordinates, atol=1e-7
    )


def test__reflection_zero_normal_raises(peroxide: Geometry) -> None:
    """Test that a zero normal vector is rejected."""
    with pytest.raises(ValueError, match="non-zero"):
        autogeom.reflect(peroxide, [0, 0, 0])


def test__translate(water: Geometry) -> None:
    """Test translation."""
    shift = [1.0, 2.0, 3.0]
    translated = autogeom.translate(water, shift)
    assert np.allclose(translated.coordinates, water.coordinates + shift)
    assert not np.allclose(translated.coordinates, water.coordinates)


def test__translate_in_place(water: Geometry) -> None:
    """Test in-place translation."""
    original = water.coordinates.copy()
    result = autogeom.translate(water, [1.0, 0.0, 0.0], in_place=True)
    assert result is water
    assert np.allclose(water.coordinates, np.add(original, [1.0, 0.0, 0.0]))


def test__translate_with_keys(water: Geometry) -> None:
    """Test translation of a subset of atoms."""
    original = water.coordinates.copy()
    translated = autogeom.translate(water, [1.0, 0.0, 0.0], keys=[0])
    assert np.allclose(translated.coordinates[0], original[0] + [1.0, 0.0, 0.0])
    assert np.allclose(translated.coordinates[1:], original[1:])


def test__rotate(water: Geometry) -> None:
    """Test rotation."""
    rot = Rotation.from_euler("z", 90, degrees=True)
    rotated = autogeom.rotate(water, rot)
    assert np.allclose(rotated.coordinates, rot.apply(water.coordinates))


def test__rotate_in_place(water: Geometry) -> None:
    """Test in-place rotation."""
    rot = Rotation.from_euler("z", 90, degrees=True)
    expected = rot.apply(water.coordinates)
    result = autogeom.rotate(water, rot, in_place=True)
    assert result is water
    assert np.allclose(water.coordinates, expected)


# A random rotation, origin, and translation, for roundtrip and invariance tests
ROT = Rotation.random(random_state=0)
ORIGIN = np.array([0.3, -1.2, 0.7])
SHIFT = np.array([1.0, 2.0, 3.0])


def _rigid_motion(geo: Geometry) -> Geometry:
    """Apply a rotation about a point and a translation."""
    return autogeom.translate(autogeom.rotate(geo, ROT, origin=ORIGIN), SHIFT)


def test__rotate_about_origin_roundtrip(propyl_oxirane: Geometry) -> None:
    """Test rotation about a point: it stays fixed, and the inverse undoes it."""
    geo = autogeom.translate(propyl_oxirane, ORIGIN - propyl_oxirane.coordinates[3])
    rotated = autogeom.rotate(geo, ROT, origin=ORIGIN)
    assert np.allclose(rotated.coordinates[3], ORIGIN)
    assert not np.allclose(rotated.coordinates, geo.coordinates)
    restored = autogeom.rotate(rotated, ROT.inv(), origin=ORIGIN)
    assert np.allclose(restored.coordinates, geo.coordinates)


def test__reflect_about_origin_roundtrip() -> None:
    """Test reflection across a plane through a point, and its inverse."""
    normal = [1.0, -2.0, 0.5]
    geo = Geometry(
        symbols=["He", "He"], coordinates=[ORIGIN, ORIGIN + 1.0], charge=0, spin=0
    )
    reflected = autogeom.reflect(geo, normal, origin=ORIGIN)
    assert np.allclose(reflected.coordinates[0], ORIGIN)
    assert not np.allclose(reflected.coordinates[1], geo.coordinates[1])
    restored = autogeom.reflect(reflected, normal, origin=ORIGIN)
    assert np.allclose(restored.coordinates, geo.coordinates)


def test__center(propyl_oxirane: Geometry) -> None:
    """Test that centering moves the center of mass to the origin."""
    centered = autogeom.center(autogeom.translate(propyl_oxirane, [5.0, -3.0, 1.0]))
    assert np.allclose(autogeom.center_of_mass(centered), 0.0)
    assert np.allclose(
        autogeom.distance_matrix(centered), autogeom.distance_matrix(propyl_oxirane)
    )


# Selecting and joining atoms


def test__select_and_join_roundtrip(propyl_oxirane: Geometry) -> None:
    """Test that splitting a geometry and joining the parts restores it."""
    keys1, keys2 = [3, 0, 7, 12], [k for k in range(15) if k not in (3, 0, 7, 12)]
    parts = [autogeom.select_atoms(propyl_oxirane, ks) for ks in (keys1, keys2)]
    joined = autogeom.join(parts).relabel_atoms(np.argsort(keys1 + keys2).tolist())
    assert joined.symbols == propyl_oxirane.symbols
    assert np.array_equal(joined.coordinates, propyl_oxirane.coordinates)


def test__remove_atoms_complements_select(propyl_oxirane: Geometry) -> None:
    """Test that removing atoms selects the rest, in order."""
    removed = autogeom.remove_atoms(propyl_oxirane, {2, 5})
    kept = autogeom.select_atoms(
        propyl_oxirane, [k for k in range(15) if k not in (2, 5)]
    )
    assert removed == kept
    assert removed.atom_count == propyl_oxirane.atom_count - 2


def test__select_atoms_charge_and_spin() -> None:
    """Test default and explicit charge and spin of a selection."""
    methane = Geometry(
        symbols=["C", "H", "H", "H", "H"],
        coordinates=np.zeros((5, 3)),
        charge=0,
        spin=0,
    )
    methyl = autogeom.remove_atoms(methane, [4])
    assert (methyl.charge, methyl.spin) == (0, 1)
    cation = autogeom.remove_atoms(methane, [4], charge=1, spin=0)
    assert (cation.charge, cation.spin) == (1, 0)
    ch2 = autogeom.remove_atoms(methane, [3, 4])
    assert (ch2.charge, ch2.spin) == (0, 0)
    # Joining two methyl radicals sums the spins (a triplet)
    assert autogeom.join([methyl, methyl]).spin == 2


# Rotational analysis and alignment


def test__alignment_rotation_recovers_rotation(propyl_oxirane: Geometry) -> None:
    """Test that the alignment rotation of a moved geometry is the rotation."""
    found = autogeom.alignment_rotation(_rigid_motion(propyl_oxirane), propyl_oxirane)
    assert found.approx_equal(ROT, atol=1e-8)


@pytest.mark.parametrize("axis", [(0, 1, 0), (1, -2, 3), (0, 0, -1)])
def test__axis_alignment_rotation(
    propyl_oxirane: Geometry, axis: tuple[float, float, float]
) -> None:
    """Test that the atom vector ends up along the axis, from any orientation."""
    unit = np.array(axis) / np.linalg.norm(axis)
    for seed in range(3):
        geo = autogeom.rotate(propyl_oxirane, Rotation.random(random_state=seed))
        rot = autogeom.axis_alignment_rotation(geo, (0, 4), axis=axis)
        aligned = autogeom.rotate(geo, rot, origin=geo.coordinates[0])
        vec = aligned.coordinates[4] - aligned.coordinates[0]
        assert np.allclose(vec / np.linalg.norm(vec), unit)
        assert np.allclose(aligned.coordinates[0], geo.coordinates[0])
        assert np.allclose(
            autogeom.distance_matrix(aligned), autogeom.distance_matrix(geo)
        )
        # Already aligned: nothing to do
        again = autogeom.axis_alignment_rotation(aligned, (0, 4), axis=axis)
        assert again.magnitude() < 1e-6


@pytest.mark.parametrize(
    ("keys", "match"), [((0,), "two distinct"), ((1, 1), "two distinct")]
)
def test__axis_alignment_rotation_raises(
    water: Geometry, keys: tuple[int, ...], match: str
) -> None:
    """Test that the atoms must be two distinct atoms."""
    with pytest.raises(ValueError, match=match):
        autogeom.axis_alignment_rotation(water, keys)


def test__alignment_rotation_mismatch_raises(
    water: Geometry, peroxide: Geometry
) -> None:
    """Test that geometries with different atoms are rejected."""
    with pytest.raises(ValueError, match="do not match"):
        autogeom.alignment_rotation(water, peroxide)


def test__inertial_frame_rotation(propyl_oxirane: Geometry) -> None:
    """Test that the inertial frame diagonalizes the inertia tensor."""
    geo = _rigid_motion(propyl_oxirane)
    moments, axes = autogeom.principal_axes(geo)
    assert np.isclose(np.linalg.det(axes), 1.0)
    assert np.all(np.diff(moments) >= 0)

    rot = autogeom.inertial_frame_rotation(geo)
    framed = autogeom.rotate(autogeom.center(geo), rot)
    assert np.allclose(autogeom.inertia_tensor(framed), np.diag(moments))
    # The moments don't depend on the starting orientation
    assert np.allclose(moments, autogeom.principal_axes(propyl_oxirane)[0])


# Measurements


def test__measurements_invariant_under_rigid_motion(propyl_oxirane: Geometry) -> None:
    """Test that distances and angles don't change under rigid motions."""
    moved = _rigid_motion(propyl_oxirane)
    for fn, keys in [
        (autogeom.distance, [0, 5]),
        (autogeom.angle, [0, 1, 2]),
        (autogeom.dihedral, [0, 1, 2, 3]),
    ]:
        assert np.isclose(fn(moved, keys), fn(propyl_oxirane, keys))
    assert np.allclose(
        autogeom.distance_matrix(moved), autogeom.distance_matrix(propyl_oxirane)
    )


def test__dihedral_reflection_flips_sign(propyl_oxirane: Geometry) -> None:
    """Test that a reflection flips the sign of a dihedral angle."""
    mirrored = autogeom.reflect(propyl_oxirane, [1.0, -2.0, 0.5])
    keys = [0, 1, 2, 3]
    assert np.isclose(
        autogeom.dihedral(mirrored, keys), -autogeom.dihedral(propyl_oxirane, keys)
    )


def test__dihedral_matches_dihedrals_table(peroxide: Geometry) -> None:
    """Test that the dihedral of specific atoms agrees with the table."""
    amat = autogeom.adjacency_matrix(peroxide)
    table = autogeom.dihedrals(peroxide, amat)
    assert np.isclose(autogeom.dihedral(peroxide, [0, 1, 2, 3]), table[0, -1])
    assert np.isclose(autogeom.dihedral(peroxide, [3, 2, 1, 0]), table[0, -1])


@pytest.mark.parametrize(
    ("fn", "keys"),
    [
        (autogeom.distance, [0]),
        (autogeom.angle, [0, 1]),
        (autogeom.dihedral, [0, 1, 2]),
    ],
)
def test__measurement_wrong_key_count_raises(
    peroxide: Geometry, fn: Callable[..., float], keys: list[int]
) -> None:
    """Test that the wrong number of atom indices is rejected."""
    with pytest.raises(ValueError, match="atom indices"):
        fn(peroxide, keys)


# Twisting groups about bonds (propyl_oxirane: chain C0-C1-C2-C3, ring C3-O4-C5)


@pytest.mark.parametrize("angle", [-120.0, 37.0, 180.0])
def test__twist(propyl_oxirane: Geometry, angle: float) -> None:
    """Test that a twist changes only the dihedral about the bond, by the angle."""
    geo = propyl_oxirane
    twisted = autogeom.twist(geo, (1, 2), angle, degrees=True)
    before = autogeom.dihedral(geo, [0, 1, 2, 3], degrees=True)
    after = autogeom.dihedral(twisted, [0, 1, 2, 3], degrees=True)
    assert np.isclose((after - before - angle + 180) % 360, 180)
    # The side of atom 1 doesn't move; the side of atom 2 moves rigidly
    fixed = [0, 1, *np.flatnonzero(autogeom.adjacency_matrix(geo)[0]).tolist()]
    assert np.allclose(twisted.coordinates[fixed], geo.coordinates[fixed])
    moving = [2, 3, 4, 5]
    assert np.allclose(
        autogeom.distance_matrix(autogeom.select_atoms(twisted, moving)),
        autogeom.distance_matrix(autogeom.select_atoms(geo, moving)),
    )
    # Twisting back is a roundtrip
    back = autogeom.twist(twisted, (1, 2), -angle, degrees=True)
    assert np.allclose(back.coordinates, geo.coordinates)


def test__twist_angle_roundtrip(propyl_oxirane: Geometry) -> None:
    """Test that twisting by `twist_angle` turns a direction onto the target."""
    geo = propyl_oxirane
    axis = geo.coordinates[2] - geo.coordinates[1]
    axis /= np.linalg.norm(axis)
    vec = geo.coordinates[3] - geo.coordinates[2]
    for seed in range(5):
        target = np.random.default_rng(seed).normal(size=3)
        angle = autogeom.twist_angle(geo, (1, 2), vec, target)
        twisted = autogeom.twist(geo, (1, 2), angle)
        new = twisted.coordinates[3] - twisted.coordinates[2]
        new_perp = new - np.dot(new, axis) * axis
        target_perp = target - np.dot(target, axis) * axis
        assert np.allclose(
            new_perp / np.linalg.norm(new_perp),
            target_perp / np.linalg.norm(target_perp),
        )
    assert np.isclose(autogeom.twist_angle(geo, (1, 2), vec, vec), 0)


@pytest.mark.parametrize(
    ("keys", "match"),
    [((3, 4), "in a ring"), ((0, 3), "not bonded"), ((1, 1), "two distinct")],
)
def test__twist_raises(propyl_oxirane: Geometry, keys: tuple, match: str) -> None:
    """Test that ring bonds, non-bonds, and non-pairs can't be twisted."""
    with pytest.raises(ValueError, match=match):
        autogeom.twist(propyl_oxirane, keys, 1.0)


def test__twist_angle_parallel_raises(propyl_oxirane: Geometry) -> None:
    """Test that a direction along the bond has no twist angle."""
    axis = propyl_oxirane.coordinates[2] - propyl_oxirane.coordinates[1]
    with pytest.raises(ValueError, match="parallel"):
        autogeom.twist_angle(propyl_oxirane, (1, 2), axis, [1, 0, 0])
