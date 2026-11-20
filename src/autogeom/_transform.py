"""Rigid-body transformations, twists, and alignment."""

from collections.abc import Collection, Sequence

import numpy as np
import pint
from numpy.typing import ArrayLike
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from ._geometry import Geometry
from ._measure import adjacency_matrix, center_of_mass
from .types import FloatArray

__all__ = [
    "alignment_rotation",
    "axis_alignment_rotation",
    "center",
    "place",
    "reflect",
    "rotate",
    "translate",
    "twist",
    "twist_angle",
]


def translate(
    geo: Geometry,
    arr: ArrayLike,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Translate a geometry.

    Args:
        geo: Geometry.
        arr: Translation vector, or one vector per selected atom.
        keys: Atoms to translate. If `None`, translate all atoms.
        in_place: If `True`, modify `geo` instead of returning a copy.

    Returns:
        The translated geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    mask = slice(None) if keys is None else list(keys)
    geo.coordinates[mask] = np.add(geo.coordinates[mask], arr)
    return geo


def reflect(
    geo: Geometry,
    normal: ArrayLike,
    *,
    origin: ArrayLike = (0.0, 0.0, 0.0),
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Reflect a geometry across a plane.

    Args:
        geo: Geometry.
        normal: Normal vector of the reflection plane.
        origin: A point on the reflection plane.
        keys: Atoms to reflect. If `None`, reflect all atoms.
        in_place: If `True`, modify `geo` instead of returning a copy.

    Returns:
        The reflected geometry.

    Raises:
        ValueError: If `normal` is a zero vector.
    """
    normal = np.asarray(normal, dtype=float)
    norm2 = np.dot(normal, normal)
    if not norm2 > 0:
        msg = f"Reflection plane normal must be a non-zero vector, got {normal}."
        raise ValueError(msg)

    geo = geo if in_place else geo.model_copy(deep=True)
    proj = np.outer(normal, normal) / norm2
    mask = slice(None) if keys is None else list(keys)
    origin = np.asarray(origin, dtype=float)
    coords = geo.coordinates[mask] - origin
    geo.coordinates[mask] = coords - 2 * coords @ proj + origin
    return geo


def rotate(
    geo: Geometry,
    rot: Rotation,
    *,
    origin: ArrayLike = (0.0, 0.0, 0.0),
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Rotate a geometry about a point.

    Args:
        geo: Geometry.
        rot: Rotation to apply.
        origin: Point to rotate about.
        keys: Atoms to rotate. If `None`, rotate all atoms.
        in_place: If `True`, modify `geo` instead of returning a copy.

    Returns:
        The rotated geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    mask = slice(None) if keys is None else list(keys)
    origin = np.asarray(origin, dtype=float)
    geo.coordinates[mask] = rot.apply(geo.coordinates[mask] - origin) + origin
    return geo


def twist(
    geo: Geometry, keys: Sequence[int], angle: float, *, degrees: bool = False
) -> Geometry:
    """Rotate a group about a bond, e.g. a methyl or amine group.

    Everything on the side of atom `j` of the bond ``(i, j)`` rotates about the
    axis from `i` to `j`, by the right-hand rule (thumb from `i` to `j`). Bond
    lengths and angles are kept; only the dihedral angles about the bond change.

    Args:
        geo: Geometry.
        keys: The bond, ``(i, j)``. The side of `j` moves.
        angle: Angle to rotate by.
        degrees: If `True`, `angle` is in degrees instead of radians.

    Returns:
        The twisted geometry.

    Raises:
        ValueError: If `i` and `j` aren't bonded, or the bond is in a ring.

    Example:
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["H", "O", "O", "H"],
        ...     coordinates=[[1, 0, 0], [0, 0, 0], [0, 0, 1.4], [1, 0, 1.4]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> geo = autogeom.twist(geo, (1, 2), 90, degrees=True)
        >>> round(autogeom.dihedral(geo, [0, 1, 2, 3], degrees=True), 6)
        90.0
    """
    i, j = _bond_keys(keys)
    moving = _bond_side(geo, i, j)
    axis = geo.coordinates[j] - geo.coordinates[i]
    angle = float(np.radians(angle)) if degrees else angle
    rot = Rotation.from_rotvec(angle * axis / np.linalg.norm(axis))
    return rotate(geo, rot, origin=geo.coordinates[j], keys=moving)


def twist_angle(
    geo: Geometry,
    keys: Sequence[int],
    vec: ArrayLike,
    target: ArrayLike,
    *,
    degrees: bool = False,
) -> float:
    """Find the twist about a bond that turns one direction onto another.

    Both directions are projected onto the plane perpendicular to the bond, and
    the result is the signed angle between the projections, so that
    ``twist(geo, keys, twist_angle(geo, keys, vec, target))`` turns `vec` (as
    part of the moving group) to point as close as possible to `target`.

    Args:
        geo: Geometry.
        keys: The bond, ``(i, j)``, as for `twist`.
        vec: Direction attached to the moving group, e.g. a bond direction.
        target: Direction to turn it toward.
        degrees: If `True`, return the angle in degrees instead of radians.

    Returns:
        The angle, between -pi and pi (or -180 and 180 degrees).

    Raises:
        ValueError: If `vec` or `target` is parallel to the bond.
    """
    i, j = _bond_keys(keys)
    axis = geo.coordinates[j] - geo.coordinates[i]
    axis = axis / np.linalg.norm(axis)
    vec_perp, target_perp = (
        np.asarray(v, dtype=float) - np.dot(v, axis) * axis for v in (vec, target)
    )
    for name, v in (("vec", vec_perp), ("target", target_perp)):
        if np.isclose(np.linalg.norm(v), 0.0):
            msg = f"`{name}` is parallel to the bond {keys}."
            raise ValueError(msg)
    y = np.dot(np.cross(vec_perp, target_perp), axis)
    ang = float(np.arctan2(y, np.dot(vec_perp, target_perp)))
    return float(np.degrees(ang)) if degrees else ang


def _bond_keys(keys: Sequence[int]) -> tuple[int, int]:
    """Check that `keys` is a pair of distinct atoms."""
    if len(keys) != 2 or keys[0] == keys[1]:  # noqa: PLR2004
        msg = f"Expected a bond of two distinct atoms, got {keys}."
        raise ValueError(msg)
    return int(keys[0]), int(keys[1])


def _bond_side(geo: Geometry, i: int, j: int) -> list[int]:
    """List the atoms on the side of `j` of the bond ``(i, j)``, without `j`."""
    amat = adjacency_matrix(geo)
    if not amat[i, j]:
        msg = f"Atoms {i} and {j} are not bonded."
        raise ValueError(msg)
    amat[i, j] = amat[j, i] = 0
    _, labels = connected_components(amat, directed=False)
    if labels[i] == labels[j]:
        msg = f"The bond ({i}, {j}) is in a ring, so its sides can't twist."
        raise ValueError(msg)
    return [k for k in np.flatnonzero(labels == labels[j]).tolist() if k != j]


def center(geo: Geometry, *, in_place: bool = False) -> Geometry:
    """Translate a geometry so that its center of mass is at the origin.

    Args:
        geo: Geometry.
        in_place: If `True`, modify `geo` instead of returning a copy.

    Returns:
        The centered geometry.
    """
    return translate(geo, -center_of_mass(geo), in_place=in_place)


def alignment_rotation(geo: Geometry, ref: Geometry) -> Rotation:
    """Find the rotation that best aligns a reference geometry onto a geometry.

    The rotation is found by the Kabsch algorithm, after moving both centers of
    mass to the origin. Atoms are matched by index.

    Args:
        geo: Target geometry.
        ref: Reference geometry, with the same atoms as `geo`.

    Returns:
        The rotation `rot` minimizing the RMSD between ``rotate(center(ref), rot)``
        and ``center(geo)``.

    Raises:
        ValueError: If the geometries have different symbols.

    Example:
        >>> from scipy.spatial.transform import Rotation
        >>> import autogeom
        >>> ref = autogeom.Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> rot = Rotation.from_euler("z", 30, degrees=True)
        >>> geo = autogeom.rotate(ref, rot)
        >>> round(float(autogeom.alignment_rotation(geo, ref).magnitude()), 6)
        0.523599
    """
    if geo.symbols != ref.symbols:
        msg = f"Symbols do not match: {geo.symbols} != {ref.symbols}"
        raise ValueError(msg)

    # The second return value is the root-sum-square distance
    return Rotation.align_vectors(
        center(geo).coordinates, center(ref).coordinates, weights=geo.masses
    )[0]


def axis_alignment_rotation(
    geo: Geometry, keys: Sequence[int], axis: ArrayLike = (0.0, 0.0, 1.0)
) -> Rotation:
    """Find the smallest rotation pointing the vector between two atoms along an axis.

    Use it to put a bond or other molecular axis along a coordinate axis, e.g.
    before spinning the geometry about that axis.

    Args:
        geo: Geometry.
        keys: Two atoms, ``(i, j)``. The vector from `i` to `j` is aligned.
        axis: Direction to point it in.

    Returns:
        The rotation, to apply with `rotate` (about atom `i` to keep it in
        place, e.g. ``rotate(geo, rot, origin=geo.coordinates[i])``).

    Raises:
        ValueError: If `keys` isn't two distinct atoms at different positions.

    Example:
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["H", "H"], coordinates=[[0, 0, 0], [1, 0, 0]], charge=0, spin=0
        ... )
        >>> rot = autogeom.axis_alignment_rotation(geo, [0, 1], axis=[0, 1, 0])
        >>> autogeom.rotate(geo, rot).coordinates.round(6).tolist()
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    """
    if len(keys) != 2 or keys[0] == keys[1]:  # noqa: PLR2004
        msg = f"Expected two distinct atoms, got {keys}."
        raise ValueError(msg)
    i, j = keys
    vec = geo.coordinates[j] - geo.coordinates[i]
    if np.isclose(np.linalg.norm(vec), 0.0):
        msg = f"Atoms {i} and {j} are at the same position."
        raise ValueError(msg)
    # For a single pair of vectors, scipy returns the smallest rotation
    return Rotation.align_vectors([axis], [vec])[0]


def place(
    geo: Geometry,
    ref: Geometry,
    keys: Sequence[int],
    distance: float | pint.Quantity,
    *,
    direction: ArrayLike | None = None,
) -> Geometry:
    """Move a geometry next to a reference geometry, e.g. to build a complex.

    Atom `j` of `geo` is put `distance` from atom `i` of `ref` along `direction`,
    and `geo` is turned rigidly so its other atoms point away from `ref`. Only
    the distance to atom `i` is set, so other atoms of `ref` may clash with
    `geo`.

    Args:
        geo: Geometry to move.
        ref: Reference geometry, which stays fixed.
        keys: The atoms ``(j, i)``: `j` of `geo` and `i` of `ref`.
        distance: Distance between the atoms, in Angstroms or as a `pint`
            quantity with length units.
        direction: Direction from atom `i` to atom `j`. If `None`, the normal
            to the best-fit plane of `ref`, on the side of atom `i`, or for a
            linear `ref`, the direction from its centroid to atom `i`.

    Returns:
        The moved copy of `geo`; join it to `ref` with `join`.

    Raises:
        ValueError: If `direction` is zero, or is `None` and can't be chosen
            (`ref` is an atom, or atom `i` is the center of a linear `ref`).

    Example:
        >>> import autogeom
        >>> water = autogeom.Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> h2 = autogeom.Geometry(
        ...     symbols=["H", "H"],
        ...     coordinates=[[0, 0, 0], [0, 0.74, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> h2 = autogeom.place(h2, water, (0, 0), 2.5, direction=[0, 0, 1])
        >>> h2.coordinates.round(6).tolist()
        [[0.0, 0.0, 2.5], [0.0, 0.0, 3.24]]
        >>> complex_ = autogeom.join([water, h2])
        >>> round(autogeom.distance(complex_, (0, 3)), 6)
        2.5
    """
    j, i = (int(k) for k in keys)
    if isinstance(distance, pint.Quantity):
        distance = distance.m_as("angstrom")
    vec = _place_direction(ref, i) if direction is None else _unit(direction)

    xyz = geo.coordinates
    rest = np.delete(xyz, j, axis=0)
    rest_vec = rest.mean(axis=0) - xyz[j] if len(rest) else np.zeros(3)
    if not np.isclose(np.linalg.norm(rest_vec), 0.0):
        # For a single pair of vectors, scipy returns the smallest rotation
        rot = Rotation.align_vectors([vec], [rest_vec])[0]
        geo = rotate(geo, rot, origin=xyz[j])
    return translate(geo, ref.coordinates[i] + distance * vec - geo.coordinates[j])


# Singular values below this (in Angstroms) count as zero when fitting a plane
_PLANE_TOL = 1e-3


def _place_direction(ref: Geometry, i: int) -> FloatArray:
    """Choose the default direction for `place` from atom `i` of `ref`."""
    xyz = ref.coordinates
    centroid = xyz.mean(axis=0)
    _, sing, vt = np.linalg.svd(xyz - centroid)
    out = xyz[i] - centroid
    if len(sing) > 1 and sing[1] > _PLANE_TOL:
        normal = vt[-1]
        return -normal if np.dot(out, normal) < 0 else normal
    if np.isclose(np.linalg.norm(out), 0.0):
        msg = f"Can't choose a direction from atom {i} of `ref`, so pass `direction`."
        raise ValueError(msg)
    return out / np.linalg.norm(out)


def _unit(vec: ArrayLike) -> FloatArray:
    """Normalize a vector, rejecting a zero vector."""
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if np.isclose(norm, 0.0):
        msg = "Expected a non-zero `direction`."
        raise ValueError(msg)
    return vec / norm
