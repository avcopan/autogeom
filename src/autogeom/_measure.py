"""Properties, measurements, and internal coordinates of a geometry."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import spatial
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from . import rd
from ._geometry import Geometry
from .types import FloatArray

__all__ = [
    "adjacency_matrix",
    "angle",
    "angles",
    "bonds",
    "center_of_mass",
    "dihedral",
    "dihedrals",
    "distance",
    "distance_keys",
    "distance_matrix",
    "inertia_tensor",
    "inertial_frame_rotation",
    "principal_axes",
]

# Properties
_FLOOD_FILL_STEP = 0.05


def center_of_mass(geo: Geometry) -> FloatArray:
    """Calculate the center of mass.

    Args:
        geo: Geometry.

    Returns:
        Center of mass coordinates (the origin for an empty geometry).
    """
    masses = np.asarray(geo.masses)
    if not masses.size:
        return np.zeros(3)
    return masses @ geo.coordinates / np.sum(masses)


def distance_matrix(geo: Geometry) -> np.ndarray:
    """Calculate the interatomic distance matrix.

    Args:
        geo: Geometry.

    Returns:
        Distance matrix, shape ``(N, N)``.
    """
    return spatial.distance.cdist(geo.coordinates, geo.coordinates)


def distance_keys(geo: Geometry) -> np.ndarray:
    """Build a sorted distance descriptor.

    Each unique atom pair gives a row ``[z_low, z_high, distance]``. Rows are
    sorted by atomic numbers, then by distance.

    Args:
        geo: Geometry.

    Returns:
        Descriptor array, shape ``(npairs, 3)``.
    """
    z = np.asarray(geo.atomic_numbers)
    dmat = distance_matrix(geo)
    iu, ju = np.triu_indices(geo.atom_count, k=1)

    pairs = np.sort(np.stack([z[iu], z[ju]], axis=1), axis=1)
    key = np.column_stack([pairs.astype(float), dmat[iu, ju]])

    return _sort_rows(key)


def adjacency_matrix(
    geo: Geometry,
    *,
    sigma: float = 1.3,
    flood_fill: bool = False,
    enforce_valence: bool = False,
) -> np.ndarray:
    """Determine connectivity from covalent radii.

    Two atoms are bonded if their distance is less than `sigma` times the sum of
    their covalent radii.

    Args:
        geo: Geometry.
        sigma: Scaling factor for the sum of covalent radii.
        flood_fill: If `True`, increase `sigma` in steps of 0.05 until the
            molecule is a single fragment, or until no more bonds can be added.
        enforce_valence: If `True`, cap the bonds to each atom at
            `autogeom.rd.max_bond_count`. Bonds are accepted shortest first,
            relative to the sum of covalent radii, so the result does not depend
            on atom order.

    Returns:
        Symmetric 0/1 adjacency matrix, shape ``(N, N)``.
    """
    natms = geo.atom_count
    radii = np.asarray(geo.covalent_radii)
    iu, ju = np.triu_indices(natms, k=1)
    ratios = distance_matrix(geo)[iu, ju] / (radii[iu] + radii[ju])
    order = np.argsort(ratios, kind="stable")
    iu, ju, ratios = iu[order], ju[order], ratios[order]

    max_bonds = [rd.max_bond_count(s) for s in geo.symbols] if enforce_valence else None

    while True:
        amat = np.zeros((natms, natms), dtype=int)
        ncand = int(np.searchsorted(ratios, sigma, side="left"))
        if max_bonds is None:
            amat[iu[:ncand], ju[:ncand]] = amat[ju[:ncand], iu[:ncand]] = 1
        else:
            counts = np.zeros(natms, dtype=int)
            for i, j in zip(iu[:ncand], ju[:ncand], strict=True):
                if counts[i] < max_bonds[i] and counts[j] < max_bonds[j]:
                    amat[i, j] = amat[j, i] = 1
                    counts[[i, j]] += 1

        if not flood_fill or ncand == len(ratios):
            break

        n_components, _ = connected_components(amat, directed=False)
        if n_components <= 1:
            break
        sigma += _FLOOD_FILL_STEP

    return amat


def inertia_tensor(geo: Geometry) -> np.ndarray:
    """Calculate the inertia tensor about the center of mass.

    Args:
        geo: Geometry.

    Returns:
        Inertia tensor, shape ``(3, 3)`` (zero for an empty geometry).

    Example:
        >>> import numpy as np
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> tensor = autogeom.inertia_tensor(geo)
        >>> tensor.shape
        (3, 3)
        >>> bool(np.allclose(tensor, tensor.T))
        True
    """
    masses = np.asarray(geo.masses)
    coords = geo.coordinates - center_of_mass(geo)
    mr2 = np.einsum("i,ij,ij->", masses, coords, coords)
    return mr2 * np.eye(3) - np.einsum("i,ij,ik->jk", masses, coords, coords)


def principal_axes(geo: Geometry) -> tuple[FloatArray, FloatArray]:
    """Calculate the principal moments and axes of inertia.

    Args:
        geo: Geometry.

    Returns:
        The principal moments, in ascending order, and the principal axes, as
        the columns of a ``(3, 3)`` rotation matrix (a right-handed frame).
    """
    moments, axes = np.linalg.eigh(inertia_tensor(geo))
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return moments, axes


def inertial_frame_rotation(geo: Geometry) -> Rotation:
    """Find the rotation into the principal axis frame.

    After ``rotate(center(geo), rot)``, the principal axes with the smallest to
    largest moments lie along x, y, and z, so the inertia tensor is diagonal.

    Args:
        geo: Geometry.

    Returns:
        The rotation.
    """
    _, axes = principal_axes(geo)
    return Rotation.from_matrix(axes.T)


# Measurements
def distance(geo: Geometry, keys: Sequence[int]) -> float:
    """Measure the distance between two atoms.

    Args:
        geo: Geometry.
        keys: Indices of the two atoms.

    Returns:
        The distance, in Angstroms.
    """
    r1, r2 = _key_coordinates(geo, keys, 2)
    return float(np.linalg.norm(r2 - r1))


def angle(geo: Geometry, keys: Sequence[int], *, degrees: bool = False) -> float:
    """Measure the angle between three atoms.

    Args:
        geo: Geometry.
        keys: Indices of the three atoms, with the vertex in the middle.
        degrees: If `True`, return degrees instead of radians.

    Returns:
        The angle, between 0 and pi.

    Example:
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> autogeom.angle(geo, [1, 0, 2], degrees=True)
        90.0
    """
    r1, r2, r3 = _key_coordinates(geo, keys, 3)
    v1, v2 = r1 - r2, r3 - r2
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    ang = float(np.arccos(np.clip(cos, -1.0, 1.0)))
    return float(np.degrees(ang)) if degrees else ang


def dihedral(geo: Geometry, keys: Sequence[int], *, degrees: bool = False) -> float:
    """Measure the dihedral angle between four atoms.

    Looking down the 2-3 bond, the angle is positive when atom 4 is rotated
    clockwise from atom 1.

    Args:
        geo: Geometry.
        keys: Indices of the four atoms.
        degrees: If `True`, return degrees instead of radians.

    Returns:
        The dihedral angle, between -pi and pi.

    Example:
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["H", "O", "O", "H"],
        ...     coordinates=[[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> autogeom.dihedral(geo, [0, 1, 2, 3], degrees=True)
        90.0
    """
    r1, r2, r3, r4 = _key_coordinates(geo, keys, 4)
    b1, b2, b3 = r2 - r1, r3 - r2, r4 - r3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2))
    ang = float(np.arctan2(y, x))
    return float(np.degrees(ang)) if degrees else ang


# Internal coordinates
def bonds(geo: Geometry, amat: ArrayLike) -> np.ndarray:
    """List bond distances.

    Args:
        geo: Geometry.
        amat: Adjacency matrix.

    Returns:
        Sorted rows ``[z1, z2, distance]`` with ``z1 <= z2``, shape
        ``(nbonds, 3)``.

    Example:
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> b = autogeom.bonds(geo, autogeom.adjacency_matrix(geo))
        >>> b.shape
        (2, 3)
        >>> [tuple(float(x) for x in row) for row in b]
        [(1.0, 8.0, 1.0), (1.0, 8.0, 1.0)]
    """
    zs = np.asarray(geo.atomic_numbers)
    xyz = geo.coordinates
    i, j = np.nonzero(np.triu(_as_adjacency(amat, geo.atom_count)))

    dist = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    rows = np.column_stack([np.minimum(zs[i], zs[j]), np.maximum(zs[i], zs[j]), dist])
    return _sort_rows(rows.reshape(-1, 3))


def angles(geo: Geometry, amat: ArrayLike) -> np.ndarray:
    """List bond angles.

    Args:
        geo: Geometry.
        amat: Adjacency matrix.

    Returns:
        Sorted rows ``[z1, z2, z3, theta]`` with `theta` in radians, `z2` the
        central atom, and ``z1 <= z3``, shape ``(nangles, 4)``.

    Example:
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> a = autogeom.angles(geo, autogeom.adjacency_matrix(geo))
        >>> a.shape
        (1, 4)
        >>> round(float(a[0][-1]), 6)
        1.570796
    """
    zs = np.asarray(geo.atomic_numbers)
    xyz = geo.coordinates
    nbrs = _neighbors(_as_adjacency(amat, geo.atom_count))

    triples = [
        (i, j, k)
        for j, nbr in enumerate(nbrs)
        for pos, i in enumerate(nbr)
        for k in nbr[pos + 1 :]
    ]
    i, j, k = np.array(triples, dtype=int).reshape(-1, 3).T

    v1 = xyz[i] - xyz[j]
    v2 = xyz[k] - xyz[j]
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos_theta = np.clip(np.einsum("nt,nt->n", v1, v2) / norms, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    rows = np.column_stack(
        [np.minimum(zs[i], zs[k]), zs[j], np.maximum(zs[i], zs[k]), theta]
    )
    return _sort_rows(rows.reshape(-1, 4))


def dihedrals(geo: Geometry, amat: ArrayLike) -> np.ndarray:
    """List dihedral angles.

    Args:
        geo: Geometry.
        amat: Adjacency matrix.

    Returns:
        Sorted rows ``[z1, z2, z3, z4, phi]`` with `phi` in radians, shape
        ``(ndihedrals, 5)``. Each row is oriented so that
        ``(z1, z2, z3, z4) <= (z4, z3, z2, z1)``, which leaves `phi` unchanged.

    Example:
        >>> import numpy as np
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["H", "O", "O", "H"],
        ...     coordinates=[[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> amat = np.array(
        ...     [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
        ... )
        >>> d = autogeom.dihedrals(geo, amat)
        >>> d.shape
        (1, 5)
        >>> [float(x) for x in d[0][:4]]
        [1.0, 8.0, 8.0, 1.0]
        >>> round(float(d[0][-1]), 6)
        1.570796
    """
    zs = np.asarray(geo.atomic_numbers)
    xyz = geo.coordinates
    amat = _as_adjacency(amat, geo.atom_count)
    nbrs = _neighbors(amat)

    # Enumerate each dihedral once, about its unordered central bond (j, k)
    quads = [
        (i, j, k, l)
        for j, k in zip(*np.nonzero(np.triu(amat)), strict=True)
        for i in nbrs[j]
        if i != k
        for l in nbrs[k]  # noqa: E741
        if l not in (i, j)
    ]
    i, j, k, l = np.array(quads, dtype=int).reshape(-1, 4).T  # noqa: E741

    b1 = xyz[j] - xyz[i]
    b2 = xyz[k] - xyz[j]
    b3 = xyz[l] - xyz[k]

    nb2 = np.linalg.norm(b2, axis=1, keepdims=True)
    nb2 = np.where(nb2 == 0.0, 1.0, nb2)
    ub2 = b2 / nb2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    x = np.einsum("nt,nt->n", n1, n2)
    y = np.einsum("nt,nt->n", np.cross(n1, n2), ub2)
    phi = np.arctan2(y, x)

    zquads = zs[np.column_stack([i, j, k, l]).astype(int)].reshape(-1, 4)
    zquads = np.array(
        [min(tuple(zq), tuple(zq[::-1])) for zq in zquads.tolist()], dtype=float
    ).reshape(-1, 4)
    return _sort_rows(np.column_stack([zquads, phi]))


# Helpers
def _key_coordinates(geo: Geometry, keys: Sequence[int], count: int) -> np.ndarray:
    """Get the coordinates of `count` atoms, checking the number of keys."""
    keys = list(keys)
    if len(keys) != count:
        msg = f"Expected {count} atom indices, got {keys}."
        raise ValueError(msg)
    return geo.coordinates[keys]


def _as_adjacency(amat: ArrayLike, natms: int) -> np.ndarray:
    """Convert to a boolean adjacency matrix, checking its shape."""
    amat = np.asarray(amat).astype(bool)
    if amat.shape != (natms, natms):
        msg = f"Expected adjacency matrix of shape {(natms, natms)}, got {amat.shape}."
        raise ValueError(msg)
    return amat


def _neighbors(amat: np.ndarray) -> list[list[int]]:
    """Get the sorted neighbor indices of each atom."""
    return [np.flatnonzero(row).tolist() for row in amat]


def _sort_rows(arr: np.ndarray) -> np.ndarray:
    """Sort the rows of a 2D array lexicographically."""
    return arr[np.lexsort(arr.T[::-1])]
