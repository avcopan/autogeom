"""Vibrational analysis from a Cartesian Hessian."""

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._geometry import Geometry, RotorKind, rotor_kind
from .types import FloatArray, FloatArrayField
from .units import ureg

_FREQ_FACTOR = np.sqrt((1 * ureg.hartree / ureg.bohr**2 / ureg.amu).m_as("1/s**2")) / (
    2 * np.pi * (1 * ureg.speed_of_light).m_as("cm/s")
)
"""Converts √(Hartree/(Bohr² amu)) to cm⁻¹."""


class PESPoint(BaseModel):
    """Local quadratic model of the potential energy surface at a geometry.

    Attributes:
        geometry: Molecular geometry with N atoms.
        hessian: ``(3N, 3N)`` Hessian in Hartree/Bohr², symmetrized on input.
        gradient: ``(N, 3)`` gradient in Hartree/Bohr.
        energy: Energy in Hartree.
        rotor_kind: Atom, linear or nonlinear; it sets the number of rotations
            projected out. The default is `rotor_kind` of the geometry.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    geometry: Geometry
    hessian: FloatArrayField
    gradient: FloatArrayField | None = None
    energy: float | None = None
    rotor_kind: RotorKind = Field(
        default_factory=lambda data: rotor_kind(data["geometry"])
    )

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        n = self.geometry.atom_count

        hess_shape = (3 * n, 3 * n)
        if self.hessian.shape != hess_shape:
            msg = f"Hessian shape {self.hessian.shape} should be {hess_shape}."
            raise ValueError(msg)
        if not np.isfinite(self.hessian).all():
            msg = "Hessian has non-finite values."
            raise ValueError(msg)

        if self.gradient is not None:
            grad_shape = (n, 3)
            if self.gradient.shape != grad_shape:
                msg = f"Gradient shape {self.gradient.shape} should be {grad_shape}."
                raise ValueError(msg)
            if not np.isfinite(self.gradient).all():
                msg = "Gradient has non-finite values."
                raise ValueError(msg)

        # The model is frozen, so bypass Pydantic's __setattr__
        object.__setattr__(self, "hessian", (self.hessian + self.hessian.T) / 2)
        return self


class VibrationalAnalysis(BaseModel):
    """Normal modes of the Hessian, with translations and rotations projected out.

    Attributes:
        frequencies: Frequencies in cm⁻¹, imaginary ones negative, ascending.
        eigenvalues: Eigenvalues of the projected, mass-weighted Hessian, in
            Hartree/(Bohr² amu), ascending.
        modes: ``(n_vib, N, 3)`` Cartesian displacements, each normalized to 1.
        eigenvectors: ``(n_vib, 3N)`` orthonormal eigenvectors of the projected,
            mass-weighted Hessian, one per row.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    frequencies: FloatArrayField
    eigenvalues: FloatArrayField
    modes: FloatArrayField
    eigenvectors: FloatArrayField


def _external_basis(
    geo: Geometry, masses: Sequence[float], kind: RotorKind
) -> FloatArray:
    """Orthonormal complement of the mass-weighted translations and rotations.

    Returns:
        The ``(3N, 3N - 3 - kind.rotation_count)`` complement.
    """
    m = np.asarray(masses, dtype=np.float64)
    n = len(m)
    if not n:
        return np.zeros((0, 0))

    r = geo.coordinates - m @ geo.coordinates / m.sum()
    sqrt_m = np.sqrt(m)[:, None, None]
    eye = np.eye(3)
    trans = np.broadcast_to(eye, (n, 3, 3))
    rots = np.cross(eye[None, :, :], r[:, None, :])  # (atom, k, xyz) = e_k x r
    t_mat = (sqrt_m * np.concatenate([trans, rots], axis=1)).transpose(0, 2, 1)
    t_mat = t_mat.reshape(3 * n, 6)

    u, _, _ = np.linalg.svd(t_mat, full_matrices=True)
    return u[:, 3 + kind.rotation_count :]


def vibrational_analysis(
    point: PESPoint, *, masses: Sequence[float] | None = None
) -> VibrationalAnalysis:
    """Calculate frequencies and normal modes from the Hessian.

    Translations and rotations are projected out, so there are
    ``3N - 3 - point.rotor_kind.rotation_count`` modes: 3N - 6 for a nonlinear
    molecule, 3N - 5 for a linear one, and none for an atom.

    Args:
        point: Geometry and Hessian.
        masses: Atomic masses in amu; the default is `Geometry.masses`
            (most-abundant isotopes).

    Returns:
        Frequencies and modes.

    Examples:
        >>> import numpy as np
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["H", "H"],
        ...     coordinates=[[0, 0, 0], [0, 0, 0.74]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> k = 0.37  # Hartree/Bohr², a rough H2 force constant
        >>> hess = np.zeros((6, 6))
        >>> hess[2, 2] = hess[5, 5] = k
        >>> hess[2, 5] = hess[5, 2] = -k
        >>> point = autogeom.PESPoint(geometry=geo, hessian=hess)
        >>> vib = autogeom.vibrational_analysis(point)
        >>> vib.frequencies.round()
        array([4405.])
    """
    geo = point.geometry
    n = geo.atom_count
    masses = geo.masses if masses is None else list(masses)
    if len(masses) != n:
        msg = f"Got {len(masses)} masses for {n} atoms."
        raise ValueError(msg)

    w = np.repeat(1 / np.sqrt(masses), 3)
    hess_mw = w[:, None] * point.hessian * w[None, :]

    comp = _external_basis(geo, masses, point.rotor_kind)
    eigvals, eigvecs = np.linalg.eigh(comp.T @ hess_mw @ comp)
    eigvecs = (comp @ eigvecs).T
    modes = eigvecs * w[None, :]
    if modes.size:
        modes /= np.linalg.norm(modes, axis=1, keepdims=True)

    freqs = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * _FREQ_FACTOR
    return VibrationalAnalysis(
        frequencies=freqs,
        eigenvalues=eigvals,
        modes=modes.reshape(-1, n, 3),
        eigenvectors=eigvecs,
    )


def stationary_point_order(vib: VibrationalAnalysis) -> int:
    """Count the negative eigenvalues of the projected Hessian.

    There is no threshold: every sign counts, so a poorly converged structure
    can give the wrong order. A nearly free internal rotor gives a mode within
    a few cm⁻¹ of zero, whose sign is physically ambiguous.

    Args:
        vib: Vibrational analysis.

    Returns:
        0 for a minimum, 1 for a first-order saddle point, and so on.

    Examples:
        >>> import numpy as np
        >>> import autogeom
        >>> geo = autogeom.Geometry(
        ...     symbols=["H", "H"],
        ...     coordinates=[[0, 0, 0], [0, 0, 0.74]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> hess = np.zeros((6, 6))
        >>> hess[2, 2] = hess[5, 5] = -0.1
        >>> hess[2, 5] = hess[5, 2] = 0.1
        >>> point = autogeom.PESPoint(geometry=geo, hessian=hess)
        >>> autogeom.stationary_point_order(autogeom.vibrational_analysis(point))
        1
    """
    return int(np.sum(vib.eigenvalues < 0))
