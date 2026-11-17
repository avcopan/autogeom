"""Vibrational analysis tests.

The reference data is described in ``tests/data/vib/README.md``.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

import autogeom
from autogeom import Geometry, PESPoint, RotorKind
from autogeom._vib import _external_basis

DATA_PATH = Path(__file__).parent / "data"
VIB_PATH = DATA_PATH / "vib"

# Standard atomic weights, as ORCA uses
STANDARD_WEIGHTS = {"H": 1.008, "C": 12.011, "O": 15.999}


@pytest.fixture
def propyl_oxirane_hessian() -> np.ndarray:
    """(45, 45) Hessian of propyl oxirane, in Hartree/Bohr²."""
    return np.loadtxt(DATA_PATH / "propyl_oxirane_hessian.gz")


@pytest.fixture
def propyl_oxirane_point(
    propyl_oxirane: Geometry, propyl_oxirane_hessian: np.ndarray
) -> PESPoint:
    """Propyl oxirane PES point."""
    return PESPoint(geometry=propyl_oxirane, hessian=propyl_oxirane_hessian)


def _co2(bend: float = 0.0) -> Geometry:
    """CO2, with the carbon displaced perpendicular to the axis by ``bend`` Å."""
    return Geometry(
        symbols=["O", "C", "O"],
        coordinates=[[0, 0, -1.16], [0, bend, 0], [0, 0, 1.16]],
        charge=0,
        spin=0,
    )


# PESPoint


@pytest.mark.parametrize(
    ("field", "shape", "match"),
    [("hessian", (44, 44), "Hessian shape"), ("gradient", (45,), "Gradient shape")],
)
def test__pes_point_bad_shape_raises(
    propyl_oxirane: Geometry,
    propyl_oxirane_hessian: np.ndarray,
    field: str,
    shape: tuple[int, ...],
    match: str,
) -> None:
    """A Hessian or gradient whose shape doesn't match the geometry raises."""
    arrays = {"hessian": propyl_oxirane_hessian, "gradient": np.zeros((15, 3))}
    arrays[field] = np.zeros(shape)
    with pytest.raises(ValidationError, match=match):
        PESPoint(
            geometry=propyl_oxirane,
            hessian=arrays["hessian"],
            gradient=arrays["gradient"],
        )


def test__pes_point_symmetrizes_hessian(
    propyl_oxirane: Geometry, propyl_oxirane_hessian: np.ndarray
) -> None:
    """An asymmetric Hessian is replaced by (H + Hᵀ)/2."""
    rng = np.random.default_rng(0)
    hess = propyl_oxirane_hessian + 1e-6 * rng.standard_normal((45, 45))
    point = PESPoint(geometry=propyl_oxirane, hessian=hess)
    np.testing.assert_array_equal(point.hessian, point.hessian.T)
    np.testing.assert_allclose(point.hessian, (hess + hess.T) / 2)


@pytest.mark.parametrize("field", ["hessian", "gradient"])
def test__pes_point_non_finite_raises(
    propyl_oxirane: Geometry, propyl_oxirane_hessian: np.ndarray, field: str
) -> None:
    """NaN in the Hessian or gradient raises."""
    hess = propyl_oxirane_hessian.copy()
    grad = np.zeros((15, 3))
    {"hessian": hess, "gradient": grad}[field][0, 0] = np.nan
    with pytest.raises(ValidationError, match="non-finite"):
        PESPoint(geometry=propyl_oxirane, hessian=hess, gradient=grad)


# External (translation/rotation) basis


def test__external_basis_complement(propyl_oxirane: Geometry) -> None:
    """The complement is orthonormal and orthogonal to translations and rotations."""
    masses = np.array(propyl_oxirane.masses)
    comp = _external_basis(propyl_oxirane, list(masses), RotorKind.NONLINEAR)
    np.testing.assert_allclose(comp.T @ comp, np.eye(comp.shape[1]), atol=1e-12)

    sqrt_m = np.sqrt(masses)[:, None]
    ext = []
    for axis in np.eye(3):
        ext.append((sqrt_m * axis).ravel())
        ext.append((sqrt_m * np.cross(axis, propyl_oxirane.coordinates)).ravel())
    np.testing.assert_allclose(comp.T @ np.array(ext).T, 0, atol=1e-12)


# Vibrational analysis against reference data

# (name, spin, order, tolerance in cm⁻¹)
REFERENCE_CASES = [
    ("automol_oh", 1, 0, 0.06),
    ("automol_h2o2", 0, 0, 0.06),
    ("automol_c4h7_h2", 1, 1, 0.06),
    ("projrot_oh", 1, 0, 0.02),
    ("projrot_c2h2", 0, 0, 0.02),
    ("projrot_hono", 0, 0, 0.02),
    ("projrot_ts", 1, 1, 0.02),
    ("gaussian_mol857", 0, 0, 1e-3),
    ("gaussian_mol108629", 0, 0, 1e-3),
    ("gaussian_mol1", 0, 0, 1e-3),
    ("gaussian_MR_511256_0", 0, 1, 1e-3),
    ("gaussian_MR_469204_0", 0, 1, 1e-3),
    ("gaussian_MR_429334_1", 0, 1, 1e-3),
]


@pytest.mark.parametrize(
    ("name", "spin", "order", "atol"),
    REFERENCE_CASES,
    ids=[case[0] for case in REFERENCE_CASES],
)
def test__vibrational_analysis_reference(
    name: str, spin: int, order: int, atol: float
) -> None:
    """Frequencies and order match AutoMol, ProjRot and Gaussian."""
    geo = Geometry.from_xyz_file(VIB_PATH / f"{name}.xyz", charge=0, spin=spin)
    hess = np.loadtxt(VIB_PATH / f"{name}_hess.txt")
    ref_freqs = np.loadtxt(VIB_PATH / f"{name}_freqs.txt", ndmin=1)

    vib = autogeom.vibrational_analysis(PESPoint(geometry=geo, hessian=hess))
    np.testing.assert_allclose(vib.frequencies, ref_freqs, atol=atol)
    assert autogeom.stationary_point_order(vib) == order


def test__vibrational_analysis_orca(propyl_oxirane_point: PESPoint) -> None:
    """Frequencies match ORCA, given its standard atomic weights."""
    ref_freqs = np.loadtxt(DATA_PATH / "propyl_oxirane_frequencies.gz")
    masses = [STANDARD_WEIGHTS[s] for s in propyl_oxirane_point.geometry.symbols]
    vib = autogeom.vibrational_analysis(propyl_oxirane_point, masses=masses)
    np.testing.assert_allclose(vib.frequencies, ref_freqs[6:], atol=1e-3)


def test__vibrational_analysis_output(propyl_oxirane_point: PESPoint) -> None:
    """The output has the documented shapes, order and normalization."""
    vib = autogeom.vibrational_analysis(propyl_oxirane_point)
    n_vib = 3 * 15 - 6
    assert vib.frequencies.shape == vib.eigenvalues.shape == (n_vib,)
    assert vib.modes.shape == (n_vib, 15, 3)
    assert vib.eigenvectors.shape == (n_vib, 45)
    assert np.all(np.diff(vib.frequencies) >= 0)
    assert vib.frequencies[0] == pytest.approx(58.76, abs=0.01)
    np.testing.assert_allclose(np.linalg.norm(vib.modes, axis=(1, 2)), 1)
    np.testing.assert_allclose(
        vib.eigenvectors @ vib.eigenvectors.T, np.eye(n_vib), atol=1e-12
    )


def test__vibrational_analysis_atom() -> None:
    """An atom has no vibrations."""
    geo = Geometry(symbols=["Ar"], coordinates=[[0, 0, 0]], charge=0, spin=0)
    point = PESPoint(geometry=geo, hessian=np.zeros((3, 3)))
    vib = autogeom.vibrational_analysis(point)
    assert vib.frequencies.shape == (0,)
    assert vib.modes.shape == (0, 1, 3)
    assert autogeom.stationary_point_order(vib) == 0


@pytest.mark.parametrize(
    ("kind", "n_vib"),
    [(None, 4), (RotorKind.LINEAR, 4), (RotorKind.NONLINEAR, 3)],
)
def test__vibrational_analysis_rotor_kind(kind: RotorKind | None, n_vib: int) -> None:
    """The rotor kind sets the mode count; by default CO2 bent by 1e-4 Å is linear."""
    geo = _co2(bend=1e-4)
    hess = np.eye(9)
    point = (
        PESPoint(geometry=geo, hessian=hess)
        if kind is None
        else PESPoint(geometry=geo, hessian=hess, rotor_kind=kind)
    )
    assert len(autogeom.vibrational_analysis(point).frequencies) == n_vib


def test__vibrational_analysis_bad_masses_raises(
    propyl_oxirane_point: PESPoint,
) -> None:
    """The number of masses must match the number of atoms."""
    with pytest.raises(ValueError, match="15 atoms"):
        autogeom.vibrational_analysis(propyl_oxirane_point, masses=[1.0] * 14)


# Invariance and precision


def test__vibrational_analysis_invariance(
    propyl_oxirane: Geometry, propyl_oxirane_hessian: np.ndarray
) -> None:
    """Translating and rotating the molecule leave the frequencies unchanged."""
    ref = autogeom.vibrational_analysis(
        PESPoint(geometry=propyl_oxirane, hessian=propyl_oxirane_hessian)
    )

    rot = Rotation.random(rng=np.random.default_rng(0)).as_matrix()
    coords = propyl_oxirane.coordinates @ rot.T + [1.0, -2.0, 3.0]
    geo = propyl_oxirane.model_copy(update={"coordinates": coords})
    big_rot = np.kron(np.eye(15), rot)
    hess = big_rot @ propyl_oxirane_hessian @ big_rot.T

    vib = autogeom.vibrational_analysis(PESPoint(geometry=geo, hessian=hess))
    np.testing.assert_allclose(vib.frequencies, ref.frequencies, atol=1e-6)


def test__vibrational_analysis_float32(
    propyl_oxirane: Geometry, propyl_oxirane_hessian: np.ndarray
) -> None:
    """Storing the Hessian as float32 barely moves the frequencies."""
    hess32 = propyl_oxirane_hessian.astype(np.float32)
    vib64 = autogeom.vibrational_analysis(
        PESPoint(geometry=propyl_oxirane, hessian=propyl_oxirane_hessian)
    )
    vib32 = autogeom.vibrational_analysis(
        PESPoint(geometry=propyl_oxirane, hessian=hess32)
    )
    np.testing.assert_allclose(vib32.frequencies, vib64.frequencies, atol=0.01)
    order32 = autogeom.stationary_point_order(vib32)
    assert order32 == autogeom.stationary_point_order(vib64)
