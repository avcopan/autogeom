"""View tests."""

import subprocess
import sys
from pathlib import Path

import pytest

from autogeom import Geometry, View, render_gif, render_svg
from autogeom import view as build_view


@pytest.fixture
def view() -> View:
    """Empty view for testing."""
    return View()


def test__add_geometry(view: View, water: Geometry) -> None:
    """Test add geometry."""
    view.add_geometry(water, label=True)


def test__add_multiple_geometries(view: View, water: Geometry) -> None:
    """Test that labels and styles only target the most recently added model."""
    view.add_geometry(water, label=True)
    view.add_geometry(water, label=True)
    js = view.startjs
    assert js.count('"model": -1') == 2 * (water.atom_count + 1)


def test__add_axes(view: View) -> None:
    """Test add axes."""
    view.add_xyz_axes(scale=2)


def test__add_vectors_mismatched_colors_raises(view: View) -> None:
    """Test that mismatched coords/colors lengths are rejected."""
    with pytest.raises(ValueError, match="do not match"):
        view.add_vectors([[1, 0, 0], [0, 1, 0]], colors=["red"])


def test__add_vector_direction(view: View) -> None:
    """Test adding a vector as a direction from a start coordinate."""
    view.add_vector([1, 0, 0], start_coord=[1, 1, 1], direction=True)


@pytest.mark.parametrize("label", [True, False])
def test__view_function(water: Geometry, *, label: bool) -> None:
    """Test py3Dmol view construction with and without atom labels."""
    assert build_view(water, label=label) is not None


def test__render_svg(water: Geometry, tmp_path: Path) -> None:
    """Test svg rendering."""
    out = tmp_path / "water"
    result = render_svg(water, out=out)
    assert (tmp_path / "water.svg").exists()
    assert "<svg" in str(result)


def test__render_gif(water: Geometry, tmp_path: Path) -> None:
    """Test gif rendering."""
    out = tmp_path / "water"
    render_gif(water, out=out)
    assert (tmp_path / "water.gif").exists()


def test__import_does_not_require_xyzrender() -> None:
    """Test that importing autogeom doesn't import xyzrender (PyPI-only)."""
    code = "import sys, autogeom; assert 'xyzrender' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test__render_without_xyzrender_raises(
    water: Geometry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that rendering without xyzrender explains how to install it."""
    monkeypatch.setitem(sys.modules, "xyzrender", None)
    with pytest.raises(ImportError, match="PyPI"):
        render_svg(water)
