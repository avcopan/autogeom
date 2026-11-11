"""Visualization with py3Dmol and xyzrender."""

import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import py3Dmol
from numpy.typing import ArrayLike

from ._geometry import Geometry, xyz_file

if TYPE_CHECKING:
    import xyzrender

__all__ = [
    "View",
    "render_gif",
    "render_svg",
    "view",
]


class View(py3Dmol.view):
    """A `py3Dmol.view` with helpers for geometries and arrows."""

    def add_geometry(self, geo: Geometry, *, label: bool = False) -> None:
        """Add a geometry to the view.

        Args:
            geo: Geometry.
            label: If `True`, label atoms by index.
        """
        view(geo, view=self, label=label)

    def add_xyz_axes(
        self,
        *,
        scale: float = 1,
        colors: tuple[str, str, str] = ("red", "green", "blue"),
    ) -> None:
        """Add x, y, and z axis arrows from the origin.

        Args:
            scale: Length of each arrow.
            colors: Colors of the x, y, and z arrows.
        """
        self.add_vectors(np.eye(3) * scale, colors=colors)

    def add_vectors(
        self,
        coords: ArrayLike,
        start_coord: ArrayLike = (0, 0, 0),
        *,
        direction: bool = False,
        colors: Sequence[str] | None = None,
    ) -> None:
        """Add arrows that share a start point.

        Args:
            coords: Arrow tips, one row per arrow.
            start_coord: Start point shared by all arrows.
            direction: If `True`, treat each row of `coords` as a direction
                from `start_coord` rather than an end point.
            colors: One color per arrow. Defaults to black.

        Raises:
            ValueError: If the numbers of arrows and colors differ.
        """
        coords = np.asarray(coords, dtype=np.float64)
        colors = colors or ["black"] * len(coords)
        if len(coords) != len(colors):
            msg = f"Coordinates and colors do not match: {coords = }, {colors = }"
            raise ValueError(msg)

        for coord, color in zip(coords, colors, strict=True):
            self.add_vector(coord, start_coord, direction=direction, color=color)

    def add_vector(
        self,
        coord: ArrayLike,
        start_coord: ArrayLike = (0, 0, 0),
        *,
        direction: bool = False,
        color: str = "black",
    ) -> None:
        """Add an arrow.

        Args:
            coord: Arrow tip.
            start_coord: Arrow start point.
            direction: If `True`, treat `coord` as a direction from
                `start_coord` rather than an end point.
            color: Arrow color.
        """
        if direction:
            coord = np.add(coord, start_coord)

        start = np.asarray(start_coord).tolist()
        end = np.asarray(coord).tolist()
        self.addArrow(
            {
                "start": {"x": start[0], "y": start[1], "z": start[2]},
                "end": {"x": end[0], "y": end[1], "z": end[2]},
                "color": color,
            }
        )


def view(
    geo: Geometry, *, view: py3Dmol.view | None = None, label: bool = False
) -> py3Dmol.view:
    """Display a geometry with py3Dmol.

    Args:
        geo: Geometry.
        view: Existing view to add the geometry to. If `None`, create one.
        label: If `True`, label atoms by index.

    Returns:
        The view containing the geometry.
    """
    view = py3Dmol.view(width=400, height=400) if view is None else view
    view.addModel(geo.xyz_block(), "xyz")
    # Model -1 is the one just added, so earlier models keep their styling
    view.setStyle({"model": -1}, {"stick": {}, "sphere": {"scale": 0.3}})
    if label:
        for key in range(geo.atom_count):
            view.addLabel(
                str(key),
                {
                    "backgroundOpacity": 0.0,
                    "fontColor": "black",
                    "alignment": "center",
                    "inFront": True,
                },
                {"model": -1, "index": key},
            )
    return view


def render_svg(
    geo: Geometry,
    *,
    out: str | Path | None = None,
    config: "str | xyzrender.RenderConfig" = "default",
    include_h: bool = True,
) -> "xyzrender.SVGResult":
    """Render a geometry as an SVG image with xyzrender.

    The result displays inline in Jupyter.

    Args:
        geo: Geometry.
        out: Output path. The suffix is replaced with ``.svg``.
        config: xyzrender preset name or `xyzrender.RenderConfig`.
        include_h: If `True`, include hydrogen atoms.

    Returns:
        The SVG render result.
    """
    xyzrender = _import_xyzrender()
    out = Path(out).with_suffix(".svg") if out else out
    mol = _xyzrender_molecule(geo)
    return xyzrender.render(mol, config=config, hy=include_h, output=out)


def render_gif(
    geo: Geometry,
    *,
    out: str | Path | None = None,
    config: "str | xyzrender.RenderConfig" = "default",
    include_h: bool = True,
    rotation_axis: str = "x",
) -> "xyzrender.GIFResult":
    """Render a geometry rotating about an axis as a GIF with xyzrender.

    The result displays inline in Jupyter.

    Args:
        geo: Geometry.
        out: Output path. The suffix is replaced with ``.gif``.
        config: xyzrender preset name or `xyzrender.RenderConfig`.
        include_h: If `True`, include hydrogen atoms.
        rotation_axis: Axis to rotate about.

    Returns:
        The GIF render result.
    """
    xyzrender = _import_xyzrender()
    out = Path(out).with_suffix(".gif") if out else out
    mol = _xyzrender_molecule(geo)
    return xyzrender.render_gif(
        mol, config=config, hy=include_h, output=out, gif_rot=rotation_axis
    )


def _import_xyzrender() -> ModuleType:
    """Import xyzrender, which is optional for conda installs (PyPI-only)."""
    try:
        import xyzrender  # noqa: PLC0415
    except ImportError as err:
        msg = (
            "Rendering requires xyzrender, which is only on PyPI. "
            "Install it with `pip install xyzrender` or, in a pixi workspace, "
            "add it under [pypi-dependencies]."
        )
        raise ImportError(msg) from err
    return xyzrender


def _xyzrender_molecule(geo: Geometry) -> "xyzrender.Molecule":
    """Load a geometry into xyzrender via a temporary xyz file."""
    xyzrender = _import_xyzrender()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "geometry.xyz"
        xyz_file(geo, path=path)
        return xyzrender.load(path)
