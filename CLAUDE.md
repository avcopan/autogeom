# CLAUDE.md

autogeom is a Python library for 3D molecular structures, centered on the Pydantic `Geometry` model (symbols, coordinates in Å, charge, and spin as the number of unpaired electrons).
Read `CONTRIBUTING.md` for the workflow and the design rules (functions at the package root, private modules, the import layers, where conversions live).
The user guide is in `user_guide/`, and `.claude/skills/` and `.claude/rules/` come from the copier template.

## Commands

- Run everything through Pixi (`pixi run <task>`, or `pixi run -e test python ...`).
  The dependencies are only installed in the Pixi environments, not in the system Python.
- `pixi run pre-commit` runs all checks. A single test: `pixi run -e test pytest tests/test_geom.py::test__center_of_mass`.

## Things that aren't obvious from the code

- The public API is flat: users write `autogeom.center_of_mass(geo)`. A new function goes in a private module (`_measure.py`, …) and must be added to `autogeom/__init__.py`.
- Docstring examples run as doctests and start with `import autogeom`, so they read as user code.
- `autogeom.rd` only handles RDKit `Mol` objects and never imports `Geometry`.
- xyzrender is only on PyPI, so it is imported lazily and left out of the conda package (see `pixi.toml`).
- autostorage's `GeometryRow` subclasses `Geometry`, so methods must keep working for subclasses (e.g. `relabel_atoms` uses `model_copy`).

## Style

Keep docstrings and comments short: one-line summaries where possible, without restating what a name or type hint already says.
