# autogeom

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)

autogeom is a Python library for 3D molecular structures. It centers on a single `Geometry` model (atomic symbols, Cartesian coordinates, charge, and spin) and the functions that operate on it: xyz I/O, measurements, transformations, alignment, 3D visualization, and conversions to and from InChI, SMILES, and RDKit.

See the [documentation](https://avcopan.github.io/autogeom/) for the user guide and API reference.

autogeom continues the `Geometry` part of [automol](https://github.com/avcopan/automol) 0.0.31 under a new name. See the [changelog](https://github.com/avcopan/autogeom/blob/main/CHANGELOG.md) for what changed.

## Installation

autogeom requires Python 3.13 or later.
With [Pixi](https://pixi.sh), add it from the [`avcopan/automech`](https://prefix.dev/channels/avcopan/automech) channel on prefix.dev:

```bash
pixi workspace channel add https://repo.prefix.dev/avcopan/automech
pixi add autogeom
pixi add --pypi xyzrender  # optional: only on PyPI, needed for render_svg/render_gif
```

Or install it from PyPI, which includes xyzrender:

```bash
pip install autogeom  # or: uv add autogeom
```

For local development, see [Contributing](https://github.com/avcopan/autogeom/blob/main/CONTRIBUTING.md).

## Usage

Everything is imported from the package root.

### Geometry

```python
import autogeom
from autogeom import Geometry

water = Geometry(
    symbols=["O", "H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.93, 0.0, -0.24]],
    charge=0,
    spin=0,
)

com = autogeom.center_of_mass(water)
angle = autogeom.angle(water, [1, 0, 2], degrees=True)
```

### InChI, SMILES, and RDKit

```python
autogeom.inchi(water)  # "InChI=1S/H2O/h1H2"
autogeom.smiles(water)  # "O"
autogeom.formula(water)  # "H2O"

ethanol = autogeom.from_smiles("CCO")  # embeds 3D coordinates
mol = autogeom.rdkit_mol(ethanol)  # rdkit.Chem.Mol
```

### Visualization

```python
from autogeom import View

# Visualization (e.g. in a Jupyter notebook)
view = View()
view.add_geometry(water, label=True)
```

In a notebook, a `Geometry` also displays itself as a 3D view.

See the [documentation](https://avcopan.github.io/autogeom/) for more on geometries, conversions, and visualization.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. See [Contributing](https://github.com/avcopan/autogeom/blob/main/CONTRIBUTING.md) for the design rules and workflow.

## License

This project is licensed under the [MIT License](https://github.com/avcopan/autogeom/blob/main/LICENSE).

## Credits

This package was created with [copier](https://copier.readthedocs.io/) and the [avcopan/copier-pixi-python](https://github.com/avcopan/copier-pixi-python) project template.
