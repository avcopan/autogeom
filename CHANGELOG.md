# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

autogeom continues the `Geometry` part of [automol](https://github.com/avcopan/automol) 0.0.31 under a new name, which no longer clashes with the `automol` module of the older AutoMech codes.
The changes below are relative to automol 0.0.31.

### Added
- `inchi`, `smiles`, and `formula` (Hill order), which generate strings from a `Geometry`, and `from_inchi` and `from_smiles`, which embed a `Geometry` from one. They replace `rdkit_inchi.identity_fn`, `rdkit_smiles.identity_fn`, `hill_formula.identity_fn`, and the `geometry_fn`s.
- `ElementNotFoundError`, `GeometryConversionError`, `XYZFormatError`, and `ureg` are exported from the package root.

### Changed
- **Breaking:** the package is imported as `autogeom`.
- **Breaking:** everything in `automol.geom` is exported from the package root, and the `geom` module is gone: `automol.geom.center_of_mass(geo)` is now `autogeom.center_of_mass(geo)`.
- **Breaking:** `automol.rd.mol` is now `autogeom.rd`, e.g. `rd.mol.from_smiles` is `rd.from_smiles`.
- **Breaking:** `automol.utils.element`, `automol.utils.types`, and `automol.utils.exc` are now `autogeom.element`, `autogeom.types`, and `autogeom.exc`.

### Removed
- The identity framework (`Algorithm`, `AlgorithmRegistry`, `IdentityKind`, and the `rdkit_inchi`, `rdkit_smiles`, and `hill_formula` algorithms), which moves to [autostorage](https://github.com/avcopan/autostorage).
