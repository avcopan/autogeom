"""Geometry -> RDKit Mol perception tests across a broad set of C/H/O species.

Species are written as SMILES for readability and converted to InChI. Each InChI is
embedded with a fixed seed, stripped down to a `Geometry` (symbols, coordinates,
charge, spin), and passed through `autogeom.rdkit_mol`. The InChI of the perceived Mol
must match that of the embedded reference.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

import autogeom
from autogeom import Geometry, rd

SEED = 0xF00D

SMALL = [
    "[H][H]",
    "[H]",
    "O",
    "OO",
    "O=O",
    "[O][O]",
    "C",
    "[C-]#[O+]",
    "O=C=O",
    "[O-][O+]=O",
]

ALKANES = [
    "CC",
    "CC(C)(C)C",
    "CCCCCCCCCCCCCCCC",
    "C1CC1",
    "C1CCCCC1",
    "C1C2CC21",
    "C1CC2CCC1C2",
    "C1C2CC3CC1CC(C2)C3",
    "C12C3C4C1C5C2C3C45",
]

ALKENES = [
    "C=C",
    "C/C=C/C",
    "C/C=C\\C",
    "CC(C)=C(C)C",
    "C=CC=C",
    "C=C=C",
    "C1=CC1",
    "C=C1CC1",
    "C1=CCC=CC1",
    "C1=CC=CC=CC=C1",
    "CC1=CC[C@@H](CC1)C(C)=C",
]

ALKYNES = [
    "C#C",
    "CC#CC",
    "C#CC#C",
    "C=CC#C",
    "C#Cc1ccccc1",
    "C#CC1CC1",
    "OCC#C",
    "CC#CC(=O)O",
]

AROMATICS = [
    "c1ccccc1",
    "Cc1ccccc1",
    "C=Cc1ccccc1",
    "c1ccc2ccccc2c1",
    "c1ccc2c(c1)ccc1ccccc12",
    "c1cc2ccc3cccc4ccc(c1)c2c34",
    "c1ccc2cccc2cc1",
    "c1ccoc1",
]

ALCOHOLS_AND_ETHERS = [
    "CO",
    "CC(C)(C)O",
    "OCC(O)CO",
    "C=C(C)O",
    "C=COC",
    "C1CO1",
    "C1CCOC1",
    "C1COCCOCCOCCOCCOCCO1",
]

CARBONYLS = [
    "C=O",
    "CC=O",
    "CC(C)=O",
    "O=CC=O",
    "C=CC=O",
    "C=C=O",
    "O=C=C=C=O",
    "C/C=C(\\C)C=O",
]

ACIDS_AND_ESTERS = [
    "O=CO",
    "CC(=O)O",
    "CC(=O)OC",
    "CC(=O)OC(C)=O",
    "O=C1CCCO1",
    "O=C1C=CC(=O)O1",
    "OC(=O)/C=C\\C(=O)O",
    "OC(=O)/C=C/C(=O)O",
]

PEROXIDES = [
    "COOC",
    "CC(=O)OO",
    "C1COO1",
    "C1OCOO1",
]

AROMATIC_OXYGENATES = [
    "Oc1ccccc1",
    "COc1ccccc1",
    "O=Cc1ccccc1",
    "O=C(O)c1ccccc1",
    "O=C1C=CC(=O)C=C1",
]

RADICALS = [
    "[CH3]",
    "[CH2]",
    "[OH]",
    "[O]",
    "C[C](C)C",
    "[CH2]C=C",
    "[CH2]c1ccccc1",
    "[C]#C",
    "C[O]",
    "CO[O]",
    "[CH]=O",
    "[CH2]C=O",
    "CC(=O)[O]",
    "[O]c1ccccc1",
]

IONS = [
    "[OH-]",
    "[CH3+]",
    "C[O-]",
    "CC(=O)[O-]",
    "O=C([O-])[O-]",
    "[O-]c1ccccc1",
    "c1cc[cH-]c1",
    "c1cc[cH+]ccc1",
    "C[C+](C)C",
    "C=[O+]C",
    "CC#[O+]",
    "c1cc[o+]cc1",
]

STEREO = [
    "C[C@H](O)CC",
    "C[C@@H](O)CC",
    "O[C@H]1CCCC[C@@H]1O",
    "O=C(O)[C@H](O)[C@@H](O)C(=O)O",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O",
]

NATURAL_PRODUCTS_AND_DRUGS = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
    "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O",
    "C[C@@H]1CC[C@H]2[C@@H](C)C(=O)O[C@@H]3O[C@@]4(C)CC[C@@H]1[C@]32OO4",
    "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
    "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
    "CC1=C(C(C)(C)CCC1)/C=C/C(C)=C/C=C/C(C)=C/C=O",
]


KNOWN_FAILURES = [
    pytest.param(
        "C=[OH+]",
        marks=pytest.mark.xfail(
            reason="perceived as the octet-deficient carbocation [CH2+]O", strict=True
        ),
    ),
]

ALL_SMILES = [
    *SMALL,
    *ALKANES,
    *ALKENES,
    *ALKYNES,
    *AROMATICS,
    *ALCOHOLS_AND_ETHERS,
    *CARBONYLS,
    *ACIDS_AND_ESTERS,
    *PEROXIDES,
    *AROMATIC_OXYGENATES,
    *RADICALS,
    *IONS,
    *STEREO,
    *NATURAL_PRODUCTS_AND_DRUGS,
]


def _reference_geometry(smiles: str) -> tuple[Geometry, str]:
    """Embed a species from its InChI, returning its Geometry and 3D InChI.

    The reference InChI is taken from the embedded coordinates, so that stereo left
    unspecified by the SMILES but fixed by the embedding (e.g. bridgeheads, anomeric
    centers) is included. Spin is taken from the SMILES, since InChI does not encode
    spin multiplicity.
    """
    src = Chem.MolFromSmiles(smiles)
    ref = rd.from_inchi(rd.inchi(src))
    assert EmbedMolecule(ref, randomSeed=SEED) == 0
    geo = Geometry(
        symbols=rd.symbols(ref),
        coordinates=rd.coordinates(ref),
        charge=rd.charge(ref),
        spin=rd.spin(src),
    )
    assert geo.charge == rd.charge(src)
    return geo, rd.inchi(ref)


@pytest.mark.parametrize("smiles", [*ALL_SMILES, *KNOWN_FAILURES])
def test__rdkit_mol(smiles: str) -> None:
    """Test that the perceived Mol reproduces the reference InChI."""
    geo, inchi = _reference_geometry(smiles)

    mol = autogeom.rdkit_mol(geo)

    assert rd.symbols(mol) == geo.symbols
    assert np.allclose(rd.coordinates(mol), geo.coordinates)
    assert rd.charge(mol) == geo.charge
    assert rd.spin(mol) == geo.spin
    assert rd.inchi(mol) == inchi


def test__rdkit_mol_species_are_distinct_cho() -> None:
    """Test that the species list contains only distinct C/H/O species."""
    mols = [Chem.MolFromSmiles(smi) for smi in ALL_SMILES]
    symbols = {atom.GetSymbol() for mol in mols for atom in mol.GetAtoms()}
    assert symbols <= {"C", "H", "O"}
    assert len({Chem.MolToSmiles(mol) for mol in mols}) == len(mols)
