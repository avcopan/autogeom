# Vibrational analysis reference data

Each case has `<name>.xyz` (Å), `<name>_hess.txt` (Cartesian Hessian, Hartree/Bohr², in the frame of the `.xyz`) and `<name>_freqs.txt` (reference frequencies in cm⁻¹, imaginary ones negative, without the translations and rotations).
`tests/test_vib.py` lists the spin, order and tolerance of each case.

| prefix | source | masses | printed precision |
|---|---|---|---|
| `automol_` | the old AutoMol test data (`automol/tests/data/{oh,h2o2,c4h7_h2}_*` in [autochem](https://github.com/Auto-Mech/autochem)), copied unchanged | isotopic | 0.1 cm⁻¹ |
| `projrot_` | [ProjRot](https://github.com/Auto-Mech/mechtools) `examples/{oh,c2h2,hono,ts}`: geometry and Hessian from `RPHt_input_data.dat`, frequencies from `RTproj_freq.dat` | isotopic, ~5 digits | Hessian to 6 significant figures; 0.01 cm⁻¹ |
| `gaussian_` | [RGD1](https://doi.org/10.1038/s41597-023-02043-z) raw Gaussian 16 output (B3LYP/TZVP): geometry and Hessian from the archive block, frequencies from the `Frequencies --` lines. The `.xyz` comment line gives the charge, multiplicity and `NImag`. Minima `mol857` (HCN, linear), `mol108629` (H2O) and `mol1` (CH4); saddle points `MR_*` | isotopic | 1e-4 cm⁻¹ |

The ORCA case, `tests/data/propyl_oxirane*`, is shared with the geometry tests. Its frequencies include 6 leading zeros and were computed with standard atomic weights.
