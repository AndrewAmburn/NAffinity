# NAffinity: Nucleic Acid-Ligand Affinity Classifier

NAffinity is a machine learning-based tool for predicting nucleic acid-ligand binding strength from a nucleic acid-ligand complex structure.

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate naffinity
```

To use the `naffinity` command-line interface, install NAffinity from the repository root:

```bash
pip install -e ".[dev]"
```

## Required Input Folder Structure

NAffinity expects one folder per nucleic acid-ligand complex. Each prediction is run on a single complex folder containing a PDB file of the bound complex.

- **Folder name:** should match the PDB filename
- **Required input file:**
  - `<folder_name>.pdb` — PDB structure of the nucleic acid-ligand complex, including both receptor and bound ligand

Example folder structure:

```text
3GAO/
  3GAO.pdb
```

During execution, NAffinity generates the following files inside the same folder:

- `<folder_name>_lig.sd` — ligand extracted from the PDB file
- `rdkit.txt` — RDKit ligand descriptors
- `electro_hydro.txt` — electrostatic and hydrogen-bond proximity descriptors
- `descriptors.txt` — additional ligand and binding-site descriptors
- `receptor_descriptors.txt` — receptor binding-site descriptors
- `naffinity_predicted_binding_class.txt` — final prediction output

## Usage

Run the full pipeline for a single complex:

```bash
naffinity run <folder_path>
```

Example:

```bash
naffinity run example/3GAO
```

Optional arguments:

```bash
naffinity run <folder_path> --overwrite
naffinity run <folder_path> --jobs 4
naffinity run <folder_path> --python /path/to/python
```

### Output  
Successful execution produces:

```text
<folder_path>/naffinity_predicted_binding_class.txt
```

This file contains:
- `PredictedClass` — `Strong binder` or `Weak/moderate binder`
- `ProbabilityStrongBinder` — probability score (0.00–1.00)

---

## Notes  
- NAffinity assumes the input structure is already a **bound complex** with the ligand placed in the binding site.
- Ligand extraction is heuristic and is based on HETATM residue grouping and proximity to the nucleic acid. If a PDB contains multiple small molecules (e.g., additives, cofactors), verify that the extracted ligand file `<folder_name>_lig.sd` corresponds to the intended ligand.

---

## Included Example

NAffinity includes an example nucleic acid–ligand complex to verify correct installation.

### Example Complex  
`3GAO` — Crystal structure of the guanine riboswitch bound to xanthine. 
Citation: DOI: https://doi.org/10.1016/j.str.2009.04.009 
Reference affinity for the bound ligand: `Kd (M): 3.9e-5`.
Correct classification is: Weak/moderate binder.

### Example Directory  
```text
NAffinity/example/3GAO
```

### Run the Example  
```bash
cd NAffinity
conda activate naffinity
pip install -e ".[dev]" 

naffinity run example/3GAO
```

### Expected Output  
The folder:
```text
NAffinity/example/3GAO_expected_output
```
contains the expected outputs produced by a successful run (feature files and affinity prediction output). After running the example, your generated outputs in `example/3GAO/` can be compared to the expected outputs.

### Notes on the Example  
- The example is intended to validate environment setup and end-to-end NAffinity execution.
- If your outputs differ slightly from the expected outputs, confirm that you are using the dependency versions listed above.

---
