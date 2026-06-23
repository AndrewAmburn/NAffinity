# NAffinity: Nucleic Acid-Ligand Affinity Classifier

NAffinity is a machine learning-based tool for predicting nucleic acid–ligand binding strength from three-dimensional nucleic acid–ligand complex structures. The software supports both single-complex prediction and batch processing workflows for large-scale screening of experimentally determined, docked, or modeled nucleic acid–ligand complexes.

# Installation

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

NAffinity operates on one nucleic acid–ligand complex per folder. Each complex must contain a PDB structure in which the ligand has already been positioned within the nucleic acid binding site. Ligand placement may originate from an experimental holo structure or from an external docking workflow.

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

During batch execution, NAffinity additionally produces:

- `batch_results.csv` — summary table containing predictions for all processed complexes

Note: NAffinity does not perform docking internally. Input structures must already contain the ligand in its predicted or experimentally determined binding pose.

## Reproducibility

**NAffinity includes the complete dataset, train/test split assignments, training scripts, and hyperparameter optimization workflows used in the manuscript.**

### **Dataset**

**The dataset used for model development is provided in:**

```text
data/naffinity_dataset.csv
```

**The dataset contains a `Split` column indicating the exact train/test assignments used throughout the study:**

- **Train** — complexes used for model training and hyperparameter optimization
- **Stratified** — held-out test complexes used for final model evaluation

### **Training**

**The complete model training workflow is provided in:**

```text
training/train_naffinity.py
```

**This script includes:**

- **Data loading**
- **Label parsing**
- **Feature preprocessing**
- **Removal of constant features**
- **Random Forest model fitting**
- **Held-out test set evaluation**

### **Hyperparameter Optimization**

**The hyperparameter optimization workflow is provided in:**

```text
training/hyperparameter_search.py
```

**The following hyperparameters were optimized using ten-fold stratified cross-validation:**

- **n_estimators**
- **max_depth**
- **min_samples_split**
- **min_samples_leaf**
- **bootstrap**
- **class_weight**

### **Random Seeds**

**All analyses reported in the manuscript used:**

```text
RANDOM_STATE = 42
```

### **Final Model Parameters**

**The final model selected through hyperparameter optimization used:**

```text
n_estimators = 300
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
bootstrap = False
class_weight = "balanced"
```

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
naffinity run <folder_path> --jobs 4
```

### Output  
Successful execution produces:

```text
<folder_path>/naffinity_predicted_binding_class.txt
```

This file contains:
- `PredictedClass` — `Strong binder` or `Weak/moderate binder`
- `ProbabilityStrongBinder` — probability score (0.00–1.00)

Predictions are reported as:

- Strong binder: Kd < 1 μM

- Weak/moderate binder: Kd ≥ 1 μM

---


## Batch Processing

NAffinity supports batch processing of multiple nucleic acid–ligand complexes for large-scale screening workflows. 

Run all complex folders contained within a parent directory:

```bash
naffinity batch <parent_directory>
```

Example:

```bash
naffinity batch example/
```

Optional arguments:

```bash
naffinity batch <parent_directory> --jobs 4
```

During batch execution, each subdirectory is processed independently and predictions are aggregated into a summary file:

```text
<parent_directory>/batch_results.csv
```
The summary file contains:

- `Complex` — complex folder name
- `PredictedClass` — predicted affinity class (`Strong binder` or `Weak/moderate binder`)
- `ProbabilityStrongBinder` — predicted probability of strong binding

Example:

```text
Complex,PredictedClass,ProbabilityStrongBinder
1AJU,Weak/moderate binder,0.28
1EVV_Neomycin,Strong binder,0.51
2GDI,Strong binder,0.81
```

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
