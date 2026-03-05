#!/usr/bin/env python3
"""
Compute RDKit 2D + 3D ligand descriptors for a single NAffinity complex folder.

Input (same pattern as ligand_extraction.py):
  - a directory whose basename is the complex/folder name, e.g. .../9CPG
  - expects a ligand file named: (folder_name)_lig.sd inside that directory

Output:
  - writes rdkit.txt in the same directory

Usage:
  python3 rdkit_features.py /path/to/complex_folder
  python3 rdkit_features.py /path/to/complex_folder --ligand-sd custom_ligand.sd
  python3 rdkit_features.py /path/to/complex_folder --out rdkit.txt
"""

import argparse
import os
import sys
import time

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors as rd3d
from rdkit.Chem import AllChem


def getMolDescriptors2D(mol, missingVal=None):
    res = {}
    for name, fn in Descriptors._descList:
        try:
            res[name] = fn(mol)
        except Exception:
            res[name] = missingVal
    return res


def getMolDescriptors3D(mol, missingVal=None):
    res = {}
    keys = [
        "PMI1", "PMI2", "PMI3", "NPR1", "NPR2",
        "RadiusOfGyration", "InertialShapeFactor",
        "Eccentricity", "Asphericity", "SpherocityIndex",
    ]
    try:
        if mol.GetNumConformers() == 0:
            # If the ligand has no coordinates, embed as a fallback.
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        res["PMI1"] = rd3d.CalcPMI1(mol)
        res["PMI2"] = rd3d.CalcPMI2(mol)
        res["PMI3"] = rd3d.CalcPMI3(mol)
        res["NPR1"] = rd3d.CalcNPR1(mol)
        res["NPR2"] = rd3d.CalcNPR2(mol)
        res["RadiusOfGyration"] = rd3d.CalcRadiusOfGyration(mol)
        res["InertialShapeFactor"] = rd3d.CalcInertialShapeFactor(mol)
        res["Eccentricity"] = rd3d.CalcEccentricity(mol)
        res["Asphericity"] = rd3d.CalcAsphericity(mol)
        res["SpherocityIndex"] = rd3d.CalcSpherocityIndex(mol)

    except Exception as e:
        # Do not crash the full pipeline; write missing values consistently.
        print(f"⚠️ 3D descriptor error: {e}", file=sys.stderr)
        for k in keys:
            res[k] = missingVal

    return res


def load_sdf_with_retry(sdf_path, retries=3, delay=2):
    for attempt in range(retries):
        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            if suppl and len(suppl) > 0 and suppl[0] is not None:
                return suppl
        except Exception as e:
            print(f"[Retry {attempt+1}] Failed to load {sdf_path}: {e}", file=sys.stderr)
            time.sleep(delay)
    print(f"❌ Failed to load {sdf_path} after {retries} retries.", file=sys.stderr)
    return None


def compute_rdkit_descriptors_from_sd(sd_path: str):
    suppl = load_sdf_with_retry(sd_path)
    if not suppl:
        raise RuntimeError(f"Could not load ligand SD file: {sd_path}")

    mol = suppl[0]
    if mol is None:
        raise RuntimeError(f"RDKit returned None for first molecule in: {sd_path}")

    # Kekulization fix (kept from your notebook logic)
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception as e:
            print(f"⚠️ Kekulization failed: {e}", file=sys.stderr)

    desc_2d = getMolDescriptors2D(mol)
    desc_3d = getMolDescriptors3D(mol)
    return {**desc_2d, **desc_3d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Directory containing (folder_name)_lig.sd")
    ap.add_argument("--ligand-sd", default=None, help="Optional override ligand SD filename (in the directory)")
    ap.add_argument("--out", default="rdkit.txt", help="Output filename to write inside the directory (default: rdkit.txt)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output file if present")
    args = ap.parse_args()

    folder = os.path.abspath(args.dir)
    if not os.path.isdir(folder):
        print(f"ERROR: Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(folder))

    ligand_sd = args.ligand_sd if args.ligand_sd else f"{folder_name}_lig.sd"
    sd_path = os.path.join(folder, ligand_sd)
    out_path = os.path.join(folder, args.out)

    if not os.path.exists(sd_path):
        print(f"ERROR: Missing ligand SD file: {sd_path}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(out_path) and not args.overwrite:
        print(f"⏭️ Skipping {folder_name} ({args.out} already exists)")
        print(f"Wrote: {out_path}")
        return

    try:
        desc = compute_rdkit_descriptors_from_sd(sd_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w") as f:
        for k, v in desc.items():
            f.write(f"{k}: {v}\n")

    print(f"✅ Wrote {args.out} for {folder_name}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()