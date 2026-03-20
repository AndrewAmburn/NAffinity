#!/usr/bin/env python3
"""
Generate receptor-focused binding-site descriptors for a single NAffinity complex folder.

Input (same convention as the other scripts):
  - directory basename = folder_name
  - expects:
      (folder_name).pdb
      (folder_name)_lig.sd

Output:
  - writes receptor_descriptors.txt in the same directory

Usage:
  python3 receptor_descriptors_features.py /path/to/complex_folder
  python3 receptor_descriptors_features.py /path/to/complex_folder --radius 6.0
  python3 receptor_descriptors_features.py /path/to/complex_folder --overwrite
"""

import argparse
import os
import sys
from collections import defaultdict
from math import sqrt

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree


def generate_receptor_descriptors(ligand_file: str, pdb_file: str, out_file: str, radius: float = 6.0):
    bases_fixed = ["A", "U", "C", "G"]
    base_aliases = {
        "A": "A", "ADE": "A", "DA": "A", "RA": "A",
        "U": "U", "URA": "U", "DU": "U", "RU": "U",
        "C": "C", "CYT": "C", "DC": "C", "RC": "C",
        "G": "G", "GUA": "G", "DG": "G", "RG": "G",
    }
    monovalent_metals = {"NA", "K"}
    divalent_metals = {"MG", "CA", "ZN", "MN"}

    suppl = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=False)
    if not suppl or suppl[0] is None:
        raise ValueError("RDKit failed to load ligand molecule from SD.")

    lig_mol = suppl[0]
    try:
        Chem.SanitizeMol(lig_mol)
    except Exception:
        try:
            for atom in lig_mol.GetAtoms():
                atom.SetIsAromatic(False)
            for bond in lig_mol.GetBonds():
                bond.SetIsAromatic(False)
            Chem.SanitizeMol(lig_mol)
        except Exception:
            Chem.Kekulize(lig_mol, clearAromaticFlags=True)

    AllChem.ComputeGasteigerCharges(lig_mol)
    if lig_mol.GetNumConformers() == 0:
        raise ValueError("Ligand SD has no conformer/coordinates.")

    lig_coords = lig_mol.GetConformer().GetPositions()
    ligand_charges = [float(atom.GetProp("_GasteigerCharge") or 0.0) for atom in lig_mol.GetAtoms()]
    ligand_total_charge = float(sum(ligand_charges))

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", pdb_file)
    model = structure[0]

    receptor_atoms = []
    receptor_meta = []
    for chain in model:
        for residue in chain:
            hetfield, resseq, _ = residue.get_id()
            is_hetatm = hetfield.strip() != ""
            resname = residue.get_resname().upper()
            for atom in residue:
                coord = atom.get_coord()
                receptor_atoms.append(coord)
                receptor_meta.append(
                    {
                        "name": atom.get_name(),
                        "element": (atom.element or "").upper(),
                        "resname": resname,
                        "resid": residue.get_id(),
                        "chain": chain.id,
                        "is_hetatm": is_hetatm,
                    }
                )

    receptor_coords = np.array(receptor_atoms, dtype=float)
    lig_tree = cKDTree(lig_coords)
    receptor_tree = cKDTree(receptor_coords)

    close_indices = lig_tree.query_ball_tree(receptor_tree, r=float(radius))
    close_atoms = set(idx for sublist in close_indices for idx in sublist)

    features = defaultdict(int)
    all_distances = []
    monovalent_distances = []
    divalent_distances = []
    phosphate_distances = []
    base_residues = set()

    ligand_centroid = np.mean(lig_coords, axis=0)
    dummy_receptor_charges = []
    weighted_charges = []
    distance_weights = []
    hbond_acceptors = 0
    hbond_donors = 0
    steric_clash_count = 0

    element_charge_map = {"O": -0.4, "N": -0.3, "P": 0.2, "S": 0.1, "C": 0.0}

    for idx in close_atoms:
        atom = receptor_meta[idx]
        receptor_coord = receptor_coords[idx]
        dists = np.linalg.norm(lig_coords - receptor_coord, axis=1)
        dist = float(np.min(dists))

        resname = atom["resname"]
        atom_name = atom["name"]
        element = atom["element"]

        if resname in base_aliases:
            base_letter = base_aliases[resname]
            features[f"Residue_{base_letter}"] += 1
            base_residues.add(atom["resid"])

        if resname in ("HOH", "WAT"):
            features["Residue_HOH"] += 1

        if resname in monovalent_metals:
            features["MonovalentMetalCount"] += 1
            monovalent_distances.append(dist)

        if resname in divalent_metals:
            features["DivalentMetalCount"] += 1
            divalent_distances.append(dist)

        if atom_name.strip().upper() in ("P", "OP1", "OP2", "OP3") and dist > 0:
            phosphate_distances.append(dist)

        if not atom["is_hetatm"]:
            all_distances.append(dist)

        q = float(element_charge_map.get(element, 0.0))
        dummy_receptor_charges.append(q)
        weighted_charges.append(q / (dist + 1e-3))
        distance_weights.append(1.0 / (dist + 1e-3))

        if element in {"O", "N"}:
            hbond_acceptors += 1
        if atom_name.upper().startswith("H"):
            hbond_donors += 1
        if dist < 2.0:
            steric_clash_count += 1

    features["NumNearbyAtoms"] = int(len(close_atoms))
    features["MeanDistanceToNearbyAtoms"] = round(float(np.mean(all_distances)), 3) if all_distances else 0
    features["MinDistanceToReceptor"] = round(float(np.min(all_distances)), 3) if all_distances else 0.001
    features["MaxDistanceToReceptor"] = round(float(np.max(all_distances)), 3) if all_distances else 0.001
    features["DistanceToClosestPhosphate"] = round(float(min(phosphate_distances)), 3) if phosphate_distances else 10
    features["DistanceToClosestMonovalent"] = round(float(min(monovalent_distances)), 3) if monovalent_distances else 10
    features["DistanceToClosestDivalent"] = round(float(min(divalent_distances)), 3) if divalent_distances else 10
    features["NumBaseResidues"] = int(len(base_residues))

    features["LigandToReceptorCOM"] = round(
        float(sqrt(np.sum((ligand_centroid - np.mean(receptor_coords, axis=0)) ** 2))), 3
    )
    features["LigandCoordRange"] = round(float(np.ptp(lig_coords)), 3)

    features["TotalReceptorPartialCharge"] = round(float(sum(dummy_receptor_charges)), 3)
    features["NetChargeBalance"] = round(float(ligand_total_charge - sum(dummy_receptor_charges)), 3)
    features["AbsChargeSum"] = round(float(sum(abs(x) for x in dummy_receptor_charges)), 3)
    features["ChargeVariance"] = round(float(np.var(dummy_receptor_charges)), 5)
    features["ChargeDensity"] = round(float(np.mean(dummy_receptor_charges)), 5)
    features["DistWeightedChargeSum"] = round(float(sum(weighted_charges)), 3)
    features["InverseDistanceSum"] = round(float(sum(distance_weights)), 3)
    features["NearbyHBondAcceptors"] = int(hbond_acceptors)
    features["NearbyHBondDonors"] = int(hbond_donors)
    features["StericClashCount"] = int(steric_clash_count)

    ordered_keys = [
        "NumNearbyAtoms",
        "MeanDistanceToNearbyAtoms",
        "MinDistanceToReceptor",
        "MaxDistanceToReceptor",
        "DistanceToClosestPhosphate",
        "MonovalentMetalCount",
        "DivalentMetalCount",
        "DistanceToClosestMonovalent",
        "DistanceToClosestDivalent",
        "NumBaseResidues",
        "LigandToReceptorCOM",
        "LigandCoordRange",
        "TotalReceptorPartialCharge",
        "NetChargeBalance",
        "AbsChargeSum",
        "ChargeVariance",
        "ChargeDensity",
        "DistWeightedChargeSum",
        "InverseDistanceSum",
        "NearbyHBondAcceptors",
        "NearbyHBondDonors",
        "StericClashCount",
    ] + [f"Residue_{b}" for b in bases_fixed] + ["Residue_HOH"]

    with open(out_file, "w") as f:
        for key in ordered_keys:
            f.write(f"{key}: {features.get(key, 0)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Directory containing (folder_name).pdb and (folder_name)_lig.sd")
    ap.add_argument("--radius", type=float, default=6.0, help="Binding-site radius in Å (default: 6.0)")
    ap.add_argument("--out", default="receptor_descriptors.txt", help="Output filename (default: receptor_descriptors.txt)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    args = ap.parse_args()

    folder = os.path.abspath(args.dir)
    if not os.path.isdir(folder):
        print(f"ERROR: Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(folder))
    ligand_path = os.path.join(folder, f"{folder_name}_lig.sd")
    pdb_path = os.path.join(folder, f"{folder_name}.pdb")
    out_path = os.path.join(folder, args.out)

    if not os.path.exists(ligand_path):
        print(f"ERROR: Missing ligand file: {ligand_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(pdb_path):
        print(f"ERROR: Missing PDB file: {pdb_path}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(out_path) and not args.overwrite:
        print(f"⏭️ Skipping {folder_name} ({args.out} already exists)")
        print(f"Wrote: {out_path}")
        return

    try:
        generate_receptor_descriptors(ligand_path, pdb_path, out_path, radius=args.radius)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Wrote {args.out} for {folder_name}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()