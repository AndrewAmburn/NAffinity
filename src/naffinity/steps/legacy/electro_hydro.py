#!/usr/bin/env python3
"""
Compute electrostatic + H-bond proximity features for a single NAffinity complex folder.

Input (same folder-based convention as ligand_extraction.py):
  - directory basename is the complex name (folder_name)
  - expects:
      (folder_name).pdb
      (folder_name)_lig.sd

Output:
  - writes electro_hydro.txt in the same directory

Usage:
  python3 electro_hydro_features.py /path/to/complex_folder
  python3 electro_hydro_features.py /path/to/complex_folder --radius 6.0
  python3 electro_hydro_features.py /path/to/complex_folder --overwrite
"""

import argparse
import os
import sys
from collections import defaultdict
from math import cos, radians

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree


# Keep this list stable so failures still write a complete file with zeros.
FEATURE_KEYS = [
    "MeanLigandReceptorChargeProduct",
    "MinAttractiveElectrostaticEnergy",
    "MaxRepulsiveElectrostaticEnergy",
    "SumPositiveReceptorCharge",
    "SumNegativeReceptorCharge",
    "ChargeRatioPosNeg",
    "ElectrostaticComplementarity",
    "DistWeightedChargeSum",
    "NetChargeGradient",
    "ElectrostaticVariance",
    "StericClashCount",
    "IdealHBondPairCount",
    "MedianHbondDistance",
    "MeanHbondAngleScore",
    "MaxHbondAngleScore",
    "HBondSaturationIndex",
    "NearbyDonors",
    "NearbyAcceptors",
    "BuriedHBondScore",
    "AvgLigandChargeMagnitude",
    "MaxLigandPartialCharge",
    "MinLigandPartialCharge",
    "AvgReceptorChargeMagnitude",
    "PolarSurfaceContactCount",
    "PolarHydrophobicProximityScore",
]


def compute_features(ligand_file: str, pdb_file: str, radius: float = 6.0):
    feats = {k: 0.0 for k in FEATURE_KEYS}

    try:
        # --- Ligand load + sanitize ---
        suppl = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=False)
        mol = suppl[0] if suppl and len(suppl) > 0 and suppl[0] is not None else None
        if mol is None:
            raise ValueError("RDKit failed to load ligand from SD.")

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            for atom in mol.GetAtoms():
                atom.SetIsAromatic(False)
            for bond in mol.GetBonds():
                bond.SetIsAromatic(False)
            Chem.SanitizeMol(mol)

        # Gasteiger charges
        AllChem.ComputeGasteigerCharges(mol)
        if mol.GetNumConformers() == 0:
            raise ValueError("Ligand has no conformer/coordinates in SD.")

        lig_coords = mol.GetConformer().GetPositions()
        lig_charges = [float(atom.GetProp("_GasteigerCharge") or 0.0) for atom in mol.GetAtoms()]
        lig_elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        lig_donors = [a for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8)]

        # --- Receptor parse ---
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("receptor", pdb_file)
        model = structure[0]

        receptor_coords = []
        receptor_charge_entries = []

        # Fixed per-element proxy charges (your notebook mapping)
        charge_map = {"O": -0.4, "N": -0.3, "P": 0.2, "S": 0.1, "C": 0.0}

        for chain in model:
            for residue in chain:
                for atom in residue:
                    coord = atom.coord
                    element = (atom.element or "").strip().upper()
                    receptor_coords.append(coord)
                    receptor_charge_entries.append(
                        {
                            "coord": coord,
                            "charge": charge_map.get(element, 0.0),
                            "element": element,
                        }
                    )

        if len(receptor_coords) == 0:
            raise ValueError("No receptor atoms parsed from PDB.")

        r_coords = np.array(receptor_coords, dtype=float)

        # --- Neighbor search within radius ---
        lig_tree = cKDTree(lig_coords)
        r_tree = cKDTree(r_coords)
        indices = lig_tree.query_ball_tree(r_tree, r=radius)

        electro_energies = []
        charge_products = []
        dist_weights = []
        attract_energies = []
        repel_energies = []
        pos_sum = 0.0
        neg_sum = 0.0
        hbond_scores = []
        hbond_distances = []
        hbond_count = 0
        steric_clashes = 0
        near_donors = 0
        near_acceptors = 0
        polar_surface_contact = 0
        polar_hydrophobic_contacts = 0

        for i, neighbors in enumerate(indices):
            l_coord = lig_coords[i]
            l_charge = lig_charges[i]
            l_element = lig_elements[i]

            for j in neighbors:
                r_coord = r_coords[j]
                r_entry = receptor_charge_entries[j]
                r_element = r_entry["element"]
                r_charge = r_entry["charge"]

                dist = float(np.linalg.norm(l_coord - r_coord))
                if dist < 1e-2:
                    continue

                qprod = l_charge * r_charge
                energy = qprod / (dist**2)

                electro_energies.append(energy)
                charge_products.append(qprod)
                dist_weights.append(1.0 / dist)

                if qprod < 0:
                    attract_energies.append(energy)
                elif qprod > 0:
                    repel_energies.append(energy)

                if r_charge > 0:
                    pos_sum += r_charge
                elif r_charge < 0:
                    neg_sum += r_charge

                if dist < 2.0:
                    steric_clashes += 1

                # Very simple H-bond proxy from your notebook: any nearby O/N contributes
                if r_element in {"O", "N"}:
                    hbond_scores.append(cos(radians(180)) ** 2 / (dist**2))
                    hbond_distances.append(dist)
                    hbond_count += 1
                    near_acceptors += 1
                    polar_surface_contact += 1

                if r_element == "H":
                    near_donors += 1

                if r_element in {"O", "N"} and l_element == "C":
                    polar_hydrophobic_contacts += 1

        # --- Write features ---
        feats["MeanLigandReceptorChargeProduct"] = float(np.mean(charge_products)) if charge_products else 0.0
        feats["MinAttractiveElectrostaticEnergy"] = float(min(attract_energies)) if attract_energies else 0.0
        feats["MaxRepulsiveElectrostaticEnergy"] = float(max(repel_energies)) if repel_energies else 0.0
        feats["SumPositiveReceptorCharge"] = float(pos_sum)
        feats["SumNegativeReceptorCharge"] = float(neg_sum)
        feats["ChargeRatioPosNeg"] = float(abs(pos_sum) / (abs(neg_sum) + 1e-6))
        feats["ElectrostaticComplementarity"] = (
            float(np.mean([-1.0 if q < 0 else 1.0 for q in charge_products])) if charge_products else 0.0
        )

        # Keep your notebook behavior (even though algebraically unusual): sum(q / (1/d)) = sum(q * d)
        feats["DistWeightedChargeSum"] = float(
            sum(q / (1.0 / d) for q, d in zip(charge_products, dist_weights))
        ) if dist_weights else 0.0

        feats["NetChargeGradient"] = float(sum(abs(e) for e in electro_energies)) if electro_energies else 0.0
        feats["ElectrostaticVariance"] = float(np.var(electro_energies)) if electro_energies else 0.0
        feats["StericClashCount"] = float(steric_clashes)

        feats["IdealHBondPairCount"] = float(hbond_count)
        feats["MedianHbondDistance"] = float(np.median(hbond_distances)) if hbond_distances else 0.0
        feats["MeanHbondAngleScore"] = float(np.mean(hbond_scores)) if hbond_scores else 0.0
        feats["MaxHbondAngleScore"] = float(max(hbond_scores)) if hbond_scores else 0.0
        feats["HBondSaturationIndex"] = float(hbond_count / (len(lig_donors) + 1e-6))

        feats["NearbyDonors"] = float(near_donors)
        feats["NearbyAcceptors"] = float(near_acceptors)
        feats["BuriedHBondScore"] = float(sum(1 for d in hbond_distances if d < 3.5))

        feats["AvgLigandChargeMagnitude"] = float(np.mean([abs(q) for q in lig_charges])) if lig_charges else 0.0
        feats["MaxLigandPartialCharge"] = float(max(lig_charges)) if lig_charges else 0.0
        feats["MinLigandPartialCharge"] = float(min(lig_charges)) if lig_charges else 0.0
        feats["AvgReceptorChargeMagnitude"] = float(
            np.mean([abs(d["charge"]) for d in receptor_charge_entries])
        ) if receptor_charge_entries else 0.0

        feats["PolarSurfaceContactCount"] = float(polar_surface_contact)
        feats["PolarHydrophobicProximityScore"] = float(polar_hydrophobic_contacts)

        return feats

    except Exception as e:
        # Do not crash a master pipeline: return all-zeros with a clear stderr message.
        print(f"ERROR electro_hydro: {e}", file=sys.stderr)
        return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Directory containing (folder_name).pdb and (folder_name)_lig.sd")
    ap.add_argument("--radius", type=float, default=6.0, help="Neighborhood radius in Å (default: 6.0)")
    ap.add_argument("--out", default="electro_hydro.txt", help="Output filename (default: electro_hydro.txt)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    args = ap.parse_args()

    folder = os.path.abspath(args.dir)
    if not os.path.isdir(folder):
        print(f"ERROR: Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(folder))
    pdb_path = os.path.join(folder, f"{folder_name}.pdb")
    lig_path = os.path.join(folder, f"{folder_name}_lig.sd")
    out_path = os.path.join(folder, args.out)

    if not os.path.exists(pdb_path):
        print(f"ERROR: Missing PDB file: {pdb_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(lig_path):
        print(f"ERROR: Missing ligand SD file: {lig_path}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(out_path) and not args.overwrite:
        print(f"⏭️ Skipping {folder_name} ({args.out} already exists)")
        print(f"Wrote: {out_path}")
        return

    feats = compute_features(lig_path, pdb_path, radius=args.radius)

    with open(out_path, "w") as f:
        for k in sorted(feats.keys()):
            f.write(f"{k}: {feats[k]:.4f}\n")

    print(f"✅ Wrote {args.out} for {folder_name}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()