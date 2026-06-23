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


import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree
import warnings

from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter("ignore", PDBConstructionWarning)

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

RNA_DONORS = {
    ("A", "N6"),
    ("G", "N1"),
    ("G", "N2"),
    ("C", "N4"),
    ("U", "N3"),
    ("I", "N1"),
    ("DA", "N6"),
    ("DG", "N1"),
    ("DG", "N2"),
    ("DC", "N4"),
    ("DT", "N3"),
}

RNA_ACCEPTORS = {
    ("A", "N1"),
    ("A", "N3"),
    ("A", "N7"),

    ("G", "O6"),
    ("G", "N7"),

    ("C", "O2"),

    ("U", "O2"),
    ("U", "O4"),

    ("I", "O6"),
    ("DA", "N1"),
    ("DA", "N3"),
    ("DA", "N7"),

    ("DG", "O6"),
    ("DG", "N7"),

    ("DC", "O2"),

    ("DT", "O2"),
    ("DT", "O4"),
}


RNA_PARTIAL_CHARGES = {

    "P": 1.0,
    "OP1": -1.0,
    "OP2": -1.0,
    "O1P": -1.0,
    "O2P": -1.0,

    # ribose oxygens
    "O2'": -0.40,
    "O3'": -0.40,
    "O4'": -0.35,
    "O5'": -0.35,

    # carbonyl oxygens
    "O2": -0.50,
    "O4": -0.50,
    "O6": -0.50,

    # donor nitrogens
    "N1": -0.20,
    "N2": -0.25,
    "N3": -0.20,
    "N4": -0.25,
    "N6": -0.25,

    # aromatic acceptor nitrogens
    "N7": -0.30,
    "N9": -0.15,
}

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

            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=
                    Chem.SanitizeFlags.SANITIZE_ALL ^
                    Chem.SanitizeFlags.SANITIZE_PROPERTIES
                )

            except Exception:
                pass

        # Gasteiger charges
        AllChem.ComputeGasteigerCharges(mol)
        if mol.GetNumConformers() == 0:
            raise ValueError("Ligand has no conformer/coordinates in SD.")

        lig_coords = mol.GetConformer().GetPositions()
        lig_charges = []
        for atom in mol.GetAtoms():
            try:
                q = float(atom.GetProp("_GasteigerCharge"))

                if np.isnan(q) or np.isinf(q):
                    q = 0.0

            except Exception:
                q = 0.0

            lig_charges.append(q)
        lig_elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        lig_atoms = list(mol.GetAtoms())

        lig_is_donor = []
        lig_is_acceptor = []

        for atom in lig_atoms:

            atomic_num = atom.GetAtomicNum()

            try:
                n_h = atom.GetTotalNumHs()
            except Exception:
                n_h = 0

            # Donor: N/O carrying at least one hydrogen
            is_donor = (
                atomic_num in (7, 8)
                and n_h > 0
            )

            # Acceptor: exclude positively charged donor nitrogens
            is_acceptor = (
                atomic_num in (7, 8)
                and atom.GetFormalCharge() <= 0
                and not is_donor
            )

            lig_is_donor.append(is_donor)
            lig_is_acceptor.append(is_acceptor)

        lig_donors = [
            a
            for a, is_donor in zip(lig_atoms, lig_is_donor)
            if is_donor
        ]
        # --- Receptor parse ---
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("receptor", pdb_file)
        model = structure[0]

        receptor_coords = []
        receptor_charge_entries = []

        def get_rna_partial_charge(atom_name, element):

            atom_name = atom_name.strip().upper()
            if atom_name.startswith("OP"):
                return -1.0

            if atom_name in RNA_PARTIAL_CHARGES:
                return RNA_PARTIAL_CHARGES[atom_name]

            if element == "P":
                return 1.0

            if element == "O":
                return -0.35

            if element == "N":
                return -0.20

            return 0.0

        for chain in model:
            for residue in chain:
                for atom in residue:
                    coord = atom.coord
                    element = (atom.element or "").strip().upper()
                    receptor_coords.append(coord)
                    atom_name = atom.get_name().strip().upper()

                    receptor_charge_entries.append(
                    {
                        "coord": coord,
                        "charge": get_rna_partial_charge(
                            atom_name,
                            element
                        ),
                        "element": element,
                        "atom_name": atom_name,
                        "resname": residue.get_resname().strip().upper(),
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
        distances = []
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
        observed_polar_hydrophobic = set()
        observed_donors = set()
        observed_acceptors = set()
        
        for i, neighbors in enumerate(indices):
            l_coord = lig_coords[i]
            l_charge = lig_charges[i]
            l_element = lig_elements[i]
            l_is_donor = lig_is_donor[i]
            l_is_acceptor = lig_is_acceptor[i]

            for j in neighbors:
                r_coord = r_coords[j]
                r_entry = receptor_charge_entries[j]
                r_element = r_entry["element"]
                r_charge = r_entry["charge"]

                r_atom_name = r_entry["atom_name"]
                r_resname = r_entry["resname"]

                dist = float(np.linalg.norm(l_coord - r_coord))
                if dist < 1e-2:
                    continue
                        
                qprod = l_charge * r_charge
                energy = qprod / max(dist, 1e-6)

                electro_energies.append(energy)
                charge_products.append(qprod)
                distances.append(dist)

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


                # --------------------------------------------------
                # Polar-hydrophobic proximity
                # --------------------------------------------------

                if (
                    dist <= 4.5
                    and l_element == "C"
                    and r_element in {"O", "N"}
                ):
                    observed_polar_hydrophobic.add(j)




                # --------------------------------------------------
                # Plausible hydrogen-bond partner identification
                # --------------------------------------------------

                receptor_donor = (
                    (r_resname, r_atom_name)
                    in RNA_DONORS
                )

                receptor_acceptor = (
                    (r_resname, r_atom_name)
                    in RNA_ACCEPTORS
                )
                if dist <= 3.5:

                    if receptor_donor:
                        observed_donors.add(j)

                    if receptor_acceptor:
                        observed_acceptors.add(j)

                plausible_hbond = (
                    (l_is_donor and receptor_acceptor)
                    or
                    (l_is_acceptor and receptor_donor)
                )
                if plausible_hbond and dist <= 3.5:

                    neighbor_candidates = []

                    for k, other_coord in enumerate(r_coords):

                        if k == j:
                            continue

                        d_neighbor = np.linalg.norm(
                            other_coord - r_coord
                        )

                        if 0.5 < d_neighbor < 2.2:
                            neighbor_candidates.append(
                                (d_neighbor, other_coord)
                            )

                    angle_score = 0.0

                    if neighbor_candidates:

                        neighbor_coord = min(
                            neighbor_candidates,
                            key=lambda x: x[0]
                        )[1]

                        v1 = neighbor_coord - r_coord
                        v2 = l_coord - r_coord

                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)

                        if norm1 > 1e-6 and norm2 > 1e-6:

                            cos_theta = np.dot(
                                v1,
                                v2
                            ) / (norm1 * norm2)

                            cos_theta = np.clip(
                                cos_theta,
                                -1.0,
                                1.0
                            )

                            theta = np.degrees(
                                np.arccos(cos_theta)
                            )

                            angle_score = (
                                np.cos(
                                    np.radians(
                                        180.0 - theta
                                    )
                                ) ** 2
                            )

                    hbond_score = (
                        angle_score /
                        (dist ** 2)
                    )

                    hbond_scores.append(
                        hbond_score
                    )

                    hbond_distances.append(
                        dist
                    )
                    hbond_count += 1
                    polar_surface_contact += 1


    
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

        # Physically motivated distance-weighted charge interaction
        feats["DistWeightedChargeSum"] = float(
            sum(q / max(d, 1e-6) for q, d in zip(charge_products, distances))
        ) if distances else 0.0

        feats["NetChargeGradient"] = float(sum(abs(e) for e in electro_energies)) if electro_energies else 0.0
        feats["ElectrostaticVariance"] = float(np.var(electro_energies)) if electro_energies else 0.0
        feats["StericClashCount"] = float(steric_clashes)

        feats["IdealHBondPairCount"] = float(hbond_count)
        feats["MedianHbondDistance"] = float(np.median(hbond_distances)) if hbond_distances else 0.0
        feats["MeanHbondAngleScore"] = float(np.mean(hbond_scores)) if hbond_scores else 0.0
        feats["MaxHbondAngleScore"] = float(max(hbond_scores)) if hbond_scores else 0.0
        feats["HBondSaturationIndex"] = float(hbond_count / (len(lig_donors) + 1e-6))
        near_donors = len(observed_donors)
        near_acceptors = len(observed_acceptors)
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
        feats["PolarHydrophobicProximityScore"] = float(
            len(observed_polar_hydrophobic)
        )
        return feats

    except Exception as e:
        # Do not crash a master pipeline: return all-zeros with a clear stderr message.
        print(f"ERROR electro_hydro: {e}", file=sys.stderr)
        return feats

def run(
    folder,
    radius=6.0,
    out="electro_hydro.txt",
):
    folder = os.path.abspath(folder)

    folder_name = os.path.basename(
        os.path.normpath(folder)
    )

    pdb_path = os.path.join(
        folder,
        f"{folder_name}.pdb"
    )

    lig_path = os.path.join(
        folder,
        f"{folder_name}_lig.sd"
    )

    out_path = os.path.join(
        folder,
        out
    )

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(
            f"Missing PDB file: {pdb_path}"
        )

    if not os.path.exists(lig_path):
        raise FileNotFoundError(
            f"Missing ligand SD file: {lig_path}"
        )

    feats = compute_features(
        lig_path,
        pdb_path,
        radius=radius,
    )

    with open(out_path, "w") as f:
        for k in sorted(feats.keys()):
            f.write(f"{k}: {feats[k]:.4f}\n")

    return out_path

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "dir",
        help="Directory containing (folder_name).pdb and (folder_name)_lig.sd"
    )

    ap.add_argument(
        "--radius",
        type=float,
        default=6.0,
        help="Neighborhood radius in Å (default: 6.0)"
    )

    ap.add_argument(
        "--out",
        default="electro_hydro.txt",
        help="Output filename (default: electro_hydro.txt)"
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file"
    )

    args = ap.parse_args()

    folder = os.path.abspath(args.dir)

    if not os.path.isdir(folder):
        print(
            f"ERROR: Not a directory: {folder}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = os.path.join(
        folder,
        args.out,
    )

    if os.path.exists(out_path) and not args.overwrite:
        folder_name = os.path.basename(
            os.path.normpath(folder)
        )

        print(
            f"⏭️ Skipping {folder_name} "
            f"({args.out} already exists)"
        )

        print(f"Wrote: {out_path}")
        return

    try:
        written = run(
            folder,
            radius=args.radius,
            out=args.out,
        )

    except Exception as e:
        print(
            f"ERROR: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    folder_name = os.path.basename(
        os.path.normpath(folder)
    )


if __name__ == "__main__":
    main()

