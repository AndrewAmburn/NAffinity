#!/usr/bin/env python3
"""
Generate NAffinity "descriptors.txt" for a single complex folder.

Input (same convention as ligand_extraction.py / rdkit_features.py):
  - directory basename = folder_name
  - expects:
      (folder_name).pdb
      (folder_name)_lig.sd

Output:
  - writes descriptors.txt inside the same directory

Usage:
  python3 descriptors_features.py /path/to/complex_folder
  python3 descriptors_features.py /path/to/complex_folder --radius 6.0
  python3 descriptors_features.py /path/to/complex_folder --overwrite
"""

import argparse
import os
import re
import sys

import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
from scipy.spatial import ConvexHull


# --- Settings (kept from your notebook defaults) ---
BINDING_SITE_RADIUS = 6.0
HBOND_DISTANCE = 3.5
PI_STACKING_DISTANCE = 5.0
ELECTROSTATIC_DISTANCE_CUTOFF = 6.0

MONOVALENT_METALS = {"NA", "K"}
DIVALENT_METALS = {"MG", "ZN", "MN", "FE", "CA"}
METAL_ELEMENTS = MONOVALENT_METALS.union(DIVALENT_METALS)

WATER_NAMES = {"HOH", "WAT"}
PHOSPHATE_CHARGE = -1.0


def is_rna_residue(res):
    return res.get_id()[0] == " " and res.get_resname().strip().upper() in {"A", "U", "G", "C", "I", "DA", "DT", "DG", "DC"}


def is_water(res):
    return res.get_resname().strip().upper() in WATER_NAMES


def is_metal(atom):
    elem = (atom.element or "").strip().upper()
    return elem in METAL_ELEMENTS


def manual_metal_distances(pdb_path, ligand_coords):
    """
    Reads PDB HETATM records, collects metal coordinates (by element), returns:
      - counts per metal element
      - min distance per metal element (if present)
    """
    metals = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            atom_name = line[12:16].strip()
            # crude but consistent with your notebook
            elem = re.sub("[^A-Z]", "", atom_name.strip().upper())
            if elem in METAL_ELEMENTS:
                x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])
                metals.append((np.array([x, y, z], dtype=float), elem))

    feats = {}
    if not metals or ligand_coords is None or len(ligand_coords) == 0:
        # still return 0-filled features for known metals so downstream is stable
        for m in sorted(METAL_ELEMENTS):
            feats[f"NumMetal_{m}"] = 0
            feats[f"MinDistMetal_{m}"] = 0.0
        return feats

    lig = np.array(ligand_coords, dtype=float)
    by_elem = {}
    for coord, elem in metals:
        by_elem.setdefault(elem, []).append(coord)

    for m in sorted(METAL_ELEMENTS):
        coords = by_elem.get(m, [])
        feats[f"NumMetal_{m}"] = len(coords)
        if coords:
            dmin = min(float(np.min(np.linalg.norm(lig - c, axis=1))) for c in coords)
            feats[f"MinDistMetal_{m}"] = dmin
        else:
            feats[f"MinDistMetal_{m}"] = 0.0

    return feats


def extract_ligand_mol_from_sd(sd_path):
    suppl = Chem.SDMolSupplier(sd_path, removeHs=False, sanitize=False)
    mol = suppl[0] if suppl and len(suppl) > 0 else None
    if mol is None:
        return None

    # suppress noisy RDKit warnings
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        for atom in mol.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in mol.GetBonds():
            bond.SetIsAromatic(False)
        Chem.SanitizeMol(mol)

    if mol.GetNumConformers() == 0:
        raise ValueError("Ligand SD has no conformer/coordinates.")
    return mol


def extract_ligand_features(mol):
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "NumChiralCenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "MolarRefractivity": Descriptors.MolMR(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
    }


def compute_binding_site_features(ligand_coords, rna_atoms, water_atoms, metal_atoms, cofactor_atoms, ligand_charge, pdb_path, radius):
    combined_atoms = [a for a in (rna_atoms + water_atoms + metal_atoms) if hasattr(a, "coord") and len(a.coord) == 3]
    if not combined_atoms:
        return {}

    ns = NeighborSearch(combined_atoms)
    nearby_atoms = set()
    for coord in ligand_coords:
        nearby_atoms.update(ns.search(coord, radius))

    local_rna_atoms = [a for a in nearby_atoms if a in rna_atoms]
    local_water_atoms = [a for a in nearby_atoms if a in water_atoms]

    def min_dist_atoms_to_coords(coords, atoms):
        if coords is None or len(coords) == 0 or not atoms:
            return 0.0
        return float(min(np.linalg.norm(c - a.coord) for c in coords for a in atoms))

    min_cofactor_dist = min_dist_atoms_to_coords(ligand_coords, cofactor_atoms)

    local_residues = {a.get_parent() for a in local_rna_atoms}
    base_counts = {"A": 0, "U": 0, "G": 0, "C": 0, "I": 0}
    phosphate_contacts = 0  # kept for compatibility with your logic
    phosphate_atoms = []

    for res in local_residues:
        base = res.get_resname().strip().upper()
        if base in base_counts:
            base_counts[base] += 1
        phosphate_atoms += [a for a in res.get_atoms() if "P" in a.get_id()]

    # simple proximity counts
    hbond_count = sum(
        1
        for la in ligand_coords
        for ra in local_rna_atoms
        if np.linalg.norm(la - ra.coord) < HBOND_DISTANCE
    )

    pi_stack_count = 0
    for res in local_residues:
        rname = res.get_resname().strip().upper()
        if rname in base_counts:
            base_atoms = [a.coord for a in res.get_atoms() if (a.element or "").upper() != "H"]
            if base_atoms:
                rna_centroid = np.mean(base_atoms, axis=0)
                if any(np.linalg.norm(rna_centroid - la) < PI_STACKING_DISTANCE for la in ligand_coords):
                    pi_stack_count += 1

    try:
        pocket_volume = float(ConvexHull(np.array([a.coord for a in local_rna_atoms], dtype=float)).volume) if local_rna_atoms else 0.0
    except Exception:
        pocket_volume = 0.0

    # phosphate electrostatic proxy
    electrostatic_score = 0.0
    if phosphate_atoms and ligand_coords is not None and len(ligand_coords) > 0:
        for la in ligand_coords:
            for pa in phosphate_atoms:
                d = np.linalg.norm(la - pa.coord)
                if d < ELECTROSTATIC_DISTANCE_CUTOFF:
                    electrostatic_score += PHOSPHATE_CHARGE * float(ligand_charge) / (float(d) + 1e-3)

    buried_atoms = sum(1 for la in ligand_coords if any(np.linalg.norm(la - ra.coord) < 4.0 for ra in local_rna_atoms))
    sasa_proxy = float(buried_atoms / len(ligand_coords)) if ligand_coords is not None and len(ligand_coords) > 0 else 0.0

    is_helix = phosphate_contacts > 4 and (base_counts["A"] + base_counts["U"] > 2)
    is_loop = phosphate_contacts <= 2 and (base_counts["G"] >= 2)
    is_junction = (not is_helix) and (not is_loop)

    base_total = sum(base_counts.values())
    base_fractions = {f"Frac{b}": (base_counts[b] / base_total if base_total else 0.0) for b in base_counts}

    metal_feats = manual_metal_distances(pdb_path, ligand_coords)

    return {
        "NumBindingSiteAtoms": float(len(local_rna_atoms)),
        "NumWaterMolecules": float(len(set(a.get_parent() for a in local_water_atoms))),
        "MinDistCofactor": float(min_cofactor_dist),
        "NumPhosphateContacts": float(phosphate_contacts),
        "NumHbondCandidates": float(hbond_count),
        "NumPiStacking": float(pi_stack_count),
        "BindingPocketVolume": float(pocket_volume),
        "ElectrostaticScore": float(electrostatic_score),
        "LigandSASAProxy": float(sasa_proxy),
        "IsHelixSite": float(int(is_helix)),
        "IsLoopSite": float(int(is_loop)),
        "IsJunctionSite": float(int(is_junction)),
        **base_fractions,
        **metal_feats,
    }


def process_single_folder(folder, radius):
    folder_name = os.path.basename(os.path.normpath(folder))
    pdb_path = os.path.join(folder, f"{folder_name}.pdb")
    sd_path = os.path.join(folder, f"{folder_name}_lig.sd")
    out_path = os.path.join(folder, "descriptors.txt")

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Missing PDB file: {pdb_path}")
    if not os.path.exists(sd_path):
        raise FileNotFoundError(f"Missing ligand SD file: {sd_path}")

    mol = extract_ligand_mol_from_sd(sd_path)
    if mol is None:
        raise ValueError(f"Could not load ligand from {sd_path}")

    ligand_coords = mol.GetConformer().GetPositions()
    ligand_charge = Chem.GetFormalCharge(mol)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(folder_name, pdb_path)
    model = structure[0]

    rna_atoms = []
    water_atoms = []
    metal_atoms = []
    cofactor_atoms = []

    for chain in model:
        for res in chain:
            if is_rna_residue(res):
                rna_atoms.extend(list(res.get_atoms()))
            elif is_water(res):
                water_atoms.extend(list(res.get_atoms()))
            else:
                for atom in res.get_atoms():
                    if is_metal(atom):
                        metal_atoms.append(atom)
                    elif res.id[0].startswith("H_"):  # optional: cofactors
                        cofactor_atoms.append(atom)

    ligand_feats = extract_ligand_features(mol)
    site_feats = compute_binding_site_features(
        ligand_coords=np.array(ligand_coords, dtype=float),
        rna_atoms=rna_atoms,
        water_atoms=water_atoms,
        metal_atoms=metal_atoms,
        cofactor_atoms=cofactor_atoms,
        ligand_charge=ligand_charge,
        pdb_path=pdb_path,
        radius=radius,
    )

    feats = {}
    feats.update(ligand_feats)
    feats.update(site_feats)

    with open(out_path, "w") as f:
        for k, v in feats.items():
            f.write(f"{k}: {v}\n")

    return out_path


def main():
    RDLogger.DisableLog("rdApp.*")

    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Directory containing (folder_name).pdb and (folder_name)_lig.sd")
    ap.add_argument("--radius", type=float, default=BINDING_SITE_RADIUS, help=f"Binding-site radius in Å (default: {BINDING_SITE_RADIUS})")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite descriptors.txt if it exists")
    args = ap.parse_args()

    folder = os.path.abspath(args.dir)
    if not os.path.isdir(folder):
        print(f"ERROR: Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    out_path = os.path.join(folder, "descriptors.txt")
    if os.path.exists(out_path) and not args.overwrite:
        folder_name = os.path.basename(os.path.normpath(folder))
        print(f"⏭️ Skipping {folder_name} (descriptors.txt already exists)")
        print(f"Wrote: {out_path}")
        return

    try:
        written = process_single_folder(folder, radius=args.radius)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(folder))
    print(f"✅ Wrote descriptors.txt for {folder_name}")
    print(f"Wrote: {written}")


if __name__ == "__main__":
    main()