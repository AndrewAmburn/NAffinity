#!/usr/bin/env python3
"""
Extract the most likely ligand from a nucleic acid–ligand complex PDB and write it as:
  (folder_name)_lig.sd

Usage:
  python extract_ligand_sd.py /path/to/complex_folder
  python extract_ligand_sd.py /path/to/complex_folder --ligand-resname AMP
  python extract_ligand_sd.py /path/to/complex_folder --ligand-chain A --ligand-resid 401
"""

import argparse
import os
import sys
from math import sqrt
from typing import Dict, List, Tuple, Optional

from rdkit import Chem


WATER_RESNAMES = {"HOH", "WAT", "DOD"}
# Common ions / crystallization components to ignore as "ligand"
IGNORE_RESNAMES = {
    "NA", "K", "CL", "BR", "I",
    "MG", "CA", "ZN", "MN", "FE", "CO", "NI", "CU", "CD",
    "SO4", "PO4", "NO3",
}
# Nucleic acid residue names commonly found in PDBs (RNA/DNA, common variants)
NA_RESNAMES = {
    "A", "U", "G", "C", "I",
    "DA", "DT", "DG", "DC", "DI",
    "RA", "RU", "RG", "RC",
    "ADE", "URA", "GUA", "CYT", "THY",
}

def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def parse_pdb_atoms(pdb_path: str):
    """
    Returns:
      na_coords: list of coords for nucleic-acid ATOM records
      het_groups: dict keyed by (resname, chain, resid, icode) -> list of hetatm lines + coords + elements
    """
    na_coords: List[Tuple[float, float, float]] = []
    het_groups: Dict[Tuple[str, str, int, str], Dict[str, list]] = {}

    with open(pdb_path, "r") as f:
        for line in f:
            if len(line) < 54:
                continue

            rec = line[0:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                continue

            resname = line[17:20].strip().upper()
            chain = (line[21] or " ").strip() or " "
            resid_str = line[22:26].strip()
            icode = (line[26] or " ").strip() or " "

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            # Element: prefer columns 76-78, fallback to atom name first letter(s)
            element = (line[76:78].strip() or line[12:16].strip()[0]).upper()
            # Normalize common PDB quirks
            if element and len(element) == 2 and element[1].islower():
                element = element[0] + element[1].upper()

            if rec == "ATOM":
                if resname in NA_RESNAMES:
                    na_coords.append((x, y, z))

            elif rec == "HETATM":
                if resname in WATER_RESNAMES:
                    continue
                key = (resname, chain, int(resid_str) if resid_str else -9999, icode)
                if key not in het_groups:
                    het_groups[key] = {"lines": [], "coords": [], "elements": []}
                het_groups[key]["lines"].append(line.rstrip("\n"))
                het_groups[key]["coords"].append((x, y, z))
                het_groups[key]["elements"].append(element)

    return na_coords, het_groups

def score_candidate(
    key: Tuple[str, str, int, str],
    group: Dict[str, list],
    na_coords: List[Tuple[float, float, float]],
) -> Tuple[float, Dict[str, float]]:
    """
    Heuristic scoring to pick the "ligand" residue among HETATM groups.
    Returns (score, diagnostics)
    """
    resname, chain, resid, icode = key
    elements = group["elements"]

    # Ignore obvious non-ligand residues
    if resname in IGNORE_RESNAMES:
        return -1e9, {"reason": "ignored_resname"}

    heavy = sum(1 for e in elements if e != "H")
    if heavy < 6:
        return -1e9, {"reason": "too_small", "heavy_atoms": heavy}

    # Compute min distance to any nucleic acid atom (if NA coords exist)
    min_d = 1e9
    if na_coords:
        for c in group["coords"]:
            # small optimization: early break if already very close
            for n in na_coords:
                d = _dist(c, n)
                if d < min_d:
                    min_d = d
                    if min_d < 2.0:
                        break
            if min_d < 2.0:
                break

    # Favor residues near NA (typical binding site); penalize far-away hetero groups
    # The score is dominated by size among "near" candidates, but still considers distance.
    near_bonus = 0.0
    far_penalty = 0.0
    if min_d <= 6.0:
        near_bonus = 50.0
    elif min_d <= 10.0:
        near_bonus = 10.0
    else:
        far_penalty = 50.0

    # Organic-likeness: reward presence of carbon, penalize if it's mostly metals
    has_c = any(e == "C" for e in elements)
    metal_count = sum(1 for e in elements if e in {"MG","ZN","MN","FE","CA","CO","NI","CU","CD","NA","K","CL","BR","I"})
    organic_bonus = 10.0 if has_c else 0.0
    metal_penalty = 10.0 * metal_count

    # Final score
    score = (
        near_bonus
        + organic_bonus
        + 2.0 * heavy
        - 1.5 * min(min_d, 20.0)  # small preference for closer
        - far_penalty
        - metal_penalty
    )

    diag = {
        "heavy_atoms": float(heavy),
        "min_dist_to_na": float(min_d if min_d < 1e8 else -1.0),
        "metal_atoms": float(metal_count),
        "has_carbon": float(1.0 if has_c else 0.0),
    }
    return score, diag

def choose_ligand_group(
    het_groups: Dict[Tuple[str, str, int, str], Dict[str, list]],
    na_coords: List[Tuple[float, float, float]],
    override_resname: Optional[str] = None,
    override_chain: Optional[str] = None,
    override_resid: Optional[int] = None,
) -> Tuple[Tuple[str, str, int, str], Dict[str, float]]:
    # If overrides are provided, try to match exactly.
    if override_resname or override_chain or override_resid is not None:
        candidates = []
        for key in het_groups.keys():
            resname, chain, resid, icode = key
            if override_resname and resname != override_resname.upper():
                continue
            if override_chain and chain != override_chain:
                continue
            if override_resid is not None and resid != override_resid:
                continue
            candidates.append(key)

        if len(candidates) == 1:
            return candidates[0], {"override_used": 1.0}
        if len(candidates) > 1:
            # If multiple match (e.g., multiple residues with same resname), pick best-scoring among them
            best = None
            best_score = -1e18
            best_diag = {}
            for key in candidates:
                s, d = score_candidate(key, het_groups[key], na_coords)
                if s > best_score:
                    best, best_score, best_diag = key, s, d
            best_diag["override_used"] = 1.0
            return best, best_diag

        raise ValueError("Override did not match any HETATM residue group in the PDB.")

    # Otherwise, score all and pick best
    best_key = None
    best_score = -1e18
    best_diag = {}
    for key, group in het_groups.items():
        s, d = score_candidate(key, group, na_coords)
        if s > best_score:
            best_key, best_score, best_diag = key, s, d

    if best_key is None or best_score < -1e8:
        raise ValueError("No suitable ligand candidate found (all HETATM groups filtered out).")

    best_diag["override_used"] = 0.0
    best_diag["score"] = float(best_score)
    return best_key, best_diag

def ligand_group_to_rdkit_mol(lines: List[str]) -> Chem.Mol:
    """
    Convert a residue's PDB HETATM block to an RDKit Mol.
    Note: RDKit may fail for some PDBs that lack connectivity. If it fails, the script errors clearly.
    """
    pdb_block = "\n".join(lines) + "\nEND\n"
    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("RDKit could not parse ligand PDB block into a molecule (likely missing/ambiguous connectivity).")

    # Try sanitization; if aromaticity causes trouble, de-aromatize and retry
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            for atom in mol.GetAtoms():
                atom.SetIsAromatic(False)
            for bond in mol.GetBonds():
                bond.SetIsAromatic(False)
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise ValueError(f"RDKit failed to sanitize ligand molecule: {e}")

    return mol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Directory containing (folder_name).pdb")
    ap.add_argument("--ligand-resname", default=None, help="Force ligand residue name (e.g., AMP)")
    ap.add_argument("--ligand-chain", default=None, help="Force ligand chain ID (e.g., A)")
    ap.add_argument("--ligand-resid", type=int, default=None, help="Force ligand residue number (e.g., 401)")
    args = ap.parse_args()

    folder = os.path.abspath(args.dir)
    if not os.path.isdir(folder):
        print(f"ERROR: Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(folder))
    pdb_path = os.path.join(folder, f"{folder_name}.pdb")
    out_sd = os.path.join(folder, f"{folder_name}_lig.sd")

    if not os.path.exists(pdb_path):
        print(f"ERROR: Expected PDB not found: {pdb_path}", file=sys.stderr)
        sys.exit(1)

    na_coords, het_groups = parse_pdb_atoms(pdb_path)
    if not het_groups:
        print("ERROR: No HETATM residue groups found in the PDB.", file=sys.stderr)
        sys.exit(1)

    try:
        best_key, diag = choose_ligand_group(
            het_groups,
            na_coords,
            override_resname=args.ligand_resname,
            override_chain=args.ligand_chain,
            override_resid=args.ligand_resid,
        )
    except Exception as e:
        print(f"ERROR selecting ligand: {e}", file=sys.stderr)
        sys.exit(1)

    resname, chain, resid, icode = best_key
    lines = het_groups[best_key]["lines"]

    try:
        mol = ligand_group_to_rdkit_mol(lines)
    except Exception as e:
        print(f"ERROR converting ligand to RDKit mol: {e}", file=sys.stderr)
        print("Fallback suggestion: write the ligand HETATM block to a .pdb and convert with OpenBabel (obabel) if installed.", file=sys.stderr)
        sys.exit(1)

    w = Chem.SDWriter(out_sd)
    w.write(mol)
    w.close()

    print(f"Selected ligand: resname={resname} chain={chain} resid={resid} icode={icode}")
    if "min_dist_to_na" in diag:
        print(f"  heavy atoms={int(diag.get('heavy_atoms', -1))} minimum distance to receptor={diag.get('min_dist_to_na', -1):.2f} Å")
    print(f"Wrote: {out_sd}")

if __name__ == "__main__":
    main()