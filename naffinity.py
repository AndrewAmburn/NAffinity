#!/usr/bin/env python3
"""
NAffinity master runner (single folder).

Pipeline:
1) ligand_extraction.py
2) [PARALLEL] rdkit_features.py, electro_hydro.py, descriptors.py, receptor_descriptors.py
3) naffinity_predict.py  -> writes naffinity_predicted_binding_class.txt

Printing:
- Only prints:
  Running NAffinity on: <folder>
  Prediction complete.
  Output: <path>

Usage:
  python3 naffinity.py /path/to/complex_folder
  python3 naffinity.py /path/to/complex_folder --overwrite
  python3 naffinity.py /path/to/complex_folder --jobs -1
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


LIGAND_STEP = "ligand_extraction.py"
PARALLEL_STEPS = [
    "rdkit_features.py",
    "electro_hydro.py",
    "descriptors.py",
    "receptor_descriptors.py",
]
PREDICT_STEP = "naffinity_predict.py"

OVERWRITE_SUPPORTED = set(PARALLEL_STEPS)


def _run_cmd(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr, " ".join(cmd)


def run_step(python_exe: str, script_path: Path, folder: Path, overwrite: bool) -> None:
    cmd = [python_exe, str(script_path), str(folder)]
    if overwrite and script_path.name in OVERWRITE_SUPPORTED:
        cmd.append("--overwrite")

    code, out, err, cmd_str = _run_cmd(cmd)
    if code != 0:
        if out:
            print(out, end="")
        if err:
            print(err, end="", file=sys.stderr)
        raise RuntimeError(f"Step failed: {script_path.name} (exit code {code})\nCommand: {cmd_str}")


def max_workers_from_jobs(jobs: int) -> int:
    if jobs == -1:
        return os.cpu_count() or 1
    return max(1, int(jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Complex folder containing (folder_name).pdb")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use (default: current interpreter)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs where supported")
    ap.add_argument("--jobs", type=int, default=-1, help="Parallel workers for feature steps (-1 = all cores)")
    args = ap.parse_args()

    folder = Path(args.dir).expanduser().resolve()
    if not folder.is_dir():
        print(f"ERROR: Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    folder_name = folder.name
    pdb_path = folder / f"{folder_name}.pdb"
    if not pdb_path.exists():
        print(f"ERROR: Missing PDB file: {pdb_path}", file=sys.stderr)
        sys.exit(1)

    repo_dir = Path(__file__).resolve().parent

    # Validate scripts exist
    required = [LIGAND_STEP, *PARALLEL_STEPS, PREDICT_STEP]
    missing = [s for s in required if not (repo_dir / s).exists()]
    if missing:
        print("ERROR: Missing pipeline scripts in repo root:", file=sys.stderr)
        for s in missing:
            print(f"  - {s}", file=sys.stderr)
        sys.exit(1)

    print(f"Running NAffinity on: {folder}")

    try:
        # 1) ligand extraction (must be first)
        run_step(args.python, repo_dir / LIGAND_STEP, folder, overwrite=args.overwrite)

        # 2) feature steps in parallel (all depend only on PDB + _lig.sd)
        workers = max_workers_from_jobs(args.jobs)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for s in PARALLEL_STEPS:
                script_path = repo_dir / s
                cmd = [args.python, str(script_path), str(folder)]
                if args.overwrite and s in OVERWRITE_SUPPORTED:
                    cmd.append("--overwrite")
                futures.append(ex.submit(_run_cmd, cmd))

            for fut in as_completed(futures):
                code, out, err, cmd_str = fut.result()
                if code != 0:
                    if out:
                        print(out, end="")
                    if err:
                        print(err, end="", file=sys.stderr)
                    raise RuntimeError(f"Feature step failed (exit code {code})\nCommand: {cmd_str}")

        # 3) prediction (depends on feature txt files)
        run_step(args.python, repo_dir / PREDICT_STEP, folder, overwrite=args.overwrite)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("Prediction complete.")

    out_file = folder / "naffinity_predicted_binding_class.txt"
    if out_file.exists():
        print(f"Output: {out_file}")
    else:
        print("WARNING: Prediction output file not found (naffinity_predicted_binding_class.txt).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()