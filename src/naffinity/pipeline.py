from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .steps.ligand_extraction import run as ligand_run
from .steps.rdkit_features import run as rdkit_run
from .steps.electro_hydro import run as electro_run
from .steps.descriptors import run as descriptors_run
from .steps.receptor_descriptors import run as receptor_run
from .steps.naffinity_predict import run as predict_run

def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def get_default_model_path() -> Path:
    return get_repo_root() / "model" / "naffinity.joblib"

def max_workers_from_jobs(jobs: int) -> int:
    if jobs == -1:
        return os.cpu_count() or 1
    return max(1, int(jobs))

def run_pipeline(
    input_dir: str | Path,
    jobs: int = -1,
    model_path: str | Path | None = None,
) -> Path:

    folder = Path(input_dir).expanduser().resolve()

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {folder}"
        )

    folder_name = folder.name

    pdb_path = folder / f"{folder_name}.pdb"

    if not pdb_path.exists():
        raise FileNotFoundError(
            f"Missing PDB file: {pdb_path}"
        )

    resolved_model_path = (
        Path(model_path).expanduser().resolve()
        if model_path is not None
        else get_default_model_path()
    )

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: "
            f"{resolved_model_path}"
        )


    # -----------------------------
    # Ligand extraction
    # -----------------------------

    ligand_run(str(folder))

    # -----------------------------
    # Feature generation
    # -----------------------------

    workers = max_workers_from_jobs(jobs)

    feature_functions = [
        rdkit_run,
        electro_run,
        descriptors_run,
        receptor_run,
    ]

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = [
            executor.submit(
                func,
                str(folder),
            )
            for func in feature_functions
        ]

        for future in as_completed(futures):
            future.result()

    # -----------------------------
    # Prediction
    # -----------------------------

    predict_run(
        str(folder),
        model_path=str(resolved_model_path),
    )

    out_file = (
        folder /
        "naffinity_predicted_binding_class.txt"
    )

    if not out_file.exists():
        raise FileNotFoundError(
            f"Prediction output file not found: "
            f"{out_file}"
        )

    return out_file
