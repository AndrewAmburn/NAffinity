from __future__ import annotations

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


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_steps_dir() -> Path:
    return Path(__file__).resolve().parent / "steps"


def get_default_model_path() -> Path:
    return get_repo_root() / "model" / "naffinity.joblib"


def _run_cmd(cmd: list[str]):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr, " ".join(cmd)


def run_step(
    python_executable: str,
    script_path: Path,
    folder: Path,
    overwrite: bool,
    model_path: Path | None = None,
) -> None:
    cmd = [python_executable, str(script_path), str(folder)]

    if model_path is not None and script_path.name == PREDICT_STEP:
        cmd.extend(["--model", str(model_path)])

    if overwrite and script_path.name in OVERWRITE_SUPPORTED:
        cmd.append("--overwrite")

    code, out, err, cmd_str = _run_cmd(cmd)
    if code != 0:
        if out:
            print(out, end="")
        if err:
            print(err, end="", file=sys.stderr)
        raise RuntimeError(
            f"Step failed: {script_path.name} (exit code {code})\nCommand: {cmd_str}"
        )


def max_workers_from_jobs(jobs: int) -> int:
    if jobs == -1:
        return os.cpu_count() or 1
    return max(1, int(jobs))


def run_pipeline(
    input_dir: str | Path,
    python_executable: str | None = None,
    overwrite: bool = False,
    jobs: int = -1,
    model_path: str | Path | None = None,
) -> Path:
    folder = Path(input_dir).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    folder_name = folder.name
    pdb_path = folder / f"{folder_name}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"Missing PDB file: {pdb_path}")

    steps_dir = get_steps_dir()
    python_executable = python_executable or sys.executable
    resolved_model_path = (
        Path(model_path).expanduser().resolve()
        if model_path is not None
        else get_default_model_path()
    )

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

    required = [LIGAND_STEP, *PARALLEL_STEPS, PREDICT_STEP]
    missing = [s for s in required if not (steps_dir / s).exists()]
    if missing:
        missing_str = "\n".join(f"  - {s}" for s in missing)
        raise FileNotFoundError(
            "Missing pipeline scripts in src/naffinity/steps:\n"
            f"{missing_str}"
        )

    print(f"Running NAffinity on: {folder}")

    run_step(
        python_executable=python_executable,
        script_path=steps_dir / LIGAND_STEP,
        folder=folder,
        overwrite=overwrite,
    )

    workers = max_workers_from_jobs(jobs)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for script_name in PARALLEL_STEPS:
            script_path = steps_dir / script_name
            cmd = [python_executable, str(script_path), str(folder)]
            if overwrite and script_name in OVERWRITE_SUPPORTED:
                cmd.append("--overwrite")
            futures.append(executor.submit(_run_cmd, cmd))

        for future in as_completed(futures):
            code, out, err, cmd_str = future.result()
            if code != 0:
                if out:
                    print(out, end="")
                if err:
                    print(err, end="", file=sys.stderr)
                raise RuntimeError(
                    f"Feature step failed (exit code {code})\nCommand: {cmd_str}"
                )

    run_step(
        python_executable=python_executable,
        script_path=steps_dir / PREDICT_STEP,
        folder=folder,
        overwrite=overwrite,
        model_path=resolved_model_path,
    )

    out_file = folder / "naffinity_predicted_binding_class.txt"
    if not out_file.exists():
        raise FileNotFoundError(f"Prediction output file not found: {out_file}")

    return out_file