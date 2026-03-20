from pathlib import Path
from unittest.mock import patch

import pytest

from naffinity.pipeline import (
    get_default_model_path,
    get_repo_root,
    get_steps_dir,
    max_workers_from_jobs,
    run_pipeline,
    run_step,
)


def test_get_repo_root_exists():
    repo_root = get_repo_root()
    assert repo_root.exists()
    assert repo_root.is_dir()


def test_get_steps_dir_is_path_object():
    steps_dir = get_steps_dir()
    assert isinstance(steps_dir, Path)


def test_get_default_model_path_points_to_model_folder():
    model_path = get_default_model_path()
    assert model_path.name == "naffinity.joblib"
    assert model_path.parent.name == "model"


def test_max_workers_from_jobs_all_cores():
    workers = max_workers_from_jobs(-1)
    assert workers >= 1


def test_max_workers_from_jobs_positive_value():
    assert max_workers_from_jobs(4) == 4
    assert max_workers_from_jobs(1) == 1


def test_max_workers_from_jobs_zero_defaults_to_one():
    assert max_workers_from_jobs(0) == 1


def test_run_step_adds_model_flag_for_predict_step(tmp_path):
    script_path = tmp_path / "naffinity_predict.py"
    script_path.write_text("# dummy\n")
    folder = tmp_path / "3GAO"
    folder.mkdir()
    model_path = tmp_path / "model" / "naffinity.joblib"
    model_path.parent.mkdir()
    model_path.write_text("dummy model")

    with patch("naffinity.pipeline._run_cmd", return_value=(0, "", "", "")) as mock_run_cmd:
        run_step(
            python_executable="python",
            script_path=script_path,
            folder=folder,
            overwrite=False,
            model_path=model_path,
        )

    cmd = mock_run_cmd.call_args[0][0]
    assert cmd == [
        "python",
        str(script_path),
        str(folder),
        "--model",
        str(model_path),
    ]


def test_run_step_adds_overwrite_for_supported_steps(tmp_path):
    script_path = tmp_path / "rdkit_features.py"
    script_path.write_text("# dummy\n")
    folder = tmp_path / "3GAO"
    folder.mkdir()

    with patch("naffinity.pipeline._run_cmd", return_value=(0, "", "", "")) as mock_run_cmd:
        run_step(
            python_executable="python",
            script_path=script_path,
            folder=folder,
            overwrite=True,
        )

    cmd = mock_run_cmd.call_args[0][0]
    assert cmd == [
        "python",
        str(script_path),
        str(folder),
        "--overwrite",
    ]


def test_run_step_raises_runtime_error_on_failure(tmp_path):
    script_path = tmp_path / "dummy.py"
    script_path.write_text("print('hello')\n")
    folder = tmp_path / "complex"
    folder.mkdir()

    with patch(
        "naffinity.pipeline._run_cmd",
        return_value=(1, "out", "err", "python dummy.py"),
    ):
        with pytest.raises(RuntimeError, match="Step failed"):
            run_step(
                python_executable="python",
                script_path=script_path,
                folder=folder,
                overwrite=False,
            )


def test_run_pipeline_raises_for_missing_directory():
    with pytest.raises(NotADirectoryError):
        run_pipeline("/this/path/does/not/exist")


def test_run_pipeline_raises_for_missing_pdb(tmp_path):
    folder = tmp_path / "3GAO"
    folder.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing PDB file"):
        run_pipeline(folder)


def test_run_pipeline_raises_for_missing_model(tmp_path):
    folder = tmp_path / "3GAO"
    folder.mkdir()
    (folder / "3GAO.pdb").write_text("HEADER TEST\n")

    fake_steps_dir = tmp_path / "steps"
    fake_steps_dir.mkdir()

    for script_name in [
        "ligand_extraction.py",
        "rdkit_features.py",
        "electro_hydro.py",
        "descriptors.py",
        "receptor_descriptors.py",
        "naffinity_predict.py",
    ]:
        (fake_steps_dir / script_name).write_text("# dummy script\n")

    missing_model = tmp_path / "model" / "naffinity.joblib"

    with patch("naffinity.pipeline.get_steps_dir", return_value=fake_steps_dir):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            run_pipeline(folder, model_path=missing_model)


def test_run_pipeline_raises_for_missing_scripts(tmp_path):
    folder = tmp_path / "3GAO"
    folder.mkdir()
    (folder / "3GAO.pdb").write_text("HEADER TEST\n")

    fake_steps_dir = tmp_path / "steps"
    fake_steps_dir.mkdir()

    fake_model = tmp_path / "model" / "naffinity.joblib"
    fake_model.parent.mkdir()
    fake_model.write_text("dummy model")

    with patch("naffinity.pipeline.get_steps_dir", return_value=fake_steps_dir):
        with pytest.raises(FileNotFoundError, match="Missing pipeline scripts"):
            run_pipeline(folder, model_path=fake_model)


def test_run_pipeline_returns_output_path_when_successful(tmp_path):
    folder = tmp_path / "3GAO"
    folder.mkdir()
    (folder / "3GAO.pdb").write_text("HEADER TEST\n")

    fake_steps_dir = tmp_path / "steps"
    fake_steps_dir.mkdir()

    for script_name in [
        "ligand_extraction.py",
        "rdkit_features.py",
        "electro_hydro.py",
        "descriptors.py",
        "receptor_descriptors.py",
        "naffinity_predict.py",
    ]:
        (fake_steps_dir / script_name).write_text("# dummy script\n")

    fake_model = tmp_path / "model" / "naffinity.joblib"
    fake_model.parent.mkdir()
    fake_model.write_text("dummy model")

    expected_output = folder / "naffinity_predicted_binding_class.txt"

    def fake_run_step(*args, **kwargs):
        script_path = kwargs["script_path"]
        if script_path.name == "naffinity_predict.py":
            expected_output.write_text("PredictedClass: Weak/moderate binder\n")

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class DummyExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.futures = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, cmd):
            future = DummyFuture((0, "", "", " ".join(cmd)))
            self.futures.append(future)
            return future

    def fake_as_completed(futures):
        for future in futures:
            yield future

    with patch("naffinity.pipeline.get_steps_dir", return_value=fake_steps_dir), \
         patch("naffinity.pipeline.run_step", side_effect=fake_run_step), \
         patch("naffinity.pipeline.ProcessPoolExecutor", DummyExecutor), \
         patch("naffinity.pipeline.as_completed", fake_as_completed):
        output = run_pipeline(folder, jobs=2, model_path=fake_model)

    assert output == expected_output
    assert expected_output.exists()


def test_run_pipeline_raises_if_prediction_output_missing(tmp_path):
    folder = tmp_path / "3GAO"
    folder.mkdir()
    (folder / "3GAO.pdb").write_text("HEADER TEST\n")

    fake_steps_dir = tmp_path / "steps"
    fake_steps_dir.mkdir()

    for script_name in [
        "ligand_extraction.py",
        "rdkit_features.py",
        "electro_hydro.py",
        "descriptors.py",
        "receptor_descriptors.py",
        "naffinity_predict.py",
    ]:
        (fake_steps_dir / script_name).write_text("# dummy script\n")

    fake_model = tmp_path / "model" / "naffinity.joblib"
    fake_model.parent.mkdir()
    fake_model.write_text("dummy model")

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class DummyExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.futures = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, cmd):
            future = DummyFuture((0, "", "", " ".join(cmd)))
            self.futures.append(future)
            return future

    def fake_as_completed(futures):
        for future in futures:
            yield future

    with patch("naffinity.pipeline.get_steps_dir", return_value=fake_steps_dir), \
         patch("naffinity.pipeline.run_step"), \
         patch("naffinity.pipeline.ProcessPoolExecutor", DummyExecutor), \
         patch("naffinity.pipeline.as_completed", fake_as_completed):
        with pytest.raises(FileNotFoundError, match="Prediction output file not found"):
            run_pipeline(folder, jobs=2, model_path=fake_model)