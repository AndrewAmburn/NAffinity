import sys
from pathlib import Path
from unittest.mock import patch

from naffinity.cli import main


def test_cli_without_subcommand_prints_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["naffinity"])
    main()
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "run" in captured.out


def test_cli_run_subcommand_calls_pipeline(capsys, monkeypatch, tmp_path):
    input_dir = tmp_path / "3GAO"
    input_dir.mkdir()

    expected_output = input_dir / "naffinity_predicted_binding_class.txt"

    with patch("naffinity.cli.run_pipeline", return_value=expected_output) as mock_run_pipeline:
        monkeypatch.setattr(
            sys,
            "argv",
            ["naffinity", "run", str(input_dir)],
        )
        main()

    mock_run_pipeline.assert_called_once_with(
        str(input_dir),
        python_executable=None,
        overwrite=False,
        jobs=-1,
    )

    captured = capsys.readouterr()
    assert "Prediction complete." in captured.out
    assert str(expected_output) in captured.out


def test_cli_run_subcommand_accepts_optional_arguments(capsys, monkeypatch, tmp_path):
    input_dir = tmp_path / "3GAO"
    input_dir.mkdir()

    expected_output = input_dir / "naffinity_predicted_binding_class.txt"

    with patch("naffinity.cli.run_pipeline", return_value=expected_output) as mock_run_pipeline:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "naffinity",
                "run",
                str(input_dir),
                "--python",
                "/usr/bin/python3",
                "--overwrite",
                "--jobs",
                "4",
            ],
        )
        main()

    mock_run_pipeline.assert_called_once_with(
        str(input_dir),
        python_executable="/usr/bin/python3",
        overwrite=True,
        jobs=4,
    )

    captured = capsys.readouterr()
    assert "Prediction complete." in captured.out
    assert str(expected_output) in captured.out