import argparse

from naffinity.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NAffinity: nucleic acid-ligand affinity classification pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run the full NAffinity pipeline on a single complex directory",
    )
    run_parser.add_argument(
        "input_dir",
        help="Path to the input complex directory",
    )
    run_parser.add_argument(
        "--python",
        default=None,
        help="Python executable to use for legacy helper scripts (default: current interpreter)",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs where supported",
    )
    run_parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Parallel workers for feature steps (-1 = all cores)",
    )

    args = parser.parse_args()

    if args.command == "run":
        output_path = run_pipeline(
            args.input_dir,
            python_executable=args.python,
            overwrite=args.overwrite,
            jobs=args.jobs,
        )
        print("Prediction complete.")
        print(f"Output: {output_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()