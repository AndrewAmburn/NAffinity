import argparse

from naffinity.pipeline import run_pipeline
from naffinity.batch import run_batch


def main() -> None:

    parser = argparse.ArgumentParser(
        description="NAffinity: nucleic acid-ligand affinity classification pipeline"
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    # --------------------------------------------------
    # Single-complex mode
    # --------------------------------------------------

    run_parser = subparsers.add_parser(
        "run",
        help="Run the full NAffinity pipeline on a single complex directory",
    )

    run_parser.add_argument(
        "input_dir",
        help="Path to the input complex directory",
    )

    run_parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Parallel workers for feature generation (-1 = all cores)",
    )

    # --------------------------------------------------
    # Batch mode
    # --------------------------------------------------

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run NAffinity on all complex folders within a parent directory",
    )

    batch_parser.add_argument(
        "input_dir",
        help="Directory containing complex subfolders",
    )

    batch_parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Parallel workers (-1 = all cores)",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Single complex
    # --------------------------------------------------

    if args.command == "run":

        output_path = run_pipeline(
            args.input_dir,
            jobs=args.jobs,
        )
        print(f"Running NAffinity on: {args.input_dir}")
        print("Prediction complete.")
        print(f"Output: {output_path}")
        return

    # --------------------------------------------------
    # Batch mode
    # --------------------------------------------------

    if args.command == "batch":

        run_batch(
            args.input_dir,
            jobs=args.jobs,
        )

        print("Batch complete.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()