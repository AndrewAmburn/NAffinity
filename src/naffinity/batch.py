from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import csv
from .pipeline import run_pipeline


def parse_prediction_file(pred_file):

    predicted_class = "NA"
    probability = "NA"

    with open(pred_file, "r") as f:

        for line in f:

            if line.startswith("PredictedClass:"):
                predicted_class = line.split(":", 1)[1].strip()

            elif line.startswith("ProbabilityStrongBinder:"):
                
                try:

                    probability = float(

                        line.split(":", 1)[1].strip()

                    )

                except ValueError:

                    probability = None

    return predicted_class, probability

def run_batch(
    parent_dir,
    jobs=-1,
    model_path=None,
):
    parent = Path(parent_dir).expanduser().resolve()

    if not parent.is_dir():
        raise NotADirectoryError(parent)

    folders = sorted(
        [
            p for p in parent.iterdir()
            if p.is_dir()
        ]
    )

    results = []

    with ThreadPoolExecutor(max_workers=jobs if jobs > 0 else None) as executor:

        future_map = {
            executor.submit(
                run_pipeline,
                str(folder),
                jobs=jobs,
                model_path=model_path,
            ): folder
            for folder in folders
        }

        for future in as_completed(future_map):

            folder = future_map[future]

            try:
                output = future.result()

                pred_class, prob = parse_prediction_file(output)

                results.append(
                    {
                        "Complex": folder.name,
                        "PredictedClass": pred_class,
                        "ProbabilityStrongBinder": prob,
                    }
                )

                print(f"✓ {folder.name}")

            except Exception as e:

                results.append(
                    {
                        "Complex": folder.name,
                        "PredictedClass": "FAILED",
                        "ProbabilityStrongBinder": "NA",
                    }
                )

    summary_file = parent / "batch_results.csv"

    with open(summary_file, "w", newline="") as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[

                "Complex",

                "PredictedClass",

                "ProbabilityStrongBinder",

            ],

        )

        writer.writeheader()

        for row in sorted(

            results,

            key=lambda x: x["Complex"]

        ):

            writer.writerow(row)

    print(f"\nSummary written to: {summary_file}")

    return results

