from __future__ import annotations

import json

from model import RadicalizationModelManager


def main() -> None:
    manager = RadicalizationModelManager()
    result = manager.train_from_csv()
    metrics = result["metrics"]
    print("Training complete.")
    print(json.dumps(
        {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "dataset_size": metrics["dataset_size"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

