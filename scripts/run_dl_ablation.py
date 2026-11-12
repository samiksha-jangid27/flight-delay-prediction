"""Compare DL hyperparameters using the same seeds for every configuration."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from flight_delay.deep_learning.dataset import build_dl_dataset
from flight_delay.deep_learning.encoding import DLTensorEncoder
from flight_delay.deep_learning.train import _fit_trial, _read_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dl_config.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 46, 50])
    parser.add_argument("--output", type=Path, default=Path("reports/dl/controlled_ablation.csv"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Seeds must be distinct nonnegative integers")
    config = _read_config(args.config)
    data = args.config.resolve().parent.parent / config["data_path"]
    encoder = DLTensorEncoder(build_dl_dataset(data))
    train = encoder.transform_partition("train")
    validation = encoder.transform_partition("validation")
    settings = config["training"]

    rows = []
    for seed in args.seeds:
        for index, trial in enumerate(config["search"]["trials"]):
            result = _fit_trial(
                encoder, train, validation, trial, seed=seed,
                epochs=settings["epochs"], patience=settings["patience"],
                min_delta=settings["min_delta"], optimizer_name=settings["optimizer"],
                device=torch.device("cpu"),
            )
            rows.append({
                "seed": seed, "trial": index, **trial,
                "best_epoch": result["best_epoch"],
                "validation_average_precision": result["validation_average_precision"],
            })
    results = pd.DataFrame(rows)
    baseline = results.loc[results.trial.eq(0)].set_index("seed").validation_average_precision
    results["ap_difference_from_same_seed_baseline"] = (
        results.validation_average_precision - results.seed.map(baseline)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.groupby("trial").ap_difference_from_same_seed_baseline.agg(["mean", "min", "max"]))


if __name__ == "__main__":
    main()
