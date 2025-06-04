import json
import os
import re

import pandas as pd

from modelforge.datasets.datasets import (
    anomaly_dataset,
    heating_dataset,
    immoscout_house_price_dataset,
    weather_dataset_probabilistic_regression,
)

datasets = ["anomaly", "heating", "house_price", "weather_probabilistic"]
dataset_sizes = [
    anomaly_dataset("score", "../../data/processed/anomaly_models").size,
    heating_dataset("../../data/processed/heating_models/data").size,
    immoscout_house_price_dataset(
        "../../data/processed/immoscout24_models/house_price/data"
    ).size,
    weather_dataset_probabilistic_regression(
        "../../data/processed/weather_models/prob/data"
    ).size,
]
datasets_retraining = [f"{dataset}_retraining" for dataset in datasets]
datasets_representative = [f"{dataset}_representative" for dataset in datasets]
datasets = datasets_representative + datasets_retraining
dataset_sizes = dataset_sizes * 2
losses = ["roc_auc", "mae", "mape", "crps"]
losses = losses * 2
losses_maximize = [True, False, False, False]
losses_maximize = losses_maximize * 2


def read_experiment_results_consolidation(experiment: str, experiment_date: str):

    pipeline_regex = re.compile(r"^(.*)_\d{4}-\d{2}-\d{2}-\d{2}:\d{2}$")

    rows = []
    for dataset, loss, loss_maximize in zip(datasets, losses, losses_maximize):
        pipeline_groups = [
            dir
            for dir in os.listdir(f"../../data/results/{experiment}/{dataset}")
            if os.path.isdir(f"../../data/results/{experiment}/{dataset}/{dir}")
            if experiment_date in dir
        ]
        pipeline_groups_clean = [
            pipeline_regex.match(pipeline).group(1).replace("_uniform", "")
            for pipeline in pipeline_groups
        ]
        for pipeline_group, pipeline_group_clean in zip(
            pipeline_groups, pipeline_groups_clean
        ):
            pipelines = [
                dir
                for dir in os.listdir(
                    f"../../data/results/{experiment}/{dataset}/{pipeline_group}"
                )
                if os.path.isdir(
                    f"../../data/results/{experiment}/{dataset}/{pipeline_group}/{dir}"
                )
            ]
            for pipeline in pipelines:
                cluster_amounts = os.listdir(
                    f"../../data/results/{experiment}/{dataset}/{pipeline_group}/{pipeline}"
                )
                for cluster_amount in cluster_amounts:
                    with open(
                        f"../../data/results/{experiment}/{dataset}/{pipeline_group}/{pipeline}/{cluster_amount}/params.json"
                    ) as f:
                        params = json.load(f)
                        num_cluster = params[
                            "modelclusterer__cluster_mixin__n_clusters"
                        ]
                    with open(
                        f"../../data/results/{experiment}/{dataset}/{pipeline_group}/{pipeline}/{cluster_amount}/scores.json"
                    ) as f:
                        try:
                            scores = json.load(f)
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "loss": loss,
                                    "pipeline": pipeline,
                                    "pipeline_group": pipeline_group_clean,
                                    **scores,
                                    "num_clusters": int(num_cluster),
                                }
                            )
                        except json.JSONDecodeError:
                            print(
                                f"Could not load scores for {dataset}/{pipeline_group}/{pipeline}/{cluster_amount}"
                            )
                            print(f)
                rows.append(
                    {
                        "dataset": dataset,
                        "pipeline": pipeline,
                        "pipeline_group": pipeline_group_clean,
                        **scores,
                    }
                )
    return pd.DataFrame(rows), pipeline_groups, pipeline_groups_clean
