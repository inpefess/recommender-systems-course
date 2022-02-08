""" generate course project tasks """
from typing import List

import numpy as np
import pandas as pd

STUDENTS: List[str] = []
DATASETS = [
    "https://darel13712.github.io/rs_datasets/Datasets/msd/",
    "https://darel13712.github.io/rs_datasets/Datasets/netflix/",
    "https://darel13712.github.io/rs_datasets/Datasets/goodreads/",
    "https://darel13712.github.io/rs_datasets/Datasets/epinions/",
    "https://darel13712.github.io/rs_datasets/Datasets/bookx/",
    "https://darel13712.github.io/rs_datasets/Datasets/dating_agency/",
    "https://darel13712.github.io/rs_datasets/Datasets/jester/",
    "books from https://darel13712.github.io/rs_datasets/Datasets/amazon/",
    "https://darel13712.github.io/rs_datasets/Datasets/rekko/",
    "https://darel13712.github.io/rs_datasets/Datasets/steam/",
    "https://darel13712.github.io/rs_datasets/Datasets/anime/",
    "https://darel13712.github.io/rs_datasets/Datasets/retail_rocket/",
    "https://darel13712.github.io/rs_datasets/Datasets/diginetica/",
    "clothing from https://darel13712.github.io/rs_datasets/Datasets/amazon/",
    "kitchen from https://darel13712.github.io/rs_datasets/Datasets/amazon/",
    "electronics from https://darel13712.github.io/rs_datasets/Datasets/amazon/",
    "sports from https://darel13712.github.io/rs_datasets/Datasets/amazon/",
    "phones from https://darel13712.github.io/rs_datasets/Datasets/amazon/",
    "tools from https://darel13712.github.io/rs_datasets/Datasets/amazon/",
    "automotive from https://darel13712.github.io/rs_datasets/Datasets/amazon/",
    "toys from https://darel13712.github.io/rs_datasets/Datasets/amazon/",
    "pet from https://darel13712.github.io/rs_datasets/Datasets/amazon/",
]
METRICS = [
    "https://darel13712.github.io/rs_metrics/metrics/#hitrate",
    "https://darel13712.github.io/rs_metrics/metrics/#precision",
    "https://darel13712.github.io/rs_metrics/metrics/#mean-average-precision",
    "https://darel13712.github.io/rs_metrics/metrics/#recall",
    "https://darel13712.github.io/rs_metrics/metrics/#coverage",
    "https://darel13712.github.io/rs_metrics/metrics/#ndcg",
    "https://darel13712.github.io/rs_metrics/metrics/#mrr",
    "https://darel13712.github.io/rs_metrics/metrics/#-ndcg",
    "https://darel13712.github.io/rs_metrics/metrics/#popularity",
    "https://darel13712.github.io/rs_metrics/metrics/#surprisal",
]
MODELS = [
    "SVD",
    "implicit ALS",
    "implicit KNN",
    "popular",
    "LightFM",
    "turi",
    "spotlight",
]


def generate_tasks() -> None:
    """generate a CSV with tasks"""
    models = 3 * MODELS + MODELS[:1]
    metrics = 2 * METRICS + METRICS[:2]
    print(len(STUDENTS), len(DATASETS), len(models), len(metrics))
    good = False
    while not good:
        table = pd.DataFrame(
            zip(  # type: ignore
                STUDENTS,
                np.random.permutation(DATASETS),  # type: ignore
                np.random.permutation(models),  # type: ignore
                np.random.permutation(models),  # type: ignore
                np.random.permutation(metrics),  # type: ignore
                np.random.permutation(metrics),  # type: ignore
            ),
            columns=[
                "student",
                "dataset",
                "model 1",
                "model 2",
                "metric 1",
                "metric 2",
            ],
        )
        good = (table["model 1"] == table["model 2"]).sum() == 0 and (
            table["metric 1"] == table["metric 2"]
        ).sum() == 0
    table.to_csv("project_task.csv", index=False)
