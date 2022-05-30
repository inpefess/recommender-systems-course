# Copyright 2022 Boris Shminke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DNN Recommender Example
=======================

"""
import os
import shutil
from typing import Any, Dict

import pandas as pd
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset import Dataset
from recbole.model.general_recommender import ConvNCF
from recbole.trainer import Trainer
from rs_metrics import hitrate

from rs_course.lightfm_bpr import get_lightfm_predictions
from rs_course.utils import (
    enumerate_users_and_items,
    movielens_split,
    pandas_to_scipy,
)


def prepare_recbole_data(
    data_name: str, dataframe: pd.DataFrame, config: Config
) -> Dataset:
    """
    creates a directory and writes an interactions 'Atomic File' there
    Attention! The directory ``data_name`` will be removed no questions asked

    :param data_name: a name for the folder and the main file
    :param dataframe: a Pandas dataframe to convert to
        ``recbole``'s Atomic Files
    :param config: a ``recbole`` config
    :returns:
    """
    shutil.rmtree(data_name, ignore_errors=True)
    os.mkdir(data_name)
    dataframe.columns = pd.Index(
        [
            "user_id:token",
            "item_id:token",
            "label:float",
            "timestamp:float",
        ]
    )
    dataframe.to_csv(
        os.path.join(".", data_name, f"{data_name}.inter"),
        sep="\t",
        index=False,
    )
    return Dataset(config)


def get_recbole_trained_recommender(
    config: Config, train_data: Dataset
) -> ConvNCF:
    """
    :param config: a ``recbole`` config
    :param train_data: a training dataset in the ``recbole`` format
    :returns: a trained model ready for evaluation
    """
    train_data, valid_data, _ = data_preparation(config, train_data)
    recommender = ConvNCF(config, train_data.dataset)
    Trainer(config, recommender).fit(
        train_data, valid_data, saved=True, show_progress=True
    )
    recommender.eval()
    return recommender


def dnn_recommender(
    ratings: pd.DataFrame, model_config: Dict[str, Any]
) -> float:
    """
    >>> import os
    >>> model_config = {
    ...     "data_path": ".",
    ...     "eval_args": {
    ...         "group_by": None,
    ...         "order": "RO",
    ...         "split": {"RS": [0.95, 0.05, 0.0]},
    ...         "mode": "pop10",
    ...     },
    ...     "epochs": 1,
    ...     "use_gpu": os.environ.get("TEST_ON_GPU", False),
    ... }
    >>> test_ratings = getfixture("recbole_test_data").ratings
    >>> isinstance(
    ...     dnn_recommender(test_ratings, model_config),
    ...     float
    ... )
    True

    :param ratings: a dataset of user-items intersection
    :param model_config: ``config_dict`` of a ``recbole`` model
    :returns: hitrate@10
    """
    enumerate_users_and_items(ratings)
    train, test, shape = movielens_split(ratings, 0.95, True)
    train_sparse = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    recbole_config = Config(
        model="ConvNCF",
        dataset="train",
        config_dict=model_config,
    )
    recbole_train = prepare_recbole_data("train", train, recbole_config)
    recommender = get_recbole_trained_recommender(
        recbole_config, recbole_train
    )
    pred = get_lightfm_predictions(
        recommender,
        train_sparse,
        test,
        10,
        min(test["user_id"].unique().size, 1000),
    )
    return hitrate(test, pred)
