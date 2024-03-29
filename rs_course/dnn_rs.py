# Copyright 2022-2023 Boris Shminke
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
# noqa: D205, D400
"""
DNN Recommender Example
=======================

"""
import os
import shutil
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset import Dataset
from recbole.model.general_recommender import ConvNCF
from recbole.trainer import Trainer
from rs_metrics import hitrate
from scipy.sparse import csr_matrix
from tqdm import tqdm

from rs_course.utils import (
    enumerate_users_and_items,
    movielens_split,
    pandas_to_scipy,
)


def _get_dnn_weights(
    sparse_train: csr_matrix,
    split_test_users_into: int,
    recommender,
    test_users_part: np.ndarray,
) -> np.ndarray:
    test_items_parts = np.array_split(
        np.arange(sparse_train.shape[1]), split_test_users_into
    )
    raw_weights_small_parts = []
    for test_items_part in test_items_parts:
        raw_weights_small_parts.append(
            recommender.predict(
                {
                    "user_id": torch.tensor(
                        np.repeat(test_users_part, test_items_part.shape[0]),
                        dtype=torch.long,
                        device=recommender.device,
                    ),
                    "item_id": torch.tensor(
                        np.tile(test_items_part, test_users_part.shape[0]),
                        dtype=torch.long,
                        device=recommender.device,
                    ),
                }
            )
            .cpu()
            .detach()
            .numpy()
            .reshape(test_users_part.shape[0], test_items_part.shape[0])
        )
    return np.hstack(raw_weights_small_parts)


def get_dnn_predictions(
    recommender: Any,
    sparse_train: csr_matrix,
    test: pd.DataFrame,
    top_k: int,
    split_test_users_into: int,
) -> Dict[int, List[int]]:
    """
    Get recommendations given a DNN recommender.

    :param recommender: a recommender
    :param sparse_train: a ``scr_matrix`` representation of the train data
    :param test: test data
    :param top_k: how many recommendations to return for each user
    :param split_test_users_into: split ``test`` by users into several chunks
        to fit into memory
    :returns: recommendations in ``rs_metrics`` compatible format
    """
    test_users_parts = np.array_split(
        test.user_id.unique(), split_test_users_into
    )
    pred = {}
    for test_users_part in tqdm(test_users_parts):
        raw_weights_part = _get_dnn_weights(
            sparse_train,
            split_test_users_into,
            recommender,
            test_users_part,
        )
        no_train_weights = np.where(
            (sparse_train[test_users_part].toarray() > 0),
            0.0,
            raw_weights_part,
        )
        pred.update(
            dict(
                zip(
                    test_users_part,
                    np.argpartition(  # type: ignore
                        -no_train_weights, top_k, axis=1
                    )[:, :top_k],
                )
            )
        )
    return pred


def prepare_recbole_data(
    data_name: str, dataframe: pd.DataFrame, config: Config
) -> Dataset:
    """
    Create a directory and write an interactions 'Atomic File' there.

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
    Train a DNN recommender from RecBole.

    :param config: a ``recbole`` config
    :param train_data: a training dataset in the ``recbole`` format
    :returns: a trained model ready for evaluation
    """
    train_data, valid_data, _ = data_preparation(config, train_data)
    recommender = ConvNCF(config, train_data.dataset).to(
        config.final_config_dict["device"]
    )
    Trainer(config, recommender).fit(
        train_data, valid_data, saved=True, verbose=True
    )
    recommender.eval()
    return recommender


def dnn_recommender(
    ratings: pd.DataFrame,
    model_config: Dict[str, Any],
    split_test_users_into: int,
    top_k: int,
    train_percentage: float,
) -> float:
    """
    Build a RecBole model.

    >>> import os
    >>> model_config = {
    ...     "data_path": ".",
    ...     "eval_step": 0,
    ...     "embedding_size": 1,
    ...     "cnn_channels": [1, 1],
    ...     "cnn_kernels": [1],
    ...     "cnn_strides": [1],
    ...     "epochs": 1,
    ...     "use_gpu": os.environ.get("TEST_ON_GPU", False),
    ... }
    >>> test_ratings = getfixture("recbole_test_data").ratings  # noqa: F821
    >>> isinstance(
    ...     dnn_recommender(test_ratings, model_config, 100, 10, 0.95),
    ...     float
    ... )
    True

    :param ratings: a dataset of user-items interactions
    :param model_config: ``config_dict`` of a ``recbole`` model
    :param split_test_users_into: split ``test`` by users into several chunks
        to fit into memory
    :param top_k: the number of items to recommend
    :param train_percentage: percentage of user-item pairs to leave in the
        training set
    :returns: hitrate@top_k
    """
    enumerate_users_and_items(ratings)
    train, test, shape = movielens_split(ratings, train_percentage, True)
    # here we want train set to include all the items and users
    # we achieve that by filling the histories for user and item
    # with indices 0 (not a real ID); we need that because ``recbole``
    # enumerates users and items itself
    # also we add a pair (1, 1) explicitly, because ``recbole`` doesn't
    # want any user to have all items in a history. That spoils a real user
    # history (1 is a real ID), but hopefully not much
    train = pd.concat(
        [
            train,
            pd.DataFrame(
                [
                    {
                        "user_id": 0,
                        "item_id": i,
                        "rating": 1,
                        "timestamp": 1,
                    }
                    for i in range(2, shape[1])
                ]
                + [
                    {
                        "user_id": 1,
                        "item_id": 1,
                        "rating": 1,
                        "timestamp": 1,
                    }
                ]
                + [
                    {
                        "user_id": i,
                        "item_id": 0,
                        "rating": 1,
                        "timestamp": 1,
                    }
                    for i in range(2, shape[0])
                ]
            ),
        ]
    )
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
    pred = get_dnn_predictions(
        recommender,
        train_sparse,
        test,
        top_k,
        min(test["user_id"].unique().size, split_test_users_into),
    )
    return hitrate(test, pred)
