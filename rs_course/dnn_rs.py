# Copyright 2021-2022 Boris Shminke
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
from typing import Any, Dict

import pandas as pd
from rs_metrics import hitrate
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions

from rs_course.lightfm_bpr import get_lightfm_predictions
from rs_course.utils import movielens_split, pandas_to_scipy


def dnn_recommender(
    ratings: pd.DataFrame, model_config: Dict[str, Any], verbose: bool
) -> float:
    """
    >>> import os
    >>> model_config = {
    ...     "embedding_dim": 2,
    ...     "batch_size": 2,
    ...     "use_cuda": os.environ.get("TEST_ON_GPU", False),
    ...     "loss": "bpr",
    ...     "n_iter": 1,
    ...     "num_negative_samples": 1,
    ...     "random_state": 0,
    ... }
    >>> test_ratings = getfixture("test_dataset").ratings
    >>> isinstance(
    ...     dnn_recommender(test_ratings, model_config, False),
    ...     float
    ... )
    True

    :param ratings: a dataset of user-items intersection
    :param model_config: a dict of ``ImplicitFactorizationModel`` arguments
    :param verbose: print diagnostic info during training
    :param dataset_size: a size of MovieLens dataset to use
    :returns: hitrate@10
    """
    train, test, shape = movielens_split(ratings, 0.95, True)
    train_sparse = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    recommender = ImplicitFactorizationModel(**model_config)
    recommender.fit(
        Interactions(
            user_ids=train.user_id.values,
            item_ids=train.item_id.values,
            weights=train.rating.values,
            num_users=shape[0],
            num_items=shape[1],
        ),
        verbose=verbose,
    )
    pred = get_lightfm_predictions(
        recommender,
        train_sparse,
        test,
        10,
        min(test["user_id"].unique().size, 1000),
    )
    return hitrate(test, pred)
