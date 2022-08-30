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
# noqa: D205, D400
"""
Collaborative Filtering ALS Recommender
=======================================

"""
import os
from typing import Any, Dict, Tuple

import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from rs_course.utils import (
    evaluate_implicit_recommender,
    movielens_split,
    pandas_to_scipy,
)


def als_recommendations(
    ratings: pd.DataFrame,
    model_params: Dict[str, Any],
    split_test_users_into: int,
    top_k: int,
    train_percentage: float,
) -> Tuple[
    csr_matrix, AlternatingLeastSquares, float, pd.DataFrame, pd.DataFrame
]:
    """
    Build an ALS recommender from ``implicit``.

    >>> ratings = getfixture("test_dataset").ratings  # noqa: F821
    >>> import os
    >>> model_params = {
    ...      "factors": 1,
    ...      "use_gpu": os.environ.get("TEST_ON_GPU", False),
    ...      "random_state": 0,
    ...      "iterations": 1,
    ... }
    >>> _, _, hit_rate, _, _ = als_recommendations(
    ...     ratings=ratings,
    ...     model_params=model_params,
    ...     split_test_users_into=1,
    ...     top_k=10,
    ...     train_percentage=0.95
    ... )
    >>> print(hit_rate)
    1.0

    :param ratings: a dataset of user-items intersection
    :param model_params: ALS training parameters
    :param split_test_users_into: a number of chunks for testing
    :param top_k: the number of items to recommend
    :param train_percentage: percentage of user-item pairs to leave in the
        training set
    :returns: a tuple of train set in sparse format, trained recommender,
        hit_rate@10, train, and test test in ``pandas format``
    """
    train, test, shape = movielens_split(ratings, train_percentage, True)
    sparse_train = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    recommender = AlternatingLeastSquares(**model_params)
    recommender.fit(sparse_train)
    return (
        sparse_train,
        recommender,
        evaluate_implicit_recommender(
            recommender, sparse_train, test, split_test_users_into, top_k
        ),
        train,
        test,
    )
