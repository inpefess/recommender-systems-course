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
Cold Start Recommender Example
==============================

"""
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from rs_datasets import MovieLens

try:
    # pylint: disable=ungrouped-imports
    from implicit.gpu import Matrix  # pylint: disable=no-name-in-module
except ImportError:
    from scipy.sparse import csr_matrix as Matrix

# pylint: disable=ungrouped-imports
from implicit.nearest_neighbours import ItemItemRecommender

from rs_course.cf_als import als_recommendations
from rs_course.content_based_knn import (
    evaluate_implicit_recommender,
    get_content_based_recommender,
)


def get_cold_items(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[List[int], pd.DataFrame]:
    """
    :param dataset_size: a size of MovieLens dataset to use
    :returns: a list of cold items and the test set
    """
    cold_items = list(
        set(test.item_id.values).difference(set(train.item_id.values))
    )
    print(
        "cold items percentage in test:",
        len(cold_items) / test.item_id.unique().size,
    )
    print(
        "cold rows percentage in test:",
        len(test[test.item_id.isin(cold_items)]) / len(test),
    )
    print(
        "users with cold items percentage in test:",
        test[test.item_id.isin(cold_items)].user_id.unique().size
        / test.user_id.unique().size,
    )
    print(
        "users with no cold items percentage in test:",
        test[~test.item_id.isin(cold_items)].user_id.unique().size
        / test.user_id.unique().size,
    )
    return cold_items, test


def compute_cold_factors(
    cold_items: List[int],
    content_based_recommender: ItemItemRecommender,
    recommender: AlternatingLeastSquares,
) -> Dict[int, np.ndarray]:
    """

    :param cold_items: a list of cold items to which compute factors
    :param content_based_recommender: a content-based recommender to induce new
        factors
    :param recommender: an ALS recommender with known factors (for non-cold
        items)
    :returns: a dictionary with cold item IDs as keys and their computed
        factors as values
    """
    cold_factors = {}
    for item_id in cold_items:
        if content_based_recommender.similarity[item_id].nnz > 1:
            new_factors = np.zeros((1, recommender.factors))
            norm = 0
            for sim_item, weight in (
                content_based_recommender.similarity[item_id].todok().items()
            ):
                known_factors = recommender.item_factors[int(sim_item[1])]
                known_factors_array = (
                    known_factors
                    if isinstance(known_factors, np.ndarray)
                    else known_factors.to_numpy()
                )
                new_factors += weight * known_factors_array
                norm += weight**2
            cold_factors[item_id] = new_factors / np.sqrt(norm)
    return cold_factors


def cold_start(
    movielens: MovieLens,
    als_config: Dict[str, Any],
    split_test_users_into: int,
) -> None:
    """
    >>> import os
    >>> als_config = {
    ...      "factors": 1,
    ...      "use_gpu": os.environ.get("TEST_ON_GPU", False),
    ...      "random_state": 0
    ... }
    >>> cold_start(getfixture("test_dataset"), als_config, 1)  # noqa: F821
    Collaborative Filtering Hit-Rate: 1.0
    Content-Based Hit-Rate: 1.0
    cold items percentage in test: 0.0
    cold rows percentage in test: 0.0
    users with cold items percentage in test: 0.0
    users with no cold items percentage in test: 1.0
    Hybrid Hit-Rate: 1.0

    :param movielens: MovieLens dataset
    :param als_config: collaborative model training params
    :param split_test_users_into: a number of chunks for testing
    :returns:
    """
    sparse_train, recommender, cf_hitrate, train, test = als_recommendations(
        movielens.ratings,
        als_config,
        split_test_users_into,
    )
    print("Collaborative Filtering Hit-Rate:", cf_hitrate)
    content_based_recommender = get_content_based_recommender(
        movielens, split_test_users_into
    )
    cold_items, test = get_cold_items(train, test)
    cold_factors = compute_cold_factors(
        cold_items, content_based_recommender, recommender
    )
    new_item_factors = (
        recommender.item_factors
        if isinstance(recommender.item_factors, np.ndarray)
        else recommender.item_factors.to_numpy()
    )
    for item_id, item_factors in cold_factors.items():
        new_item_factors[int(item_id)] = item_factors
    recommender.item_factors = (
        Matrix(new_item_factors) if als_config["use_gpu"] else new_item_factors
    )
    print(
        "Hybrid Hit-Rate:",
        evaluate_implicit_recommender(
            recommender, sparse_train, test, split_test_users_into, 10
        ),
    )
