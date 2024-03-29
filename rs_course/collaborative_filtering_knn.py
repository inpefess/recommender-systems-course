# Copyright 2021-2023 Boris Shminke
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
Collaborative Filtering KNN Recommender
=======================================

"""
import pandas as pd
from implicit.nearest_neighbours import CosineRecommender

from rs_course.utils import (
    evaluate_implicit_recommender,
    movielens_split,
    pandas_to_scipy,
)


def collaborative_filtering_knn(
    ratings: pd.DataFrame,
    number_of_neighbours: int,
    split_test_users_into: int,
    top_k: int,
    train_percentage: float,
) -> None:
    """
    Build a collaborative filtering KNN model.

    >>> collaborative_filtering_knn(
    ...     getfixture("test_dataset").ratings,  # noqa: F821
    ...     7,
    ...     1,
    ...     top_k=10,
    ...     train_percentage=0.95
    ... )
    1.0

    :param ratings: dataset of user-items interactions
    :param number_of_neighbours: number of neighbours for KNN
    :param split_test_users_into: a number of chunks for testing
    :param top_k: the number of items to recommend
    :param train_percentage: percentage of user-item pairs to leave in the
        training set
    """
    train, test, shape = movielens_split(ratings, train_percentage, True)
    sparse_train = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    recommender = CosineRecommender(K=number_of_neighbours)
    recommender.fit(sparse_train)
    print(
        evaluate_implicit_recommender(
            recommender, sparse_train, test, split_test_users_into, top_k
        )
    )
