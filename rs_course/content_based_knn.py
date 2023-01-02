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
Content-based item2item KNN Recommender
=======================================

"""
from implicit.nearest_neighbours import CosineRecommender, ItemItemRecommender
from rs_datasets import MovieLens

from rs_course.utils import (
    evaluate_implicit_recommender,
    get_sparse_item_features,
    movielens_split,
    pandas_to_scipy,
)


def get_content_based_recommender(
    movielens: MovieLens,
    split_test_users_into: int,
    top_k: int,
    train_percentage: float,
) -> ItemItemRecommender:
    """
    Build a content-based recommender.

    >>> _ = get_content_based_recommender(
    ...     getfixture("test_dataset"),  # noqa: F821
    ...     1,
    ...     10,
    ...     0.95
    ... )
    Content-Based Hit-Rate: 1.0

    :param movielens: a MovieLens dataset
    :param split_test_users_into: into how many chunks to split the test set
    :param top_k: the number of items to recommend
    :param train_percentage: percentage of user-item pairs to leave in the
        training set
    :returns: a trained recommender
    """
    train, test, shape = movielens_split(
        movielens.ratings, train_percentage, True
    )
    sparse_train = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    item_features, _ = get_sparse_item_features(movielens, movielens.ratings)
    recommender = CosineRecommender()
    recommender.fit(item_features.T)
    print(
        "Content-Based Hit-Rate:",
        evaluate_implicit_recommender(
            recommender, sparse_train, test, split_test_users_into, top_k
        ),
    )
    return recommender
