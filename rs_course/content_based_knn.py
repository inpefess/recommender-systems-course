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
Content-based item2item KNN Recommender
=======================================

"""
from implicit.nearest_neighbours import ItemItemRecommender, TFIDFRecommender
from rs_datasets import MovieLens

from rs_course.utils import (
    evaluate_implicit_recommender,
    get_sparse_item_features,
    movielens_split,
    pandas_to_scipy,
)


def get_content_based_recommender(dataset_size: str) -> ItemItemRecommender:
    """
    main function of the module
    >>> _ = get_content_based_recommender("small")
    Content-Based Hit-Rate: 0.2

    :param dataset_size: a size of MovieLens dataset to use
    """
    movielens = MovieLens(dataset_size)
    train, test, shape = movielens_split(movielens.ratings, 0.95, True)
    sparse_train = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    item_features, _ = get_sparse_item_features(movielens)
    recommender = TFIDFRecommender()
    recommender.fit(item_features)
    print(
        "Content-Based Hit-Rate:",
        evaluate_implicit_recommender(recommender, sparse_train, test),
    )
    return recommender
