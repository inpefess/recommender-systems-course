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
Collaborative Filtering KNN Recommender
=======================================

"""
from implicit.nearest_neighbours import TFIDFRecommender
from rs_datasets import MovieLens

from rs_course.utils import (
    evaluate_implicit_recommender,
    movielens_split,
    pandas_to_scipy,
)


def collaborative_filtering_knn(dataset_size: str) -> None:
    """
    >>> collaborative_filtering_knn("small")
    0.5

    :param dataset_size: a size of MovieLens dataset to use
    """
    train, test, shape = movielens_split(
        MovieLens(dataset_size).ratings, 0.95, True
    )
    sparse_train = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    recommender = TFIDFRecommender()
    recommender.fit(sparse_train)
    print(
        evaluate_implicit_recommender(recommender, sparse_train, test, 3, 10)
    )
