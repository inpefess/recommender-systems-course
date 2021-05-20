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
collaborative filtering ALS recommender
"""
import os
from typing import Tuple

from implicit.als import AlternatingLeastSquares
from rs_datasets import MovieLens
from scipy.sparse import csr_matrix

from rs_course.utils import (
    evaluate_implicit_recommender,
    movielens_split,
    pandas_to_scipy,
)


def als_recommendations(
    dataset_size: str, use_gpu: bool
) -> Tuple[csr_matrix, AlternatingLeastSquares, float]:
    """
    >>> import os
    >>> _, _, hitrate = als_recommendations(
    ...     "small", os.environ.get("TEST_ON_GPU", False)
    ... )
    >>> print(hitrate)
    0...

    :param dataset_size: a size of MovieLens dataset to use
    :param use_gpu: whether to use GPU or not
    """
    ratings = MovieLens(dataset_size).ratings
    train, test, shape = movielens_split(ratings, 0.95, True)
    sparse_train = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    recommender = AlternatingLeastSquares(
        factors=128, use_gpu=use_gpu, random_state=0
    )
    recommender.fit(sparse_train.T)
    return (
        sparse_train,
        recommender,
        evaluate_implicit_recommender(recommender, sparse_train, test),
    )
