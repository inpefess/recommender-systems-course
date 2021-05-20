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
Pure SVD recommender
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from rs_datasets import MovieLens
from rs_metrics import hitrate
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from rs_course.utils import movielens_split, pandas_to_scipy


def get_svd_recs(
    recommender: TruncatedSVD, sparse_train: csr_matrix, test: pd.DataFrame
) -> Dict[int, List[int]]:
    """
    get recommendations given a truncated SVD decomposition

    :param recommender: a truncated SVD decomposition
    :param sparse_train: a ``scr_matrix`` representation of the train data
    :param test: test data
    :returns: recommendations in ``rs_metrics`` compatible format
    """
    user_embeddings = recommender.transform(sparse_train)
    test_users = test.user_id.unique()
    test_parts = np.array_split(test_users, 3)
    pred = {}
    for test_part in tqdm(test_parts):
        raw_weights = user_embeddings[test_part].dot(recommender.components_)
        no_train_weights = np.where(
            (sparse_train[test_part].toarray() > 0),
            0.0,
            raw_weights,
        )
        pred.update(
            dict(
                zip(
                    test_part,
                    np.argpartition(  # type: ignore
                        -no_train_weights, 10, axis=1
                    )[:, :10],
                )
            )
        )
    return pred


def main(dataset_size: str):
    """
    >>> main("small")
    0.3

    :param dataset_size: a size of MovieLens dataset to use
    """
    ratings = MovieLens(dataset_size).ratings
    train, test, shape = movielens_split(ratings, 0.95, True)
    sparse_train = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    recommender = TruncatedSVD(n_components=100, random_state=0).fit(
        sparse_train
    )
    pred = get_svd_recs(recommender, sparse_train, test)
    print(hitrate(test, pred))
