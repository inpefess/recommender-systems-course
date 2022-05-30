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
Pure SVD Recommender
====================

"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from rs_metrics import hitrate
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from rs_course.utils import movielens_split, pandas_to_scipy


def get_svd_recs(
    recommender: TruncatedSVD,
    sparse_train: csr_matrix,
    test: pd.DataFrame,
    split_test_users_into: int,
) -> Dict[int, List[int]]:
    """
    get recommendations given a truncated SVD decomposition

    :param recommender: a truncated SVD decomposition
    :param sparse_train: a ``scr_matrix`` representation of the train data
    :param test: test data
    :param split_test_users_into: a number of chunks for testing
    :returns: recommendations in ``rs_metrics`` compatible format
    """
    user_embeddings = recommender.transform(sparse_train)
    test_users = test.user_id.unique()
    test_parts = np.array_split(test_users, split_test_users_into)
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


def pure_svd_recommender(
    ratings: pd.DataFrame,
    split_test_users_into: int,
    model_config: Dict[str, Any],
) -> None:
    """
    >>> pure_svd_recommender(
    ...     getfixture("test_dataset").ratings,  # noqa: F821
    ...     1,
    ...     {"n_components": 1, "random_state": 0},
    ... )
    1.0

    :param ratings: a dataset of user-items intersection
    :param split_test_users_into: a number of chunks for testing
    :param model_config: a dict of ``TruncatedSVD`` argument for model training
    :returns:
    """
    train, test, shape = movielens_split(ratings, 0.95, True)
    sparse_train = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    recommender = TruncatedSVD(**model_config).fit(sparse_train)
    pred = get_svd_recs(recommender, sparse_train, test, split_test_users_into)
    print(hitrate(test, pred))
