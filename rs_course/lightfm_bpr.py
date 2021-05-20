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
LightFM BPR example
"""
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from lightfm import LightFM
from rs_datasets import MovieLens
from rs_metrics import hitrate
from scipy.sparse import csr_matrix
from tqdm import tqdm

from rs_course.utils import movielens_split, pandas_to_scipy


def get_lightfm_predictions(
    recommender: Any,
    sparse_train: csr_matrix,
    test: pd.DataFrame,
    top_k: int,
    split_test_users_into: int,
) -> Dict[int, List[int]]:
    """
    get recommendations given a LightFM recommender

    :param recommender: a recommender
    :param sparse_train: a ``scr_matrix`` representation of the train data
    :param test: test data
    :param top_k: how many recommendations to return for each user
    :param split_test_users_into: split ``test`` by users into several chunks
        to fit into memory
    :returns: recommendations in ``rs_metrics`` compatible format
    """
    item_ids = np.arange(sparse_train.shape[1], dtype=np.int32)
    test_users_parts = np.array_split(
        test.user_id.unique(), split_test_users_into
    )
    pred = {}
    for test_users_part in tqdm(test_users_parts):
        if isinstance(recommender, LightFM):
            raw_weights_part = recommender.predict(
                np.repeat(test_users_part, item_ids.shape[0]),
                np.tile(item_ids, test_users_part.shape[0]),
                num_threads=os.cpu_count(),
            ).reshape(test_users_part.shape[0], item_ids.shape[0])
        else:
            raw_weights_part = recommender.predict(
                np.repeat(test_users_part, item_ids.shape[0]),
                np.tile(item_ids, test_users_part.shape[0]),
            ).reshape(test_users_part.shape[0], item_ids.shape[0])
        no_train_weights = np.where(
            (sparse_train[test_users_part].toarray() > 0),
            0.0,
            raw_weights_part,
        )
        pred.update(
            dict(
                zip(
                    test_users_part,
                    np.argpartition(  # type: ignore
                        -no_train_weights, top_k, axis=1
                    )[:, :top_k],
                )
            )
        )
    return pred


def main(dataset_size: str):
    """
    >>> main("small")
    0.1

    :param dataset_size: a size of MovieLens dataset to use
    """
    movielens = MovieLens(dataset_size)
    train_data, test_data, shape = movielens_split(
        movielens.ratings, 0.95, True
    )
    sparse_train = pandas_to_scipy(
        train_data, "rating", "user_id", "item_id", shape
    )
    model = LightFM(
        no_components=100, loss="bpr", learning_rate=0.01, random_state=0
    )
    model.fit_partial(
        sparse_train, num_threads=os.cpu_count(), epochs=15, verbose=True
    )
    pred = get_lightfm_predictions(model, sparse_train, test_data, 10, 3)
    print(hitrate(test_data, pred))
