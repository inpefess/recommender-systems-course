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
DNN recommender example
"""
import torch

if torch.cuda.is_available():
    torch.cuda.current_device()

from rs_datasets import MovieLens
from rs_metrics import hitrate
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions

from rs_course.lightfm_bpr import get_lightfm_predictions
from rs_course.utils import movielens_split, pandas_to_scipy


def main(dataset_size: str, use_gpu: bool, verbose: bool):
    """
    >>> import os
    >>> main("small", os.environ.get("TEST_ON_GPU", False), False)
    0...

    :param dataset_size: a size of MovieLens dataset to use
    :param use_gpu: whether to use GPU or not
    :param verbose: whether to print training log
    """
    movielens = MovieLens(dataset_size)
    train, test, shape = movielens_split(movielens.ratings, 0.95, True)
    train_sparse = pandas_to_scipy(
        train, "rating", "user_id", "item_id", shape
    )
    recommender = ImplicitFactorizationModel(
        embedding_dim=128,
        batch_size=2 ** 18,
        use_cuda=use_gpu,
        loss="bpr",
        n_iter=15,
        num_negative_samples=1,
        random_state=0,
    )
    recommender.fit(
        Interactions(
            user_ids=train.user_id.values,
            item_ids=train.item_id.values,
            weights=train.rating.values,
            num_users=shape[0],
            num_items=shape[1],
        ),
        verbose=verbose,
    )
    pred = get_lightfm_predictions(
        recommender,
        train_sparse,
        test,
        10,
        min(test["user_id"].unique().size, 1000),
    )
    print(hitrate(test, pred))
