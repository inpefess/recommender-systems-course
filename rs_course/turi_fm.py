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
Turi Recommender Example
========================

"""
import pandas as pd
import turicreate as tc
from rs_datasets import MovieLens
from rs_metrics import hitrate

from rs_course.utils import get_sparse_item_features, movielens_split


def turi_recommender(
    training_data: tc.SFrame,
    item_data: tc.SFrame,
    test: pd.DataFrame,
    verbose: bool,
) -> None:
    """
    :param train_data: training data (interactions) in `turi` format
    :param item_data: item features in `turi` format
    :param test: test data in `pandas` format
    :param verbose: whether to print training log
    """
    model = tc.recommender.ranking_factorization_recommender.create(
        training_data,
        num_factors=100,
        num_sampled_negative_examples=1,
        max_iterations=1,
        item_data=item_data,
        sgd_step_size=0.01,
        verbose=verbose,
    )
    results = model.recommend(users=test.user_id.unique())
    print(hitrate(test, results.to_dataframe()))


def evaluate_turi_recommender(dataset_size: str, verbose: bool) -> None:
    """
    >>> evaluate_turi_recommender("small", False)
    0...
    0...

    :param dataset_size: a size of MovieLens dataset to use
    :param verbose: whether to print training log
    """
    movielens = MovieLens(dataset_size)
    train, test, _ = movielens_split(movielens.ratings, 0.95, True)
    training_data = tc.SFrame(data=train)
    turi_recommender(training_data, None, test, verbose)
    _, agg_tags = get_sparse_item_features(movielens)
    item_features = (
        agg_tags[["item_id", "tag"]].groupby("item_id").agg(list).reset_index()
    )
    item_data = tc.SFrame(data=item_features)
    turi_recommender(training_data, item_data, test, verbose)
