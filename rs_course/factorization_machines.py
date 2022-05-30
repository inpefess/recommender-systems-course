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
Factorization Machines with Vowpal Wabbit
=========================================
"""
import pandas as pd
import vowpalwabbit
from sklearn.metrics import roc_auc_score
from vowpalwabbit.dftovw import DFtoVW

from rs_course.utils import movielens_split


def _prepare_data(ratings: pd.DataFrame) -> None:
    ratings["user_id"] = (
        ratings["user_id"]
        .astype(pd.CategoricalDtype())  # type: ignore
        .cat.codes
        + 1
    )
    ratings["item_id"] = (
        ratings["item_id"]
        .astype(pd.CategoricalDtype())  # type: ignore
        .cat.codes
        + 1
    )
    ratings["label"] = 1
    ratings["label"] = ratings["label"].where(
        ratings["rating"] > 3, -1  # type: ignore
    )


def factorization_machines(
    ratings: pd.DataFrame,
    num_epochs: int,
    verbose: bool,
    seed: int,
    bit_precision: int,
) -> float:
    """
    factorization machine example

    >>> factorization_machines(
    ...     getfixture("test_dataset").ratings, 1, False, 0, 1  # noqa: F821
    ... )
    0.8333333333333333

    :param dataset_size: a size of MovieLens dataset to use
    :param num_epochs: number of epochs (``vw`` passes)
    :param verbose: an opposite of ``vw`` quiet
    :param seed: a random seed for testing
    :param bit_precision: a VW argument
    :returns:
    """
    _prepare_data(ratings)
    train, test, _ = movielens_split(ratings, 0.95)
    model = vowpalwabbit.Workspace(
        lrq="ui100 uu100 ii100",
        cache=True,
        learning_rate=0.01,
        holdout_period=5,
        loss_function="logistic",
        bit_precision=bit_precision,
        l2=0.001,
        quiet=not verbose,
        random_seed=seed,
    )
    vw_train = DFtoVW.from_colnames(
        df=train, y="label", x=["user_id", "item_id"]
    ).convert_df()
    vw_test = DFtoVW.from_colnames(
        df=test, y="label", x=["user_id", "item_id"]
    ).convert_df()
    for _ in range(num_epochs):
        for example in vw_train:
            model.learn(example)
    pred = []
    for example in vw_test:
        pred.append(model.predict(example))
    return roc_auc_score(test.label, pred)
