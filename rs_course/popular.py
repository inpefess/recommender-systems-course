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
# noqa: D205, D400
"""
Popular (not Personalised) Recommender
======================================

"""
import pandas as pd
from rs_metrics import hitrate

from rs_course.utils import movielens_split


def popular_recommender(ratings: pd.DataFrame, warm_users_only: bool) -> None:
    """
    Build a non-personalised recommender based on item popularity.

    >>> popular_recommender(
    ...     getfixture("test_dataset").ratings,  # noqa: F821
    ...     False
    ... )
    1.0

    :param ratings: a dataset of user-items intersection
    :param warm_users_only: test on only those users, who were in training set
    """
    train, test, _ = movielens_split(ratings, 0.95, warm_users_only)
    top_k = (
        train[["user_id", "item_id"]]
        .groupby("item_id")
        .count()
        .sort_values("user_id")[-10:]
        .reset_index()
        .drop(columns=["user_id"])
    )
    all_recs_pd = pd.merge(
        pd.DataFrame(test.user_id.unique(), columns=["user_id"]),
        top_k,
        how="cross",
    )
    print(hitrate(test, all_recs_pd))
