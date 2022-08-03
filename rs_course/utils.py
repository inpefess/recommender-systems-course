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
Useful Function for the Whole Course
====================================
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from implicit.nearest_neighbours import ItemItemRecommender
from rs_datasets import MovieLens
from rs_metrics import hitrate
from scipy.sparse import csr_matrix
from tqdm import tqdm


def pandas_to_scipy(
    pd_dataframe: pd.DataFrame,
    data_name: str,
    rows_name: str,
    cols_name: str,
    shape: Tuple[int, int],
) -> csr_matrix:
    """
    Transform pandas dataset with three columns to a sparse matrix.

    :param data_name: column name with values for the matrix cells
    :param rows_name: column name with row numbers of the cells
    :param cols_name: column name with column numbers of the cells
    :param shape: a pair (total number of rows, total number of columns)
    :returns: a ``csr_matrix``
    """
    return csr_matrix(
        (
            pd_dataframe[data_name].astype(float),
            (pd_dataframe[rows_name], pd_dataframe[cols_name]),
        ),
        shape=shape,
    )


def movielens_split(
    ratings: pd.DataFrame,
    train_percentage: float,
    warm_users_only: bool = False,
) -> Tuple[csr_matrix, pd.DataFrame, Tuple[int, int]]:
    """
    Split ``ratings`` dataset to train and test.

    :param ratings: ratings dataset from MovieLens
    :param train_percentage: percentage of data to put into training dataset
    :param warm_users_only: test on only those users, who were in training set
    :returns: sparse matrix for training and pandas dataset for testing
    """
    time_split = ratings.timestamp.quantile(train_percentage)  # type: ignore
    train = ratings[ratings.timestamp < time_split]
    test = ratings[ratings.timestamp >= time_split]
    if warm_users_only:
        warm_users = list(set(train.user_id).intersection(set(test.user_id)))
        final_test = test[test.user_id.isin(warm_users)]
    else:
        final_test = test
    return (
        train,
        final_test,
        (ratings.user_id.max() + 1, ratings.item_id.max() + 1),
    )


def evaluate_implicit_recommender(
    recommender: ItemItemRecommender,
    train: csr_matrix,
    test: pd.DataFrame,
    split_test_users_into: int,
    top_k: int,
) -> float:
    """
    Compute hit-rate for a recommender from ``implicit`` package.

    :param recommender: some recommender from ``implicit`` package
    :param train: sparse matrix of ratings
    :param test: pandas dataset of ratings for testing
    :param split_test_users_into: split ``test`` by users into several chunks
        to fit into memory
    :param top_k: how many items to recommend to each user
    :returns: hitrate@10
    """
    all_recs = []
    test_users_parts = np.array_split(
        test.user_id.unique(), split_test_users_into
    )
    for test_users_part in tqdm(test_users_parts):
        item_ids, weights = recommender.recommend(
            test_users_part, train[test_users_part], top_k
        )
        user_recs = pd.DataFrame(
            np.vstack([item_ids.reshape((1, -1)), weights.reshape((1, -1))]).T,
            columns=["item_id", "weight"],
        )
        user_recs["user_id"] = np.repeat(test_users_part, top_k)
        all_recs.append(user_recs)
    all_recs_pd = pd.concat(all_recs)
    return hitrate(test, all_recs_pd)


def get_sparse_item_features(
    movielens: MovieLens, ratings: pd.DataFrame
) -> Tuple[csr_matrix, pd.DataFrame]:
    """
    Extract item features from ``tags`` dataset.

    :param movielens: full MovieLens dataset
    :returns: sparse matrix and a `pandas` DataFrame of item features (tags)
    """
    genres_data = movielens.items[["item_id", "genres"]]
    genres_data["user_id"] = -1
    genres_data["tag"] = genres_data.genres.str.split("|")
    genres_tags = genres_data.explode("tag")[["item_id", "user_id", "tag"]]
    all_tags = pd.concat(
        [movielens.tags.drop(columns=["timestamp"]), genres_tags]
    )
    agg_tags = (
        all_tags[all_tags.item_id.isin(ratings.item_id)]
        .groupby(["item_id", "tag"])
        .count()
        .reset_index()
    )
    agg_tags["tag_id"] = agg_tags.tag.astype(  # type: ignore
        "category"  # type: ignore
    ).cat.codes  # type: ignore
    return (
        pandas_to_scipy(
            agg_tags,
            "user_id",
            "item_id",
            "tag_id",
            (ratings.item_id.max() + 1, agg_tags.tag_id.max() + 1),
        ),
        agg_tags,
    )


def enumerate_users_and_items(ratings: pd.DataFrame) -> None:
    """Inplace change of user and item IDs into numbers."""
    ratings["user_id"] = (
        ratings.user_id.astype("category").cat.codes + 1  # type: ignore
    )
    ratings["item_id"] = (
        ratings.item_id.astype("category").cat.codes + 1  # type: ignore
    )


def filter_users_and_items(
    ratings: pd.DataFrame,
    min_items_per_user: Optional[int],
    min_users_per_item: Optional[int],
) -> pd.DataFrame:
    """
    Leave only items with at least ``min_users_per_item`` users who rated them.

    (and only users who rated at least ``min_items_per_user``)

    :param min_items_per_user: if ``None`` then don't filter
    :param min_users_per_item: if ``None`` then don't filter
    :returns: filtered ratings dataset
    """
    filtered_ratings = ratings
    if min_items_per_user is not None:
        item_counts = ratings.groupby("user_id").count().item_id
        active_users = item_counts[
            item_counts >= min_items_per_user
        ].reset_index()["user_id"]
        filtered_ratings = filtered_ratings[
            filtered_ratings.user_id.isin(active_users)
        ]
    if min_users_per_item is not None:
        user_counts = ratings.groupby("item_id").count().user_id
        popular_items = user_counts[
            user_counts >= min_users_per_item
        ].reset_index()["item_id"]
        filtered_ratings = filtered_ratings[
            filtered_ratings.item_id.isin(popular_items)
        ]
    return filtered_ratings
